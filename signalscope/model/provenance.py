"""
provenance.py — Module D: read C2PA Content Credentials and EXIF metadata,
and combine that evidence with the model's visual verdict.

Design: metadata is treated as a *prior*, not an override. A visually
convincing verdict should never be silently flipped by possibly-stripped
or possibly-forged metadata -- images are routinely re-saved without EXIF,
and C2PA manifests can be absent for legitimate real photos. We only ever
nudge the confidence, and we always say so explicitly in the output so the
combination is transparent to the judge/user (Section 3.2 Module D
requirement: "explain how you combine metadata evidence with the model's
visual verdict").
"""

import json
from typing import Optional

import piexif
from PIL import Image

try:
    import c2pa  # optional; not all environments have the native library
    _HAS_C2PA = True
except ImportError:
    _HAS_C2PA = False


def read_exif(image_path: str) -> dict:
    try:
        exif_dict = piexif.load(image_path)
    except Exception:
        return {"present": False}

    ifd0 = exif_dict.get("0th", {})
    software = ifd0.get(piexif.ImageIFD.Software)
    make = ifd0.get(piexif.ImageIFD.Make)
    has_camera_tags = bool(make) or (piexif.ExifIFD.LensModel in exif_dict.get("Exif", {}))
    return {
        "present": bool(exif_dict.get("0th")) or bool(exif_dict.get("Exif")),
        "software": software.decode(errors="ignore") if isinstance(software, bytes) else software,
        "camera_make": make.decode(errors="ignore") if isinstance(make, bytes) else make,
        "has_camera_tags": has_camera_tags,
    }


def read_c2pa(image_path: str) -> dict:
    if not _HAS_C2PA:
        return {"present": False, "note": "c2pa-python not installed in this environment"}
    try:
        manifest = c2pa.Reader(image_path).json()
        data = json.loads(manifest)
        active = data.get("active_manifest")
        claim_generator = None
        if active and "manifests" in data:
            claim_generator = data["manifests"].get(active, {}).get("claim_generator")
        return {"present": bool(active), "claim_generator": claim_generator, "raw": data}
    except Exception as e:
        return {"present": False, "error": str(e)}


def combine_with_visual_verdict(visual_p_fake: float, image_path: str) -> dict:
    exif = read_exif(image_path)
    c2pa_info = read_c2pa(image_path)

    adjustment = 0.0
    notes = []

    if c2pa_info.get("present"):
        gen = (c2pa_info.get("claim_generator") or "").lower()
        if any(k in gen for k in ("diffusion", "dall-e", "midjourney", "firefly", "stable")):
            adjustment += 0.15
            notes.append("C2PA manifest names a generative-AI tool as claim generator.")
        else:
            adjustment -= 0.10
            notes.append("C2PA manifest present with a non-generative claim generator "
                          "(weak evidence toward 'real', not conclusive).")
    else:
        notes.append("No C2PA manifest found -- common for both real and synthetic "
                      "images since adoption is partial; treated as neutral.")

    if exif.get("present") and exif.get("has_camera_tags"):
        adjustment -= 0.10
        notes.append("EXIF contains camera make/lens tags, weakly consistent with a "
                      "real capture (metadata can still be stripped or forged).")
    elif not exif.get("present"):
        notes.append("No EXIF found -- common for both re-saved real photos and "
                      "synthetic images; treated as neutral, not evidence of fakery.")

    combined = min(max(visual_p_fake + adjustment, 0.0), 1.0)
    return {
        "visual_p_fake": round(visual_p_fake, 3),
        "combined_p_fake": round(combined, 3),
        "adjustment_applied": round(adjustment, 3),
        "exif": exif,
        "c2pa": {k: v for k, v in c2pa_info.items() if k != "raw"},
        "reasoning": notes,
    }
