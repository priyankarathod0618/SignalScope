"""
adversarial.py — Module G: how easily is the detector fooled, and what
mitigations help. Runs simple, honest attacks (no exotic gradient-based
adversarial patches required by the brief) and reports flip rates.

Usage:
    python model/adversarial.py --manifest data/sample_manifest.csv \
        --checkpoint checkpoints/signalscope.pt
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageFilter

from predict import predict_image


def add_noise(img: Image.Image, sigma: float = 8.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def blur(img: Image.Image, radius: float = 1.2) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def mild_crop_pad(img: Image.Image, frac: float = 0.05) -> Image.Image:
    w, h = img.size
    dx, dy = int(w * frac), int(h * frac)
    cropped = img.crop((dx, dy, w - dx, h - dy))
    return cropped.resize((w, h))


ATTACKS = {
    "gaussian_noise": add_noise,
    "gaussian_blur": blur,
    "crop_pad_5pct": mild_crop_pad,
}


def run_attacks(image_label_pairs, checkpoint_path="checkpoints/signalscope.pt",
                 out_json="report/adversarial_findings.json"):
    findings = {}
    for name, attack_fn in ATTACKS.items():
        flips, total = 0, 0
        for path, true_label in image_label_pairs:
            img = Image.open(path).convert("RGB")
            base_pred = predict_image(path, checkpoint_path)
            base_label = 1 if base_pred["label"] == "ai-generated" else 0

            attacked = attack_fn(img)
            tmp = "/tmp/_adv_tmp.jpg"
            attacked.save(tmp)
            new_pred = predict_image(tmp, checkpoint_path)
            new_label = 1 if new_pred["label"] == "ai-generated" else 0

            if new_label != base_label:
                flips += 1
            total += 1
        findings[name] = {
            "flip_rate": round(flips / max(total, 1), 3),
            "n": total,
        }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(findings, f, indent=2)

    # Honest, human-readable mitigation notes -- fill in after inspecting results
    mitigations = {
        "gaussian_noise": ("If flip_rate is high, add noise augmentation "
                            "during training (model/dataset.py transforms)."),
        "gaussian_blur": ("High flip rate suggests over-reliance on "
                           "high-frequency artefacts alone; combine with "
                           "lower-frequency cues or blur-augment training."),
        "crop_pad_5pct": ("High flip rate suggests the model is sensitive "
                           "to exact framing; add random-crop augmentation."),
    }
    findings["_mitigation_notes"] = mitigations
    with open(out_json, "w") as f:
        json.dump(findings, f, indent=2)
    return findings


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                     help="CSV with columns: path,label (0=real,1=fake)")
    ap.add_argument("--checkpoint", default="checkpoints/signalscope.pt")
    args = ap.parse_args()

    import csv
    pairs = []
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            pairs.append((row["path"], int(row["label"])))

    print(json.dumps(run_attacks(pairs, args.checkpoint), indent=2))
