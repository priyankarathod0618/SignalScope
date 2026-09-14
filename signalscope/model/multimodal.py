"""
multimodal.py — Module E: image + caption/claim consistency as an added
authenticity signal (generic captions only -- never political/real-event
claims, per Section 1 scope rules).

Uses open_clip to score image-text similarity. Low similarity flags a
possible mismatch (e.g. a fabricated product listing whose caption doesn't
match what's actually shown) -- an authenticity signal independent of
whether the pixels themselves are synthetic.
"""

from typing import Optional

import torch
import open_clip
from PIL import Image

_MODEL_CACHE = {}


def _load_clip(device: str):
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device]
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    _MODEL_CACHE[device] = (model, preprocess, tokenizer)
    return _MODEL_CACHE[device]


def image_text_consistency(image_path: str, caption: str,
                            device: Optional[str] = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = _load_clip(device)

    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer([caption]).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(img)
        txt_feat = model.encode_text(text)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        similarity = float((img_feat @ txt_feat.T).item())

    # CLIP cosine similarities for genuinely matching image/caption pairs
    # typically fall ~0.25-0.35; well below ~0.18 usually indicates mismatch.
    # These are documented rule-of-thumb bands, not calibrated probabilities --
    # report them as such in the model report.
    if similarity >= 0.25:
        verdict = "consistent"
    elif similarity >= 0.18:
        verdict = "uncertain"
    else:
        verdict = "likely mismatched"

    return {
        "similarity": round(similarity, 4),
        "verdict": verdict,
        "note": ("Rule-of-thumb CLIP similarity bands, not a calibrated "
                 "probability. Caption must be a generic description, not a "
                 "claim about a real event or person (see scope rules)."),
    }
