"""
predict.py — the required core prediction interface (Section 4.1).

Organizers / judges call `predict_image(path_or_pil)` to score the held-out
set. This is the ONLY entry point that should be treated as the scored
core-task interface; keep its signature stable.

CLI:
    python model/predict.py --checkpoint checkpoints/signalscope.pt --image path/to/img.jpg
"""

import argparse
import json
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import build_transforms
from train import build_model
from calibration import apply_temperature

_CACHE = {}


def load_checkpoint(checkpoint_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path in _CACHE:
        return _CACHE[checkpoint_path]
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(ckpt["backbone"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    bundle = {"model": model, "temperature": ckpt.get("temperature", 1.0),
              "device": device, "classes": ckpt.get("classes", ["REAL", "FAKE"])}
    _CACHE[checkpoint_path] = bundle
    return bundle


def predict_image(image: Union[str, Image.Image], checkpoint_path: str = "checkpoints/signalscope.pt"):
    """Returns: {"label": "real"|"ai-generated", "confidence": float in [0,1],
                 "p_fake": float, "p_real": float}
    Confidence is the calibrated probability of the predicted class — this is
    the number reported as ROC-AUC's operating output and shown in the UI.
    """
    bundle = load_checkpoint(checkpoint_path)
    model, temperature, device = bundle["model"], bundle["temperature"], bundle["device"]

    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    tf = build_transforms(train=False)
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        logits = apply_temperature(logits, temperature)
        probs = F.softmax(logits, dim=1)[0]

    p_real, p_fake = float(probs[0]), float(probs[1])
    label = "ai-generated" if p_fake >= 0.5 else "real"
    confidence = max(p_real, p_fake)
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "p_fake": round(p_fake, 4),
        "p_real": round(p_real, 4),
        "presentation": f"Likely {label}" if confidence < 0.9 else f"Very likely {label}",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/signalscope.pt")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    result = predict_image(args.image, args.checkpoint)
    print(json.dumps(result, indent=2))
