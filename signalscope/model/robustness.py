"""
robustness.py — Module C: robustness to degradation (JPEG re-compression,
resize, screenshot simulation, light edits) with a degradation-vs-accuracy
report as required by the PDF.
"""

import io
import json
import os
from typing import List

import numpy as np
from PIL import Image, ImageEnhance
from sklearn.metrics import roc_auc_score

from predict import predict_image


def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_down_up(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return small.resize((w, h))


def simulate_screenshot(img: Image.Image) -> Image.Image:
    """Approximates a phone-screenshot-of-a-screenshot: downscale, mild
    JPEG loss, and a slight sharpness/contrast shift from re-rendering."""
    img = resize_down_up(img, 0.6)
    img = jpeg_recompress(img, quality=60)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    img = ImageEnhance.Contrast(img).enhance(1.05)
    return img


DEGRADATIONS = {
    "clean": lambda im: im,
    "jpeg_q30": lambda im: jpeg_recompress(im, 30),
    "jpeg_q60": lambda im: jpeg_recompress(im, 60),
    "resize_0.5x": lambda im: resize_down_up(im, 0.5),
    "resize_0.25x": lambda im: resize_down_up(im, 0.25),
    "screenshot_sim": simulate_screenshot,
}


def evaluate_robustness(image_label_pairs: List[tuple],
                         checkpoint_path: str = "checkpoints/signalscope.pt",
                         out_json: str = "report/robustness_report.json") -> dict:
    """image_label_pairs: list of (path, true_label) with true_label in {0,1},
    0=real, 1=ai-generated -- use a held-out sample, never the training set."""
    results = {}
    for name, fn in DEGRADATIONS.items():
        probs, labels = [], []
        for path, label in image_label_pairs:
            img = Image.open(path).convert("RGB")
            degraded = fn(img)
            tmp_path = "/tmp/_degraded_tmp.jpg"
            degraded.save(tmp_path)
            pred = predict_image(tmp_path, checkpoint_path)
            probs.append(pred["p_fake"])
            labels.append(label)
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
        results[name] = {"auc": round(float(auc), 4), "n": len(labels)}

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                     help="CSV/JSON with columns: path,label (0=real,1=fake)")
    ap.add_argument("--checkpoint", default="checkpoints/signalscope.pt")
    args = ap.parse_args()

    import csv
    pairs = []
    with open(args.manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["path"], int(row["label"])))

    print(json.dumps(evaluate_robustness(pairs, args.checkpoint), indent=2))
