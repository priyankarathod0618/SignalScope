"""
attribution.py — Module B: generator attribution (GAN vs diffusion vs
specific model family), scored as its own multi-class metric per the PDF.

Two supported modes:

1. Trained head (preferred): if your training data has family sub-labels
   (folder names like FAKE_gan/, FAKE_sd/, FAKE_midjourney/), train a small
   auxiliary classifier `train_attribution_head()` on frozen backbone
   features from the core model, then use `predict_family()`.

2. Heuristic fallback (works with zero extra labels): frequency-domain
   artefact statistics differ systematically between GAN and diffusion
   outputs (GANs: strong high-frequency grid/checkerboard periodicity from
   transposed-conv upsampling; diffusion: smoother spectra with distinct
   denoising-step artefacts). This is a real, citable signal
   (Zhang et al., "Detecting and Simulating Artifacts in GAN Fake Images"),
   used here only as a coarse fallback — report it as heuristic, not
   ground truth, in your model report's Limitations section.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.fft import fft2, fftshift


def _radial_spectrum(gray: np.ndarray, n_bins: int = 32) -> np.ndarray:
    f = fftshift(fft2(torch.from_numpy(gray.astype(np.float32))))
    mag = torch.log1p(f.abs()).numpy()
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max = r.max()
    bins = np.linspace(0, r_max, n_bins + 1)
    radial = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        radial[i] = mag[mask].mean() if mask.any() else 0.0
    return radial / (radial.sum() + 1e-8)


def heuristic_family(image_path: str) -> dict:
    """Fallback attribution using radial frequency-spectrum shape. Reports
    a confidence and is explicitly labeled heuristic, not a trained metric."""
    img = Image.open(image_path).convert("L").resize((256, 256))
    gray = np.array(img)
    spectrum = _radial_spectrum(gray)
    # GAN outputs tend to show a secondary high-frequency energy bump from
    # checkerboard/transposed-conv upsampling; diffusion spectra decay more
    # monotonically. This threshold is a coarse, documented heuristic.
    high_freq_bump = float(spectrum[-6:].sum() / (spectrum[:6].sum() + 1e-8))
    family = "GAN-like" if high_freq_bump > 0.55 else "diffusion-like"
    confidence = min(0.5 + abs(high_freq_bump - 0.55), 0.85)
    return {
        "family": family,
        "confidence": round(confidence, 2),
        "method": "heuristic (radial frequency spectrum) -- not a trained classifier",
        "high_freq_ratio": round(high_freq_bump, 3),
    }


class AttributionHead(nn.Module):
    """Small trained head on top of frozen backbone features, used when
    sub-labeled training data (per-generator folders) is available."""
    def __init__(self, in_features: int, num_families: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_families)
        )

    def forward(self, feats):
        return self.net(feats)


def predict_family(image_path: str, head: AttributionHead, feature_extractor,
                    family_names: list, device: str = "cpu") -> dict:
    """Use once you've trained `head` on backbone features + family labels.
    See README 'Training Module B' for the expected folder-label convention."""
    from dataset import build_transforms
    img = Image.open(image_path).convert("RGB")
    x = build_transforms(train=False)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feature_extractor(x)
        logits = head(feats)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return {"family": family_names[idx], "confidence": round(float(probs[idx]), 3),
            "method": "trained attribution head",
            "distribution": {n: round(float(p), 3) for n, p in zip(family_names, probs)}}
