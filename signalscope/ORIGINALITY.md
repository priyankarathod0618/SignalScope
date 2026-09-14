# Originality Declaration

All substantive code in this repository (`model/`, `backend/`, `frontend/`)
was written during the SIH-2026 development window (10–15 September) for
this challenge (SignalScope, C-433).

## Third-party code / libraries referenced

- **PyTorch / torchvision** — model backbone, training loop primitives (BSD).
- **timm** — ViT / EfficientNet backbone implementations (Apache-2.0).
- **pytorch-grad-cam** (jacobgil) — Grad-CAM implementation used in
  `model/gradcam.py`, called via its public API; no code copied verbatim.
- **open_clip** — CLIP model/weights for Module E image–text consistency.
- **piexif**, **c2pa-python** — EXIF and C2PA manifest parsing.
- **FastAPI**, **Vite**, **React** — application scaffolding.

## Datasets referenced

- CIFAKE (Bird & Lotfi, MIT/open license) — core training/validation data.
- Optionally GenImage — cite explicitly in your README if you add it to
  `--extra-dirs` during training.

## Notebooks / tutorials referenced

- None copied wholesale. The frequency-domain attribution heuristic in
  `model/attribution.py` is inspired by published findings on GAN
  upsampling artefacts (Zhang et al., "Detecting and Simulating Artifacts
  in GAN Fake Images", WIFS 2019) — implemented independently, not copied
  from a public notebook.

## AI coding assistance

Portions of this codebase were drafted with AI coding assistance. The
working system, its evaluation, and this declaration are what is
submitted for scoring, per Section 8.
