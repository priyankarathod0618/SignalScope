# SignalScope — Telling Real From Synthetic

Real-vs-AI-generated image classifier with a faithful, localised
explanation, generator attribution, degradation robustness, provenance
signals, image–caption consistency, and a deployable web app. Built for
SIH-2026 (LJIET, C-433).

**Scope note (read first):** this system classifies general synthetic
imagery (scenes, objects, product shots). It is **not** for face-swap
deepfakes of real people and does **not** adjudicate political or
real-world-event claims. Every verdict is presented as a likelihood
("likely AI-generated"), never an accusation.

## 1. What's built

| | Module | Status |
|---|---|---|
| Core | Real-vs-AI-generated classification + calibrated confidence | ✅ |
| A | Faithful explanation (Grad-CAM heat-map + grounded text) | ✅ |
| B | Generator attribution (GAN vs diffusion) | ✅ (heuristic fallback; trainable head included) |
| C | Robustness to degradation (JPEG, resize, screenshot-sim) | ✅ (evaluation script) |
| D | Provenance & metadata (C2PA / EXIF) | ✅ |
| E | Multimodal image + caption consistency (CLIP) | ✅ |
| F | Deployable interface (FastAPI + React drag-and-drop) | ✅ |
| G | Active-defence / adversarial failure analysis | ⚠️ analysis harness included, run and fill in `report/adversarial_findings.md` |

## 2. Setup & run (reproduce a prediction in under 10 minutes)

```bash
# 1. Clone and install
git clone <this-repo-url> && cd signalscope
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the data (core dataset, cited in Section 3)
# Download CIFAKE from Kaggle and unzip into data/train and data/test
# so the layout is data/train/REAL, data/train/FAKE, data/test/REAL, data/test/FAKE
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data --unzip

# 3. Train the core model (transfer learning, ResNet-50 by default)
python model/train.py --data-dir data/train --epochs 8 --backbone resnet50 \
    --out checkpoints/signalscope.pt
# Swap --backbone vit_b16 or efficientnet_b0 to try other encoders.

# 4. Run a single prediction (this is the organizers' scored interface)
python model/predict.py --checkpoint checkpoints/signalscope.pt --image path/to/img.jpg

# 5. Start the backend API
uvicorn backend.app:app --reload --port 8000

# 6. Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
# open http://localhost:5173
```

Organizers evaluating the held-out set should call `model/predict.py` (or
`backend/app.py`'s `/api/predict`) directly, per Section 4.1 — this
interface is never retrained or fine-tuned on that set.

## 3. Datasets used

- **Core training/validation:** [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
  (real CIFAR-10 photos vs. Stable-Diffusion-generated equivalents),
  MIT/open-licensed, ~120k balanced images.
- **Held-out test:** organizers' unseen-generator set — never trained on.
- **Optional extra training data** (cite if used): e.g.
  [GenImage](https://github.com/GenImage-Dataset/GenImage) for broader
  generator coverage. Add via `--extra-dirs`.

## 4. Reported metrics

Fill in after training (auto-written to `report/train_log.json` and the
checkpoint's metadata):

- Overall held-out AUC: `TBD`
- Unseen-generator-split AUC (primary): `TBD`
- Macro-F1: `TBD`
- Confusion matrix: `TBD`
- Accuracy & FPR at operating threshold 0.5: `TBD`

See `report/model_report.md` for the full one-page report and
`report/robustness_report.json` for the Module C degradation curve.

## 5. Architecture overview

```
Image (+ optional caption / metadata)
   → Pre-processing & Augmentation (torchvision transforms)
   → CV Detector (ResNet-50 / ViT-B/16 / EfficientNet-B0, transfer-learned)
   → Temperature-scaled calibrated verdict
   → Explainer (Grad-CAM heat-map + grounded, signal-based text — model/gradcam.py)
   → Auxiliary signals (attribution, provenance, multimodal — model/*.py)
   → Responsible UI ("likely AI-generated", never "certain" — frontend/)
```

Backbone: transfer-learned ResNet-50 (ImageNet-pretrained) by default —
chosen for training-time stability and speed within a hackathon window;
`model/train.py --backbone vit_b16` swaps in a ViT-B/16 for teams with more
compute/time, using the same pipeline.

Calibration: temperature scaling fit on the validation split only
(`model/calibration.py`), so confidence numbers mean what they say.

## 6. Known limitations

- The generator-attribution module (B) defaults to a frequency-spectrum
  **heuristic**, not a trained classifier, unless you provide per-generator
  sub-labels and train `attribution.AttributionHead` — see that file's
  docstring. Report attribution results as heuristic in any write-up.
- C2PA reading (Module D) degrades gracefully (`present: false`) if the
  `c2pa-python` native library isn't available in your environment.
- The multimodal consistency thresholds (Module E) are documented
  rules-of-thumb from typical CLIP similarity distributions, not a
  calibrated probability — do not present them as such.
- Robustness (Module C) is evaluated, not adversarially hardened; see
  Module G's harness for known failure modes once you've run it.

## 7. Demo video & deployed app

Link here once recorded (3–5 min, showing core + bonus modules per
Section 7.4): `TBD`

## 8. Repository structure

```
signalscope/
  README.md
  ORIGINALITY.md
  requirements.txt
  model/          # dataset, training, predict interface, explainability, bonus modules
  backend/         # FastAPI service
  frontend/        # React drag-and-drop UI
  report/          # model report, train log, robustness report
```
