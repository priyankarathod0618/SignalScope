# SignalScope — Telling Real From Synthetic

<p align="center">
  <strong>AI-Powered Image Authenticity & Media Forensics</strong>
</p>

<p align="center">
  Detect AI-generated images, understand why a model made its decision, and inspect additional signals such as generator attribution, provenance, and image–caption consistency.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square\&logo=pytorch\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square\&logo=fastapi\&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square\&logo=react\&logoColor=black)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-ML-6C63FF?style=flat-square)

</p>

> Built for **Smart India Hackathon 2026 — LJIET, C-433**

---

## Overview

**SignalScope** is a computer-vision based media forensics system designed to distinguish **real images from AI-generated synthetic images**.

Instead of returning only a binary prediction, SignalScope combines classification with **explainability, confidence calibration, generator attribution, robustness analysis, provenance signals, and multimodal consistency checks**.

The goal is to make AI-image detection more **interpretable, transparent, and useful for real-world analysis**.

---

## What SignalScope Does

| Capability                 | Description                                                                        |
| -------------------------- | ---------------------------------------------------------------------------------- |
| **AI Image Detection**     | Classifies images as real or likely AI-generated                                   |
| **Confidence Calibration** | Produces calibrated confidence rather than raw model scores                        |
| **Visual Explanation**     | Grad-CAM heatmaps show where the model focused                                     |
| **Generator Attribution**  | Provides GAN vs diffusion attribution signals                                      |
| **Robustness Analysis**    | Tests predictions under JPEG compression, resizing and screenshot-like degradation |
| **Provenance Analysis**    | Inspects C2PA and EXIF metadata when available                                     |
| **Multimodal Analysis**    | Checks image–caption consistency using CLIP                                        |
| **Web Interface**          | React-based interface connected to a FastAPI backend                               |

---

## Architecture

```text
                    ┌─────────────────────┐
                    │   Image + Caption    │
                    │    + Metadata        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Pre-processing &     │
                    │ Image Augmentation   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Vision Detector   │
                    │ ResNet-50 / ViT /   │
                    │   EfficientNet      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Calibrated Verdict  │
                    │ Real / Likely AI    │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
      ┌────────────┐   ┌─────────────┐   ┌──────────────┐
      │ Grad-CAM   │   │ Attribution │   │ Provenance / │
      │ Explanation│   │ & Robustness│   │ Multimodal   │
      └────────────┘   └─────────────┘   └──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     React UI        │
                    │   + FastAPI API     │
                    └─────────────────────┘
```

---

## Tech Stack

| Layer                     | Technology                           |
| ------------------------- | ------------------------------------ |
| **Frontend**              | React, JavaScript, CSS               |
| **Backend**               | FastAPI, Uvicorn                     |
| **Machine Learning**      | PyTorch, torchvision, timm           |
| **Computer Vision**       | ResNet-50, ViT-B/16, EfficientNet-B0 |
| **Explainability**        | Grad-CAM                             |
| **Multimodal Analysis**   | CLIP                                 |
| **Evaluation**            | scikit-learn                         |
| **Metadata / Provenance** | C2PA, EXIF                           |
| **Dataset**               | CIFAKE                               |

---

## Project Structure

```text
SignalScope/
├── model/              # Dataset, training, prediction & ML modules
├── backend/             # FastAPI application and API endpoints
├── frontend/            # React web interface
├── report/              # Model, robustness & analysis reports
├── requirements.txt     # Python dependencies
├── ORIGINALITY.md       # Project originality information
└── README.md
```

---

## Dataset

SignalScope uses **CIFAKE** as its core training dataset.

* **100,000 training images**

  * 50,000 REAL
  * 50,000 FAKE
* **20,000 test images**

  * 10,000 REAL
  * 10,000 FAKE
* Real images are derived from CIFAR-10.
* Synthetic images are generated using Stable Diffusion.

Dataset:
[CIFAKE — Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

The dataset is **not included in this repository**.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/priyankarathod0618/SignalScope.git
cd SignalScope/signalscope
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare CIFAKE

Place the dataset in:

```text
data/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

### 5. Train the model

```bash
python model/train.py \
  --data-dir data/train \
  --epochs 8 \
  --backbone resnet50 \
  --out checkpoints/signalscope.pt
```

Other supported backbones:

```text
resnet50
vit_b16
efficientnet_b0
```

### 6. Run a prediction

```bash
python model/predict.py \
  --checkpoint checkpoints/signalscope.pt \
  --image path/to/image.jpg
```

### 7. Start the backend

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

### 8. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:5173
```

## Core Features

- **AI Image Detection** — classifies images as **real or likely AI-generated**, with calibrated confidence rather than relying only on raw model scores.

- **Visual Explanation** — generates **Grad-CAM heatmaps** showing where the model focused, making the prediction more interpretable.

- **Generator Attribution** — provides signals for **GAN vs diffusion-generated imagery**, with heuristic fallback when a trained attribution head is unavailable.

- **Robustness Analysis** — evaluates predictions under **JPEG compression, resizing, and screenshot-like degradation** to understand how stable the detector remains.

- **Provenance Analysis** — inspects available **C2PA and EXIF metadata** to provide additional provenance signals alongside the model prediction.

- **Multimodal Analysis** — uses **CLIP** to check image–caption consistency as an additional signal.

- **Deployable Web Interface** — a **React frontend connected to a FastAPI backend**, providing an accessible interface for running the analysis pipeline.

---

## Core ML Pipeline

SignalScope follows a multi-stage analysis pipeline:

**Image → Preprocessing → Vision Model → Calibrated Prediction → Explanation → Additional Forensic Signals**

### Classification

The default detector uses a **transfer-learned ResNet-50** model.

### Explainability

**Grad-CAM** generates a visual heatmap showing the regions that influenced the prediction.

### Calibration

Temperature scaling is applied using the validation split so that confidence scores are more meaningful.

### Additional Signals

SignalScope can additionally inspect:

* Generator characteristics
* Image degradation robustness
* C2PA provenance
* EXIF metadata
* Image–caption consistency

---

## Model Evaluation

The core model achieved the following validation performance during training:

| Metric | Result |
| --- | ---: |
| Validation AUC | **0.9935** |
| Validation Macro-F1 | **0.9519** |

The validation split is held out from training but follows the training data
distribution. Final organizers' held-out performance, including the
unseen-generator split, is evaluated separately through `model/predict.py`.

Detailed training results are stored in `report/train_log.json`.

---

## Responsible AI

SignalScope is designed as a **forensic assistance tool**, not an absolute authority.

### Scope

The system focuses on **general synthetic imagery**, such as:

* Scenes
* Objects
* Product images
* Generated visual content

It is **not designed for face-swap deepfakes of real people** and does not determine whether political or real-world-event claims are true.

Predictions are therefore expressed as:

> **Likely AI-generated**

rather than absolute statements such as:

> **Definitely fake**

---

## Known Limitations

* Generator attribution currently uses a heuristic fallback unless a dedicated attribution classifier is trained.
* C2PA functionality depends on the availability of the required native library.
* CLIP consistency thresholds are heuristic rather than calibrated probabilities.
* Robustness testing evaluates known degradations but does not make the model adversarially hardened.
* Performance may vary for generators and image distributions not represented in the training data.

---

## Project Status

| Component                 | Status                    |
| ------------------------- | ------------------------- |
| Core image classification | 🟢 Implemented  |
| Grad-CAM explanation      | 🟢 Implemented            |
| Generator attribution     | 🟢 Implemented            |
| Robustness evaluation     | 🟢 Implemented            |
| C2PA / EXIF analysis      | 🟢 Implemented            |
| CLIP multimodal analysis  | 🟢 Implemented            |
| FastAPI backend           | 🟢 Implemented            |
| React frontend            | 🟢 Implemented            |
| Final benchmark metrics   | 🟢 Implemented |

---

## SIH 2026

**Problem:** Telling Real From Synthetic in the Age of Generative Media

**Team:** TrustUsBro
**Institution:** L. J. Institute of Engineering & Technology
**Problem Code:** C-433
**Domain:** AI / Media Forensics / Trust & Safety

---

## License

Academic project developed for **Smart India Hackathon 2026**.

---
