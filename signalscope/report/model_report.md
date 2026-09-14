# SignalScope — Model Report

*Fill in every `TBD` after running `model/train.py` and the organizers'
held-out evaluation. Keep this to one page.*

| Field | Detail |
|---|---|
| **Task** | Binary real-vs-AI-generated classification (core). Bonus modules attempted: A (explanation), B (attribution), C (robustness), D (provenance), E (multimodal), F (deployable UI). |
| **Data & split** | Core: CIFAKE (~TBD images), 85/15 train/val split, seed 42 (`model/dataset.py`). Held-out test: organizers' set, never trained on. Extra public data added: `TBD` (cite). |
| **Model / approach** | Backbone: `TBD` (ResNet-50 / ViT-B/16 / EfficientNet-B0), ImageNet-pretrained, fine-tuned `TBD` epochs, AdamW lr=3e-4, cosine schedule. Augmentation: horizontal flip, mild blur/color jitter. Calibration: temperature scaling on val split (`model/calibration.py`), T=`TBD`. |
| **Metric & result** | Overall held-out AUC: `TBD`. **Unseen-generator-split AUC (primary): `TBD`.** Macro-F1: `TBD`. Accuracy @ threshold 0.5: `TBD`. FPR @ threshold 0.5: `TBD`. Confusion matrix: `TBD`. |
| **Baseline** | Provided baseline AUC: `TBD`. Our overall / unseen-split delta vs. baseline: `TBD`. |
| **Limitations** | Which generators break the classifier: `TBD` (fill in from unseen-split error analysis). Which degradations break it: see `report/robustness_report.json`. Attribution (Module B) is heuristic unless a per-generator-labeled head was trained. |

## Explanation faithfulness (Module A) self-check

Before submission, sample ~15 predictions and manually verify against the
rubric in the brief's Section 4.3:

- **Correctness** — does the highlighted region correspond to a real
  artefact you can independently see?
- **Localisation** — is the heat-map's `area_frac` small/moderate, not
  covering the whole frame? (`model/gradcam.py` reports this per image.)
- **Usefulness** — would a non-expert know what to do with the sentence?
- **No over-claiming** — does it hedge ("likely", "uncertain") rather than
  assert certainty, and does it avoid any claim about a real person/event?

Record the sample results here: `TBD`.
