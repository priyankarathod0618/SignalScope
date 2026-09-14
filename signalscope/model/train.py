"""
train.py — SignalScope core task trainer.

Transfer-learning binary classifier (REAL vs AI-generated) with an
auxiliary generator-attribution head (Module B) trained when
attribution labels are available in the folder names
(e.g. FAKE_sd, FAKE_midjourney, FAKE_gan -> family label parsed from suffix).

Usage:
    python model/train.py --data-dir data/train --epochs 8 --backbone resnet50
    python model/train.py --data-dir data/train --epochs 8 --backbone vit_b16

Outputs:
    checkpoints/signalscope_<backbone>.pt   (weights + calibration temperature)
    report/train_log.json                   (loss/AUC curves for the report)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter  # optional, safe to remove

from dataset import DataConfig, build_dataloaders
from calibration import fit_temperature


def build_model(backbone: str, num_classes: int = 2):
    if backbone in ("resnet18", "resnet50"):
        import torchvision.models as tvm
        ctor = tvm.resnet50 if backbone == "resnet50" else tvm.resnet18
        weights = "IMAGENET1K_V2" if backbone == "resnet50" else "IMAGENET1K_V1"
        m = ctor(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if backbone.startswith("vit"):
        import timm
        m = timm.create_model("vit_base_patch16_224", pretrained=True,
                               num_classes=num_classes)
        return m
    if backbone.startswith("efficientnet"):
        import timm
        m = timm.create_model("efficientnet_b0", pretrained=True,
                               num_classes=num_classes)
        return m
    raise ValueError(f"Unknown backbone {backbone}")


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds).tolist()
    return {"auc": float(auc), "macro_f1": float(f1), "confusion_matrix": cm,
            "probs": probs, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--extra-dirs", nargs="*", default=None,
                     help="extra PUBLIC datasets, e.g. GenImage, cite in README")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", default="resnet50",
                     choices=["resnet18", "resnet50", "vit_b16", "efficientnet_b0"])
    ap.add_argument("--out", default="checkpoints/signalscope.pt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("report", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} backbone={args.backbone}")

    cfg = DataConfig(train_dir=args.data_dir, extra_train_dirs=args.extra_dirs,
                      batch_size=args.batch_size)
    train_loader, val_loader, classes = build_dataloaders(cfg)

    model = build_model(args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        sched.step()
        train_loss = running_loss / len(train_loader.dataset)

        val_metrics = evaluate(model, val_loader, device)
        print(f"[epoch {epoch+1}/{args.epochs}] loss={train_loss:.4f} "
              f"val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
              f"({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                         "val_auc": val_metrics["auc"],
                         "val_macro_f1": val_metrics["macro_f1"]})

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            # Fit temperature scaling for honest, calibrated confidences
            temperature = fit_temperature(model, val_loader, device)
            torch.save({
                "model_state": model.state_dict(),
                "backbone": args.backbone,
                "temperature": temperature,
                "val_auc": val_metrics["auc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "confusion_matrix": val_metrics["confusion_matrix"],
                "classes": classes,
            }, args.out)
            print(f"  -> saved new best checkpoint (AUC={best_auc:.4f}, T={temperature:.3f})")

    with open("report/train_log.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[train] done. Best val AUC:", best_auc)
    print("IMPORTANT: this is validation AUC on held-out *training-distribution* "
          "data only. The scored metric is the organizers' held-out set "
          "(including the unseen-generator split) via model/predict.py — "
          "never train on that set.")


if __name__ == "__main__":
    main()
