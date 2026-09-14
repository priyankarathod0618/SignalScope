"""
dataset.py
Loads the CIFAKE-style real-vs-AI-generated dataset for training/validation,
and (separately) any organizer-provided held-out test set at inference time.

Expected directory layout after downloading CIFAKE
(https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images):

    data/
      train/
        REAL/*.jpg
        FAKE/*.jpg
      test/
        REAL/*.jpg
        FAKE/*.jpg

If you add extra public datasets (e.g. GenImage) for training, drop them into
train/REAL or train/FAKE following the same convention, or pass multiple
root dirs to build_dataloaders(extra_train_dirs=[...]).
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMG_SIZE = 224
LABEL_REAL = 0
LABEL_FAKE = 1


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            # Randomly re-JPEG-compress-ish via random resize/quality jitter
            # (real degradation augmentation lives in model/robustness.py and
            # can be composed in here for Module C training-time hardening).
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@dataclass
class DataConfig:
    train_dir: str = "data/train"          # expects REAL/ and FAKE/ subfolders
    held_out_dir: Optional[str] = None      # organizers' unseen-generator set (eval only)
    extra_train_dirs: Optional[List[str]] = None
    val_fraction: float = 0.15
    batch_size: int = 64
    # Keep loading in the main process by default.  This is reliable on
    # Windows (and with Python 3.14), where the locally-defined wrapper
    # dataset cannot be pickled by multiprocessing workers.
    num_workers: int = 0
    seed: int = 42


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Builds train/val loaders from the provided CIFAKE-style train_dir.
    NOTE: class_to_idx from ImageFolder is alphabetical -> {'FAKE':0,'REAL':1}
    or {'REAL':0,'FAKE':1} depending on folder names; we normalize labels
    below so FAKE == LABEL_FAKE (1) and REAL == LABEL_REAL (0) regardless.
    """
    base = ImageFolder(cfg.train_dir, transform=build_transforms(train=True))
    class_to_idx = base.class_to_idx
    idx_remap = {}
    for name, idx in class_to_idx.items():
        idx_remap[idx] = LABEL_FAKE if name.upper().startswith("FAKE") else LABEL_REAL

    class RemappedFolder(Dataset):
        def __init__(self, folder_dataset, remap, train):
            self.ds = folder_dataset
            self.remap = remap
            # re-instantiate with correct transform if val (no augmentation)
            self.tf = build_transforms(train=train)

        def __len__(self):
            return len(self.ds.samples)

        def __getitem__(self, i):
            path, orig_label = self.ds.samples[i]
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img = self.tf(img)
            return img, self.remap[orig_label]

    full = RemappedFolder(base, idx_remap, train=True)

    if cfg.extra_train_dirs:
        extras = []
        for d in cfg.extra_train_dirs:
            extra_base = ImageFolder(d)
            extra_remap = {
                idx: (LABEL_FAKE if name.upper().startswith("FAKE") else LABEL_REAL)
                for name, idx in extra_base.class_to_idx.items()
            }
            extras.append(RemappedFolder(extra_base, extra_remap, train=True))
        full = ConcatDataset([full] + extras)

    n_val = int(len(full) * cfg.val_fraction)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, ["REAL", "FAKE"]


def build_heldout_loader(held_out_dir: str, batch_size: int = 64) -> DataLoader:
    """For organizers' held-out set at judging time. Same REAL/FAKE convention.
    Splits are NEVER trained on -- see model/train.py which never touches this."""
    base = ImageFolder(held_out_dir, transform=build_transforms(train=False))
    return DataLoader(base, batch_size=batch_size, shuffle=False, num_workers=0)
