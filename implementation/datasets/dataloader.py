"""
dataloader.py
=============
Dataset loading with the paper-exact subsampling strategy.

Paper: "A Parallel Hybrid Quantum-Classical Convolutional Design Using
        Parameterized Quantum Circuits for Image Classification"
        Quantum Engineering (2026), article 6643049.

References
----------
- Table 1 (page 7)    : Dataset sizes and splits
- Section 4.2.1       : Class-balanced subsampling procedure
- DATASETS.md         : Full dataset documentation

Paper splits (Table 1)
----------------------
  MNIST         : 10,000 train (1,000/class)  /  2,000 test (200/class)
  Fashion-MNIST : 10,000 train (1,000/class)  /  2,000 test (200/class)
  Overhead-MNIST:  8,519 train (full dataset) /  1,065 test (full dataset)

The paper's subsampling was driven by quantum-simulator computational limits.
"""

from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Reproducibility seed
# ---------------------------------------------------------------------------
PAPER_SEED = 42   # Table 4, page 10


def _set_seed(seed: int = PAPER_SEED):
    random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Canonical transform: ToTensor only (pixels → [0,1])
# Paper Section 4.2 / EXPERIMENT_SETUP.md Section 6
# ---------------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),   # uint8 [0,255] → float32 [0,1]
])


# ---------------------------------------------------------------------------
# Balanced subsampling (Section 4.2.1)
# ---------------------------------------------------------------------------
def _balanced_subsample(dataset: Dataset,
                         samples_per_class: int,
                         seed: int = PAPER_SEED) -> Subset:
    """
    Randomly select `samples_per_class` images for each class.

    Parameters
    ----------
    dataset          : a PyTorch Dataset with (image, label) items
    samples_per_class: number of samples to keep per class
    seed             : random seed for reproducibility

    Returns
    -------
    Subset of `dataset` with balanced class distribution.
    """
    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        label = dataset[idx][1]
        class_indices.setdefault(label, []).append(idx)

    selected: list[int] = []
    for label in sorted(class_indices.keys()):
        pool = class_indices[label]
        k    = min(samples_per_class, len(pool))
        selected.extend(rng.sample(pool, k))

    return Subset(dataset, selected)


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------
def get_mnist(data_root: str = "data",
              batch_size: int = 32,
              seed: int = PAPER_SEED
              ) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST with paper-exact subsampling (Table 1):
      train: 10,000 images  (1,000 per class × 10 classes)
      test :  2,000 images  (  200 per class × 10 classes)

    Returns
    -------
    train_loader, test_loader
    """
    full_train = datasets.MNIST(root=data_root, train=True,  download=True, transform=TRANSFORM)
    full_test  = datasets.MNIST(root=data_root, train=False, download=True, transform=TRANSFORM)

    train_sub = _balanced_subsample(full_train, samples_per_class=1000, seed=seed)
    test_sub  = _balanced_subsample(full_test,  samples_per_class=200,  seed=seed)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_sub,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Fashion-MNIST
# ---------------------------------------------------------------------------
def get_fashion_mnist(data_root: str = "data",
                      batch_size: int = 32,
                      seed: int = PAPER_SEED
                      ) -> Tuple[DataLoader, DataLoader]:
    """
    Load Fashion-MNIST with paper-exact subsampling (Table 1):
      train: 10,000 images  (1,000 per class × 10 classes)
      test :  2,000 images  (  200 per class × 10 classes)
    """
    full_train = datasets.FashionMNIST(root=data_root, train=True,  download=True, transform=TRANSFORM)
    full_test  = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=TRANSFORM)

    train_sub = _balanced_subsample(full_train, samples_per_class=1000, seed=seed)
    test_sub  = _balanced_subsample(full_test,  samples_per_class=200,  seed=seed)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_sub,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Overhead-MNIST  (full dataset, no subsampling — Table 1)
#
# Overhead-MNIST is a remote-sensing 28×28 grayscale dataset with aerial
# categories (Car, Harbor, Helicopter, Oil_gas_field, Parking_lot, Plane,
# Runway_mark, Ship, Stadium, Storage_tank, ...).
# Repository: https://github.com/reveondivad/ov-mnist
#
# The dataset follows the MNIST binary file format, so we use a custom loader.
# ---------------------------------------------------------------------------

def _load_idx_images(path: str) -> torch.Tensor:
    """Load IDX3 image file (MNIST binary format)."""
    import struct, gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = torch.frombuffer(f.read(), dtype=torch.uint8)
    return data.view(n, 1, rows, cols).float() / 255.0


def _load_idx_labels(path: str) -> torch.Tensor:
    """Load IDX1 label file (MNIST binary format)."""
    import struct, gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = torch.frombuffer(f.read(), dtype=torch.uint8)
    return data.long()


class OverheadMNISTDataset(Dataset):
    """
    Overhead-MNIST dataset (MNIST-format binary files).

    Directory layout expected:
        overhead_data/
            train-images-idx3-ubyte   (or .gz)
            train-labels-idx1-ubyte   (or .gz)
            t10k-images-idx3-ubyte    (or .gz)
            t10k-labels-idx1-ubyte    (or .gz)

    Download from: https://github.com/reveondivad/ov-mnist
    """
    def __init__(self, data_dir: str, train: bool = True):
        prefix = "train" if train else "t10k"
        data_dir = Path(data_dir)
        # Try both .gz and plain binary
        for suffix in ["-images-idx3-ubyte.gz", "-images-idx3-ubyte"]:
            img_path = data_dir / f"{prefix}{suffix}"
            if img_path.exists():
                break
        for suffix in ["-labels-idx1-ubyte.gz", "-labels-idx1-ubyte"]:
            lbl_path = data_dir / f"{prefix}{suffix}"
            if lbl_path.exists():
                break

        self.images = _load_idx_images(str(img_path))
        self.labels = _load_idx_labels(str(lbl_path))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_overhead_mnist(data_dir: str = "data/overhead_mnist",
                       batch_size: int = 32
                       ) -> Tuple[DataLoader, DataLoader]:
    """
    Load full Overhead-MNIST dataset (Table 1):
      train: 8,519 images  (full dataset, no subsampling)
      test : 1,065 images  (full dataset, no subsampling)

    Parameters
    ----------
    data_dir : directory containing the MNIST-format binary files.
    """
    data_dir = Path(data_dir)

    # The official ov-mnist archive contains JPEGs arranged by class rather
    # than IDX files: overhead/{training,testing}/{class_name}/*.jpg.
    # Supporting this layout lets the project use the downloaded source
    # directly, while retaining IDX support for compatible mirrors.
    image_root = data_dir / "overhead"
    train_images = image_root / "training"
    test_images = image_root / "testing"
    if train_images.is_dir() and test_images.is_dir():
        overhead_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        train_ds = datasets.ImageFolder(train_images, transform=overhead_transform)
        test_ds = datasets.ImageFolder(test_images, transform=overhead_transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, test_loader

    required = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    if not data_dir.exists() or not all(
        (data_dir / name).exists() or (data_dir / f"{name}.gz").exists()
        for name in required
    ):
        raise FileNotFoundError(
            f"Overhead-MNIST IDX files were not found in '{data_dir}'. "
            "Pass the directory containing train-/t10k- image and label files."
        )
    train_ds = OverheadMNISTDataset(data_dir, train=True)
    test_ds  = OverheadMNISTDataset(data_dir, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "mnist"         : get_mnist,
    "fashion_mnist" : get_fashion_mnist,
    "overhead_mnist": get_overhead_mnist,
}


def get_dataloaders(dataset_name: str,
                    data_root: str = "data",
                    batch_size: int = 32,
                    seed: int = PAPER_SEED):
    """
    Unified entry point.  dataset_name ∈ {"mnist", "fashion_mnist", "overhead_mnist"}.
    """
    name = dataset_name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Choose from: {list(DATASET_REGISTRY.keys())}")
    if name == "overhead_mnist":
        return get_overhead_mnist(
            data_dir=str(Path(data_root) / "overhead_mnist"), batch_size=batch_size
        )
    return DATASET_REGISTRY[name](data_root=data_root,
                                   batch_size=batch_size,
                                   seed=seed)
