"""
plotting.py
===========
Reproduce Figures 6–8 (accuracy/loss curves) and confusion matrices.

Paper: "A Parallel Hybrid Quantum-Classical Convolutional Design Using
        Parameterized Quantum Circuits for Image Classification"
        Quantum Engineering (2026), article 6643049.
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List


# ---------------------------------------------------------------------------
# 1. Training curve plots  (Figures 6, 7, 8 in the paper)
# ---------------------------------------------------------------------------
def plot_training_curves(histories: Dict[str, dict],
                         metric: str = "acc",
                         title: str = "",
                         save_path: Optional[str] = None):
    """
    Plot train/test accuracy or loss curves for multiple models.

    Parameters
    ----------
    histories  : {model_name: history_dict}  — output of trainer.train()
    metric     : "acc" or "loss"
    title      : plot title
    save_path  : if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, hist in histories.items():
        if metric == "acc":
            ax.plot(hist["train_acc"], linestyle="--", label=f"{name} train")
            ax.plot(hist["test_acc"],  linestyle="-",  label=f"{name} test")
            ax.set_ylabel("Accuracy")
        else:
            ax.plot(hist["train_loss"], linestyle="--", label=f"{name} train")
            ax.plot(hist["test_loss"],  linestyle="-",  label=f"{name} test")
            ax.set_ylabel("Loss")

    ax.set_xlabel("Epoch")
    ax.set_title(title or f"Validation {metric.capitalize()} vs Epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None):
    """Plot and optionally save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 3. Noise robustness bar chart  (Tables 5–8)
# ---------------------------------------------------------------------------
def plot_noise_results(noise_results: dict,
                       noise_type: str = "bit_flip",
                       save_path: Optional[str] = None):
    """
    Reproduce the noise-robustness bar chart (Tables 5-8).

    Parameters
    ----------
    noise_results : {model_name: {error_rate: accuracy}}
                    error_rate keys may be floats (0.0, 0.1...) or strings.
    noise_type    : label for title
    save_path     : if provided, save figure to this path
    """
    models = list(noise_results.keys())
    rates  = [0.0, 0.1, 0.2, 0.3]          # canonical float keys
    x      = np.arange(len(rates))
    width  = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(models):
        # Normalise keys to float so both float-keyed and str-keyed dicts work
        raw  = noise_results[model]
        norm = {}
        for k, v in raw.items():
            try:
                norm[float(k)] = v
            except (ValueError, TypeError):
                pass   # skip non-numeric keys like "no_noise"
        vals = [norm.get(r, 0) for r in rates]
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xlabel("Error Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Noise Robustness - {noise_type.replace('_', ' ').title()}")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(["No noise (p=0)", "p=0.1", "p=0.2", "p=0.3"])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 4. Circuit metrics bar chart  (Table 2)
# ---------------------------------------------------------------------------
def plot_circuit_metrics(metrics: dict, save_path: Optional[str] = None):
    """Visualize expressibility, entanglement, and discreteness for each circuit."""
    names  = list(metrics.keys())
    expr   = [metrics[n]["expressibility"] for n in names]
    ent    = [metrics[n]["entanglement"]   for n in names]

    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, expr, width, label="Expressibility (lower=better)")
    ax.bar(x + width/2, ent,  width, label="Entanglement (higher=better)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title("PQC Metrics - Experiment 1 (Table 2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()
