"""
experiment4_ablation_study.py
==============================
Experiment 4: Ablation Study — Quantum vs Classical Branch Contribution

This experiment isolates the contribution of each branch by testing 4 model
variants on all 3 paper datasets:

  1. QC-CNN-Parallel (full)    — Classical + Quantum branches (baseline)
  2. Classical-Only            — Classical branch only (8 filters, no quantum)
  3. Quantum-Only              — Quantum branch only (Circuit 11, no classical)
  4. Classical-Extended        — 12 classical filters (parameter-matched)

All variants use the same FC head, training protocol (Table 4), and datasets
(Table 1) for fair comparison.

Key questions answered:
  - Does the quantum branch add value? (Full vs Classical-Only)
  - Can quantum work alone? (Quantum-Only performance)
  - Is quantum better than just adding more classical filters? (Full vs Extended)

Usage:
  python experiments/experiment4_ablation_study.py
  python experiments/experiment4_ablation_study.py --dataset mnist
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
import torch
from pathlib import Path

from models.qc_cnn_parallel  import QCCNNParallel
from models.ablation_models   import ClassicalOnlyCNN, QuantumOnlyCNN, ClassicalExtendedCNN
from datasets                  import get_dataloaders
from training                  import train, set_seed
from utils                     import plot_training_curves


# ---------------------------------------------------------------------------
# Configuration (Table 4)
# ---------------------------------------------------------------------------
SEED       = 42
LR         = 0.01
BATCH_SIZE = 32
EPOCHS     = 50
RESULTS    = Path("results/experiment4")
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model registry for ablation variants
# ---------------------------------------------------------------------------
ABLATION_MODELS = {
    "qc_cnn_parallel":     {"class": QCCNNParallel,      "label": "Full Hybrid (Proposed)"},
    "classical_only":      {"class": ClassicalOnlyCNN,    "label": "Classical-Only (8 filters)"},
    "quantum_only":        {"class": QuantumOnlyCNN,      "label": "Quantum-Only (Circuit 11)"},
    "classical_extended":  {"class": ClassicalExtendedCNN, "label": "Classical-Extended (12 filters)"},
}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment4(datasets_to_run=("mnist", "fashion_mnist", "overhead_mnist")):
    """Run the full ablation study across all datasets and model variants."""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    all_results = {}

    for ds_name in datasets_to_run:
        print(f"\n{'='*65}")
        print(f"  ABLATION STUDY — Dataset: {ds_name.upper()}")
        print(f"{'='*65}")

        # Load dataset
        try:
            train_loader, test_loader = get_dataloaders(
                ds_name, data_root="data", batch_size=BATCH_SIZE, seed=SEED
            )
        except FileNotFoundError as e:
            print(f"  [SKIP] Could not load {ds_name}: {e}")
            continue

        # Determine number of classes
        _, sample_labels = next(iter(train_loader))
        num_classes = int(sample_labels.max().item()) + 1
        print(f"  Classes: {num_classes}")

        ds_results = {}
        ds_histories = {}

        for model_name, model_info in ABLATION_MODELS.items():
            print(f"\n  {'─'*55}")
            print(f"  Training: {model_info['label']}")
            print(f"  {'─'*55}")

            set_seed(SEED)  # Reset seed for each variant
            model = model_info["class"](num_classes=num_classes)

            # Print parameter counts
            if hasattr(model, "count_parameters"):
                counts = model.count_parameters()
                print(f"  Conv params: {counts['conv_total']}  |  "
                      f"Quantum params: {counts.get('quantum_pqc', 0)}  |  "
                      f"Total: {counts['total']}")

            history, summary, trained_model = train(
                model        = model,
                train_loader = train_loader,
                test_loader  = test_loader,
                num_epochs   = EPOCHS,
                lr           = LR,
                seed         = SEED,
                device       = device,
                save_dir     = str(RESULTS),
                model_name   = model_name,
                dataset_name = ds_name,
            )

            # Add parameter efficiency metric
            acc = summary["best_test_acc"]
            conv_params = counts["conv_total"]
            summary["accuracy_per_param"] = acc / conv_params if conv_params > 0 else 0
            summary["conv_params"] = conv_params
            summary["quantum_params"] = counts.get("quantum_pqc", 0)

            ds_results[model_name] = summary
            ds_histories[model_name] = history

        all_results[ds_name] = ds_results

        # --- Plot comparison curves ---
        plot_training_curves(
            ds_histories,
            metric    = "acc",
            title     = f"Ablation: Accuracy — {ds_name}",
            save_path = str(RESULTS / ds_name / "ablation_accuracy.png"),
        )
        plot_training_curves(
            ds_histories,
            metric    = "loss",
            title     = f"Ablation: Loss — {ds_name}",
            save_path = str(RESULTS / ds_name / "ablation_loss.png"),
        )

    # Save all summaries
    with open(RESULTS / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Print final comparison table ---
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT 4 — ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    print(f"  {'Model':<28} {'Dataset':<18} {'Conv Params':>12} "
          f"{'Best Acc':>10} {'F1':>8} {'Time(s)':>10}")
    print(f"  {'─'*92}")

    for ds_name, models in all_results.items():
        for model_name, summary in models.items():
            label = ABLATION_MODELS[model_name]["label"]
            print(f"  {label:<28} {ds_name:<18} "
                  f"{summary['conv_params']:>12} "
                  f"{summary['best_test_acc']:>10.4f} "
                  f"{summary['final_test_f1']:>8.4f} "
                  f"{summary['total_train_time_s']:>10.1f}")
        print()

    # --- Key findings ---
    for ds_name, models in all_results.items():
        if "qc_cnn_parallel" in models and "classical_only" in models:
            full_acc = models["qc_cnn_parallel"]["best_test_acc"]
            class_acc = models["classical_only"]["best_test_acc"]
            quantum_gain = full_acc - class_acc
            print(f"  {ds_name}: Quantum contribution = {quantum_gain:+.4f} "
                  f"({full_acc:.4f} vs {class_acc:.4f})")
        if "qc_cnn_parallel" in models and "classical_extended" in models:
            full_acc = models["qc_cnn_parallel"]["best_test_acc"]
            ext_acc = models["classical_extended"]["best_test_acc"]
            print(f"  {ds_name}: Hybrid vs Extended-Classical = {full_acc - ext_acc:+.4f}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4: Ablation Study")
    parser.add_argument(
        "--dataset", nargs="+",
        default=["mnist", "fashion_mnist", "overhead_mnist"],
        choices=["mnist", "fashion_mnist", "overhead_mnist"],
    )
    args = parser.parse_args()
    run_experiment4(datasets_to_run=args.dataset)
