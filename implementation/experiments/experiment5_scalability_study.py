"""
experiment5_scalability_study.py
=================================
Experiment 5: Scalability Study — Qubit Count & Circuit Depth Sensitivity

Systematically varies the quantum circuit configuration to:
  Part A: Sweep qubit count (2, 4, 6, 8) at fixed depth=2
  Part B: Sweep circuit depth (1, 2, 3, 4, 5) at fixed qubits=4

Validates the paper's choice of 4 qubits / depth 3, and detects
barren plateaus at higher qubit counts.

Addresses paper Section 5.1 Future Work:
  "further optimizing the QC-CNN-Parallel architecture"

Usage:
  python experiments/experiment5_scalability_study.py
  python experiments/experiment5_scalability_study.py --part A
  python experiments/experiment5_scalability_study.py --part B
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
import torch
import numpy as np
from pathlib import Path

from models.scalable_quantum_circuit import (
    ScalableQCCNNParallel,
    measure_gradient_variance,
)
from datasets  import get_dataloaders
from training  import train, set_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED       = 42
LR         = 0.01
BATCH_SIZE = 32
EPOCHS     = 50
RESULTS    = Path("results/experiment5")
RESULTS.mkdir(parents=True, exist_ok=True)

# Part A: Qubit count sweep (fixed depth=2, matching paper's Circuit 11)
QUBIT_SWEEP = [2, 4, 6, 8]
FIXED_DEPTH = 2   # Paper's Circuit 11 has 2 variational layers

# Part B: Depth sweep (fixed qubits=4, matching paper's Circuit 11)
DEPTH_SWEEP = [1, 2, 3, 4, 5]
FIXED_QUBITS = 4


# ---------------------------------------------------------------------------
# Part A: Qubit Count Scaling
# ---------------------------------------------------------------------------
def run_qubit_sweep(dataset_name="mnist"):
    """
    Vary qubit count while keeping depth fixed.
    Tests: 2, 4, 6, 8 qubits with depth=2.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  PART A: QUBIT COUNT SWEEP (depth={FIXED_DEPTH})")
    print(f"  Dataset: {dataset_name.upper()}  |  Device: {device}")
    print(f"{'='*65}")

    train_loader, test_loader = get_dataloaders(
        dataset_name, data_root="data", batch_size=BATCH_SIZE, seed=SEED
    )
    _, sample_labels = next(iter(train_loader))
    num_classes = int(sample_labels.max().item()) + 1

    results = {}

    for n_qubits in QUBIT_SWEEP:
        print(f"\n  {'─'*55}")
        print(f"  Qubits: {n_qubits}  |  Depth: {FIXED_DEPTH}")
        print(f"  {'─'*55}")

        set_seed(SEED)
        model = ScalableQCCNNParallel(
            num_classes=num_classes,
            n_qubits=n_qubits,
            n_layers=FIXED_DEPTH,
        )
        counts = model.count_parameters()
        print(f"  Quantum params: {counts['quantum_pqc']}  |  "
              f"Conv total: {counts['conv_total']}  |  "
              f"Total: {counts['total']}")

        # Measure gradient variance BEFORE training (barren plateau detection)
        print(f"  Measuring gradient variance...")
        grad_info = measure_gradient_variance(model, train_loader, device, n_batches=5)
        print(f"  Gradient variance: {grad_info['grad_variance']:.6f}  "
              f"Mean norm: {grad_info['mean_grad_norm']:.6f}")

        config_name = f"qubits_{n_qubits}_depth_{FIXED_DEPTH}"
        history, summary, trained_model = train(
            model        = model,
            train_loader = train_loader,
            test_loader  = test_loader,
            num_epochs   = EPOCHS,
            lr           = LR,
            seed         = SEED,
            device       = device,
            save_dir     = str(RESULTS),
            model_name   = config_name,
            dataset_name = dataset_name,
        )

        # Find convergence epoch (first epoch to reach 85% accuracy)
        convergence_epoch = None
        for ep, acc in enumerate(history["test_acc"], 1):
            if acc >= 0.85:
                convergence_epoch = ep
                break

        results[config_name] = {
            "n_qubits": n_qubits,
            "n_layers": FIXED_DEPTH,
            "quantum_params": counts["quantum_pqc"],
            "conv_total": counts["conv_total"],
            "total_params": counts["total"],
            "best_test_acc": summary["best_test_acc"],
            "final_test_f1": summary["final_test_f1"],
            "total_train_time_s": summary["total_train_time_s"],
            "avg_epoch_time_s": summary["total_train_time_s"] / EPOCHS,
            "convergence_epoch_85": convergence_epoch,
            "gradient_variance": grad_info["grad_variance"],
            "gradient_mean_norm": grad_info["mean_grad_norm"],
        }

    return results


# ---------------------------------------------------------------------------
# Part B: Circuit Depth Scaling
# ---------------------------------------------------------------------------
def run_depth_sweep(dataset_name="mnist"):
    """
    Vary circuit depth while keeping qubit count fixed.
    Tests: depth 1, 2, 3, 4, 5 with 4 qubits.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  PART B: CIRCUIT DEPTH SWEEP (qubits={FIXED_QUBITS})")
    print(f"  Dataset: {dataset_name.upper()}  |  Device: {device}")
    print(f"{'='*65}")

    train_loader, test_loader = get_dataloaders(
        dataset_name, data_root="data", batch_size=BATCH_SIZE, seed=SEED
    )
    _, sample_labels = next(iter(train_loader))
    num_classes = int(sample_labels.max().item()) + 1

    results = {}

    for n_layers in DEPTH_SWEEP:
        print(f"\n  {'─'*55}")
        print(f"  Qubits: {FIXED_QUBITS}  |  Depth: {n_layers}")
        print(f"  {'─'*55}")

        set_seed(SEED)
        model = ScalableQCCNNParallel(
            num_classes=num_classes,
            n_qubits=FIXED_QUBITS,
            n_layers=n_layers,
        )
        counts = model.count_parameters()
        print(f"  Quantum params: {counts['quantum_pqc']}  |  "
              f"Conv total: {counts['conv_total']}  |  "
              f"Total: {counts['total']}")

        # Measure gradient variance
        print(f"  Measuring gradient variance...")
        grad_info = measure_gradient_variance(model, train_loader, device, n_batches=5)
        print(f"  Gradient variance: {grad_info['grad_variance']:.6f}  "
              f"Mean norm: {grad_info['mean_grad_norm']:.6f}")

        config_name = f"qubits_{FIXED_QUBITS}_depth_{n_layers}"
        history, summary, trained_model = train(
            model        = model,
            train_loader = train_loader,
            test_loader  = test_loader,
            num_epochs   = EPOCHS,
            lr           = LR,
            seed         = SEED,
            device       = device,
            save_dir     = str(RESULTS),
            model_name   = config_name,
            dataset_name = dataset_name,
        )

        convergence_epoch = None
        for ep, acc in enumerate(history["test_acc"], 1):
            if acc >= 0.85:
                convergence_epoch = ep
                break

        results[config_name] = {
            "n_qubits": FIXED_QUBITS,
            "n_layers": n_layers,
            "quantum_params": counts["quantum_pqc"],
            "conv_total": counts["conv_total"],
            "total_params": counts["total"],
            "best_test_acc": summary["best_test_acc"],
            "final_test_f1": summary["final_test_f1"],
            "total_train_time_s": summary["total_train_time_s"],
            "avg_epoch_time_s": summary["total_train_time_s"] / EPOCHS,
            "convergence_epoch_85": convergence_epoch,
            "gradient_variance": grad_info["grad_variance"],
            "gradient_mean_norm": grad_info["mean_grad_norm"],
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_experiment5(parts=("A", "B"), dataset_name="mnist"):
    """Run the full scalability study."""
    all_results = {}

    if "A" in parts:
        qubit_results = run_qubit_sweep(dataset_name)
        all_results["qubit_sweep"] = qubit_results

    if "B" in parts:
        depth_results = run_depth_sweep(dataset_name)
        all_results["depth_sweep"] = depth_results

    # Save results
    with open(RESULTS / "scalability_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Print summary tables ---
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT 5 — SCALABILITY STUDY RESULTS")
    print(f"{'='*80}")

    if "qubit_sweep" in all_results:
        print(f"\n  Part A: Qubit Count Sweep (depth={FIXED_DEPTH})")
        print(f"  {'Qubits':>8} {'Q-Params':>10} {'Best Acc':>10} "
              f"{'F1':>8} {'Conv@85%':>10} {'Grad Var':>12} {'Time/Ep':>10}")
        print(f"  {'─'*78}")
        for config, res in all_results["qubit_sweep"].items():
            conv_ep = str(res["convergence_epoch_85"]) if res["convergence_epoch_85"] else "N/A"
            print(f"  {res['n_qubits']:>8} {res['quantum_params']:>10} "
                  f"{res['best_test_acc']:>10.4f} {res['final_test_f1']:>8.4f} "
                  f"{conv_ep:>10} {res['gradient_variance']:>12.6f} "
                  f"{res['avg_epoch_time_s']:>10.1f}s")

    if "depth_sweep" in all_results:
        print(f"\n  Part B: Circuit Depth Sweep (qubits={FIXED_QUBITS})")
        print(f"  {'Depth':>8} {'Q-Params':>10} {'Best Acc':>10} "
              f"{'F1':>8} {'Conv@85%':>10} {'Grad Var':>12} {'Time/Ep':>10}")
        print(f"  {'─'*78}")
        for config, res in all_results["depth_sweep"].items():
            conv_ep = str(res["convergence_epoch_85"]) if res["convergence_epoch_85"] else "N/A"
            print(f"  {res['n_layers']:>8} {res['quantum_params']:>10} "
                  f"{res['best_test_acc']:>10.4f} {res['final_test_f1']:>8.4f} "
                  f"{conv_ep:>10} {res['gradient_variance']:>12.6f} "
                  f"{res['avg_epoch_time_s']:>10.1f}s")

    # --- Key findings ---
    if "qubit_sweep" in all_results:
        sweep = all_results["qubit_sweep"]
        best_config = max(sweep.items(), key=lambda x: x[1]["best_test_acc"])
        print(f"\n  Best qubit config: {best_config[1]['n_qubits']} qubits "
              f"(acc={best_config[1]['best_test_acc']:.4f})")

        # Detect barren plateau (gradient variance drops significantly)
        grad_vars = [(r["n_qubits"], r["gradient_variance"])
                    for r in sweep.values()]
        grad_vars.sort()
        if len(grad_vars) >= 2:
            if grad_vars[-1][1] < grad_vars[0][1] * 0.1:
                print(f"  ⚠ Barren plateau detected: gradient variance drops "
                      f"from {grad_vars[0][1]:.6f} ({grad_vars[0][0]}q) "
                      f"to {grad_vars[-1][1]:.6f} ({grad_vars[-1][0]}q)")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 5: Scalability Study")
    parser.add_argument("--part", nargs="+", default=["A", "B"],
                       choices=["A", "B"], help="Which parts to run")
    parser.add_argument("--dataset", default="mnist",
                       choices=["mnist", "fashion_mnist", "overhead_mnist"])
    args = parser.parse_args()
    run_experiment5(parts=args.part, dataset_name=args.dataset)
