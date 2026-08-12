"""
run_all.py
==========
Master entry point to run all three paper experiments sequentially.

Paper: "A Parallel Hybrid Quantum-Classical Convolutional Design Using
        Parameterized Quantum Circuits for Image Classification"
        Quantum Engineering (2026), article 6643049.

Usage:
  # Run Experiment 2 only (classification — fastest to validate correctness)
  python run_all.py --exp 2 --dataset mnist

  # Run Experiment 2 on all datasets
  python run_all.py --exp 2

  # Run all experiments in order
  python run_all.py

  # Smoke test with tiny subset (verify forward/backward pass works)
  python run_all.py --smoke_test

Experiment map
--------------
  Experiment 1 → experiments/experiment1_circuit_selection.py
                 Reproduces Tables 2 and 3 (PQC selection study)

  Experiment 2 → experiments/experiment2_classification.py
                 Reproduces Figures 6, 7, 8 and Table 4 (main classification)

  Experiment 3 → experiments/experiment3_noise_robustness.py
                 Reproduces Tables 5, 6, 7, 8 (noise robustness)
"""

from __future__ import annotations

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Smoke test (no real data needed — verifies model and gradient flow)
# ---------------------------------------------------------------------------
def run_smoke_test():
    """Verify the model forward and backward pass on dummy data."""
    from models import QCCNNParallel, ClassicalCNN
    from models.quantum_circuit import make_noisy_circuit

    print("\n" + "=" * 60)
    print("  SMOKE TEST — verifying model, optimizer, gradient flow")
    print("=" * 60)

    device  = torch.device("cpu")
    B       = 2       # tiny batch
    images  = torch.rand(B, 1, 28, 28)
    labels  = torch.randint(0, 10, (B,))
    loss_fn = torch.nn.CrossEntropyLoss()

    for name, model in [("QCCNNParallel", QCCNNParallel(10)),
                         ("ClassicalCNN",  ClassicalCNN(10))]:
        model = model.to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=0.01)
        opt.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {name:<20}  loss={loss.item():.4f}  total_params={params:,}")

    # Noisy circuit test
    print("\n  Testing noisy circuits (default.mixed)...")
    for nt in ["bit_flip", "phase_flip", "depolarizing"]:
        qnode  = make_noisy_circuit(nt, 0.1)
        model  = QCCNNParallel(10, qnode=qnode)
        logits = model(images[:1])           # batch=1 for mixed simulator
        loss   = loss_fn(logits, labels[:1])
        print(f"  ✓ Noisy ({nt:<14})  loss={loss.item():.4f}")

    print("\n  ✓ All smoke tests passed!\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="QC-CNN-Parallel: run paper experiments"
    )
    parser.add_argument(
        "--exp", nargs="+", type=int, default=[2],
        choices=[1, 2, 3],
        help="Which experiments to run (default: 2)",
    )
    parser.add_argument(
        "--dataset", nargs="+",
        default=["mnist", "fashion_mnist", "overhead_mnist"],
        choices=["mnist", "fashion_mnist", "overhead_mnist"],
        help="Datasets for Experiment 2 (default: all three)",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run a quick smoke test to verify installation and model correctness",
    )
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    if 1 in args.exp:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 1: PQC Selection Study (Tables 2 & 3)")
        print("=" * 60)
        from experiments.experiment1_circuit_selection import run_experiment1
        run_experiment1()

    if 2 in args.exp:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 2: Classification Benchmarks (Figs 6-8)")
        print("=" * 60)
        from experiments.experiment2_classification import run_experiment2
        run_experiment2(datasets_to_run=args.dataset)

    if 3 in args.exp:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 3: Noise Robustness (Tables 5-8)")
        print("=" * 60)
        from experiments.experiment3_noise_robustness import run_experiment3
        run_experiment3()


if __name__ == "__main__":
    main()
