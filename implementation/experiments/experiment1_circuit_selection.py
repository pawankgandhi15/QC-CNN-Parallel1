"""
experiment1_circuit_selection.py
=================================
Experiment 1: Analysis of the Performance Indicators of Different Quantum Circuits
(Paper Section 4.3.1, Tables 2 and 3)

Reproduces:
  Table 2 — Expressibility, Entangling Capability, Discreteness for 11 circuits
  Table 3 — Classification accuracy of each circuit on MNIST and Fashion-MNIST

The paper's methodology:
  • Each circuit uses 4 qubits and is evaluated with 5,000 numerical simulations.
  • A simplified hybrid model (1 quantum conv layer + 1 linear layer) is used
    to isolate the PQC contribution.
  • The winning circuit (Circuit 11, 16 params) is selected based on balancing
    expressibility, discreteness, and entangling capability.

Usage:
  python experiments/experiment1_circuit_selection.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
from pathlib import Path

from models.quantum_circuit import NUM_QUBITS
from datasets.dataloader    import get_mnist, get_fashion_mnist
from training.trainer       import set_seed, _train_epoch, _eval_epoch
from utils.circuit_metrics  import run_circuit_analysis


# ---------------------------------------------------------------------------
# Hyper-parameters (Table 4 / Section 4.3.1)
# ---------------------------------------------------------------------------
SEED          = 42
LR            = 0.01
BATCH_SIZE    = 32
NUM_EPOCHS    = 50      # same 50 epochs as main experiment
N_SIMS        = 5000   # circuit analysis: 5,000 simulations
RESULTS_DIR   = Path("results/experiment1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
dev = qml.device("default.qubit", wires=NUM_QUBITS)


# ---------------------------------------------------------------------------
# Circuit factory helpers
# ---------------------------------------------------------------------------
def _encode(inputs):
    """Angle encoding (H then RY, paper Eq. 6)."""
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i] * np.pi, wires=i)   # scale by π


def _rx_linear(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RX(weights[i], wires=i)
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rx_circle(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RX(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rx_alltoall(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RX(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        for j in range(NUM_QUBITS):
            if i != j:
                qml.CNOT(wires=[i, j])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _ry_linear(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RY(weights[i], wires=i)
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _ry_circle(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RY(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _ry_alltoall(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RY(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        for j in range(NUM_QUBITS):
            if i != j:
                qml.CNOT(wires=[i, j])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rz_linear(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RZ(weights[i], wires=i)
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rz_circle(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RZ(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _rz_alltoall(inputs, weights):
    _encode(inputs)
    for i in range(NUM_QUBITS):
        qml.RZ(weights[i], wires=i)
    for i in range(NUM_QUBITS):
        for j in range(NUM_QUBITS):
            if i != j:
                qml.CNOT(wires=[i, j])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _circuit10(inputs, weights):
    """
    Circuit 10 (Figure 5a, Equations 19–21): RX+RZ rotations + all-to-all CRX.
    28 parameters: 2 rot layers × 4 qubits × 2 gates + 4×(4-1) CRX = 16+12=28.
    """
    _encode(inputs)
    # Variational layer 1: RX + RZ per qubit (8 params)
    for i in range(NUM_QUBITS):
        qml.RX(weights[i],              wires=i)
        qml.RZ(weights[NUM_QUBITS + i], wires=i)
    # Entangling: all ordered pairs (i,j), i≠j  (12 CRX params for 4 qubits)
    idx = 8
    for i in range(NUM_QUBITS):
        for j in range(NUM_QUBITS):
            if i != j:
                qml.CRX(weights[idx], wires=[i, j])
                idx += 1
    # Variational layer 2: RX + RZ per qubit (8 params)
    for i in range(NUM_QUBITS):
        qml.RX(weights[idx],     wires=i); idx += 1
        qml.RZ(weights[idx],     wires=i); idx += 1
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def _circuit11(inputs, weights):
    """
    Circuit 11 — the SELECTED circuit (Figure 5b, Equations 22–24).
    16 parameters: 2 rot layers × 4 RY + 2 ent layers × 4 CRX.
    """
    _encode(inputs)
    w_rot1, w_ent1, w_rot2, w_ent2 = (
        weights[0:4], weights[4:8], weights[8:12], weights[12:16]
    )
    # Rot layer 1
    for i in range(NUM_QUBITS):
        qml.RY(w_rot1[i], wires=i)
    # Ent layer 1: circle 0→1→2→3→0
    for i in range(NUM_QUBITS):
        qml.CRX(w_ent1[i], wires=[i, (i + 1) % NUM_QUBITS])
    # Rot layer 2
    for i in range(NUM_QUBITS):
        qml.RY(w_rot2[i], wires=i)
    # Ent layer 2: circle shifted 1→2→3→0→1
    for i in range(NUM_QUBITS):
        qml.CRX(w_ent2[i], wires=[(i + 1) % NUM_QUBITS, (i + 2) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


# ---------------------------------------------------------------------------
# Build QNodes
# ---------------------------------------------------------------------------
CIRCUIT_DEFS = {
    "RX-Linear"    : (_rx_linear,    4),
    "RX-Circle"    : (_rx_circle,    4),
    "RX-All-to-All": (_rx_alltoall,  4),
    "RY-Linear"    : (_ry_linear,    4),
    "RY-Circle"    : (_ry_circle,    4),
    "RY-All-to-All": (_ry_alltoall,  4),
    "RZ-Linear"    : (_rz_linear,    4),
    "RZ-Circle"    : (_rz_circle,    4),
    "RZ-All-to-All": (_rz_alltoall,  4),
    "Circuit-10"   : (_circuit10,   28),
    "Circuit-11"   : (_circuit11,   16),
}

QNODES = {
    name: (qml.QNode(fn, dev, interface="torch"), n_params)
    for name, (fn, n_params) in CIRCUIT_DEFS.items()
}


# ---------------------------------------------------------------------------
# Simplified hybrid model for Experiment 1
# (one quantum conv layer + one linear classifier)
# ---------------------------------------------------------------------------
class SimpleHybrid(nn.Module):
    """
    Minimal hybrid model to isolate the PQC contribution (Section 4.3.1).
    Quantum conv (Circuit X) → flatten → Linear(4×14×14, num_classes)
    """
    def __init__(self, qnode, num_params: int, num_classes: int = 10):
        super().__init__()
        self.weights    = nn.Parameter(torch.randn(num_params) * 0.01)
        self._qnode     = qnode
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        out_H, out_W = H // 2, W // 2
        out = torch.zeros((B, 4, out_H, out_W), device=x.device, dtype=x.dtype)
        for b in range(B):
            for i in range(out_H):
                for j in range(out_W):
                    patch = x[b, 0, i*2:i*2+2, j*2:j*2+2].flatten()
                    result = torch.stack(self._qnode(patch, self.weights))
                    out[b, :, i, j] = result
        flat   = out.view(B, -1)
        return self.classifier(flat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_experiment1(run_metrics: bool = True, n_sims: int = N_SIMS):
    set_seed(SEED)
    device = torch.device("cpu")

    print("=" * 60)
    print("  Experiment 1: PQC Selection Study  (Table 2 & 3)")
    print("=" * 60)

    # Table 2 is the independent numerical circuit study described in the
    # paper. The previous implementation claimed to reproduce it but never
    # executed its metrics.
    table2_results = {}
    if run_metrics:
        circuits = {
            name: {"fn": fn, "num_params": n_params}
            for name, (fn, n_params) in CIRCUIT_DEFS.items()
        }
        table2_results = run_circuit_analysis(circuits, n_sims=n_sims)
        with open(RESULTS_DIR / "table2_circuit_metrics.json", "w") as f:
            json.dump(table2_results, f, indent=2)

    train_mnist,   test_mnist   = get_mnist(batch_size=BATCH_SIZE)
    train_fashion, test_fashion = get_fashion_mnist(batch_size=BATCH_SIZE)

    results_table3 = {}   # Table 3: classification accuracy per circuit

    for circuit_name, (qnode, n_params) in QNODES.items():
        print(f"\n--- Circuit: {circuit_name} ({n_params} params) ---")
        row = {}
        for ds_name, (tr_loader, te_loader) in [
            ("MNIST", (train_mnist, test_mnist)),
            ("Fashion-MNIST", (train_fashion, test_fashion)),
        ]:
            set_seed(SEED)
            model   = SimpleHybrid(qnode, n_params).to(device)
            opt     = torch.optim.Adam(model.parameters(), lr=LR)
            loss_fn = nn.CrossEntropyLoss()

            best_acc = 0.0
            for epoch in range(1, NUM_EPOCHS + 1):
                _train_epoch(model, tr_loader, opt, loss_fn, device)
                _, acc, _, _, _ = _eval_epoch(model, te_loader, loss_fn, device)
                if acc > best_acc:
                    best_acc = acc

            row[ds_name] = round(best_acc, 4)
            print(f"  {ds_name}: best_acc = {best_acc:.4f}")

        results_table3[circuit_name] = row

    # Save Table 3 results
    with open(RESULTS_DIR / "table3_circuit_classification.json", "w") as f:
        json.dump(results_table3, f, indent=2)

    # Print comparison against paper Table 3
    print("\n" + "=" * 60)
    print("  Table 3 Results vs Paper")
    print("=" * 60)
    paper_table3 = {
        "RX-Linear":     {"MNIST": 0.6560, "Fashion-MNIST": 0.7364},
        "RX-Circle":     {"MNIST": 0.6066, "Fashion-MNIST": 0.6864},
        "RX-All-to-All": {"MNIST": 0.7495, "Fashion-MNIST": 0.7647},
        "RY-Linear":     {"MNIST": 0.7530, "Fashion-MNIST": 0.7626},
        "RY-Circle":     {"MNIST": 0.7602, "Fashion-MNIST": 0.7763},
        "RY-All-to-All": {"MNIST": 0.8006, "Fashion-MNIST": 0.7846},
        "RZ-Linear":     {"MNIST": 0.5496, "Fashion-MNIST": 0.6229},
        "RZ-Circle":     {"MNIST": 0.5481, "Fashion-MNIST": 0.6213},
        "RZ-All-to-All": {"MNIST": 0.7249, "Fashion-MNIST": 0.6881},
        "Circuit-10":    {"MNIST": 0.8254, "Fashion-MNIST": 0.7946},
        "Circuit-11":    {"MNIST": 0.8057, "Fashion-MNIST": 0.7778},
    }
    print(f"{'Circuit':<20} {'MNIST (paper)':>14} {'MNIST (ours)':>14} "
          f"{'FashMNIST (paper)':>18} {'FashMNIST (ours)':>18}")
    print("-" * 90)
    for name in results_table3:
        p  = paper_table3.get(name, {})
        r  = results_table3[name]
        print(f"{name:<20} {p.get('MNIST',0):>14.4f} {r.get('MNIST',0):>14.4f} "
              f"{p.get('Fashion-MNIST',0):>18.4f} {r.get('Fashion-MNIST',0):>18.4f}")

    return {"table2": table2_results, "table3": results_table3}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip the 5,000-simulation Table 2 analysis.")
    parser.add_argument("--n-sims", type=int, default=N_SIMS,
                        help="Simulations per Table 2 metric (paper: 5000).")
    args = parser.parse_args()
    run_experiment1(run_metrics=not args.skip_metrics, n_sims=args.n_sims)
