"""
experiment3_noise_robustness.py
================================
Experiment 3: Analysis of the Effects of Noise on Model Performance
(Paper Section 4.3.3, Tables 5, 6, 7, 8)

Evaluates QC-CNN-Parallel and the implemented classical CNN baseline under
four noise conditions on the full MNIST validation/test set.

Noise types tested:
  1. Data noise     - Gaussian noise on input images (Table 5)
  2. Bit-flip       - Pauli-X applied before measurement (Table 6, Eq. 25)
  3. Phase-flip     - Pauli-Z applied before measurement (Table 7, Eq. 26)
  4. Depolarizing   - Depolarizing channel   (Table 8, Eq. 27)

Error rates: p ∈ {0.1, 0.2, 0.3}  (plus no-noise baseline)

Important paper notes (Section 4.3.3):
  - Uses `default.mixed` simulator (supports noise channels)
  - Batch size = 100 (not 32), because default.mixed lacks batched input support
  - Full MNIST validation/test set used (all 10 classes, no subsampling)
  - Models are pre-trained on clean MNIST, then evaluated under noise

Paper Table 5-8 reference values:
  No-noise:   Proposed=0.9005, HQNN-Quanv=0.8320, QNN=0.8350, CNN=0.8935

Usage:
  python experiments/experiment3_noise_robustness.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.quantum_circuit  import make_noisy_circuit
from models.qc_cnn_parallel  import QCCNNParallel, ClassicalCNN
from training.trainer        import set_seed, _eval_epoch
from utils.plotting          import plot_noise_results


# ---------------------------------------------------------------------------
# Configuration (Section 4.3.3)
# ---------------------------------------------------------------------------
SEED         = 42
BATCH_SIZE   = 100      # paper uses 100 for noise experiments
EPOCHS       = 50
LR           = 0.01
ERROR_RATES  = [0.1, 0.2, 0.3]
NOISE_TYPES  = ["data_noise", "bit_flip", "phase_flip", "depolarizing"]
RESULTS_DIR  = Path("results/experiment3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Paper reference values (Tables 5–8) for comparison
PAPER_REFS = {
    "data_noise": {
        "Proposed":   {0.0: 0.9005, 0.1: 0.8915, 0.2: 0.8900, 0.3: 0.8425},
        "HQNN-Quanv": {0.0: 0.8320, 0.1: 0.8344, 0.2: 0.7796, 0.3: 0.7125},
        "QNN":        {0.0: 0.8350, 0.1: 0.8170, 0.2: 0.7544, 0.3: 0.7100},
        "CNN":        {0.0: 0.8935, 0.1: 0.8840, 0.2: 0.8530, 0.3: 0.7844},
    },
    "bit_flip": {
        "Proposed":   {0.0: 0.9005, 0.1: 0.8769, 0.2: 0.8558, 0.3: 0.8405},
        "HQNN-Quanv": {0.0: 0.8320, 0.1: 0.6775, 0.2: 0.6523, 0.3: 0.6399},
        "QNN":        {0.0: 0.8350, 0.1: 0.7115, 0.2: 0.6124, 0.3: 0.4615},
    },
    "phase_flip": {
        "Proposed":   {0.0: 0.9005, 0.1: 0.8836, 0.2: 0.8618, 0.3: 0.8602},
        "HQNN-Quanv": {0.0: 0.8320, 0.1: 0.8279, 0.2: 0.8254, 0.3: 0.8267},
        "QNN":        {0.0: 0.8350, 0.1: 0.8272, 0.2: 0.8191, 0.3: 0.8167},
    },
    "depolarizing": {
        "Proposed":   {0.0: 0.9005, 0.1: 0.8639, 0.2: 0.8664, 0.3: 0.8327},
        "HQNN-Quanv": {0.0: 0.8320, 0.1: 0.7021, 0.2: 0.6502, 0.3: 0.6059},
        "QNN":        {0.0: 0.8350, 0.1: 0.7552, 0.2: 0.6944, 0.3: 0.5904},
    },
}


# ---------------------------------------------------------------------------
# Full MNIST loader (batch_size=100, no subsampling — paper Section 4.3.3)
# ---------------------------------------------------------------------------
def get_full_mnist(batch_size: int = BATCH_SIZE, max_samples: int = None):
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds   = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    if max_samples is not None and max_samples < len(test_ds):
        from torch.utils.data import Subset
        indices = list(range(max_samples))
        test_ds = Subset(test_ds, indices)
        print(f"  [INFO] Using {max_samples} test samples (of 10,000) for speed.")
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Data-noise evaluation (Gaussian noise on input, Table 5)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_data_noise(model: nn.Module,
                         loader: DataLoader,
                         noise_std: float,
                         device: torch.device) -> float:
    """Add Gaussian noise to images and evaluate accuracy (Table 5)."""
    model.eval()
    correct, total = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if noise_std > 0:
            images = (images + torch.randn_like(images) * noise_std).clamp(0, 1)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Quantum noise evaluation (bit-flip, phase-flip, depolarizing — Tables 6–8)
# Requires replacing the QNode with a noisy one from make_noisy_circuit().
# ---------------------------------------------------------------------------
def evaluate_quantum_noise(model_factory,
                            loader: DataLoader,
                            noise_type: str,
                            noise_prob: float,
                            device: torch.device) -> float:
    """
    Evaluate a model with quantum noise channels (Tables 6–8).
    model_factory: callable() → QCCNNParallel with noisy QNode
    """
    noisy_qnode = make_noisy_circuit(noise_type, noise_prob)
    model       = model_factory(noisy_qnode).to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    _, acc, _, _, _ = _eval_epoch(model, loader, loss_fn, device)
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_experiment3(max_test_samples: int = None):
    set_seed(SEED)
    device   = torch.device("cpu")   # default.mixed runs on CPU
    results  = {}

    test_loader = get_full_mnist(max_samples=max_test_samples)
    print("=" * 60)
    print("  Experiment 3: Noise Robustness (Tables 5-8)")
    print("  Note: Pre-train models first using experiment2_classification.py")
    print("=" * 60)

    # --- Load clean models trained for Experiment 2 ---
    # Noise robustness is meaningful only when every evaluated model uses its
    # own trained clean checkpoint.  The old code evaluated a newly initialized
    # CNN, making its noise comparison invalid.
    def load_or_train_weights(model_name, model_cls):
        model_path = Path("results/experiment2/mnist") / f"{model_name}_best.pt"
        if model_path.exists():
            return torch.load(model_path, map_location=device, weights_only=True)

        print(f"\n  [WARNING] Pre-trained {model_name} not found at {model_path}.")
        print("  Training a clean model for noise evaluation (50 epochs)...")
        from datasets.dataloader import get_mnist
        from training.trainer import train as run_train
        # Table 4: training uses batch_size=32
        train_loader, clean_test_loader = get_mnist(batch_size=32)
        _, _, trained = run_train(
            model_cls(num_classes=10), train_loader, clean_test_loader,
            num_epochs=EPOCHS, lr=LR, seed=SEED, device=device,
            save_dir="results/experiment2", model_name=model_name,
            dataset_name="mnist",
        )
        return {name: value.detach().cpu() for name, value in trained.state_dict().items()}

    proposed_weights = load_or_train_weights("qc_cnn_parallel", QCCNNParallel)
    cnn_weights = load_or_train_weights("classical_cnn", ClassicalCNN)

    # -----------------------------------------------------------------------
    # Table 5: Data noise
    # -----------------------------------------------------------------------
    print("\n  Table 5: Data noise (Gaussian noise on input)")
    base_model = QCCNNParallel(num_classes=10).to(device)
    base_model.load_state_dict(proposed_weights)
    cnn_model  = ClassicalCNN(num_classes=10).to(device)
    cnn_model.load_state_dict(cnn_weights)

    data_noise_results = {"Proposed": {}, "CNN": {}}
    for p in [0.0] + ERROR_RATES:
        prop_acc = evaluate_data_noise(base_model, test_loader, p, device)
        cnn_acc  = evaluate_data_noise(cnn_model,  test_loader, p, device)
        data_noise_results["Proposed"][p] = round(prop_acc, 4)
        data_noise_results["CNN"     ][p] = round(cnn_acc,  4)
        print(f"    p={p:.1f}  Proposed={prop_acc:.4f}  CNN={cnn_acc:.4f}  "
              f"(paper: Prop={PAPER_REFS['data_noise']['Proposed'].get(p,'?')},"
              f" CNN={PAPER_REFS['data_noise']['CNN'].get(p,'?')})")
    results["data_noise"] = data_noise_results

    # -----------------------------------------------------------------------
    # Tables 6-8: Quantum noise channels
    # -----------------------------------------------------------------------
    for noise_type in ["bit_flip", "phase_flip", "depolarizing"]:
        tname = noise_type.replace("_", "-").title()
        print(f"\n  Table 6/7/8: {tname} noise")
        noise_results = {"Proposed": {}}

        # No-noise baseline (re-use clean weights)
        no_noise_model = QCCNNParallel(num_classes=10).to(device)
        no_noise_model.load_state_dict(proposed_weights)
        loss_fn = nn.CrossEntropyLoss()
        _, no_noise_acc, _, _, _ = _eval_epoch(no_noise_model, test_loader, loss_fn, device)
        noise_results["Proposed"][0.0] = round(no_noise_acc, 4)

        for p in ERROR_RATES:
            def factory(qnode, _p=p):
                m = QCCNNParallel(num_classes=10, qnode=qnode)
                m.load_state_dict(proposed_weights, strict=False)
                return m

            noisy_qnode = make_noisy_circuit(noise_type, p)
            noisy_model = factory(noisy_qnode)
            _, noisy_acc, _, _, _ = _eval_epoch(noisy_model.to(device), test_loader, loss_fn, device)
            noise_results["Proposed"][p] = round(noisy_acc, 4)

            paper_val = PAPER_REFS[noise_type]["Proposed"].get(p, "?")
            print(f"    p={p:.1f}  Proposed={noisy_acc:.4f}  "
                  f"(paper: {paper_val})")

        results[noise_type] = noise_results

        # Save bar chart
        plot_noise_results(
            noise_results,
            noise_type = tname,
            save_path  = str(RESULTS_DIR / f"noise_{noise_type}.png"),
        )

    # Save all results
    with open(RESULTS_DIR / "noise_results.json", "w") as f:
        # Convert float keys to str for JSON
        json_results = {
            ntype: {
                model: {str(k): v for k, v in acc_dict.items()}
                for model, acc_dict in model_dict.items()
            }
            for ntype, model_dict in results.items()
        }
        json.dump(json_results, f, indent=2)

    print(f"\n  ✓ Results saved to: {RESULTS_DIR}")

    # Print final comparison table
    print("\n" + "=" * 60)
    print("  Final Noise Comparison vs Paper (Proposed model)")
    print("=" * 60)
    for noise_type, model_dict in results.items():
        print(f"\n  {noise_type}:")
        for p in [0.0] + ERROR_RATES:
            got   = model_dict.get("Proposed", {}).get(p, "N/A")
            paper = PAPER_REFS.get(noise_type, {}).get("Proposed", {}).get(p, "N/A")
            print(f"    p={p:.1f}  Ours={got}  Paper={paper}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-test-samples", type=int, default=None,
        help="Limit test set size for faster runs (paper uses full 10,000; "
             "try 200-500 for quick testing on CPU).",
    )
    args = parser.parse_args()
    run_experiment3(max_test_samples=args.max_test_samples)
