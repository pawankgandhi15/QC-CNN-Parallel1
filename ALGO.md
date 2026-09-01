# ALGO.md — Algorithm Reference for QC-CNN-Parallel Experiments

> **Source Paper:** *A Parallel Hybrid Quantum-Classical Convolutional Design Using Parameterized Quantum Circuits for Image Classification*
> **Journal:** Quantum Engineering (2026), Article 6643049
> **Authors:** Haoxuan Liu, Xiaoping Lou — Hunan Normal University

---

## Table of Contents

1. [Core Algorithm Overview](#1-core-algorithm-overview)
2. [Algorithms Used — By Component](#2-algorithms-used--by-component)
3. [Experiment 1 — PQC Circuit Selection](#3-experiment-1--pqc-circuit-selection)
4. [Experiment 2 — General Classification](#4-experiment-2--general-classification)
5. [Experiment 3 — Noise Robustness](#5-experiment-3--noise-robustness)
6. [How All Experiments Connect](#6-how-all-experiments-connect)
7. [Parameters & Conditions Required to Run](#7-parameters--conditions-required-to-run)
8. [Software & Hardware Requirements](#8-software--hardware-requirements)

---

## 1. Core Algorithm Overview

The QC-CNN-Parallel model is a **hybrid quantum-classical convolutional neural network** that works as follows:

```
Input Image [28×28×1]
       │
       ├──────────────────────────────────┐
       │  Classical Branch                │  Quantum Branch
       │  Conv2d(1→8, 4×4, stride=2)      │  QuantumConvLayer (Circuit-11)
       │  ↓                               │  2×2 patches → 4 qubits → PQC
       │  [B, 8, 14, 14]                  │  [B, 4, 14, 14]
       └──────────────┬───────────────────┘
                      │  Concatenate (channel-wise)
                      ↓
               [B, 12, 14, 14]
                      │  Flatten
                      ↓
               [B, 2352]
                      │  FC1: 2352→128 + ReLU
                      │  FC2: 128→64  + ReLU
                      │  FC3: 64→C    (logits)
                      ↓
               Softmax → Class Label
```

**Three-stage quantum pipeline per patch:**

```
Classical pixel values
        ↓
[1] ENCODING:  H gate + RY(xi × π) on each qubit   (Angle Encoding, Eq. 6)
        ↓
[2] PQC:       Variational RY layers + CRX entangling layers  (Circuit-11, Eq. 24)
        ↓
[3] MEASUREMENT: ⟨Z⟩ expectation value per qubit   (Pauli-Z observable, Eq. 14)
        ↓
4-dimensional feature vector → classical dense layers
```

---

## 2. Algorithms Used — By Component

### 2.1 Angle Encoding (Feature Mapping)
- **Paper Section:** 3.1, Equation 6
- **What it does:** Converts classical pixel values into quantum states
- **Algorithm:**
  ```
  For each pixel xi in the 2×2 patch:
      Apply Hadamard gate H on qubit i       → creates superposition
      Apply RY(xi × π) rotation on qubit i   → encodes pixel value as angle
  Result: |ψ⟩ = RY(xi·π) · H|0⟩
  ```
- **Gate used:** H (Hadamard), RY (Y-rotation)
- **Input:** 4 pixel values from a 2×2 image patch, scaled by π
- **Output:** 4-qubit quantum state

---

### 2.2 Parameterized Quantum Circuit — Circuit 11
- **Paper Section:** 3.2, Equations 22–24, Figure 5b
- **What it does:** Trainable quantum feature transformation (the "quantum convolution")
- **Structure:** 4 layers — RY rot → CRX circle → RY rot → CRX shifted circle
- **Algorithm:**
  ```
  U11(θ, ϕ) = U_ent2(ϕ2) · U_rot2(θ2) · U_ent1(ϕ1) · U_rot1(θ1)

  Variational Layer (RY):
    For each qubit i: apply RY(θ_i)                    [Eq. 22]

  Entangling Layer 1 (Circle, starting at q0):
    CRX(ϕ0): q0 → q1
    CRX(ϕ1): q1 → q2
    CRX(ϕ2): q2 → q3
    CRX(ϕ3): q3 → q0                                  [Eq. 23]

  Entangling Layer 2 (Circle, starting at q1, shifted):
    CRX(ϕ4): q1 → q2
    CRX(ϕ5): q2 → q3
    CRX(ϕ6): q3 → q0
    CRX(ϕ7): q0 → q1
  ```
- **Total trainable parameters:** 16 (8 for RY rotations + 8 for CRX gates)
- **Why Circuit-11 was chosen:** Best balance of Expressibility (0.0071), Entanglement (0.5463), Discreteness (0.0191)

---

### 2.3 Pauli-Z Measurement
- **Paper Section:** 3.3, Equation 14
- **What it does:** Converts quantum state back to classical values
- **Algorithm:**
  ```
  E(θ) = ⟨ψ_out | Z | ψ_out⟩    for each qubit
  Output range: [-1, 1] per qubit
  ```
- **Why Pauli-Z:** Eigenstates |0⟩ and |1⟩ map directly to computational basis; output is bounded and stable for classical layers

---

### 2.4 Parameter-Shift Rule (Gradient Computation)
- **Paper Section:** 3.5, Equation 17
- **What it does:** Computes gradients of quantum circuit parameters (replaces backpropagation for quantum layers)
- **Algorithm:**
  ```
  ∂E(θ)/∂θ_i = ½ [E(θ + π/2·e_i) - E(θ - π/2·e_i)]
  ```
  - Shift each parameter by +π/2 and -π/2
  - Gradient = half the difference of outputs
  - Works natively on real quantum hardware
- **Implemented via:** PennyLane's PyTorch interface (`TorchLayer`) + autograd

---

### 2.5 Adam Optimizer
- **Paper Section:** 3.5, Table 4, Equation 18
- **What it does:** Updates all model parameters (classical + quantum jointly)
- **Update rule:** `θ'_i = θ_i - η · ∂L/∂θ_i`
- **Parameters used in paper:**
  - Learning rate η = 0.01
  - β1 = 0.9 (default), β2 = 0.999 (default)

---

### 2.6 Cross-Entropy Loss
- **Paper Section:** 3.4 / 3.5, Equation 15
- **Formula:**
  ```
  L(θ) = -Σ_i  y_i · log( exp(E(θ_i)) / Σ_j exp(E(θ_j)) )
  ```
  where y_i = one-hot ground truth label, E(θ_i) = quantum circuit output for class i

---

## 3. Experiment 1 — PQC Circuit Selection

### 3.1 Purpose
Identify the best quantum circuit from 11 candidates by evaluating three structural metrics and classification accuracy.

### 3.2 Algorithms Used

#### Algorithm A: Expressibility (Eq. 9)
```
Expr = D_KL( P_PQC(F; θ) || P_Haar(F) )

Steps:
  1. Sample N=5000 parameter pairs (θ1, θ2) uniformly from [0, 2π]
  2. For each pair, compute state vectors ψ1, ψ2 from the circuit
  3. Compute fidelity F = |⟨ψ1|ψ2⟩|²
  4. Build histogram of fidelities → estimated P_PQC(F)
  5. Compare to Haar-random reference P_Haar(F) = (N-1)(1-F)^(N-2)
  6. Compute KL divergence between the two distributions
```
- **Output:** KL divergence value (lower = better = more Haar-random = more expressive)
- **Code:** `circuit_metrics.py → expressibility()`

#### Algorithm B: Meyer-Wallach Entangling Capability (Eq. 10)
```
Ent = (1/|S|) Σ_{θ_i ∈ S} Q(|ψ_{θ_i}⟩)

Steps:
  1. Sample N=5000 random parameter vectors from [0, 2π]
  2. For each parameter set, compute the output state vector
  3. Compute Meyer-Wallach measure Q for each state:
     Q = (4/n) Σ_k [1 - Tr(ρ_k²)]
     where ρ_k = partial trace over all qubits except k
  4. Average Q over all samples
```
- **Output:** Entanglement score in [0, 1] (higher = more entangled)
- **Code:** `circuit_metrics.py → entangling_capability()`

#### Algorithm C: Discreteness — Novel Metric (Eqs. 12–13)
```
Disc = (1/M) Σ_i Var(g_i)
Var(g_i) = (1/N) Σ_j (g_i^(j) - ḡ_i)²

Steps:
  1. Sample N=5000 random parameter initializations
  2. For each initialization, compute gradients of all M parameters
     using the parameter-shift rule (shift = π/2)
  3. For each parameter i, compute variance of its gradient across all N initializations
  4. Average variance across all M parameters
```
- **What it measures:** Gradient heterogeneity across random initializations
  - High Discreteness = informative gradients = avoids barren plateau
  - Low Discreteness = flat gradients = training difficulty
- **Code:** `circuit_metrics.py → discreteness()`

#### Algorithm D: Classification Accuracy per Circuit
```
Steps:
  For each of the 11 circuits:
    1. Build SimpleHybrid model (1 quantum conv layer + 1 linear classifier)
    2. Train for 50 epochs on MNIST (10,000 train / 2,000 test)
    3. Train for 50 epochs on Fashion-MNIST (10,000 train / 2,000 test)
    4. Record best test accuracy
```

### 3.3 Circuits Evaluated (11 Total)

| Circuit | Rotation Gate | Entangling Topology | Params |
|---|---|---|---|
| RX-Linear | RX | Linear (chain) | 4 |
| RX-Circle | RX | Circle (ring) | 4 |
| RX-All-to-All | RX | All-to-All | 4 |
| RY-Linear | RY | Linear | 4 |
| RY-Circle | RY | Circle | 4 |
| RY-All-to-All | RY | All-to-All | 4 |
| RZ-Linear | RZ | Linear | 4 |
| RZ-Circle | RZ | Circle | 4 |
| RZ-All-to-All | RZ | All-to-All | 4 |
| Circuit-10 | RX+RZ / CRX all-to-all | Complex | 28 |
| **Circuit-11** | **RY / CRX circle (×2)** | **Shifted Circle** | **16** |

### 3.4 Paper Results (Table 2 — Circuit Metrics)

| Circuit | Expr ↓ | Ent ↑ | Disc ↑ | MNIST Acc |
|---|---|---|---|---|
| RX-Linear | 0.1755 | 0.5618 | 0.0280 | 0.6560 |
| RY-All-to-All | 0.3454 | 0.4520 | 0.1260 | 0.8006 |
| Circuit-10 | 0.0013 | 0.7180 | 0.0208 | 0.8254 |
| **Circuit-11** | **0.0071** | **0.5463** | **0.0191** | **0.8057** |

**Winner: Circuit-11** — best balance of all three metrics, selected for Experiments 2 & 3.

### 3.5 Parameters & Conditions for Experiment 1

| Parameter | Value | Source |
|---|---|---|
| Qubits | 4 | Paper Fig. 2 |
| N_SIMS (simulations) | 5,000 | Paper Section 4.3.1 |
| Epochs per circuit per dataset | 50 | Paper Table 4 |
| Learning rate | 0.01 | Paper Table 4 |
| Batch size | 32 | Paper Table 4 |
| Optimizer | Adam | Paper Table 4 |
| Random seed | 42 | Paper Table 4 |
| Loss function | Cross-entropy | Paper Eq. 15 |
| Quantum device | `default.qubit` (PennyLane) | CPU simulation |
| Model (Exp 1 only) | SimpleHybrid (1 QConv + 1 Linear) | Paper Section 4.3.1 |
| MNIST training samples | 10,000 (1,000/class) | Paper Table 1 |
| MNIST test samples | 2,000 (200/class) | Paper Table 1 |
| Fashion-MNIST training | 10,000 (1,000/class) | Paper Table 1 |
| Fashion-MNIST test | 2,000 (200/class) | Paper Table 1 |

**Run command:**
```bash
cd implementation
python experiments/experiment1_circuit_selection.py
# Quick test (fewer sims):
python experiments/experiment1_circuit_selection.py --n-sims 500 --skip-metrics
```

**Output files:**
```
results/experiment1/table2_circuit_metrics.json
results/experiment1/table3_circuit_classification.json
```

---

## 4. Experiment 2 — General Classification

### 4.1 Purpose
Validate QC-CNN-Parallel (using Circuit-11 from Experiment 1) against classical and other hybrid quantum models across 3 datasets.

### 4.2 Algorithms Used

#### Algorithm: Full QC-CNN-Parallel Training Loop
```
For each dataset in {MNIST, Fashion-MNIST, Overhead-MNIST}:
  For each model in {QC-CNN-Parallel, Classical CNN}:

    1. FORWARD PASS:
       a. Classical branch: Conv2d(1→8, 4×4) + ReLU → [B,8,14,14]
       b. Quantum branch:
          For each 2×2 patch in the image:
            i.  Angle-encode 4 pixels → 4 qubits  (Eq. 6)
            ii. Apply Circuit-11 PQC U11(θ,ϕ)    (Eq. 24)
            iii. Measure ⟨Z⟩ for each qubit       (Eq. 14)
          → Output: [B, 4, 14, 14]
       c. Concatenate branches → [B, 12, 14, 14]
       d. Flatten → [B, 2352]
       e. FC1(2352→128) + ReLU
       f. FC2(128→64)   + ReLU
       g. FC3(64→C)     → logits

    2. LOSS:
       L = CrossEntropyLoss(logits, labels)     (Eq. 15)

    3. BACKWARD PASS:
       - Classical params: standard autograd backprop
       - Quantum params:   parameter-shift rule  (Eq. 17)
       Combined via PennyLane's TorchLayer interface

    4. UPDATE:
       Adam optimizer step                      (Eq. 18)
       θ'_i = θ_i - η · ∂L/∂θ_i

    5. Save best model checkpoint (best test accuracy)
    6. Record per-epoch: train_loss, train_acc, test_loss, test_acc, F1
```

### 4.3 Models Compared (Table 4)

| Model | Conv Params | Qubits | Architecture |
|---|---|---|---|
| Classical CNN (LeNet-5) | 464 | — | Conv2d stack |
| QC-CNN | 448 | 4 | Fixed (non-trainable) quantum circuits |
| HQNN-Quanv | 448 | 4 | Trainable PQC |
| VCNN | 456 | 4 | Variational quantum-classical conv |
| QC-ResNet | 512 | 4 | ResNet + quantum conv |
| QC-Inception | 304 | 4 | Inception + quantum conv |
| **QC-CNN-Parallel (Proposed)** | **136** | **4** | **Parallel quantum + classical branches** |

### 4.4 Paper Target Results (MNIST, Table 4 / Figure 6)

| Model | MNIST Accuracy |
|---|---|
| Classical CNN | 0.8935 |
| HQNN-Quanv | 0.8320 |
| **QC-CNN-Parallel** | **0.9005** |

### 4.5 Parameters & Conditions for Experiment 2

| Parameter | Value | Source |
|---|---|---|
| Epochs | 50 | Paper Table 4 |
| Learning rate | 0.01 | Paper Table 4 |
| Batch size | 32 | Paper Table 4 |
| Optimizer | Adam | Paper Table 4 |
| Random seed | 42 | Paper Table 4 |
| Loss function | Cross-entropy | Paper Eq. 15 |
| Quantum device | `default.qubit` (PennyLane) | Clean simulation |
| Quantum circuit | Circuit-11 (16 params) | Result of Experiment 1 |
| Device | CPU or GPU (GPU accelerates classical CNN) | Auto-detected |
| Checkpoint saved as | `results/experiment2/{dataset}/{model}_best.pt` | Required for Exp 3 |

**Datasets:**

| Dataset | Training | Test | Classes | Source |
|---|---|---|---|---|
| MNIST | 10,000 (1,000/class) | 2,000 (200/class) | 10 | torchvision auto-download |
| Fashion-MNIST | 10,000 (1,000/class) | 2,000 (200/class) | 10 | torchvision auto-download |
| Overhead-MNIST | 8,519 (full) | 1,065 (full) | 10 | Manual download required* |

> *Overhead-MNIST: Download from https://www.kaggle.com/datasets/andrewmvd/aircraft-mnist and place in `implementation/data/overhead_mnist/`. Or remove it from `EXP2_DATASETS` to skip.

**Run command:**
```bash
cd implementation
python experiments/experiment2_classification.py
# Only MNIST:
python experiments/experiment2_classification.py --dataset mnist
# MNIST + Fashion:
python experiments/experiment2_classification.py --dataset mnist fashion_mnist
```

**Output files:**
```
results/experiment2/mnist/qc_cnn_parallel_best.pt      ← needed by Experiment 3
results/experiment2/mnist/classical_cnn_best.pt         ← needed by Experiment 3
results/experiment2/mnist/qc_cnn_parallel_history.json
results/experiment2/mnist/accuracy_curve.png
results/experiment2/mnist/loss_curve.png
results/experiment2/all_summaries.json
```

---

## 5. Experiment 3 — Noise Robustness

### 5.1 Purpose
Evaluate the trained QC-CNN-Parallel under 4 types of noise at 3 error levels. Tests real-world NISQ device robustness.

### 5.2 Prerequisite
> **Experiment 2 MUST run first.** Experiment 3 loads pre-trained weights from `results/experiment2/mnist/qc_cnn_parallel_best.pt`. If the file does not exist, it auto-trains a new model (adds ~2 hours).

### 5.3 Algorithms Used

#### Noise Type 1 — Data Noise (Gaussian, Table 5)
```
Algorithm:
  1. Load pre-trained clean model (from Experiment 2)
  2. For each error rate p ∈ {0.0, 0.1, 0.2, 0.3}:
     a. For each test image:
        - Add Gaussian noise: x_noisy = clip(x + N(0, p), 0, 1)
     b. Run model forward pass on x_noisy
     c. Record accuracy
```
- **Simulates:** Sensor noise, image corruption
- **Quantum device:** `default.qubit` (no quantum noise, only input perturbation)

#### Noise Type 2 — Bit-Flip Noise (Table 6, Eq. 25)
```
Quantum noise channel applied before measurement:
  ρ → (1-p)·ρ + p·X·ρ·X
  (Pauli-X flips |0⟩↔|1⟩ with probability p)

Algorithm:
  1. Replace clean QNode with noisy QNode using default.mixed
  2. Load pre-trained weights into noisy model
  3. For each p ∈ {0.0, 0.1, 0.2, 0.3}:
     a. Add BitFlip(p) channel before Pauli-Z measurement on each qubit
     b. Evaluate on full MNIST test set (10,000 samples)
     c. Record accuracy
```
- **Simulates:** Physical qubit bit-flip errors on NISQ hardware

#### Noise Type 3 — Phase-Flip Noise (Table 7, Eq. 26)
```
Quantum noise channel:
  ρ → (1-p)·ρ + p·Z·ρ·Z
  (Pauli-Z flips phase of |1⟩ with probability p)

Algorithm: same as Bit-Flip but with PhaseFlip(p) channel
```
- **Simulates:** Phase decoherence errors

#### Noise Type 4 — Depolarizing Noise (Table 8, Eq. 27)
```
Quantum noise channel:
  ρ → (1-p)·ρ + (p/3)·(X·ρ·X + Y·ρ·Y + Z·ρ·Z)

Algorithm: same as Bit-Flip but with DepolarizingChannel(p) channel
```
- **Simulates:** General decoherence; the most realistic NISQ noise model

### 5.4 Paper Results

**Table 5 — Data Noise (Proposed model):**
| Error Rate | Accuracy |
|---|---|
| 0.0 (clean) | 0.9005 |
| 0.1 | 0.8915 |
| 0.2 | 0.8900 |
| 0.3 | 0.8425 |

**Table 6 — Bit-Flip Noise (Proposed vs HQNN-Quanv):**
| Error Rate | Proposed | HQNN-Quanv | QNN |
|---|---|---|---|
| 0.1 | 0.8769 | 0.6775 | 0.7115 |
| 0.2 | 0.8558 | 0.6523 | 0.6124 |
| 0.3 | 0.8405 | 0.6399 | 0.4615 |

**Table 7 — Phase-Flip Noise (Proposed):**
| Error Rate | Accuracy |
|---|---|
| 0.1 | 0.8836 |
| 0.2 | 0.8618 |
| 0.3 | 0.8602 |

**Table 8 — Depolarizing Noise (Proposed):**
| Error Rate | Accuracy |
|---|---|
| 0.1 | 0.8639 |
| 0.2 | 0.8664 |
| 0.3 | 0.8327 |

### 5.5 Parameters & Conditions for Experiment 3

| Parameter | Value | Source |
|---|---|---|
| Batch size | **100** (not 32) | Paper Section 4.3.3 |
| Epochs (if retraining) | 50 | Paper Table 4 |
| Error rates | p ∈ {0.1, 0.2, 0.3} | Paper Section 4.3.3 |
| Noise types | bit_flip, phase_flip, depolarizing, data_noise | Paper Eqs. 25–27 |
| Quantum device | **`default.mixed`** (NOT `default.qubit`) | Supports noise channels |
| GPU support | **None** (default.mixed is CPU-only) | PennyLane limitation |
| Test set | Full MNIST (10,000 samples, all 10 classes) | Paper Section 4.3.3 |
| Pre-trained weights | `results/experiment2/mnist/qc_cnn_parallel_best.pt` | From Experiment 2 |

> ⚠️ **Critical:** Experiment 3 uses `default.mixed` which does NOT support batched inputs and does NOT support GPU. It must run on CPU. This is why batch_size=100 is used (instead of 32 in Exp 1 & 2).

**Run command:**
```bash
cd implementation
python experiments/experiment3_noise_robustness.py
# Quick test with fewer samples:
python experiments/experiment3_noise_robustness.py --max-test-samples 500
```

**Output files:**
```
results/experiment3/noise_results.json
results/experiment3/noise_bit_flip.png
results/experiment3/noise_phase_flip.png
results/experiment3/noise_depolarizing.png
```

---

## 6. How All Experiments Connect

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT 1 (Circuit Selection)                  │
│                                                                       │
│  Evaluates 11 PQC candidates using 3 metrics:                        │
│    • Expressibility  (KL divergence vs Haar random)  [Eq. 9]        │
│    • Entanglement    (Meyer-Wallach measure)          [Eq. 10]       │
│    • Discreteness    (gradient variance)              [Eqs. 12-13]   │
│                                                                       │
│  + Classification accuracy on MNIST & Fashion-MNIST                  │
│                                                                       │
│  OUTPUT: Circuit-11 selected (16 params, balanced metrics)           │
└────────────────────────────┬────────────────────────────────────────┘
                              │  Circuit-11 used in
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT 2 (Classification)                      │
│                                                                       │
│  Full QC-CNN-Parallel model (with Circuit-11) trained on:            │
│    • MNIST (10K/2K)                                                  │
│    • Fashion-MNIST (10K/2K)                                          │
│    • Overhead-MNIST (8519/1065)                                      │
│                                                                       │
│  Training: Adam, LR=0.01, 50 epochs, Cross-entropy, Batch=32        │
│  Gradient: Parameter-shift rule [Eq. 17] + PyTorch autograd         │
│                                                                       │
│  OUTPUT: Trained model checkpoints (*.pt files)                      │
│          Accuracy/loss curves (Figures 6, 7, 8)                      │
└────────────────────────────┬────────────────────────────────────────┘
                              │  Pre-trained weights loaded into
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT 3 (Noise Robustness)                   │
│                                                                       │
│  Evaluates robustness of Exp 2 trained model under:                  │
│    • Data noise (Gaussian on input images)           [Table 5]       │
│    • Bit-flip quantum noise (Pauli-X)                [Table 6, Eq 25]│
│    • Phase-flip quantum noise (Pauli-Z)              [Table 7, Eq 26]│
│    • Depolarizing channel                            [Table 8, Eq 27]│
│                                                                       │
│  Error rates: p ∈ {0.0, 0.1, 0.2, 0.3}                             │
│  Device: default.mixed (CPU, mixed-state simulator)                  │
│                                                                       │
│  OUTPUT: Noise robustness tables (Tables 5–8)                        │
│          Bar chart comparisons                                        │
└─────────────────────────────────────────────────────────────────────┘
```

**Sequential dependency:**
```
Experiment 1  →  (Circuit-11)  →  Experiment 2  →  (*.pt weights)  →  Experiment 3
```

---

## 7. Parameters & Conditions Required to Run

### 7.1 Shared Parameters (All Experiments)

| Parameter | Value | Where Set |
|---|---|---|
| Random seed | 42 | `training/trainer.py → set_seed(42)` |
| Optimizer | Adam | `torch.optim.Adam` |
| Learning rate | 0.01 | `LR = 0.01` in each experiment file |
| Loss function | Cross-entropy | `nn.CrossEntropyLoss()` |
| Qubits | 4 | `NUM_QUBITS = 4` in `quantum_circuit.py` |
| Encoding | Angle encoding (H + RY) | `_angle_encode()` in `quantum_circuit.py` |
| Measurement | Pauli-Z expectation | `qml.expval(qml.PauliZ(i))` |
| PQC (Exp 2 & 3) | Circuit-11 (16 params) | `_circuit11_layers()` in `quantum_circuit.py` |
| Epochs | 50 | `NUM_EPOCHS = 50` in each experiment file |

### 7.2 Experiment-Specific Parameters

| Parameter | Experiment 1 | Experiment 2 | Experiment 3 |
|---|---|---|---|
| Batch size | 32 | 32 | **100** |
| N_SIMS | 5,000 | — | — |
| Quantum device | `default.qubit` | `default.qubit` | **`default.mixed`** |
| GPU support | ❌ No | ✅ Yes (classical CNN only) | ❌ No |
| Model | SimpleHybrid | QCCNNParallel + ClassicalCNN | QCCNNParallel |
| Datasets | MNIST + Fashion-MNIST | All 3 datasets | MNIST (full 10K) |
| Requires Exp 1? | No | No | No |
| Requires Exp 2? | No | No | ✅ **Yes** |

### 7.3 Conditions Required Before Running

**All experiments:**
- [ ] Python ≥ 3.8
- [ ] Working directory must be `E:\parallel_quantum\implementation\`
- [ ] PennyLane ≥ 0.38.0 installed
- [ ] PyTorch ≥ 2.0.0 installed
- [ ] scikit-learn, matplotlib, numpy, torchvision installed

**Experiment 1 specific:**
- [ ] Enough time: 4–8 hours on CPU (set `--n-sims 500 --skip-metrics` for a quick test)

**Experiment 2 specific:**
- [ ] Overhead-MNIST data downloaded (or removed from dataset list)

**Experiment 3 specific:**
- [ ] Experiment 2 must have been run first (checkpoints at `results/experiment2/mnist/*.pt`)
- [ ] CPU with sufficient RAM (default.mixed uses full density matrix simulation: 2^(2×4) = 256 elements per state)
- [ ] Do NOT run on GPU (default.mixed does not support CUDA)
- [ ] Do NOT use batch_size < 100 (default.mixed has no batched input support)

---

## 8. Software & Hardware Requirements

### 8.1 Required Python Packages

| Package | Version (tested) | Role |
|---|---|---|
| `pennylane` | ≥ 0.38.0 | Quantum circuit simulation, QNode, noise channels |
| `torch` | ≥ 2.0.0 | Neural network, autograd, Adam optimizer |
| `torchvision` | ≥ 0.15.0 | MNIST / Fashion-MNIST dataset loading |
| `numpy` | ≥ 1.24.0 | Matrix operations, state vector computation |
| `scikit-learn` | ≥ 1.3.0 | F1 score, confusion matrix |
| `matplotlib` | ≥ 3.7.0 | Training curves, bar charts |

**Install all:**
```bash
pip install pennylane>=0.38.0 torch>=2.0.0 torchvision>=0.15.0 numpy>=1.24.0 scikit-learn>=1.3.0 matplotlib>=3.7.0
```

### 8.2 PennyLane Devices Used

| Device | Experiment | Notes |
|---|---|---|
| `default.qubit` | Exp 1 & 2 | Statevector simulator; supports batching; fastest |
| `default.mixed` | Exp 3 only | Density matrix simulator; required for noise channels; CPU only; no batching |

### 8.3 Minimum Hardware Recommendations

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| GPU | None (quantum runs on CPU) | Any CUDA GPU for Exp 2 classical branch |
| Storage | 2 GB free | 5 GB (includes datasets, checkpoints) |
| Time (all 3 exps) | ~8–14 hours (CPU) | ~3–5 hours (with GPU for Exp 2) |

### 8.4 Platform Notes

| Platform | Suitable | Notes |
|---|---|---|
| **Kaggle** | ✅ Best free | 12 hr session, 30 hr/week GPU, existing notebook in repo |
| **AWS SageMaker Studio Lab** | ✅ Best free alternative | 8 hr CPU/day, 15 GB persistent storage |
| **Google Colab** | ⚠️ Limited | 90-min idle timeout may kill Exp 1 mid-run |
| **Local Windows** | ✅ | Use `E:\parallel_quantum\implementation\` as working dir |
| **Local Linux / WSL2** | ✅ | Faster than pure Windows for Python |

---

*Generated from: Quantum Engineering (2026), Article 6643049 + implementation source code*
*Created: 2026-09-01*

---
---

# Part 2 — Plain-Language Explanations (Session Q&A)

> The following sections are simple-word explanations of every algorithm, metric, noise type, and dataset used in this project. Generated from the interactive Q&A session on 2026-09-01.

---

## A. Experiment 1 Metrics — Simple Explanation

### The Core Question Experiment 1 Asks
> *"Which quantum circuit should we use?"* — Before training the full model, we test 11 candidate circuits using 3 scoring metrics + classification accuracy.

Think of a **quantum circuit like a cooking recipe** — it takes ingredients (pixel values) and transforms them into a dish (features). These 4 metrics judge how good a recipe is, before you even taste the food.

---

### A.1 Expressibility (KL Divergence, Eq. 9)
**"How many different dishes can this recipe make?"**

> Imagine you have a recipe. If you randomly change the amounts of each ingredient, how many **different dishes** can you produce?

- A **highly expressive** circuit = can produce **many different outcomes** — covers the full space of possibilities
- A **low expressibility** circuit = always produces the **same type of dish** no matter how you tweak it — boring, limited

**The math in simple steps:**
1. Run the circuit **5,000 times** with completely random settings
2. Compare pairs of outputs — are they different or the same?
3. Compare this spread to a "perfectly random" reference (called Haar-random)
4. If your circuit matches the "perfectly random" spread → **score near 0 = GOOD** ✅
5. If your circuit only produces the same outputs → **score is high = BAD** ❌

**Circuit-11 result:** `0.0071` — very close to 0 = very expressive ✅

---

### A.2 Meyer-Wallach Entanglement (Eq. 10)
**"How well do the qubits talk to each other?"**

> Imagine 4 workers (qubits) on a team. A good team is one where **each worker's action affects all others** — they are interconnected. A bad team has workers who **work independently**.

- **High entanglement** = strong teamwork between qubits = they share information = richer features
- **Low entanglement** = each qubit works alone = less powerful than classical computing

**The math in simple steps:**
1. Run the circuit 5,000 times with random settings
2. For each run, check: if you look at just ONE qubit, can you predict what the others are doing?
3. If you **cannot** predict the others → the qubits are entangled → **score near 1 = GOOD** ✅
4. If you **can** predict them easily → they're independent → **score near 0 = BAD** ❌

**Circuit-11 result:** `0.5463` — moderately entangled = balanced ✅

---

### A.3 Discreteness / Gradient Variance (Eqs. 12–13)
**"Does the circuit actually learn, or does it get stuck?"**

> Imagine teaching a student. If you give them a test and they always score exactly the same regardless of what you teach them → **they're not learning**. But if their scores vary — sometimes better, sometimes worse — they **are responding** to teaching.

- **High Discreteness** = the circuit **responds differently** to different starting points = gradients are varied = it can learn well = avoids barren plateau
- **Low Discreteness** = gradients are always nearly **zero** = the circuit doesn't know which direction to improve = training gets stuck

**What is the "Barren Plateau"?**
> Imagine you're lost on a perfectly flat infinite desert. No hills, no valleys — you can't tell which direction leads home. Gradient = 0 everywhere → you can't train the circuit.

**The math in simple steps:**
1. Start the circuit 5,000 times from **random starting weights**
2. Each time, compute gradients (how much each parameter needs to change)
3. Ask: **do these gradients vary a lot**, or are they always near zero?
4. High variance → informative gradients → **high Discreteness = GOOD** ✅
5. Near-zero variance everywhere → flat landscape → **low Discreteness = BAD** ❌

> 💡 **This is a novel metric invented in this paper** — it did not exist before!

**Circuit-11 result:** `0.0191` — a sweet spot (not too flat, not chaotically variable) ✅

---

### A.4 Circuit Classification Accuracy
**"After all the theory — does the circuit actually classify images correctly?"**

The simplest test — just train each circuit on real data and measure accuracy:

1. Take each of the 11 candidate circuits
2. Build a simple model: **quantum circuit + one linear layer**
3. Train for **50 epochs** on MNIST and Fashion-MNIST
4. Measure: **what % of test images were classified correctly?**

| Circuit | MNIST Accuracy |
|---|---|
| RX-Linear | 65.6% |
| RY-All-to-All | 80.1% |
| Circuit-10 | 82.5% |
| **Circuit-11 (Winner)** | **80.6%** |

> Circuit-11 isn't the highest accuracy alone — but its balance of all 3 metrics above makes it the best overall choice.

---

### A.5 Why All 4 Metrics Together?

| Metric | What it asks | Good value |
|---|---|---|
| Expressibility | "Can it produce diverse outputs?" | Near 0 |
| Entanglement | "Do qubits cooperate?" | Near 1 |
| Discreteness | "Will it actually learn?" | Not near 0 |
| Accuracy | "Does it actually work?" | Near 1 (100%) |

> You need **all 4** because a circuit can be expressive but untrainable, or trainable but have poor accuracy. Circuit-11 was the best **overall combination** — like choosing an employee who is skilled, collaborative, motivated, AND gets results. 🏅

---

## B. Experiment 2 Algorithms — Simple Explanation

These 6 algorithms form a **complete pipeline** — they run one after another, like an assembly line.

```
Image Pixels → [Encode] → [Quantum Circuit] → [Measure] → [Loss] → [Shift Rule] → [Adam]
                Eq.6          Eq.24             Eq.14      Eq.15      Eq.17          Eq.18
                                                            ↑__________________________|
                                                                feedback loop (training)
```

---

### B.1 Angle Encoding (Eq. 6)
**"Translating a photo into quantum language"**

> Computers store images as numbers (pixel values 0–255). But quantum circuits speak a **different language** — they use **rotation angles**. Angle Encoding is the **translator**.

**Simple analogy:** Imagine a clock hand. Pointing straight up = 0°, pointing right = 90°. A pixel value of 128 (medium brightness) becomes a **specific angle** on this clock.

**What actually happens:**
1. Take a **2×2 patch** of the image (4 pixels)
2. Scale each pixel: `angle = pixel_value × π`
3. For each pixel, **rotate a qubit** by that angle using the RY gate
4. First apply Hadamard (H) gate → puts qubit in superposition (both 0 and 1 at once)
5. Then apply RY(angle) → tilts it by the pixel's angle

```
Pixel value 0.5  →  angle = 0.5π  →  qubit tilted halfway between |0⟩ and |1⟩
Pixel value 0.0  →  angle = 0     →  qubit stays at |0⟩
Pixel value 1.0  →  angle = π     →  qubit flips to |1⟩
```

**Why this matters:** The quantum circuit can now **process the image** using quantum physics.

---

### B.2 Circuit-11 PQC (Eq. 24)
**"The quantum brain that extracts features"**

> Think of this as the **quantum version of a convolutional filter** in a classical CNN. It takes the encoded pixels and transforms them into richer, more meaningful features.

**Simple analogy:** A prism takes white light and splits it into a rainbow — revealing hidden structure. Circuit-11 does the same to quantum states.

**What it contains (4 layers in sequence):**
```
Layer 1: RY rotation on each qubit         ← learns "what to look for"
Layer 2: CRX ring connections              ← qubits influence each other (entangle)
           q0→q1→q2→q3→q0
Layer 3: RY rotation again                 ← refines what it found
Layer 4: CRX ring (shifted start)          ← different entanglement pattern
           q1→q2→q3→q0→q1
```

- **8 RY angles** (θ) — learned during training
- **8 CRX angles** (φ) — learned during training
- **Total: 16 trainable parameters** (very lightweight!)

---

### B.3 Pauli-Z Measurement (Eq. 14)
**"Reading the answer out of the quantum circuit"**

> After the quantum circuit processes the image, the result is still in quantum form. Measurement **collapses** it into a real number we can use.

**Simple analogy:** Imagine a spinning coin — it's both heads AND tails while spinning. The moment you **catch it**, it becomes one or the other. Measurement is "catching the coin."

**What it does:**
1. Look at each qubit through the **Z-axis lens** (Pauli-Z operator)
2. Get an **expectation value** — a weighted average between -1 and +1
3. Output: **4 numbers** (one per qubit), each in range `[-1, 1]`

```
Qubit result near +1 → qubit was mostly in |0⟩ state
Qubit result near -1 → qubit was mostly in |1⟩ state
Qubit result near  0 → qubit was in equal superposition
```

**Why Pauli-Z?** The answer `[-1, +1]` is already bounded and stable — perfect input for a neural network.

---

### B.4 Cross-Entropy Loss (Eq. 15)
**"Measuring how wrong the model is"**

> After the model predicts a class, we need to know: **how wrong was it?** Loss is the penalty score — higher = more wrong.

**Simple analogy:** You're guessing someone's age. If they're 25 and you guess 26 → small penalty. If you guess 80 → huge penalty.

**What it does:**
1. The model outputs **10 numbers** (one score per digit class 0–9)
2. Apply **softmax** → converts scores into probabilities (all sum to 1)
3. Look at the **correct class** probability
4. Loss = `-log(probability of correct class)`
   - Model was 85% sure and correct → small loss ✅
   - Model was 5% sure and wrong → large loss ❌

**Training goal:** Make this loss as **small as possible** over all training images.

---

### B.5 Parameter-Shift Rule (Eq. 17)
**"How to teach a quantum circuit — you can't use normal calculus"**

> In a classical neural network, we use backpropagation to compute gradients. But quantum circuits cannot do that. The Parameter-Shift Rule is the quantum alternative.

**Simple analogy:** You're trying to find the steepest hill to climb but you're **blindfolded**:
1. Take one step to the **right** → record your height
2. Take one step to the **left** → record your height
3. The **difference** tells you which direction is uphill

**That is exactly the parameter-shift rule:**
```
Gradient of θᵢ = ½ × [ circuit(θᵢ + π/2) - circuit(θᵢ - π/2) ]

For each trainable parameter:
  1. Run circuit with parameter shifted by +90°  → get output A
  2. Run circuit with parameter shifted by -90°  → get output B
  3. Gradient = (A - B) / 2
```

**Cost:** For 16 parameters → **32 extra circuit runs per gradient step** — this is why quantum training is slow!

---

### B.6 Adam Optimizer (Eq. 18)
**"The smart coach that updates all parameters"**

> Once we know the gradients (from step B.5), Adam decides **how much to actually change each parameter**.

**Simple analogy — Learning to shoot arrows:**
- **Simple gradient descent** = always move the same amount in the direction you missed
- **Adam** = moves more when you've been consistently wrong in one direction, moves less when you've been bouncing back and forth

**Settings used in paper:**

| Setting | Value |
|---|---|
| Learning rate (η) | **0.01** |
| β1 | 0.9 (momentum memory) |
| β2 | 0.999 (variance memory) |

---

### B.7 How All 6 Work Together (One Training Step)

```
┌──────────────────────────────────────────────────────────┐
│                   ONE TRAINING STEP                       │
│                                                           │
│  1. Take a batch of 32 images                            │
│           ↓                                              │
│  2. ANGLE ENCODING (Eq.6)                                │
│     Pixels → rotation angles → qubit states              │
│           ↓                                              │
│  3. CIRCUIT-11 PQC (Eq.24)                              │
│     Quantum circuit transforms the qubit states          │
│           ↓                                              │
│  4. PAULI-Z MEASUREMENT (Eq.14)                         │
│     Read out 4 numbers per patch → FC layers → prediction│
│           ↓                                              │
│  5. CROSS-ENTROPY LOSS (Eq.15)                          │
│     How wrong was the prediction? → penalty score        │
│           ↓                                              │
│  6. PARAMETER-SHIFT RULE (Eq.17)                        │
│     Compute gradient for each of 16 quantum parameters   │
│     (2 circuit runs per parameter = 32 extra runs)       │
│           ↓                                              │
│  7. ADAM OPTIMIZER (Eq.18)                              │
│     Update ALL parameters (quantum + classical)          │
│           ↓                                              │
│  Repeat: 50 epochs × 313 batches = 15,650 steps total   │
└──────────────────────────────────────────────────────────┘
```

---

## C. Experiment 3 Noise Types — Simple Explanation

**The big question Experiment 3 asks:**
> *"If the real world is messy and noisy, does our model still work?"*

Like testing a car — first on a smooth road (Experiment 2), then on **bumpy roads, rain, and fog** (Experiment 3).

Two categories of noise:
- **Data Noise** → the image is corrupted before entering the model
- **Quantum Noise** (3 types) → corruption happens **inside the quantum circuit itself**

---

### C.1 Gaussian Data Noise
**"Someone spilled salt and pepper on your photo"**

> Classical image noise — like taking a photo in a dark room or a blurry scan. Nothing to do with quantum.

**Simple analogy:** Reading a textbook where someone has **randomly scratched out letters**. Some pages lightly scratched (error 0.1), some heavily (error 0.3). Can you still understand the content?

**What actually happens:**
```
Original pixel:  0.75
Add noise:       0.75 + random(-0.2 to +0.2)  =  0.55 to 0.95
Clipped to [0,1]: still valid
```

**Paper results (Proposed model):**

| Noise level | Accuracy |
|---|---|
| Clean (0.0) | **90.05%** |
| Light (0.1) | 89.15% |
| Medium (0.2) | 89.00% |
| Heavy (0.3) | **84.25%** — still strong! |

---

### C.2 Bit-Flip Channel (Eq. 25)
**"Someone randomly flips light switches in your quantum circuit"**

> A quantum hardware error — real quantum computers are fragile. Sometimes a qubit randomly **flips from 0 to 1 or from 1 to 0** by itself.

**Simple analogy:** You're sending a text message "HELLO" — with bit-flip noise it arrives as "HXLLO". Higher error rate → more random letters get flipped. Can the receiver still understand?

**What actually happens:**
```
Normal qubit:    |0⟩ stays as |0⟩      (no error)
Bit-flip error:  |0⟩ flips to |1⟩     with probability p
                 |1⟩ flips to |0⟩     with probability p

Math (Eq. 25):
ρ → (1-p)·ρ + p·X·ρ·X
```

**Paper results:**

| Error rate | Proposed | HQNN-Quanv | QNN |
|---|---|---|---|
| 0.1 | 87.69% | 67.75% | 71.15% |
| 0.3 | **84.05%** | **63.99%** | **46.15%** |

> Our model barely changes. Competing models collapse at p=0.3.

---

### C.3 Phase-Flip Channel (Eq. 26)
**"Someone secretly reverses the spin direction of a qubit"**

> Phase is invisible in classical computing. A phase-flip doesn't change **what** a qubit is (0 or 1), it changes **how** it interferes with other qubits.

**Simple analogy:** Two speakers playing the same note can either add together (louder) or cancel each other (silence). Phase-flip is like **secretly reversing one speaker's wave** — the sound changes even though the speaker looks identical.

**What actually happens:**
```
|0⟩ → stays as |0⟩           (phase flip does nothing to |0⟩)
|1⟩ → becomes -|1⟩           (phase is reversed)
|+⟩ → becomes |−⟩            (superposition states are flipped)

Math (Eq. 26):
ρ → (1-p)·ρ + p·Z·ρ·Z
```

**Comparison — Bit-flip vs Phase-flip:**
```
Bit-flip:    |0⟩ → |1⟩     (flips the VALUE: 0 becomes 1)
Phase-flip:  |1⟩ → -|1⟩   (flips the PHASE: + becomes -)
```

**Paper results (Proposed):**

| Error rate | Accuracy |
|---|---|
| 0.1 | 88.36% |
| 0.3 | **86.02%** — very stable |

> Phase-flip is gentler than bit-flip because Pauli-Z measurement is naturally aligned with the Z-axis.

---

### C.4 Depolarizing Channel (Eq. 27)
**"The worst case — complete random corruption"**

> The most realistic and most damaging noise model. Combines **all three types** of errors at once (X, Y, and Z errors). Models the general decay of a qubit toward a completely random state.

**Simple analogy:**
- Bit-flip = someone flips a specific switch
- Phase-flip = someone reverses a specific wave
- Depolarizing = **dunking the whole circuit in chaos** — with probability `p`, the qubit **forgets everything** and becomes completely random

**What actually happens:**
```
With probability (1-p):  qubit survives perfectly
With probability p/3:    X error applied  (bit-flip)
With probability p/3:    Y error applied  (bit + phase flip)
With probability p/3:    Z error applied  (phase-flip)

Math (Eq. 27):
ρ → (1-p)·ρ + (p/3)·(X·ρ·X + Y·ρ·Y + Z·ρ·Z)

Extreme values:
  p = 0.0 → no noise, perfect circuit
  p = 1.0 → qubit is completely random, no information
```

**Paper results (Proposed):**

| Error rate | Accuracy |
|---|---|
| 0.1 | 86.39% |
| 0.3 | **83.27%** — best in class |

---

### C.5 All 4 Noises Side-by-Side

| Noise | Where applied | What it corrupts | Real-world analogy |
|---|---|---|---|
| **Gaussian Data** | Input image | Pixel brightness | Blurry/grainy photo |
| **Bit-Flip** | Inside quantum circuit | Qubit value (0↔1) | Typo while typing |
| **Phase-Flip** | Inside quantum circuit | Qubit phase | Speaker playing inverted sound |
| **Depolarizing** | Inside quantum circuit | Everything randomly | Full memory loss of qubit |

---

### C.6 Why Our Model Survives All 4

| Noise | Proposed (p=0.3) | Competitor best |
|---|---|---|
| Data noise | **84.25%** | 78.44% (CNN) |
| Bit-flip | **84.05%** | 63.99% (HQNN-Quanv) |
| Phase-flip | **86.02%** | 82.67% (HQNN-Quanv) |
| Depolarizing | **83.27%** | 60.59% (HQNN-Quanv) |

> **Secret to our robustness:** Circuit-11's **shallow depth** (only 4 layers) means fewer opportunities for errors to accumulate. Deep circuits have many points where noise can enter — our design minimises those points. 🛡️

---

## D. Why Experiments Take 2–3 Days

### The #1 Killer: Triple Nested Loop in `QuantumConvLayer.forward()`

```python
for b in range(B):           # 32 images per batch
    for i in range(out_H):   # 14 rows
        for j in range(out_W): # 14 columns
            result = self._qnode(patch, self.weights)
            # ☝️ ONE quantum circuit run per patch — pure Python, no parallelism
```

**One forward pass = `32 × 14 × 14 = 6,272 quantum circuit calls`** — completely sequential.

### The Full Time Math

#### Experiment 1 — Why it's the slowest

| Step | Cost per circuit | 11 circuits | Total |
|---|---|---|---|
| **Expressibility metric** | 5,000 × 2 circuit runs = 10,000 runs | ×11 | 110,000 circuit runs |
| **Entanglement metric** | 5,000 circuit runs | ×11 | 55,000 circuit runs |
| **Discreteness metric** | 5,000 × 2 × 4 params = 40,000 runs | ×11 | 440,000 circuit runs |
| **Classification (MNIST)** | 50 epochs × 313 batches × 6,272 runs/batch | ×11 | **~108 million runs** |
| **Classification (Fashion)** | same | ×11 | **~108 million runs** |

> 💀 **Experiment 1 alone = ~216 million+ quantum circuit calls, all sequential on CPU**

#### Experiment 2 — Per model per dataset

| Step | Circuit calls |
|---|---|
| 1 epoch × MNIST | 313 batches × 6,272 = ~2M calls |
| 50 epochs × MNIST | ~100M calls |
| 2 models × 3 datasets | **~600M calls total** |

#### Experiment 3 — The `default.mixed` penalty

`default.mixed` uses **density matrix simulation** = tracks a `2^4 × 2^4 = 16×16` complex matrix instead of a 16-element state vector. It is **~4× slower** than `default.qubit`. Plus, no batching = every image is processed one by one.

| Bottleneck | Cause | Impact |
|---|---|---|
| **Triple nested loop** | 6,272 sequential QNode calls per batch | 80% of total time |
| **5,000 sims × 11 circuits × 3 metrics** | Experiment 1 metric analysis | +6–8 hours |
| **Parameter-shift rule** | 2 extra circuit runs per parameter per step | 2× training slowdown |
| **default.mixed in Exp 3** | Density matrix = 4× slower than default.qubit | +3–4 hours |
| **CPU only** | No GPU for quantum simulation | Cannot be GPU-accelerated |

### Root Causes Summary

```
Reason 1 ── TRIPLE NESTED LOOP (biggest cause)
            6,272 sequential QNode calls per batch. No batching. No parallelism.

Reason 2 ── 5,000 SIMULATIONS × 11 CIRCUITS × 3 METRICS in Experiment 1
            Discreteness = 5,000 × 16 params × 2 = 160,000 circuit runs per circuit

Reason 3 ── default.mixed IN EXPERIMENT 3 (4× slower)
            Density matrix simulation required for noise channels

Reason 4 ── PARAMETER-SHIFT RULE (2× circuit runs per gradient step)
            16 params × 2 = 32 extra circuit runs per batch

Reason 5 ── CPU ONLY for all quantum simulation
            PennyLane default.qubit/default.mixed cannot use GPU
```

### Speed-Up Options

```bash
# Skip 5,000-sim metric analysis (saves 4–6 hours on Exp 1)
python experiments/experiment1_circuit_selection.py --skip-metrics

# Use fewer sims for a quick test
python experiments/experiment1_circuit_selection.py --n-sims 500

# Run only MNIST (skip Fashion-MNIST and Overhead-MNIST)
python experiments/experiment2_classification.py --dataset mnist
```

| Approach | Estimated Time |
|---|---|
| Full run, CPU only | **2–3 days** |
| `--skip-metrics` on Exp 1 | ~18–24 hours |
| `--n-sims 500` on Exp 1 | ~14–18 hours |
| All optimisations combined | **~8–12 hours** ✅ |

> ⚠️ You cannot go below ~8 hours because the 6,272 sequential circuit calls per batch is a fundamental quantum simulation limitation. The paper authors reduced the dataset to 10,000 samples (from 60,000) specifically to make training feasible.

---

## E. Dataset Sizes — All Experiments

### Full Original Dataset Size

| Dataset | Train Images | Test Images | **Total** |
|---|---|---|---|
| **MNIST** | 60,000 | 10,000 | **70,000** |
| **Fashion-MNIST** | 60,000 | 10,000 | **70,000** |
| **Overhead-MNIST** | 8,519 | 1,065 | **9,584** |
| | | **Grand Total** | **149,584** |

### What the Paper Actually Uses (Subsampled)

| Dataset | Train Used | Test Used | **Total Used** |
|---|---|---|---|
| **MNIST** | 10,000 | 2,000 | **12,000** |
| **Fashion-MNIST** | 10,000 | 2,000 | **12,000** |
| **Overhead-MNIST** | 8,519 | 1,065 | **9,584** |
| | | **Grand Total** | **33,584** |

### Per Experiment — Exact Images Used

```
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│    Dataset       │  Experiment 1│  Experiment 2│  Experiment 3│
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ MNIST            │              │              │              │
│  - Train         │  10,000      │  10,000      │  pre-trained │
│  - Test          │   2,000      │   2,000      │  10,000 ✅   │
│  - Batch size    │  32          │  32          │  100         │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Fashion-MNIST    │              │              │              │
│  - Train         │  10,000      │  10,000      │  ❌ Not used │
│  - Test          │   2,000      │   2,000      │  ❌          │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ Overhead-MNIST   │              │              │              │
│  - Train         │  ❌ Not used │   8,519      │  ❌ Not used │
│  - Test          │  ❌          │   1,065      │  ❌          │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ TOTAL            │  24,000      │  33,584      │  10,000      │
└──────────────────┴──────────────┴──────────────┴──────────────┘
```

### Per-Class Breakdown

```
MNIST / Fashion-MNIST (10 classes):
  Each class: 1,000 train + 200 test
  Total = 10 × 1,000 + 10 × 200 = 10,000 + 2,000 = 12,000

Overhead-MNIST (10 aerial classes):
  Full dataset = 8,519 train + 1,065 test = 9,584

Experiment 3 test set: 10,000 full MNIST test images (no subsampling)
```

### Why Only 12,000 from 70,000?

> The full 70,000 MNIST images would take **weeks** to train because the quantum circuit processes each image one 2×2 patch at a time (6,272 sequential calls per 32-image batch). The paper reduces to 12,000 so experiments finish in hours instead of weeks.

---

*Part 2 added: 2026-09-01 — plain-language Q&A session explanations*

