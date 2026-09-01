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
