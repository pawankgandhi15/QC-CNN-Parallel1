# Experimental Setup and Hardware Configuration

> **Cross-verification note:** This document has been verified against the paper
> PDF *A Parallel Hybrid Quantum-Classical Convolutional Design Using Parameterized
> Quantum Circuits for Image Classification*, Quantum Engineering (2026),
> article 6643049. All verified settings now reflect the paper's Sections 4.1,
> 4.2, and Table 4. Previously incorrect placeholders and missing details (dataset
> splits, Overhead-MNIST, noise simulator, batch sizes) have been corrected.

## 1. Experimental objective

The experiment evaluates a hybrid quantum-classical convolutional neural
network. The model contains:

- A classical convolution branch with 8 filters.
- A four-qubit quantum convolution branch.
- A 16-parameter parameterized quantum circuit.
- A fused fully connected classification head.

The primary output is the classification performance on an MNIST-like image
dataset. The experiment should compare the hybrid model with a classical
baseline using the same dataset split, preprocessing, optimizer, and
evaluation metrics.

## 2. Verified configuration from the implementation and paper (Table 4, page 10)

The current source code explicitly defines the following settings, all of which
have been confirmed to match the paper:

| Component | Configuration | Paper source |
|---|---|---|
| Framework | PyTorch with PennyLane | Section 4.2.2, page 7 |
| Quantum interface | PennyLane TorchLayer (QNode with `interface="torch"`) | Section 4.2.2, page 7 |
| Quantum device | `default.qubit` (main experiments) / `default.mixed` (noise experiments) | Section 4.2.2 / Section 4.3.3 |
| Number of qubits | 4 | Table 4, page 10 |
| Quantum measurements | Pauli-Z expectation on all 4 qubits | Section 3.3, page 6 |
| Classical convolution | 8 filters, kernel 4 x 4, stride 2, padding 1 | Section 3 / Figure 1 |
| Quantum window | 2 x 2, stride 2 | Section 3 / Figure 2 |
| Expected input | `[B, 1, 28, 28]` | Table 1, page 7 |
| Fused feature map | `[B, 12, 14, 14]` | Architecture derivation |
| Dense head | 2352 -> 128 -> 64 -> C | Section 3.4, page 6 |
| Optimizer | Adam | Table 4, page 10 |
| Learning rate | 0.01 | Table 4, page 10 |
| Random seed | 42 | Table 4, page 10 |
| Loss | Cross-entropy loss | Table 4 / Section 3.5 |
| Batch size (main) | **32** | Table 4, page 10 |
| Batch size (noise exp.) | **100** | Section 4.3.3, page 12 |
| Epochs | **50** | Table 4, page 10 |
| Example batch size in code | 4 (smoke test only) | `qc-cnn-parallel.py` |
| Quantum shots | Not specified; analytic expectation values | PennyLane default |
| PQC design | Circuit 11 (16 parameters, Circle topology) | Table 2–3, pages 9 |

The code currently performs one forward pass, one backward pass, and one Adam
update on randomly generated dummy data. It is therefore a model smoke test,
not a complete training experiment.

## 3. Software environment

### 3.1 Required packages

Install the minimum dependencies in an isolated Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch pennylane torchvision numpy
```

The exact versions should be recorded after installation:

```bash
python - <<'PY'
import sys
import numpy
import torch
import pennylane

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("PennyLane:", pennylane.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 3.2 Environment observed in this workspace

The available environment was inspected without changing it:

| Item | Observed value |
|---|---|
| Operating system | Linux 6.8.0-124-generic, x86_64 |
| Python | 3.10.12 |
| Visible CPU threads | 12 |
| NVIDIA utility | `nvidia-smi` was not available |
| PennyLane | Not installed when the script was first run |

The absence of `nvidia-smi` does not prove that no GPU exists, but GPU
availability must be verified with `torch.cuda.is_available()` after installing
PyTorch. The current code uses the PennyLane `default.qubit` simulator and does
not configure a physical quantum processor.

## 4. Quantum simulation hardware

### 4.1 Current simulator

The source creates the device using:

```python
dev = qml.device("default.qubit", wires=4)
```

This means the experiment is simulated on a classical computer. It is not an
experiment executed on IBM Quantum, AWS Braket, IonQ, Rigetti, or another
physical quantum device.

The simulator represents a four-qubit state vector with

$$
2^4=16
$$

complex amplitudes. Four qubits are small enough for exact state-vector
simulation, although the repeated Python-level patch loop can still make the
training procedure slow.

### 4.2 Measurement model

The circuit returns analytic Pauli-Z expectation values:

$$
\langle Z_k\rangle
=\langle\psi|Z_k|\psi\rangle,
\qquad k\in\{0,1,2,3\}.
$$

Because no `shots` argument is specified, the implementation does not model
finite-shot sampling noise. To study realistic hardware behavior, configure a
finite-shot simulator explicitly:

```python
dev = qml.device("default.qubit", wires=4, shots=1024)
```

Finite shots approximate the expectation value from measurement samples and can
introduce statistical noise into both forward values and gradients. The
analytic and finite-shot results must be reported as separate experiments.

### 4.3 Noise experiments simulator (verified from paper Section 4.3.3)

For the noise robustness experiments (Experiment 3 in the paper), the paper
uses PennyLane's **mixed-state simulator** (`default.mixed`) rather than
`default.qubit`:

```python
dev = qml.device("default.mixed", wires=4)
```

The `default.mixed` simulator supports quantum noise channels (bit-flip,
phase-flip, depolarizing) applied immediately prior to qubit measurement.
This simulator **does not support batched inputs**, which is why the paper
used a batch size of 100 (full MNIST validation/test set) for noise experiments
rather than 32 as in the main experiments.

The three noise types tested in the paper (Tables 5–8) are:
1. **Data noise** — Gaussian noise added to input images after dimensionality
   reduction (simulates perceptual-layer noise).
2. **Bit-flip noise** — Pauli-X gate applied with probability $p$:
   $\rho \to (1-p)\rho + p X\rho X$
3. **Phase-flip noise** — Pauli-Z gate applied with probability $p$:
   $\rho \to (1-p)\rho + p Z\rho Z$
4. **Depolarizing noise** — replaces state with maximally mixed state with
   probability $p$:
   $\rho \to (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$

Noise probabilities tested: $p \in \{0.1, 0.2, 0.3\}$.

## 5. Classical compute hardware

The classical computer performs the following work:

- Loads and preprocesses image batches.
- Applies the classical convolution branch.
- Executes the quantum simulator for every image patch.
- Computes gradients through the PyTorch/PennyLane interface.
- Updates all trainable parameters with Adam.

For a batch of size $B$, the quantum branch evaluates

$$
B\times14\times14=196B
$$

four-qubit circuits for each forward pass. This quantity is important when
reporting execution time. A batch size of 64 would require 12,544 circuit
evaluations per forward pass before counting additional evaluations needed by a
gradient method.

The current quantum layer allocates its output tensor on `x.device`, but the
PennyLane device itself is declared independently. A GPU can accelerate the
PyTorch operations, but it does not automatically make the default PennyLane
simulator execute on a GPU. The simulator backend and device placement should
be checked explicitly when claiming GPU acceleration.

## 6. Dataset and data split (verified from paper Table 1 and Section 4.2.1)

The paper uses three datasets. **The splits below differ from the standard
MNIST/Fashion-MNIST 60,000/10,000 split** because of quantum simulator
computational limitations:

| Dataset | Training | Test | Method |
|---|---:|---:|---|
| MNIST | 10,000 (1,000/class) | 2,000 (200/class) | Class-balanced random subsampling |
| Fashion-MNIST | 10,000 (1,000/class) | 2,000 (200/class) | Class-balanced random subsampling |
| Overhead-MNIST | 8,519 | 1,065 | Full dataset used |

For reproduction, use the same subsampling strategy:

```python
from torch.utils.data import Subset
import random

def subsample_balanced(dataset, samples_per_class):
    """Select samples_per_class samples for each class (balanced subsampling)."""
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        class_indices.setdefault(label, []).append(idx)
    selected = []
    for label, indices in class_indices.items():
        selected.extend(random.sample(indices, min(samples_per_class, len(indices))))
    return Subset(dataset, selected)

train_subset = subsample_balanced(full_train_dataset, samples_per_class=1000)
test_subset = subsample_balanced(full_test_dataset, samples_per_class=200)
```

Do not tune hyperparameters on the test set. The test set should be used only
for final evaluation. Dataset details and alternatives are documented in
`DATASETS.md`.

Recommended preprocessing:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
])
```

`ToTensor()` converts an 8-bit image to a floating-point tensor in `[0, 1]`.
This is compatible with the quantum angle mapping

$$
\boldsymbol{\alpha}=\pi\mathbf{p},
$$

which maps normalized patch pixels to angles in `[0,\pi]`.

## 7. Training protocol

The complete experiment should use a repeated training loop rather than the
single demonstration update in the current script:

```python
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
```

At the end of each epoch, evaluate on the validation set without gradient
tracking. Run the final test evaluation only after selecting the model and
hyperparameters.

### Verified optimizer settings

The current example specifies:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
```

The number of epochs, validation split, random seed, and scheduler are not
specified in the source and must be recorded for a complete experiment.

## 8. Reproducibility controls

Set seeds for Python, NumPy, and PyTorch before creating the dataset and model:

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

For strict reproducibility, also record:

- Python version.
- PyTorch, PennyLane, NumPy, and torchvision versions.
- Operating system and CPU model.
- GPU model, CUDA version, and driver version if applicable.
- Dataset version and download source.
- Random seed.
- Number of workers used by each `DataLoader`.
- Number of shots and quantum backend.
- Exact model hyperparameters.

Strict determinism can reduce performance and is not always available for every
GPU operation. Report whether the experiment prioritizes deterministic results
or maximum throughput.

## 9. Baselines and ablation experiments

To show the contribution of the quantum branch, use at least these conditions:

| Experiment | Classical branch | Quantum branch | Purpose |
|---|---|---|---|
| Classical baseline | Enabled | Replaced or removed | Measure conventional CNN performance |
| Quantum-only ablation | Removed | Enabled | Measure the quantum branch alone |
| QC-CNN-Parallel | Enabled | Enabled | Evaluate the proposed hybrid model |
| Finite-shot variant | Enabled | Enabled with finite shots | Measure sampling-noise effect |

All variants should use the same dataset split and evaluation procedure. If a
baseline has a different parameter count, report that difference rather than
presenting the comparison as parameter-matched.

## 10. Metrics

For a balanced $C$-class test set, report:

### Accuracy

$$
\operatorname{Accuracy}
=\frac{1}{N_{test}}
\sum_{n=1}^{N_{test}}
\mathbf{1}\left[\hat{y}^{(n)}=y^{(n)}\right].
$$

Accuracy is the fraction of correctly classified test examples.

### Loss

Report mean test cross-entropy using the same definition as training. Loss
captures confidence in addition to correctness.

### Macro-F1

For each class, compute precision and recall, then average the class-wise F1
scores. Macro-F1 is useful when classes are imbalanced, especially for
MedMNIST subsets.

### Efficiency

Report:

- Training time per epoch.
- Total training time.
- Inference time per batch or per image.
- Peak CPU/GPU memory.
- Number of quantum circuit evaluations.
- Number of shots for finite-shot experiments.

## 11. Suggested hardware table for the thesis

Replace the placeholders below with the actual machine used:

| Hardware item | Value to report |
|---|---|
| CPU | `[manufacturer and model]` |
| CPU cores/threads | `[value]` |
| System RAM | `[GB]` |
| GPU | `[model or None]` |
| GPU memory | `[GB or N/A]` |
| CUDA/cuDNN | `[version or N/A]` |
| Quantum processor | `None for current default.qubit simulation` |
| Quantum backend | `PennyLane default.qubit` |
| Qubits | `4` |
| Shots | `Analytic / [finite-shot value]` |
| Operating system | `Linux 6.8.0-124-generic x86_64 in current workspace` |

Do not claim that a physical quantum computer was used unless the experiment
was actually submitted to a QPU and the backend name, date, shot count, and
transpilation/device details are available.

## 12. Minimum experiment checklist

- [ ] Install PyTorch, PennyLane, NumPy, and torchvision.
- [ ] Replace dummy random data with a documented dataset.
- [ ] Confirm input shape `[B, 1, 28, 28]`.
- [ ] Normalize image pixels to `[0, 1]`.
- [ ] Set and record all random seeds.
- [ ] Record package versions and hardware information.
- [ ] Train for multiple epochs with train/validation/test separation.
- [ ] Evaluate the classical and hybrid baselines on the same split.
- [ ] Record analytic versus finite-shot simulation settings.
- [ ] Report accuracy, macro-F1, loss, runtime, and circuit evaluations.
- [ ] Store the best model checkpoint and experiment configuration.

## 13. Number of computers required

### Minimum requirement

Only **one computer** is required to run the current experiment because the
quantum circuit is simulated locally using PennyLane's `default.qubit` device.
The same computer performs dataset loading, classical neural-network
computation, quantum simulation, gradient calculation, and parameter updates.

The minimum practical configuration is:

| Resource | Minimum recommendation |
|---|---|
| Computers | **1** |
| CPU | 4 or more cores |
| System RAM | 8 GB minimum; 16 GB recommended |
| GPU | Optional; not required for four-qubit simulation |
| Quantum computer/QPU | Not required |
| Internet | Required only for the first dataset/package download |
| Storage | At least 5 GB free for environment, dataset, logs, and checkpoints |

The current workspace has one Linux computer with 12 visible CPU threads. This
is sufficient for a small MNIST or Fashion-MNIST experiment, although the
quantum convolution loop may be slow because it evaluates many circuits in
Python.

### Recommended research configuration

Use **one dedicated computer** for the main experiment and repeat each result
with multiple random seeds on that same machine. This keeps the software,
hardware, and runtime environment consistent. A GPU is useful for the classical
convolution and dense layers, but it is not automatically used by the
`default.qubit` simulator.

### When multiple computers are useful

Multiple computers are optional. They are useful when running independent jobs
in parallel, for example:

```text
Computer 1: random seed 1
Computer 2: random seed 2
Computer 3: random seed 3
```

For a hyperparameter sweep, one computer can run the jobs sequentially, or
several computers can each run different configurations. The number of
computers does not change the model architecture or the number of quantum
circuits per image; it only reduces wall-clock time when jobs are independent.

### Physical quantum hardware case

If the experiment is later executed on a remote quantum processor, a local PC
is still required for data preparation, job submission, result collection, and
classical optimization. In that case the setup has:

```text
1 local computer + 1 remote quantum processor
```

The remote processor is not counted as a second PC. Its provider, backend name,
number of shots, queue time, connectivity, gate errors, and transpilation
settings must be reported separately.

## 14. Current status

The current source is a functional architecture demonstration, but it is not
yet a complete paper-reproduction experiment because:

1. `pennylane` is not installed in the current environment.
2. The script uses random dummy images and labels.
3. It performs only one optimization step.
4. It does not define a train/validation/test split.
5. It does not record exact package versions or CPU/GPU details.
6. It uses a classical `default.qubit` simulator rather than a physical QPU.

These items should be completed before reporting experimental accuracy or
claiming reproduction of the paper's results.
