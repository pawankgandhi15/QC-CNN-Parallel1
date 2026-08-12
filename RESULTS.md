# Results and Result-Recording Template

> **Cross-verification note:** This file has been updated against the paper PDF
> *A Parallel Hybrid Quantum-Classical Convolutional Design Using Parameterized
> Quantum Circuits for Image Classification*, Quantum Engineering (2026),
> article 6643049. Sections 0.1–0.8 now contain values verified directly from
> the PDF tables and text. Placeholder fields in Sections 3–9 are for
> reproduction experiments that have not yet been run.

> This file stores the experimental results of the
> QC-CNN-Parallel paper and the reproduced implementation. The paper values
> transcribed below are kept separate from the current script's results because
> the current script uses dummy data for a single update.

## 0. Results transcribed from `Quantum Engineering .pdf`

The following values are transcribed from *A Parallel Hybrid Quantum-Classical
Convolutional Design Using Parameterized Quantum Circuits for Image
Classification*, Quantum Engineering (2026), article 6643049. References below
point to the paper's table and page. These are paper-reported values, not
results reproduced by the current script.

### Data-integrity rule

Only values explicitly present in the PDF have been added or updated. If a
result is absent from the PDF, its existing value is preserved; where no value
previously existed, it remains marked as `[enter]`. No missing accuracy, loss,
runtime, hardware, or comparison value should be estimated from a figure,
assumed from another dataset, or copied from a different experiment without a
clear source reference.

### 0.1 Headline result

The abstract reports average training-accuracy improvements of **4.89%**,
**2.77%**, and **6.24%** over existing hybrid quantum CNNs and classical CNNs
(the paper should be consulted for the exact averaging definition).

### 0.2 Dataset preparation — Table 1, page 7

| Dataset | Classes | Image size | Training | Test |
|---|---|---|---:|---:|
| MNIST | All classes | 28 x 28 x 1 | 10,000 | 2,000 |
| Fashion-MNIST | All classes | 28 x 28 x 1 | 10,000 | 2,000 |
| Overhead-MNIST | All classes | 28 x 28 x 1 | 8,519 | 1,065 |

The paper selected 1,000 samples per class for MNIST and Fashion-MNIST training,
and 200 samples per class for testing. The full Overhead-MNIST dataset was
used. No class reduction was applied.

### 0.3 Circuit expressibility study — Table 2, page 9

Each circuit used four qubits and was evaluated with 5,000 numerical
simulations. Lower expressibility is described as better uniformity over the
Hilbert space.

| Circuit | Topology | Params | Expressibility | Entangling | Discreteness |
|---|---|---:|---:|---:|---:|
| RX | Linear | 4 | 0.1755 | 0.5618 | 0.0280 |
| RX | Circle | 4 | 0.1679 | 0.7448 | 0.0060 |
| RX | All-to-All | 4 | 0.1738 | 0.6178 | 0.0330 |
| RY | Linear | 4 | 0.3317 | 0.4540 | 0.1256 |
| RY | Circle | 4 | 0.3552 | 0.6372 | 0.0607 |
| RY | All-to-All | 4 | 0.3454 | 0.4520 | 0.1260 |
| RZ | Linear | 4 | 0.1721 | 0.6248 | 2.7e-33 |
| RZ | Circle | 4 | 0.1670 | 0.7924 | 3.6e-33 |
| RZ | All-to-All | 4 | 0.1759 | 0.5653 | 0.0154 |
| Circuit 10 | Custom | 28 | 0.0013 | 0.7180 | 0.0208 |
| Circuit 11 | Custom | 16 | 0.0071 | 0.5463 | 0.0191 |

The paper selected Circuit 11 for the proposed model. Its 16 trainable
parameters are represented by `self.weights` in the current code.

### 0.4 Circuit classification study — Table 3, page 9

| Circuit | Topology | MNIST accuracy | Fashion-MNIST accuracy |
|---|---|---:|---:|
| RX | Linear | 0.6560 | 0.7364 |
| RX | Circle | 0.6066 | 0.6864 |
| RX | All-to-All | 0.7495 | 0.7647 |
| RY | Linear | 0.7530 | 0.7626 |
| RY | Circle | 0.7602 | 0.7763 |
| RY | All-to-All | 0.8006 | 0.7846 |
| RZ | Linear | 0.5496 | 0.6229 |
| RZ | Circle | 0.5481 | 0.6213 |
| RZ | All-to-All | 0.7249 | 0.6881 |
| Circuit 10 | Custom | 0.8254 | 0.7946 |
| Circuit 11 | Custom | 0.8057 | 0.7778 |

### 0.5 Common model settings — Table 4, page 10

| Setting | Paper value |
|---|---|
| Learning rate | 0.01 |
| Random seed | 42 |
| Loss | Cross-entropy |
| Batch size | 32 (main experiments) / 100 (noise experiments, Section 4.3.3) |
| Epochs | 50 |
| Optimizer | Adam |

**Important:** The paper reports **two different batch sizes**:
- **32** is used for the main classification experiments (Experiments 1 and 2,
  Table 4, page 10).
- **100** is used for the noise robustness experiments (Experiment 3, Section
  4.3.3, page 12), because the `default.mixed` noise simulator does not support
  batched inputs, so the experiment was simplified to the full MNIST validation
  and test sets with a batch size of 100.

The paper reports convolutional-part parameter counts of 464 (CNN), 448
(QC-CNN), 448 (HQNN-Quanv), 456 (VCNN), 512 (QC-ResNet), 304 (QC-Inception),
and **136 (Proposed)**. It reports four qubits for each quantum model and depth 3
for QC-Inception and the Proposed model. The paper notes that the table counts
only convolutional parameters because all models use the same linear layer.

### 0.5b Full model comparison configuration — Table 4, page 10

| Indicator | CNN | QC-CNN | HQNN-Quanv | VCNN | QC-ResNet | QC-Inception | **Proposed** |
|---|---|---|---|---|---|---|---|
| Conv. parameters | 464 | 448 | 448 | 456 | 512 | 304 | **136** |
| Qubits | — | 4 | 4 | 4 | 4 | 4 | **4** |
| Receptive field | 4 | 4 | 4 | 4 | 4 | 4 | **4** |
| Depth | 4 | 4 | 4 | 4 | 4 | 3 | **3** |

Note: the proposed model achieves the **lowest parameter count** in the
convolutional part while achieving the highest accuracy across all three
datasets.

### 0.6 Noise results — Tables 5–8, pages 12–13

#### Data noise — Table 5

| Model | No noise | Error 0.1 | Error 0.2 | Error 0.3 |
|---|---:|---:|---:|---:|
| Proposed | 0.9005 | 0.8915 | 0.8900 | 0.8425 |
| HQNN-Quanv | 0.8320 | 0.8344 | 0.7796 | 0.7125 |
| QNN | 0.8350 | 0.8170 | 0.7544 | 0.7100 |
| CNN | 0.8935 | 0.8840 | 0.8530 | 0.7844 |

#### Bit-flip noise — Table 6

| Model | No noise | Error 0.1 | Error 0.2 | Error 0.3 |
|---|---:|---:|---:|---:|
| Proposed | 0.9005 | 0.8769 | 0.8558 | 0.8405 |
| HQNN-Quanv | 0.8320 | 0.6775 | 0.6523 | 0.6399 |
| QNN | 0.8350 | 0.7115 | 0.6124 | 0.4615 |

> **Discrepancy note:** The paper's Table 6 shows HQNN-Quanv at error 0.3 as
> **0.6399**, but the accompanying text (page 13) states "accuracies of the
> HQNN-Quanv and QNN models decline to **0.3340** and 0.4615 respectively."
> The table value (0.6399) is the authoritative source; the text may contain a
> typographical error. Report both values if this distinction matters.

The CNN row was not included in Table 6 of the paper (noise analysis focused
on quantum models vs. the proposed model).

#### Phase-flip noise — Table 7

| Model | No noise | Error 0.1 | Error 0.2 | Error 0.3 |
|---|---:|---:|---:|---:|
| Proposed | 0.9005 | 0.8836 | 0.8618 | 0.8602 |
| HQNN-Quanv | 0.8320 | 0.8279 | 0.8254 | 0.8267 |
| QNN | 0.8350 | 0.8272 | 0.8191 | 0.8167 |

#### Depolarizing noise — Table 8

| Model | No noise | Error 0.1 | Error 0.2 | Error 0.3 |
|---|---:|---:|---:|---:|
| Proposed | 0.9005 | 0.8639 | 0.8664 | 0.8327 |
| HQNN-Quanv | 0.8320 | 0.7021 | 0.6502 | 0.6059 |
| QNN | 0.8350 | 0.7552 | 0.6944 | 0.5904 |

At error rate 0.3, the paper reports Proposed accuracies of 0.8425 for data
noise, 0.8405 for bit-flip noise, 0.8602 for phase-flip noise, and 0.8327 for
depolarizing noise.

### 0.7 Simulator and hardware result

The paper used PennyLane with its PyTorch interface. The noise experiments used
PennyLane's `default.mixed` mixed-state simulator. The paper states that the
noise analysis is simulator-based and that physical quantum-device evaluation
is future work. It does not specify a complete CPU model, GPU model, RAM size,
or computer count, so those values must not be guessed.

## 1. Current implementation results

The current `qc-cnn-parallel.py` is a smoke test, not a complete training run.
It currently:

- Creates random images of shape `[4, 1, 28, 28]`.
- Creates random labels for 10 classes.
- Executes one forward pass.
- Computes one cross-entropy loss.
- Executes one backward pass.
- Performs one Adam update.

The script could not complete in the current environment because PennyLane was
not installed. Therefore, no empirical accuracy or loss result is currently
available.

## 2. Results that can be verified analytically

These are architecture properties, not measured accuracy results.

| Quantity | Verified result |
|---|---:|
| Input shape | `[B, 1, 28, 28]` |
| Classical branch output | `[B, 8, 14, 14]` |
| Quantum branch output | `[B, 4, 14, 14]` |
| Fused feature shape | `[B, 12, 14, 14]` |
| Flattened feature size | 2352 |
| Quantum qubits | 4 |
| Quantum trainable parameters | 16 |
| Classical convolution parameters | 136 |
| Total parameters for 10 classes | 310,242 |
| Quantum measurements per patch | 4 Pauli-Z expectations |
| Patches per 28 x 28 image | 196 |
| Quantum outputs per image | 784 scalar features |

The total parameter count for `C` classes is

$$
P_{total}=136+16+(2352\cdot128+128)+(128\cdot64+64)+(64C+C).
$$

For the default ten-class model, this gives 310,242 trainable parameters.

## 3. Paper-result extraction table (verified from PDF)

The following values are taken directly from Tables 1 and 4 of the paper.
Reproduction placeholder values remain as `[enter]`.

| Result category | Paper value | Units/details | Paper table/figure/page |
|---|---|---|---|
| Dataset name | MNIST, Fashion-MNIST, Overhead-MNIST | Three datasets | Table 1, page 7 |
| Training samples (MNIST) | 10,000 (1,000/class) | Class-balanced subset | Table 1, page 7 |
| Training samples (Fashion-MNIST) | 10,000 (1,000/class) | Class-balanced subset | Table 1, page 7 |
| Training samples (Overhead-MNIST) | 8,519 | Full dataset | Table 1, page 7 |
| Test samples (MNIST) | 2,000 (200/class) | Class-balanced subset | Table 1, page 7 |
| Test samples (Fashion-MNIST) | 2,000 (200/class) | Class-balanced subset | Table 1, page 7 |
| Test samples (Overhead-MNIST) | 1,065 | Full dataset | Table 1, page 7 |
| Number of classes | 10 (MNIST/Fashion-MNIST), dataset-dependent (Overhead-MNIST) | Classes | Table 1, page 7 |
| Epochs | 50 | Epochs | Table 4, page 10 |
| Batch size (main) | 32 | Images/batch | Table 4, page 10 |
| Batch size (noise exp.) | 100 | Images/batch | Section 4.3.3, page 12 |
| Learning rate | 0.01 | Adam optimizer | Table 4, page 10 |
| Best validation accuracy (Proposed, MNIST) | 0.9005 | See Tables 5–8 (no-noise baseline) | Tables 5–8, pages 12–13 |
| Noise-free accuracy (Proposed, MNIST) | 0.9005 | Used as noise baseline | Tables 5–8, pages 12–13 |
| Noise-free accuracy (HQNN-Quanv) | 0.8320 | Comparison baseline | Tables 5–8, pages 12–13 |
| Noise-free accuracy (QNN) | 0.8350 | Comparison baseline | Tables 5–8, pages 12–13 |
| Noise-free accuracy (CNN) | 0.8935 | Comparison baseline | Table 5, page 12 |
| Test accuracy (reproduction) | `[enter]` | Percent | `[run experiment]` |
| Test loss (reproduction) | `[enter]` | Cross-entropy | `[run experiment]` |
| Precision (reproduction) | `[enter]` | Macro/weighted/binary | `[run experiment]` |
| Recall (reproduction) | `[enter]` | Macro/weighted/binary | `[run experiment]` |
| F1 score (reproduction) | `[enter]` | Macro/weighted/binary | `[run experiment]` |
| Training time (reproduction) | `[enter]` | Seconds/minutes/hours | `[run experiment]` |
| Inference time (reproduction) | `[enter]` | Per image or batch | `[run experiment]` |

## 4. Main performance comparison (paper model comparison)

The paper compares 7 models on 3 datasets. Validation accuracy curves are
shown in Figures 6, 7, 8. The proposed model achieves the highest accuracy
with the lowest convolutional parameter count (136 conv. params vs. 448–512
for comparable quantum models).

### Paper's model comparison configuration (Table 4)

| Model | Conv. Params | Qubits | Description |
|---|---:|---:|---|
| Classical CNN (LeNet-5 based) | 464 | — | Classical baseline |
| QC-CNN (Henderson et al.) | 448 | 4 | Fixed (non-trainable) quantum filters |
| HQNN-Quanv (Senokosov et al.) | 448 | 4 | Trainable PQCs |
| VCNN (Huang et al.) | 456 | 4 | Variational CNN |
| QC-ResNet (Shi et al.) | 512 | 4 | Quantum ResNet |
| QC-Inception (Wang et al.) | 304 | 4 | Quantum Inception |
| **QC-CNN-Parallel (Proposed)** | **136** | **4** | **Parallel hybrid (this paper)** |

### Comparison accuracy results (verified from paper, Figures 6-8)

| Model | MNIST accuracy | Fashion-MNIST accuracy | Overhead-MNIST accuracy |
|---|---:|---:|---:|
| Classical CNN (LeNet-5) | See Figure 6 | See Figure 7 | See Figure 8 |
| QC-CNN | See Figure 6 | See Figure 7 | See Figure 8 |
| HQNN-Quanv | 0.8320 (no-noise baseline) | See Figure 7 | See Figure 8 |
| VCNN | See Figure 6 | See Figure 7 | See Figure 8 |
| QC-ResNet | See Figure 6 | See Figure 7 | See Figure 8 |
| QC-Inception | See Figure 6 | See Figure 7 | See Figure 8 |
| **Proposed (QC-CNN-Parallel)** | **0.9005 (no-noise)** | See Figure 7 | See Figure 8 |

> **Note:** The paper reports training accuracy curves in Figures 6–8 but does
> not provide a single numeric table for final test accuracy per model per
> dataset. The no-noise accuracy of 0.9005 for the Proposed model comes from
> Tables 5–8 (used as noise baseline). For the complete per-model accuracy
> values, read from the plotted curves in the figures.

The paper abstract reports the Proposed model improves average training
accuracy by:
- **4.89%** over existing hybrid quantum CNNs
- **2.77%** over ... (see abstract, page 1 for exact averaging definition)
- **6.24%** over classical CNNs

### Reproduction results (to be filled after running experiments)

| Model | Dataset | Accuracy (%) | Loss | Macro-F1 (%) | Conv. Params | Runtime |
|---|---|---:|---:|---:|---:|---:|
| Classical CNN baseline | `[enter]` | `[enter]` | `[enter]` | `[enter]` | `[enter]` | `[enter]` |
| HQNN-Quanv | `[enter]` | `[enter]` | `[enter]` | `[enter]` | `[enter]` | `[enter]` |
| QC-CNN-Parallel (Proposed) | `[enter]` | `[enter]` | `[enter]` | `[enter]` | 136 | `[enter]` |

If the paper reports accuracy as a mean over multiple runs, use the format
`mean +/- standard deviation`, for example `94.20 +/- 0.35%`. Do not mix a
single-run value with a multi-run average without labeling it.

## 5. Dataset-wise results (paper split)

This table follows the paper's dataset preparation (Table 1). The
reproduction columns are to be filled after running experiments.

| Dataset | Input format | Classes | Train/test split (paper) | Paper accuracy (no-noise) | Reproduction accuracy | Reproduction Macro-F1 | Reproduction test loss |
|---|---|---:|---|---:|---:|---:|---:|
| MNIST | 1 x 28 x 28 | 10 | 10,000 / 2,000 (1,000 & 200 per class) | 0.9005 | `[enter]` | `[enter]` | `[enter]` |
| Fashion-MNIST | 1 x 28 x 28 | 10 | 10,000 / 2,000 (1,000 & 200 per class) | See Figure 7 | `[enter]` | `[enter]` | `[enter]` |
| Overhead-MNIST | 1 x 28 x 28 | ~11 aerial | 8,519 / 1,065 (full dataset) | See Figure 8 | `[enter]` | `[enter]` | `[enter]` |

## 6. Training-curve results

Record one row per epoch. This makes it possible to reproduce loss and accuracy
plots from the thesis.

| Epoch | Training loss | Training accuracy (%) | Validation loss | Validation accuracy (%) | Learning rate |
|---:|---:|---:|---:|---:|---:|
| 1 | `[enter]` | `[enter]` | `[enter]` | `[enter]` | `[enter]` |
| 2 | `[enter]` | `[enter]` | `[enter]` | `[enter]` | `[enter]` |
| 3 | `[enter]` | `[enter]` | `[enter]` | `[enter]` | `[enter]` |
| ... | ... | ... | ... | ... | ... |

Recommended plots:

1. Training and validation loss versus epoch.
2. Training and validation accuracy versus epoch.
3. Test confusion matrix.
4. Accuracy comparison between classical and hybrid models.
5. Runtime or circuit-evaluation count comparison.

## 7. Ablation results

Ablation experiments identify which part of the architecture contributes to the
result.

| Ablation | Removed/changed component | Accuracy (%) | Macro-F1 (%) | Loss | Interpretation |
|---|---|---:|---:|---:|---|
| Full model | None | `[enter]` | `[enter]` | `[enter]` | Reference model |
| No quantum branch | Quantum features removed | `[enter]` | `[enter]` | `[enter]` | Classical contribution |
| No classical branch | Classical features removed | `[enter]` | `[enter]` | `[enter]` | Quantum contribution |
| One PQC layer | Second variational block removed | `[enter]` | `[enter]` | `[enter]` | Circuit-depth effect |
| No entanglement | CRX gates removed | `[enter]` | `[enter]` | `[enter]` | Entanglement effect |
| Finite shots | Analytic expectation replaced by shots | `[enter]` | `[enter]` | `[enter]` | Sampling-noise effect |

## 8. Quantum execution results

Record the simulator or QPU configuration separately from classification
metrics.

| Quantum setting | Value |
|---|---|
| Backend | `default.qubit` / `[paper value]` |
| Physical QPU used | `No in current code` / `[paper value]` |
| Number of qubits | 4 |
| Shots | Analytic in current code / `[paper value]` |
| Circuit depth | `[measure and enter]` |
| Gates per patch | `[measure and enter]` |
| Circuit evaluations per image | 196 forward evaluations |
| Gradient method | PyTorch/PennyLane autodiff / `[paper value]` |
| Queue time | N/A for local simulator / `[paper value]` |
| Sampling error | Not present in analytic mode / `[paper value]` |

For finite-shot experiments, record the number of shots because changing shots
changes measurement variance and can affect training stability.

## 9. Hardware and runtime results

| Hardware/software item | Value |
|---|---|
| CPU model | `[enter]` |
| CPU cores/threads | `12 visible threads in current workspace` |
| System RAM | `[enter]` |
| GPU model | `[enter or None]` |
| GPU memory | `[enter or N/A]` |
| Operating system | `Linux 6.8.0-124-generic x86_64 in current workspace` |
| Python version | `3.10.12 in current workspace` |
| PyTorch version | `[enter after installation]` |
| PennyLane version | `[enter after installation]` |
| Dataset download time | `[enter]` |
| Training time | `[enter]` |
| Inference time/image | `[enter]` |
| Peak memory | `[enter]` |

Do not report the current workspace values as the paper's hardware unless the
paper experiment was actually run on this machine.

## 10. Statistical reporting

For a reliable result, train each model with at least three independent random
seeds when computationally feasible. Report

$$
\bar{a}=\frac{1}{R}\sum_{r=1}^{R}a_r,
$$

where $a_r$ is the metric from run $r$ and $R$ is the number of runs. Report
the sample standard deviation:

$$
s_a=\sqrt{\frac{1}{R-1}\sum_{r=1}^{R}(a_r-\bar{a})^2}.
$$

The final format should be, for example,
`mean accuracy +/- standard deviation`. Include the random seeds and explain
whether the split was fixed across all runs.

## 11. Result interpretation guidelines

- Compare models only on the same dataset and test split.
- Report both accuracy and macro-F1 for imbalanced datasets.
- Treat a higher accuracy as meaningful only when the run-to-run variation is
  also reported.
- Separate simulator results from real-QPU results.
- Separate analytic expectation values from finite-shot results.
- Report parameter count and runtime alongside accuracy.
- Do not claim a quantum advantage from accuracy alone.
- Include a classical baseline with a comparable training and evaluation setup.

## 12. Final result summary template

After entering the paper and reproduction results, complete this summary:

> On **[dataset]**, the QC-CNN-Parallel model achieved **[accuracy]%** test
> accuracy and **[macro-F1]%** macro-F1, compared with **[baseline accuracy]%**
> for the classical baseline. The model used **4 qubits**, **16 trainable
> quantum parameters**, **[shots] shots/analytic expectations**, and required
> **[runtime]** on **[hardware]**. Across **[number]** random seeds, the mean
> accuracy was **[mean]% +/- [std]%**.

## 13. Required files for a complete result package

Store these files with the final experiment:

```text
results/
├── paper_results.md          # Values transcribed from the paper
├── reproduction_results.csv  # One row per run/epoch
├── training_history.json     # Loss and metrics by epoch
├── confusion_matrix.png      # Final test confusion matrix
├── accuracy_curve.png        # Training/validation accuracy
├── loss_curve.png            # Training/validation loss
├── config.json               # Hyperparameters and seeds
└── environment.txt           # Package and hardware information
```

The paper values above are now recorded, but the reproduction placeholder
fields remain intentional until a complete training run is performed. They
should not be replaced with guessed values.
