# QC-CNN-Parallel Model Architecture

> **Cross-verification note:** This document has been verified against the paper
> PDF *A Parallel Hybrid Quantum-Classical Convolutional Design Using Parameterized
> Quantum Circuits for Image Classification*, Quantum Engineering (2026),
> article 6643049. All architecture details below match the paper's descriptions
> in Sections 3.1–3.5 and Figure 1 of the paper.
> The original research-paper PDF is now present in the workspace. Paper-level
> claims have been verified and confirmed.

## 1. Model objective

QC-CNN-Parallel is a hybrid quantum-classical convolutional neural network for
grayscale image classification (paper Section 3, page 3). It processes the same
image through two parallel feature-extraction branches:

1. A classical convolutional branch (4×4 kernel, stride 2, 8 output channels).
2. A quantum convolutional branch based on a four-qubit parameterized quantum
   circuit (PQC) with a 2×2 sliding window and stride 2.

The feature maps from both branches are concatenated along the channel dimension
and passed to a **three-layer** fully connected classification head (paper
Section 3.4, page 6). The classical CNN baseline used for comparison in the
paper is based on the **LeNet-5 architecture** (Table 4, page 10).

```text
Input image [B, 1, 28, 28]
             |
       -------------------
       |                 |
 Classical branch    Quantum branch
 Conv2d               2x2 quantum window
 [B, 8, 14, 14]       [B, 4, 14, 14]
       |                 |
       --------- Concatenate ---------
                 [B, 12, 14, 14]
                         |
                      Flatten
                    [B, 2352]
                         |
                 Linear 2352 -> 128
                         |
                       ReLU
                         |
                  Linear 128 -> 64
                         |
                       ReLU
                         |
                 Linear 64 -> C logits
```

Here, `B` is the batch size and `C` is the number of output classes. The
default implementation uses `C = 10`.

## 2. Input and preprocessing

- Input tensor shape: `[batch_size, 1, height, width]`.
- The implementation is designed for `28 x 28` grayscale images, such as
  MNIST.
- Pixel values are expected to be normalized to `[0, 1]` before entering the
  model.
- The quantum branch multiplies every extracted pixel patch by `pi` before
  angle encoding. Therefore, normalized values are mapped approximately to
  `[0, pi]`.
- The height and width must be even because the quantum branch uses a `2 x 2`
  window with stride `2`.

## 3. Classical convolutional branch

The classical branch is:

```text
Conv2d( in_channels=1,
        out_channels=8,
        kernel_size=4,
        stride=2,
        padding=1 )
ReLU
```

For a `28 x 28` input, the spatial output size is:

```text
floor((28 + 2*1 - 4) / 2) + 1 = 14
```

Output shape:

```text
[B, 8, 14, 14]
```

The number of trainable parameters is:

```text
Weights: 8 * 1 * 4 * 4 = 128
Biases:  8
Total:   136
```

## 4. Quantum convolutional branch

The quantum branch applies the same four-qubit PQC independently to every
non-overlapping `2 x 2` image patch.

### 4.1 Quantum sliding window

Configuration:

```text
Window:       2 x 2
Stride:       2
Input image:  28 x 28
Patches:      14 x 14
Qubits:       4
Measurements: 4
```

Each patch is flattened in row-major order:

```text
[[p00, p01],       [p00, p01, p10, p11]
 [p10, p11]]  ->
```

The resulting four values are used as four rotation angles:

```text
angles = pi * [p00, p01, p10, p11]
```

Each quantum circuit evaluation returns four expectation values, one for each
qubit. These four values become four output channels at the patch location.

Quantum branch output shape:

```text
[B, 4, 14, 14]
```

Because Pauli-Z expectation values lie in `[-1, 1]`, the quantum feature maps
are naturally bounded before feature fusion.

### 4.2 Four-qubit PQC

The quantum device uses four wires:

```text
q0 -- q1 -- q2 -- q3 -- back to q0
```

The circuit has 16 trainable parameters, divided into four groups of four:

```text
w_rot1 = weights[0:4]
w_ent1 = weights[4:8]
w_rot2 = weights[8:12]
w_ent2 = weights[12:16]
```

#### State preparation / angle encoding

For every qubit `qi`:

```text
H(qi)
RY(angle[i], qi)
```

This encodes one pixel from the `2 x 2` patch on each qubit.

#### Variational layer 1

Apply trainable single-qubit rotations:

```text
RY(w_rot1[i], qi)    for i = 0, 1, 2, 3
```

Then apply controlled rotations in a circular topology:

```text
CRX(w_ent1[0], q0 -> q1)
CRX(w_ent1[1], q1 -> q2)
CRX(w_ent1[2], q2 -> q3)
CRX(w_ent1[3], q3 -> q0)
```

#### Variational layer 2

Apply a second set of trainable rotations:

```text
RY(w_rot2[i], qi)    for i = 0, 1, 2, 3
```

#### Shifted entangling layer 2 (Circuit 11 — verified from paper Figure 5b)

The second entangling operator uses a Circle topology but **shifts the starting
control qubit** from q1 to q2. This prevents the two entangling patterns from
fully overlapping and enhances the expressive capacity of the circuit (paper,
page 8–9).

The first entangling layer starts at q0→q1 (control at q0).
The second entangling layer starts at q1→q2 (shifted: control at q1):

```text
CRX(w_ent2[0], q1 -> q2)   # starts from q1, not q0
CRX(w_ent2[1], q2 -> q3)
CRX(w_ent2[2], q3 -> q0)
CRX(w_ent2[3], q0 -> q1)
```

This matches the Circuit-11 formula from the paper (Equation 24, page 9):

$$
U_{11}(\theta, \phi) = U^{(2)}_{ent}(\phi_2) \, U^{(2)}_{rot}(\theta_2) \, U^{(1)}_{ent}(\phi_1) \, U^{(1)}_{rot}(\theta_1)
$$

where both entangling layers use a Circle topology but differ in their starting
control qubit: the first layer begins with qubit q0→q1, and the second starts
with qubit q1→q2 (shifted by one).

#### Measurement

Measure the Pauli-Z expectation value on every qubit:

```text
z[i] = <PauliZ(qi)>       for i = 0, 1, 2, 3
```

The PQC output is therefore a four-dimensional vector:

```text
[z0, z1, z2, z3], where each zi is in [-1, 1]
```

### 4.3 Quantum parameter count

The quantum convolution layer has one shared trainable parameter vector:

```text
16 parameters = 8 rotation parameters + 8 entangling parameters
```

The same PQC weights are reused for every image patch and every image in the
batch, just as a classical convolution kernel is shared spatially.

## 5. Feature fusion

The branch outputs are concatenated along the channel dimension:

```python
x_fused = torch.cat((x_class, x_quant), dim=1)
```

Shape calculation:

```text
Classical features: [B, 8, 14, 14]
Quantum features:   [B, 4, 14, 14]
Fused features:     [B, 12, 14, 14]
```

The two branches must have identical height and width. If the input size,
padding, kernel size, or stride is changed, the dense-layer input size must be
updated accordingly.

## 6. Classification head

After fusion, the feature tensor is flattened:

```text
[B, 12, 14, 14] -> [B, 12 * 14 * 14] -> [B, 2352]
```

The dense head is:

```text
Linear(2352, 128)
ReLU
Linear(128, 64)
ReLU
Linear(64, C)
```

The final layer returns raw logits. During training, use
`CrossEntropyLoss`, which applies the appropriate normalization internally.
Do not apply softmax before passing logits to `CrossEntropyLoss`.

## 7. Complete forward-pass pseudocode

```python
def forward(x):
    # Classical branch
    classical = relu(classical_conv(x))       # [B, 8, 14, 14]

    # Quantum branch
    quantum = quantum_conv(x)                 # [B, 4, 14, 14]

    # Fusion
    fused = concatenate(classical, quantum, axis=channel)  # [B, 12, 14, 14]
    flat = reshape(fused, [B, 2352])

    # Classifier
    hidden1 = relu(fc1(flat))                 # [B, 128]
    hidden2 = relu(fc2(hidden1))               # [B, 64]
    logits = fc3(hidden2)                     # [B, C]
    return logits
```

## 8. Parameter summary

| Component | Trainable parameters |
|---|---:|
| Classical `Conv2d` | 136 |
| Quantum PQC | 16 |
| `fc1`: `2352 -> 128` | 301,184 |
| `fc2`: `128 -> 64` | 8,256 |
| `fc3`: `64 -> 10` | 650 |
| **Total for 10 classes** | **310,242** |

The total is calculated as:

```text
136 + 16 + (2352*128 + 128) + (128*64 + 64) + (64*10 + 10)
= 310,242
```

## 9. Training specification

The current example uses:

```text
Optimizer: Adam
Learning rate: 0.01
Loss: CrossEntropyLoss
```

The forward and backward passes jointly update:

- Classical convolution weights and biases.
- The 16 quantum-circuit parameters.
- All fully connected layer parameters.

The quantum circuit must use a differentiable interface such as
PennyLane's PyTorch interface so gradients can flow from the classifier back
through the measured quantum outputs to the PQC parameters.

## 10. Implementation checklist

- [ ] Normalize input images consistently, normally to `[0, 1]`.
- [ ] Confirm that input height and width are even.
- [ ] Preserve the `2 x 2`, stride-2 patch ordering.
- [ ] Use exactly four qubits and four Pauli-Z measurements.
- [ ] Keep the 16 PQC parameters shared across all patches.
- [ ] Ensure both branches produce `14 x 14` feature maps for `28 x 28` input.
- [ ] Concatenate features along channels, giving 12 channels.
- [ ] Update the flatten dimension if the input resolution or convolution
      settings change.
- [ ] Feed logits directly to `CrossEntropyLoss`.
- [ ] Verify tensor shapes with a dummy batch before training.

## 11. Important reproduction notes

The current implementation evaluates the quantum circuit in nested Python
loops over batches and patch locations. This is faithful to the conceptual
sliding-window design but can be slow. A later optimization can batch quantum
patch evaluations or use a vectorized quantum-map implementation, provided
that the circuit, parameter sharing, patch ordering, and measurement scheme
remain unchanged.

The dense head is tied to `28 x 28` inputs because its first layer expects 2352
features. For a general input size, calculate the branch output dimensions
first and derive the first linear-layer size from the fused tensor shape.
