# Mathematical Methodology of the QC-CNN-Parallel Model

> **Cross-verification note:** This document has been verified against the paper
> PDF *A Parallel Hybrid Quantum-Classical Convolutional Design Using Parameterized
> Quantum Circuits for Image Classification*, Quantum Engineering (2026),
> article 6643049. All mathematical descriptions match the paper's Sections 3.1–3.5
> and the formulas in Equations 5–18. The PDF is now present in the workspace.
> Previously absent paper-specific details (Discreteness metric, Circuit-11
> selection rationale, parameter-shift rule) have been added in this version.

## 1. Problem definition

Let the training dataset be

$$
\mathcal{D} = \{(\mathbf{x}^{(n)}, y^{(n)})\}_{n=1}^{N},
$$

where $\mathbf{x}^{(n)}$ is a grayscale image and $y^{(n)}$ is its class label.
For the default experiment:

$$
\mathbf{x}^{(n)} \in [0,1]^{1 \times 28 \times 28},
\qquad y^{(n)} \in \{0,1,\ldots,C-1\},
\qquad C=10.
$$

The image has one channel, height 28, width 28, and normalized pixel values.
The model estimates a vector of class logits
$\mathbf{z}^{(n)} \in \mathbb{R}^{C}$, from which the predicted class is

$$
\hat{y}^{(n)} = \arg\max_{c \in \{0,\ldots,C-1\}} z_c^{(n)}.
$$

The `argmax` selects the class with the largest logit. Logits are unnormalized
scores; a probability distribution is obtained with softmax when required.

## 2. Overview of the hybrid computation

The model contains two parallel feature maps:

$$
\mathbf{F}_{\mathrm{class}} = f_{\mathrm{class}}(\mathbf{x};\boldsymbol{\theta}_{c}),
\qquad
\mathbf{F}_{\mathrm{quant}} = f_{\mathrm{quant}}(\mathbf{x};\boldsymbol{\theta}_{q}).
$$

Here, $f_{\mathrm{class}}$ is the classical convolutional branch,
$f_{\mathrm{quant}}$ is the quantum convolutional branch,
$\boldsymbol{\theta}_{c}$ contains classical convolution parameters, and
$\boldsymbol{\theta}_{q}$ contains the 16 trainable PQC parameters.

The feature maps are fused and classified as follows:

$$
\mathbf{F} = \operatorname{Concat}_{\mathrm{channel}}
\left(\mathbf{F}_{\mathrm{class}},\mathbf{F}_{\mathrm{quant}}\right),
$$

$$
\mathbf{z} = f_{\mathrm{head}}\left(\operatorname{vec}(\mathbf{F});
\boldsymbol{\theta}_{h}\right).
$$

`Concat` joins the branches along the channel axis. `vec` flattens the spatial
and channel dimensions into one vector. The head then maps the fused features
to class logits.

## 3. Classical convolutional branch

Let $x_{u,v}$ denote the input pixel at row $u$ and column $v$. The classical
branch uses eight learnable kernels $K^{(r)}$ of size $4 \times 4$ and biases
$b^{(r)}$, where $r \in \{1,\ldots,8\}$.

The pre-activation feature map is

$$
a^{(r)}_{i,j}
= b^{(r)} +
\sum_{u=0}^{3}\sum_{v=0}^{3}
K^{(r)}_{u,v}\,
\tilde{x}_{2i+u,\,2j+v},
$$

where $\tilde{x}$ is the input padded by one pixel on all sides. The indices
$2i$ and $2j$ implement stride 2.

The output after the ReLU activation is

$$
F_{\mathrm{class}}^{(r)}(i,j) = \operatorname{ReLU}(a^{(r)}_{i,j})
= \max(0,a^{(r)}_{i,j}).
$$

ReLU removes negative activations while preserving positive responses. For a
28 x 28 input, the output spatial dimension is

$$
H_{c}=W_{c}
= \left\lfloor\frac{28+2(1)-4}{2}\right\rfloor+1
=14.
$$

Therefore,

$$
\mathbf{F}_{\mathrm{class}} \in \mathbb{R}^{8 \times 14 \times 14}.
$$

The classical branch has

$$
8(1\cdot4\cdot4)+8=136
$$

trainable parameters: 128 kernel values and 8 biases.

## 4. Quantum patch extraction

The quantum branch divides the image into non-overlapping 2 x 2 patches. The
patch at output position $(i,j)$ is

$$
P_{i,j} =
\begin{bmatrix}
x_{2i,2j} & x_{2i,2j+1}\\
x_{2i+1,2j} & x_{2i+1,2j+1}
\end{bmatrix}.
$$

The patch is flattened in row-major order:

$$
\mathbf{p}_{i,j} =
\begin{bmatrix}
x_{2i,2j},\ x_{2i,2j+1},\ x_{2i+1,2j},\ x_{2i+1,2j+1}
\end{bmatrix}^{T}.
$$

Each patch produces four scalar inputs, one for each qubit. Since the image is
28 x 28 and the stride is 2, the number of patch locations is

$$
H_{q}=W_{q}=\frac{28}{2}=14.
$$

Thus the quantum branch evaluates the PQC 196 times per image, before batching
or implementation-level vectorization.

## 5. Quantum angle encoding

For patch vector $\mathbf{p}_{i,j}$, define the angle vector

$$
\boldsymbol{\alpha}_{i,j} = \pi\mathbf{p}_{i,j}.
$$

If pixels are normalized to $[0,1]$, then every angle lies in $[0,\pi]$.
This scaling converts an image intensity into a rotation angle and is the
interface between the classical image and the quantum circuit.

The encoding operation for each qubit $k$ is (paper Equation 6):

$$
|\psi\rangle = R_Y(x) H |0\rangle
= \frac{\cos(x/2)+\sin(x/2)}{\sqrt{2}}|0\rangle + \frac{\cos(x/2)-\sin(x/2)}{\sqrt{2}}|1\rangle
$$

The Hadamard gate creates a superposition before the $R_Y$ rotation modifies
the probability amplitudes according to the pixel value. For one qubit,

$$
R_Y(\alpha)=
\begin{bmatrix}
\cos(\alpha/2)&-\sin(\alpha/2)\\
\sin(\alpha/2)&\cos(\alpha/2)
\end{bmatrix}.
$$

Let $|0\rangle^{\otimes4}$ be the initial four-qubit state. The complete
encoding operation for all four qubits is:

$$
U_{\mathrm{enc}}(\boldsymbol{\alpha})
= \bigotimes_{k=0}^{3} R_Y(\alpha_k)H.
$$

The encoded state is

$$
|\psi_{\mathrm{enc}}\rangle
= U_{\mathrm{enc}}(\boldsymbol{\alpha})|0\rangle^{\otimes4}.
$$

The four rotations encode the four pixels independently at state-preparation
time; entangling gates later allow their features to interact.

## 6. Parameterized quantum circuit

The trainable PQC parameter vector is

$$
\boldsymbol{\vartheta} \in \mathbb{R}^{16}.
$$

It is divided into four four-dimensional vectors:

$$
\boldsymbol{\vartheta}=
\left[\boldsymbol{\vartheta}^{(1)}_{r},
\boldsymbol{\vartheta}^{(1)}_{e},
\boldsymbol{\vartheta}^{(2)}_{r},
\boldsymbol{\vartheta}^{(2)}_{e}\right],
$$

where $r$ denotes single-qubit rotations, $e$ denotes entangling rotations,
and the superscript denotes the variational layer.

### 6.1 Variational rotation layer 1

The first trainable rotation operator is

$$
U_{r}^{(1)}
= \bigotimes_{k=0}^{3}R_Y\left(\vartheta^{(1)}_{r,k}\right).
$$

It applies one trainable $R_Y$ gate to every qubit. This gives each qubit an
independent trainable transformation after data encoding.

### 6.2 Circular entangling layer 1

The first entangling operator is

$$
U_{e}^{(1)}
= \prod_{k=0}^{3}
CR_X\left(\vartheta^{(1)}_{e,k};,k\rightarrow(k+1)\bmod4\right).
$$

The four directed connections are

$$
0\rightarrow1,\quad1\rightarrow2,\quad2\rightarrow3,\quad3\rightarrow0.
$$

The controlled-$R_X$ gates couple neighboring qubits in a ring. This allows
information encoded in one patch pixel to influence other measured features.

### 6.3 Variational rotation layer 2

The second trainable rotation operator is

$$
U_{r}^{(2)}
= \bigotimes_{k=0}^{3}R_Y\left(\vartheta^{(2)}_{r,k}\right).
$$

This provides a second independent set of four trainable local rotations after
the first round of interactions.

### 6.4 Shifted entangling layer 2

The second entangling operator is

$$
U_{e}^{(2)}
= \prod_{k=0}^{3}
CR_X\left(\vartheta^{(2)}_{e,k};\,(k+1)\bmod4\rightarrow(k+2)\bmod4\right).
$$

Its directed connections are

$$
1\rightarrow2,\quad2\rightarrow3,\quad3\rightarrow0,\quad0\rightarrow1.
$$

The shifted pattern changes the control-target arrangement in the second
entangling stage and increases the circuit's interaction pattern.

### 6.5 Complete PQC unitary

The complete data-dependent circuit is

$$
U(\mathbf{p};\boldsymbol{\vartheta})
=U_{e}^{(2)}U_{r}^{(2)}U_{e}^{(1)}U_{r}^{(1)}
U_{\mathrm{enc}}(\pi\mathbf{p}).
$$

The rightmost operation acts first. Therefore, a patch is encoded first,
followed by rotation layer 1, entangling layer 1, rotation layer 2, and
entangling layer 2.

## 7. Quantum measurement and quantum feature map

For each qubit $k$, the circuit measures the Pauli-Z expectation value:

$$
q_k(\mathbf{p};\boldsymbol{\vartheta})
=\langle Z_k\rangle
=\langle0|^{\otimes4}
U^{\dagger}(\mathbf{p};\boldsymbol{\vartheta})
Z_k
U(\mathbf{p};\boldsymbol{\vartheta})
|0\rangle^{\otimes4}.
$$

The expectation value is the difference between the probabilities of measuring
the qubit as 0 and 1:

$$
\langle Z_k\rangle=P(q_k=0)-P(q_k=1),
\qquad -1\leq\langle Z_k\rangle\leq1.
$$

The four measured values form the quantum feature vector:

$$
\mathbf{q}_{i,j}=
\begin{bmatrix}
q_0(\mathbf{p}_{i,j};\boldsymbol{\vartheta})\\
q_1(\mathbf{p}_{i,j};\boldsymbol{\vartheta})\\
q_2(\mathbf{p}_{i,j};\boldsymbol{\vartheta})\\
q_3(\mathbf{p}_{i,j};\boldsymbol{\vartheta})
\end{bmatrix}.
$$

These values are placed at location $(i,j)$ in four quantum feature maps:

$$
F_{\mathrm{quant}}^{(k)}(i,j)=q_k(\mathbf{p}_{i,j};\boldsymbol{\vartheta}),
\qquad k=0,1,2,3.
$$

Consequently,

$$
\mathbf{F}_{\mathrm{quant}}\in\mathbb{R}^{4\times14\times14}.
$$

The same $\boldsymbol{\vartheta}$ is shared across all 196 patch locations,
which is the quantum analogue of weight sharing in convolution.

## 8. Feature fusion

The classical and quantum features are concatenated channel-wise:

$$
\mathbf{F}=\operatorname{Concat}_{c}
\left(\mathbf{F}_{\mathrm{class}},\mathbf{F}_{\mathrm{quant}}\right)
\in\mathbb{R}^{12\times14\times14}.
$$

The number of channels is $8+4=12$. The spatial dimensions remain 14 x 14
because both branches use stride 2 and produce aligned patch locations.

Flattening gives

$$
\mathbf{h}_0=\operatorname{vec}(\mathbf{F})
\in\mathbb{R}^{12\cdot14\cdot14}
=\mathbb{R}^{2352}.
$$

This is the reason the first dense layer requires 2352 input features.

## 9. Fully connected classification head

The first dense layer is

$$
\mathbf{h}_1=\operatorname{ReLU}(W_1\mathbf{h}_0+\mathbf{b}_1),
$$

where $W_1\in\mathbb{R}^{128\times2352}$ and
$\mathbf{b}_1\in\mathbb{R}^{128}$. It transforms the 2352 fused features into
128 learned representations.

The second dense layer is

$$
\mathbf{h}_2=\operatorname{ReLU}(W_2\mathbf{h}_1+\mathbf{b}_2),
$$

where $W_2\in\mathbb{R}^{64\times128}$ and
$\mathbf{b}_2\in\mathbb{R}^{64}$. It compresses the representation to 64
features.

The output layer is

$$
\mathbf{z}=W_3\mathbf{h}_2+\mathbf{b}_3,
$$

where $W_3\in\mathbb{R}^{C\times64}$ and
$\mathbf{b}_3\in\mathbb{R}^{C}$. It produces one logit per class.

No softmax is applied inside the model during training because
`CrossEntropyLoss` combines log-softmax and negative log-likelihood.

## 10. Probabilities and prediction

If class probabilities are required, apply softmax to the logits:

$$
p(y=c\mid\mathbf{x})
=\frac{\exp(z_c)}{\sum_{r=0}^{C-1}\exp(z_r)}.
$$

The denominator normalizes all class scores so that the probabilities sum to
one. The predicted class is the class with maximum probability, equivalently
the class with maximum logit.

## 11. Training objective

For a batch of $B$ examples, the multi-class cross-entropy loss is

$$
\mathcal{L}(\boldsymbol{\Theta})
=-\frac{1}{B}\sum_{n=1}^{B}
\log p\left(y^{(n)}\mid\mathbf{x}^{(n)}\right),
$$

where

$$
\boldsymbol{\Theta}
=\{\boldsymbol{\theta}_{c},\boldsymbol{\theta}_{q},\boldsymbol{\theta}_{h}\}
$$

contains the classical convolution parameters, quantum parameters, and dense
head parameters. The loss penalizes low probability assigned to the correct
class.

Using the logits directly, the same loss can be written as

$$
\mathcal{L}
=-\frac{1}{B}\sum_{n=1}^{B}
\left[z_{y^{(n)}}^{(n)}
-\log\left(\sum_{c=0}^{C-1}\exp(z_c^{(n)})\right)\right].
$$

This is the numerically stable form conceptually implemented by
`torch.nn.CrossEntropyLoss`.

## 12. Gradient-based optimization

The parameters are updated using Adam. In general, the optimization step is

$$
\boldsymbol{\Theta}_{t+1}
=\operatorname{Adam}\left(
\boldsymbol{\Theta}_{t},
\nabla_{\boldsymbol{\Theta}}\mathcal{L}_{t},
\eta\right),
$$

where $t$ is the training step and $\eta$ is the learning rate. The current
example uses $\eta=0.01$.

For the quantum parameters, the chain rule is

$$
\frac{\partial\mathcal{L}}{\partial\vartheta_s}
=\sum_{n=1}^{B}\sum_{i=0}^{13}\sum_{j=0}^{13}\sum_{k=0}^{3}
\frac{\partial\mathcal{L}}{\partial F_{\mathrm{quant}}^{(k)}(i,j)}
\frac{\partial q_k(\mathbf{p}^{(n)}_{i,j};\boldsymbol{\vartheta})}
{\partial\vartheta_s}.
$$

This equation explains how the classification loss reaches a shared quantum
parameter $\vartheta_s$ through every patch and every measured qubit. A
differentiable PennyLane PyTorch interface is required for this gradient path.

For analytic quantum simulation, the **parameter-shift rule** is used to
compute gradients (paper Equation 17):

$$
\frac{\partial E(\theta)}{\partial \theta_i}
= \frac{1}{2}\left[
E\left(\theta + \frac{\pi}{2}e_i\right)
- E\left(\theta - \frac{\pi}{2}e_i\right)
\right],
$$

where $e_i$ is the unit vector associated with the $i$-th parameter. The two
shifted circuit evaluations estimate the derivative of the expectation value
exactly (not approximately) for gates with a sinusoidal parameter dependence.
This rule is implemented automatically by the PennyLane PyTorch interface
when `interface="torch"` is specified (paper Section 4.2.2, page 7).

The weight update rule is (paper Equation 18):

$$
\theta'_i = \theta_i - \eta \frac{\partial L(\theta)}{\partial \theta_i},
$$

where $\eta$ is the learning rate. In practice the paper uses Adam rather than
plain gradient descent, so the effective update incorporates first- and
second-moment estimates of the gradient.

## 13. Parameter count

The total number of trainable parameters is

$$
P_{total}
=136+16+(2352\cdot128+128)+(128\cdot64+64)+(64C+C).
$$

For $C=10$:

$$
P_{total}=136+16+301184+8256+650=310242.
$$

The quantum contribution is only 16 parameters; most parameters are in the
dense classification head. This distinction should be reported when comparing
the hybrid model with a classical baseline.

**Paper comparison (Table 4, page 10):** The proposed model's convolutional
part uses only **136 parameters**, which is the lowest among all seven models
compared in the paper. The next lowest is QC-Inception (304) and the classical
CNN (464). All models share the same linear classification layer, so the
convolutional parameter count is the relevant comparison metric.

## 13b. PQC selection rationale (from paper Section 4.3.1 and Tables 2–3)

The paper used a three-metric selection process for the PQC design:

### Expressibility (Equation 9)

Measured as KL divergence between the PQC's fidelity distribution and the Haar
random distribution:

$$
\text{Expr} = D_{KL}(\hat{P}_{PQC}(F;\theta) \| P_{Haar}(F)).
$$

A value **closer to zero** indicates higher expressibility (better uniformity
over the Hilbert space). Circuit 11 achieves 0.0071 (Table 2, page 9).

### Meyer-Wallach Entanglement Measure (Equation 10)

$$
\text{Ent} = \frac{1}{|S|} \sum_{\theta_i \in S} Q(|\psi_{\theta_i}\rangle).
$$

Approaches 1 for strong entanglement. Circuit 11 achieves 0.5463.

### Discreteness — new metric introduced in the paper (Equations 12–13)

The paper introduces a new metric called **Discreteness** to capture gradient
heterogeneity, which neither expressibility nor entanglement measures:

$$
\text{Var}(g_i) = \frac{1}{N} \sum_{j=1}^{N} (g_i^{(j)} - \bar{g}_i)^2,
\qquad \bar{g}_i = \frac{1}{N} \sum_{j=1}^{N} g_i^{(j)},
$$

$$
\text{Disc} = \frac{1}{M} \sum_{i=1}^{M} \text{Var}(g_i),
$$

where $g_i^{(j)}$ is the gradient of the $i$-th parameter at the $j$-th
random initialization, $N$ is the number of random initializations, and $M$
is the total number of trainable parameters.

Circuit 11 achieves Discreteness = 0.0191, balancing between the near-zero
discreteness of RZ circuits (barren plateau risk) and the high discreteness
of RY circuits. Each circuit used 4 qubits and 5,000 numerical simulations.

### Circuit selection summary

| Circuit | Params | Expr ↓ | Ent | Disc | MNIST acc | Selected? |
|---|---:|---:|---:|---:|---:|---:|
| Circuit 10 | 28 | 0.0013 | 0.7180 | 0.0208 | 0.8254 | No (too many params) |
| **Circuit 11** | **16** | **0.0071** | **0.5463** | **0.0191** | **0.8057** | **Yes** |
| RY All-to-All | 4 | 0.3454 | 0.4520 | 0.1260 | 0.8006 | No |

Circuit 11 was chosen because it achieves the best balance of expressibility,
discreteness, and distinguishability **with only 16 parameters**, making it
hardware-efficient under NISQ constraints.

## 14. End-to-end mathematical expression

For image $\mathbf{x}$, the model can be summarized as

$$
\hat{\mathbf{z}}
=W_3\,\operatorname{ReLU}\left(
W_2\,\operatorname{ReLU}\left(
W_1\,\operatorname{vec}\left[
\operatorname{Concat}_{c}\left(
\operatorname{ReLU}(\operatorname{Conv}_{4\times4,s=2,p=1}(\mathbf{x})),
\operatorname{QConv}_{2\times2,s=2}(\mathbf{x};\boldsymbol{\vartheta})
\right)\right]
+\mathbf{b}_1\right)+\mathbf{b}_2\right)+\mathbf{b}_3.
$$

The quantum convolution is defined patch by patch as

$$
\operatorname{QConv}_{2\times2,s=2}(\mathbf{x};\boldsymbol{\vartheta})^{(k)}_{i,j}
=\langle Z_k\rangle_{\,U(\mathbf{p}_{i,j};\boldsymbol{\vartheta})|0\rangle^{\otimes4}},
$$

with $k\in\{0,1,2,3\}$. This complete expression shows the data flow from
the image, through the two parallel branches, into the final class logits.

## 15. Shape and implementation verification

For an input batch $\mathbf{x}\in\mathbb{R}^{B\times1\times28\times28}$, the
expected shapes are:

| Operation | Shape |
|---|---|
| Input | $[B,1,28,28]$ |
| Classical convolution + ReLU | $[B,8,14,14]$ |
| Quantum convolution | $[B,4,14,14]$ |
| Channel concatenation | $[B,12,14,14]$ |
| Flatten | $[B,2352]$ |
| First dense layer | $[B,128]$ |
| Second dense layer | $[B,64]$ |
| Output logits | $[B,C]$ |

Any change to image size, convolution stride, padding, patch size, or number
of output channels requires recalculating the first dense-layer dimension.
