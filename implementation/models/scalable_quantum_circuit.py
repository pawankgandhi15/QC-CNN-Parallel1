"""
scalable_quantum_circuit.py
============================
Configurable quantum circuit with variable qubit count and depth.

This is a SEPARATE file from quantum_circuit.py. It does NOT modify the
original Circuit 11 implementation. Instead, it provides a generalized
version that allows Experiment 5 to sweep across different configurations.

Parameters
----------
n_qubits : int  — number of qubits (2, 4, 6, 8)
n_layers : int  — circuit depth / number of variational layers (1–5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


# ---------------------------------------------------------------------------
# 1. Scalable QNode factory
# ---------------------------------------------------------------------------
def make_scalable_circuit(n_qubits: int = 4, n_layers: int = 2):
    """
    Create a QNode with configurable qubit count and depth.

    The circuit follows the Circuit-11 pattern (RY rotations + CRX entangling
    in circle topology) but generalized to arbitrary qubit counts and depths.

    Parameters
    ----------
    n_qubits : int  — number of qubits (determines patch size)
    n_layers : int  — number of variational layers (each = RY + CRX ring)

    Returns
    -------
    qnode      : callable QNode
    n_params   : int — total number of trainable parameters
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    # Each layer has: n_qubits RY rotations + n_qubits CRX entangling = 2*n_qubits params
    n_params = n_qubits * 2 * n_layers

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        """
        Scalable PQC: angle encoding + n_layers × (RY + CRX ring).

        inputs  : shape (n_qubits,) — patch pixels scaled by π
        weights : shape (n_params,) — trainable parameters
        """
        # Angle encoding (same as paper Eq. 6)
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(inputs[i], wires=i)

        # Variational layers
        idx = 0
        for layer in range(n_layers):
            # RY rotation layer
            for i in range(n_qubits):
                qml.RY(weights[idx], wires=i)
                idx += 1

            # CRX entangling ring (circle topology, shifted by layer)
            for i in range(n_qubits):
                ctrl   = (i + layer) % n_qubits
                target = (i + layer + 1) % n_qubits
                qml.CRX(weights[idx], wires=[ctrl, target])
                idx += 1

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit, n_params


# ---------------------------------------------------------------------------
# 2. Scalable QuantumConvLayer
# ---------------------------------------------------------------------------
class ScalableQuantumConvLayer(nn.Module):
    """
    Quantum convolutional layer with configurable qubit count and depth.

    Unlike the original QuantumConvLayer (fixed 4 qubits, 2×2 window),
    this supports arbitrary patch sizes based on qubit count:
      - 2 qubits → 1×2 patch, stride (1,2)
      - 4 qubits → 2×2 patch, stride 2   (paper default)
      - 6 qubits → 2×3 patch, stride (2,3)
      - 8 qubits → 2×4 patch, stride (2,4)
    """

    # Patch configurations for each qubit count
    PATCH_CONFIG = {
        2: {"patch_h": 1, "patch_w": 2, "stride_h": 1, "stride_w": 2},
        4: {"patch_h": 2, "patch_w": 2, "stride_h": 2, "stride_w": 2},
        6: {"patch_h": 2, "patch_w": 3, "stride_h": 2, "stride_w": 3},
        8: {"patch_h": 2, "patch_w": 4, "stride_h": 2, "stride_w": 4},
    }

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Create the scalable circuit
        self._qnode, n_params = make_scalable_circuit(n_qubits, n_layers)
        self.weights = nn.Parameter(torch.randn(n_params) * 0.1)

        # Get patch configuration
        if n_qubits not in self.PATCH_CONFIG:
            raise ValueError(f"Unsupported n_qubits={n_qubits}. "
                           f"Choose from {list(self.PATCH_CONFIG.keys())}")
        cfg = self.PATCH_CONFIG[n_qubits]
        self.patch_h  = cfg["patch_h"]
        self.patch_w  = cfg["patch_w"]
        self.stride_h = cfg["stride_h"]
        self.stride_w = cfg["stride_w"]

    def get_output_shape(self, H: int = 28, W: int = 28):
        """Compute output spatial dimensions for a given input size."""
        out_H = H // self.stride_h
        out_W = W // self.stride_w
        return out_H, out_W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : shape [B, 1, H, W]

        Returns
        -------
        out : shape [B, n_qubits, out_H, out_W]
        """
        B, C, H, W = x.shape
        out_H, out_W = self.get_output_shape(H, W)
        out = torch.zeros(B, self.n_qubits, out_H, out_W,
                         device=x.device, dtype=x.dtype)

        for b in range(B):
            for i in range(out_H):
                for j in range(out_W):
                    # Extract patch and flatten
                    r_start = i * self.stride_h
                    c_start = j * self.stride_w
                    patch = x[b, 0,
                             r_start:r_start + self.patch_h,
                             c_start:c_start + self.patch_w].flatten()

                    # Pad if patch is smaller than n_qubits (edge case)
                    if patch.shape[0] < self.n_qubits:
                        pad = torch.zeros(self.n_qubits - patch.shape[0],
                                        device=x.device, dtype=x.dtype)
                        patch = torch.cat([patch, pad])

                    # Scale by π (angle encoding, paper Eq. 6)
                    patch = patch * torch.pi
                    result = torch.stack(self._qnode(patch, self.weights))
                    out[b, :, i, j] = result

        return out


# ---------------------------------------------------------------------------
# 3. Scalable QC-CNN-Parallel model
# ---------------------------------------------------------------------------
class ScalableQCCNNParallel(nn.Module):
    """
    QC-CNN-Parallel with configurable quantum circuit parameters.

    Same architecture as the original, but allows varying:
    - n_qubits: number of qubits (affects quantum channel count)
    - n_layers: circuit depth (affects expressibility)

    The classical branch (8 filters) and FC head adapt automatically.
    """

    def __init__(self, num_classes: int = 10, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical branch (same as original)
        self.classical_conv = nn.Conv2d(
            in_channels=1, out_channels=8,
            kernel_size=4, stride=2, padding=1
        )

        # Scalable quantum branch
        self.quantum_conv = ScalableQuantumConvLayer(n_qubits, n_layers)

        # Compute fused feature size
        # Classical: [B, 8, 14, 14]
        # Quantum:   [B, n_qubits, out_H, out_W]
        q_out_H, q_out_W = self.quantum_conv.get_output_shape(28, 28)
        # Both branches must produce same spatial dims for concatenation
        # Classical always outputs 14×14 for 28×28 input
        # For quantum with different strides, we need adaptive pooling
        self._need_quantum_resize = (q_out_H != 14 or q_out_W != 14)
        if self._need_quantum_resize:
            self.quantum_pool = nn.AdaptiveAvgPool2d((14, 14))

        fused_channels = 8 + n_qubits
        fused_features = fused_channels * 14 * 14

        # FC head
        self.fc1 = nn.Linear(fused_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical branch
        x_class = F.relu(self.classical_conv(x))     # [B, 8, 14, 14]

        # Quantum branch
        x_quant = self.quantum_conv(x)               # [B, n_qubits, out_H, out_W]
        if self._need_quantum_resize:
            x_quant = self.quantum_pool(x_quant)     # [B, n_qubits, 14, 14]

        # Fuse and classify
        x_fused = torch.cat([x_class, x_quant], dim=1)
        x_flat  = x_fused.view(x_fused.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

    def count_parameters(self):
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "classical_conv": n(self.classical_conv),
            "quantum_pqc": n(self.quantum_conv),
            "conv_total": n(self.classical_conv) + n(self.quantum_conv),
            "total": n(self),
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
        }


# ---------------------------------------------------------------------------
# 4. Gradient variance measurement (barren plateau detection)
# ---------------------------------------------------------------------------
def measure_gradient_variance(model, dataloader, device, n_batches=5):
    """
    Measure gradient variance across batches to detect barren plateaus.

    High variance → healthy gradients
    Near-zero variance → barren plateau (training will stall)

    Returns
    -------
    dict with 'mean_grad_norm', 'grad_variance', 'max_grad', 'min_grad'
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    grad_norms = []

    for i, (images, labels) in enumerate(dataloader):
        if i >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()

        # Collect gradient norms for quantum parameters
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

    grad_norms = np.array(grad_norms)
    return {
        "mean_grad_norm": float(grad_norms.mean()),
        "grad_variance": float(grad_norms.var()),
        "max_grad": float(grad_norms.max()),
        "min_grad": float(grad_norms.min()),
    }


__all__ = [
    "make_scalable_circuit",
    "ScalableQuantumConvLayer",
    "ScalableQCCNNParallel",
    "measure_gradient_variance",
]
