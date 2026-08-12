"""
quantum_circuit.py
==================
Parameterized Quantum Circuit (PQC) — Circuit 11
as described in the paper:

  "A Parallel Hybrid Quantum-Classical Convolutional Design Using
  Parameterized Quantum Circuits for Image Classification"
  Quantum Engineering (2026), article 6643049.

References
----------
- Section 3.1  : Feature Mapping / Angle Encoding (Equations 5–6)
- Section 3.2  : PQC architecture — Circuit 11 (Equations 7–8, 19–24)
- Section 3.3  : Pauli-Z measurement (Equations 13–14)
- Table 2      : Circuit 11 — 16 params, Expr=0.0071, Ent=0.5463, Disc=0.0191
- Figure 2,3,5 : Circuit diagrams
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------
NUM_QUBITS = 4                         # 2×2 patch → 4 pixels → 4 qubits
dev_pure  = qml.device("default.qubit", wires=NUM_QUBITS)   # analytic (main)
dev_mixed = qml.device("default.mixed", wires=NUM_QUBITS)   # noise experiments


# ---------------------------------------------------------------------------
# 1. Angle-encoding helper
#    Paper Eq. 6: |ψ⟩ = RY(x)·H|0⟩  for each qubit
# ---------------------------------------------------------------------------
def _angle_encode(inputs):
    """Apply H then RY to each qubit (angle encoding, paper Eq. 6)."""
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i], wires=i)      # angle = pixel × π  (applied by caller)


# ---------------------------------------------------------------------------
# 2. Circuit 11 layers  (paper Equations 22–24, Figure 5b)
#    Two variational RY layers interleaved with two Circle CRX layers.
#    First entangling ring  : q0→q1→q2→q3→q0  (starts from qubit 0)
#    Second entangling ring : q1→q2→q3→q0→q1  (shifted by 1 → starts at q1)
# ---------------------------------------------------------------------------
def _circuit11_layers(w_rot1, w_ent1, w_rot2, w_ent2):
    """
    Apply Circuit-11 PQC (paper Eq. 24):
      U11 = U_ent2 · U_rot2 · U_ent1 · U_rot1
    """
    # --- Variational rotation layer 1 (paper Eq. 22) ---
    for i in range(NUM_QUBITS):
        qml.RY(w_rot1[i], wires=i)

    # --- Entangling layer 1: Circle topology starting at qubit 0 ---
    #   Connects: 0→1, 1→2, 2→3, 3→0
    for i in range(NUM_QUBITS):
        ctrl   = i
        target = (i + 1) % NUM_QUBITS
        qml.CRX(w_ent1[i], wires=[ctrl, target])

    # --- Variational rotation layer 2 (paper Eq. 22) ---
    for i in range(NUM_QUBITS):
        qml.RY(w_rot2[i], wires=i)

    # --- Entangling layer 2: Circle topology SHIFTED — starts at qubit 1 ---
    #   Connects: 1→2, 2→3, 3→0, 0→1  (paper Eq. 23, starting qubit = q1)
    for i in range(NUM_QUBITS):
        ctrl   = (i + 1) % NUM_QUBITS
        target = (i + 2) % NUM_QUBITS
        qml.CRX(w_ent2[i], wires=[ctrl, target])


# ---------------------------------------------------------------------------
# 3. QNode — analytic (main experiments, default.qubit)
# ---------------------------------------------------------------------------
@qml.qnode(dev_pure, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Full Circuit 11 QNode for analytic (noise-free) simulation.

    Parameters
    ----------
    inputs  : torch.Tensor, shape (4,)
              Pixel values of one 2×2 patch, already scaled by π.
    weights : torch.Tensor, shape (16,)
              Trainable PQC parameters.

    Returns
    -------
    list of 4 Pauli-Z expectation values, each in [-1, 1].
    """
    _angle_encode(inputs)
    w_rot1, w_ent1, w_rot2, w_ent2 = (
        weights[0:4], weights[4:8], weights[8:12], weights[12:16]
    )
    _circuit11_layers(w_rot1, w_ent1, w_rot2, w_ent2)
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


# ---------------------------------------------------------------------------
# 4. Noisy QNode factory — used in Experiment 3
#    Supported channels: bit_flip, phase_flip, depolarizing
#    Paper Section 4.3.3, Equations 25–27
# ---------------------------------------------------------------------------
def make_noisy_circuit(noise_type: str, noise_prob: float):
    """
    Return a QNode that applies a noise channel before each measurement.

    Parameters
    ----------
    noise_type : str  — "bit_flip" | "phase_flip" | "depolarizing"
    noise_prob : float — error probability p ∈ {0.1, 0.2, 0.3}
    """
    @qml.qnode(dev_mixed, interface="torch")
    def noisy_circuit(inputs, weights):
        _angle_encode(inputs)
        w_rot1, w_ent1, w_rot2, w_ent2 = (
            weights[0:4], weights[4:8], weights[8:12], weights[12:16]
        )
        _circuit11_layers(w_rot1, w_ent1, w_rot2, w_ent2)

        # Apply noise channel immediately before measurement (paper Sec 4.3.3)
        for q in range(NUM_QUBITS):
            if noise_type == "bit_flip":
                qml.BitFlip(noise_prob, wires=q)          # Eq. 25: ρ→(1-p)ρ+pXρX
            elif noise_type == "phase_flip":
                qml.PhaseFlip(noise_prob, wires=q)        # Eq. 26: ρ→(1-p)ρ+pZρZ
            elif noise_type == "depolarizing":
                qml.DepolarizingChannel(noise_prob, wires=q)  # Eq. 27
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")

        return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

    return noisy_circuit


# ---------------------------------------------------------------------------
# 5. QuantumConvLayer — 2×2 sliding window, stride 2
#    Paper Section 3 / Figure 2
# ---------------------------------------------------------------------------
class QuantumConvLayer(nn.Module):
    """
    Quantum convolutional layer that slides a 4-qubit PQC (Circuit 11)
    over non-overlapping 2×2 patches with stride 2.

    Weight sharing: the same 16 PQC parameters are reused for every patch
    position and every image in the batch (analogue of a classical conv kernel).

    Output shape: [B, 4, H/2, W/2]
    """
    def __init__(self, qnode=None):
        super().__init__()
        # 16 trainable PQC parameters — Circuit 11
        self.weights = nn.Parameter(torch.randn(16) * 0.1)
        self._qnode  = qnode if qnode is not None else quantum_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, 1, H, W]   (H and W must be even)

        Returns
        -------
        out : torch.Tensor  shape [B, 4, H//2, W//2]
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Image spatial dims must be even."
        out_H, out_W = H // 2, W // 2
        out = torch.zeros((B, 4, out_H, out_W), device=x.device, dtype=x.dtype)

        for b in range(B):
            for i in range(out_H):
                for j in range(out_W):
                    # Extract 2×2 patch, flatten to (4,), scale by π (Eq. 6)
                    patch  = x[b, 0, i*2:i*2+2, j*2:j*2+2].flatten() * torch.pi
                    result = torch.stack(self._qnode(patch, self.weights))
                    out[b, :, i, j] = result
        return out
