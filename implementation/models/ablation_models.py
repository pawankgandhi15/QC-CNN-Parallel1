"""
ablation_models.py
==================
Ablation model variants for Experiment 4.

These are standalone models that DO NOT modify the original QCCNNParallel
or ClassicalCNN classes. Each variant isolates one branch to measure its
individual contribution.

Variants
--------
1. ClassicalOnlyCNN     — Classical branch only (8 filters, no quantum)
2. QuantumOnlyCNN       — Quantum branch only (Circuit 11, no classical conv)
3. ClassicalExtendedCNN — Classical branch with 12 filters (parameter-matched)

All variants share the same 3-layer FC head architecture as the original
QCCNNParallel (Section 3.4) to ensure fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantum_circuit import QuantumConvLayer


# ---------------------------------------------------------------------------
# 1. Classical-Only  (remove quantum branch, keep 8 filters)
#    Conv params: 8*(1*4*4)+8 = 136 (same as original classical branch)
#    Fused shape: [B, 8, 14, 14] → flatten → [B, 1568]
# ---------------------------------------------------------------------------
class ClassicalOnlyCNN(nn.Module):
    """
    Classical-only ablation: uses the same 8-filter Conv2d as QCCNNParallel's
    classical branch, but WITHOUT the quantum branch.

    This isolates the classical contribution and answers:
    "Does the quantum branch add value beyond what classical conv provides?"
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Same classical conv as QCCNNParallel
        self.classical_conv = nn.Conv2d(
            in_channels=1, out_channels=8,
            kernel_size=4, stride=2, padding=1
        )
        # FC head adapted for 8-channel input (not 12)
        self.fc1 = nn.Linear(8 * 14 * 14, 128)    # 1568→128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.classical_conv(x))        # [B, 8, 14, 14]
        x = x.view(x.size(0), -1)                 # [B, 1568]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def count_parameters(self):
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "classical_conv": n(self.classical_conv),
            "quantum_pqc": 0,
            "conv_total": n(self.classical_conv),
            "total": n(self),
        }


# ---------------------------------------------------------------------------
# 2. Quantum-Only  (remove classical branch, keep Circuit 11)
#    Quantum params: 16 (PQC)
#    Output shape: [B, 4, 14, 14] → flatten → [B, 784]
# ---------------------------------------------------------------------------
class QuantumOnlyCNN(nn.Module):
    """
    Quantum-only ablation: uses only the QuantumConvLayer (Circuit 11)
    without any classical convolution.

    This isolates the quantum contribution and answers:
    "Can the quantum branch classify on its own?"
    """

    def __init__(self, num_classes: int = 10, qnode=None):
        super().__init__()
        self.quantum_conv = QuantumConvLayer(qnode=qnode)
        # FC head adapted for 4-channel quantum output
        self.fc1 = nn.Linear(4 * 14 * 14, 128)    # 784→128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantum_conv(x)                   # [B, 4, 14, 14]
        x = x.view(x.size(0), -1)                 # [B, 784]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def count_parameters(self):
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "classical_conv": 0,
            "quantum_pqc": n(self.quantum_conv),
            "conv_total": n(self.quantum_conv),
            "total": n(self),
        }


# ---------------------------------------------------------------------------
# 3. Classical-Extended  (12 filters to match hybrid param count)
#    Conv params: 12*(1*4*4)+12 = 204 (exceeds 136+16=152 slightly,
#    but matches the channel count of the fused hybrid [B,12,14,14])
#    This keeps the FC head identical to QCCNNParallel for fair comparison.
# ---------------------------------------------------------------------------
class ClassicalExtendedCNN(nn.Module):
    """
    Classical-extended ablation: uses 12 classical filters instead of
    8 classical + 4 quantum channels. This produces the SAME fused shape
    [B, 12, 14, 14] as QCCNNParallel, so the FC head is identical.

    This answers the critical question:
    "Is quantum advantage real, or could we just use more classical filters?"
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 12 filters to match the 12-channel fused output of QCCNNParallel
        self.classical_conv = nn.Conv2d(
            in_channels=1, out_channels=12,
            kernel_size=4, stride=2, padding=1
        )
        # SAME FC head as QCCNNParallel (2352→128→64→C)
        self.fc1 = nn.Linear(12 * 14 * 14, 128)   # 2352→128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.classical_conv(x))         # [B, 12, 14, 14]
        x = x.view(x.size(0), -1)                 # [B, 2352]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def count_parameters(self):
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "classical_conv": n(self.classical_conv),
            "quantum_pqc": 0,
            "conv_total": n(self.classical_conv),
            "total": n(self),
        }


__all__ = ["ClassicalOnlyCNN", "QuantumOnlyCNN", "ClassicalExtendedCNN"]
"""
Description: Ablation model variants for Experiment 4 — ClassicalOnly (8 filters, no quantum), QuantumOnly (Circuit 11, no classical), ClassicalExtended (12 filters, parameter-matched). All share the same FC head architecture as QCCNNParallel for fair comparison.
"""
