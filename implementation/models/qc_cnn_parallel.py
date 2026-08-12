"""
qc_cnn_parallel.py
==================
Full QC-CNN-Parallel model and LeNet-5 classical baseline.

Paper: "A Parallel Hybrid Quantum-Classical Convolutional Design Using
        Parameterized Quantum Circuits for Image Classification"
        Quantum Engineering (2026), article 6643049.

References
----------
- Section 3    : Full model architecture (Figure 1)
- Section 3.4  : Classical dense layer (3-layer FC head)
- Table 4      : Parameter counts — Proposed: 136 conv params
- ARCHITECTURE.md : Detailed shape derivations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantum_circuit import QuantumConvLayer


# ---------------------------------------------------------------------------
# 1. QC-CNN-Parallel  (the proposed model)
# ---------------------------------------------------------------------------
class QCCNNParallel(nn.Module):
    """
    Parallel Hybrid Quantum-Classical Convolutional Neural Network.

    Architecture (Figure 1, paper):
      Input [B,1,28,28]
        ├── Classical branch: Conv2d(1→8, 4×4, stride=2, pad=1) + ReLU → [B,8,14,14]
        └── Quantum  branch: QuantumConvLayer (2×2, stride=2, Circuit-11) → [B,4,14,14]
      Concat along channels → [B,12,14,14]
      Flatten → [B,2352]
      FC 2352→128 → ReLU
      FC  128→ 64 → ReLU
      FC   64→  C   (logits)

    Parameter counts
    ------------------------------------
    Classical Conv2d  : 8×(1×4×4)+8  =  136
    Quantum PQC       :               =   16
    FC 2352→128       : 2352×128+128 = 301,184
    FC  128→ 64       :  128×64 + 64 =   8,256
    FC   64→ 10       :   64×10 + 10 =     650
    Total (C=10)      :             = 310,242

    Table 4 reports 136 parameters for the proposed model's classical
    convolutional filter. It does not include the 16 trainable PQC angles, so
    the complete parallel feature extractor has 152 trainable parameters.
    """

    def __init__(self, num_classes: int = 10, qnode=None):
        """
        Parameters
        ----------
        num_classes : int   Number of output classes (default 10 for MNIST)
        qnode       : callable  Optional custom QNode (e.g. noisy circuit)
        """
        super().__init__()

        # --- Branch 1: Classical convolution ---
        # kernel=4×4, stride=2, padding=1 → output [B,8,14,14] for 28×28 input
        # Parameters: 8*(1*4*4) + 8 = 136  (matches Table 4)
        self.classical_conv = nn.Conv2d(
            in_channels=1, out_channels=8,
            kernel_size=4, stride=2, padding=1
        )

        # --- Branch 2: Quantum convolution ---
        # 2×2 window, stride=2, Circuit-11 (16 params) → [B,4,14,14]
        self.quantum_conv = QuantumConvLayer(qnode=qnode)

        # --- Classification head (Section 3.4) ---
        # Fused shape: [B, 12, 14, 14] → flatten → [B, 2352]
        self.fc1 = nn.Linear(12 * 14 * 14, 128)   # 2352×128+128 = 301,184
        self.fc2 = nn.Linear(128, 64)              # 128×64+64    =   8,256
        self.fc3 = nn.Linear(64, num_classes)      # 64×C+C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape [B, 1, 28, 28], pixels in [0, 1]

        Returns
        -------
        logits : torch.Tensor  shape [B, num_classes]
            Raw unnormalized scores. Pass directly to CrossEntropyLoss.
        """
        # -- Classical branch --
        x_class = F.relu(self.classical_conv(x))   # [B, 8, 14, 14]

        # -- Quantum branch --
        x_quant = self.quantum_conv(x)              # [B, 4, 14, 14]

        # -- Feature fusion (channel-wise concatenation) --
        x_fused = torch.cat([x_class, x_quant], dim=1)   # [B,12,14,14]
        x_flat  = x_fused.view(x_fused.size(0), -1)      # [B,2352]

        # -- 3-layer dense head --
        h1     = F.relu(self.fc1(x_flat))          # [B,128]
        h2     = F.relu(self.fc2(h1))              # [B, 64]
        logits = self.fc3(h2)                       # [B,  C]
        return logits

    def count_parameters(self):
        """Return dict of parameter counts per component (Table 4)."""
        def n(module): return sum(p.numel() for p in module.parameters())
        return {
            "classical_conv": n(self.classical_conv),
            "quantum_pqc"   : n(self.quantum_conv),
            "fc1"           : n(self.fc1),
            "fc2"           : n(self.fc2),
            "fc3"           : n(self.fc3),
            "total"         : n(self),
            "conv_total"    : n(self.classical_conv) + n(self.quantum_conv),
        }


# ---------------------------------------------------------------------------
# 2. Classical CNN baseline  (LeNet-5 adapted, Table 4)
#    Used as the classical-only comparison in Experiment 2 / Figures 6-8.
#    Paper Table 4: CNN has 464 convolutional parameters.
# ---------------------------------------------------------------------------
class ClassicalCNN(nn.Module):
    """Classical CNN comparison model with the Table 4 convolution budget.

    Table 4 specifies 464 convolutional parameters and a shared linear
    classifier, but does not publish individual LeNet filter widths.  This
    two-convolution feature extractor preserves the classifier input
    [12, 14, 14] while matching the reported count exactly:

      Conv2d(1→4, 2×2, stride=2)         : 20
      Conv2d(4→12, 3×3, stride=1, pad=1) : 444
      Total                              : 464
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.classical_conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.classical_conv2 = nn.Conv2d(4, 12, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(12 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.classical_conv1(x))    # [B, 4, 14, 14]
        x = F.relu(self.classical_conv2(x))    # [B,12, 14, 14]
        x_flat = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

    def count_parameters(self):
        """Return convolutional and complete parameter counts."""
        def n(module): return sum(p.numel() for p in module.parameters())
        return {
            "classical_conv1": n(self.classical_conv1),
            "classical_conv2": n(self.classical_conv2),
            "conv_total": n(self.classical_conv1) + n(self.classical_conv2),
            "total": n(self),
        }


# ---------------------------------------------------------------------------
# 3. Module __init__ exposure
# ---------------------------------------------------------------------------
__all__ = ["QCCNNParallel", "ClassicalCNN"]
