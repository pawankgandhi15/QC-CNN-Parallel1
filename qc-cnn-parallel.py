import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

# Define PennyLane Quantum Device
# 4 qubits matching the 2x2 receptive field of the quantum filter
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)

# ==========================================
# 1. PARAMETERIZED QUANTUM CIRCUIT (PQC) - CIRCUIT 11
# ==========================================
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    4-Qubit Parameterized Quantum Circuit (Circuit 11 from the paper).
    
    Args:
        inputs (tensor): Shape (4,) containing normalized 2x2 patch pixel values.
        weights (tensor): Shape (16,) containing trainable parameters for Circuit 11.
    """
    # Step A: Angle Encoding (Passage [22, 23])
    # Each pixel in the 2x2 patch is encoded on one qubit.
    # Apply Hadamard to create superposition, then RY rotation.
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i], wires=i)

    # Split the 16 parameters into rotation and entangling weights (Passage [47, 51]):
    # - 8 for single-qubit rotations (2 layers * 4 qubits)
    # - 8 for entangling CRX gates (2 layers * 4 gates)
    w_rot1 = weights[0:4]
    w_ent1 = weights[4:8]
    w_rot2 = weights[8:12]
    w_ent2 = weights[12:16]

    # --- PQC LAYER 1 ---
    # Variational Layer 1: Single-qubit RY rotations (Passage [47])
    for i in range(num_qubits):
        qml.RY(w_rot1[i], wires=i)

    # Entangling Layer 1: Circle topology starting with qubit 0 (Passage [48, 51])
    # Connects: 0->1, 1->2, 2->3, 3->0
    for i in range(num_qubits):
        control = i
        target = (i + 1) % num_qubits
        qml.CRX(w_ent1[i], wires=[control, target])

    # --- PQC LAYER 2 ---
    # Variational Layer 2: Single-qubit RY rotations
    for i in range(num_qubits):
        qml.RY(w_rot2[i], wires=i)

    # Entangling Layer 2: Shifted Circle topology starting with qubit 1 (Passage [51])
    # Shifted to prevent overlapping patterns and enhance expressive capacity.
    # Connects: 1->2, 2->3, 3->0, 0->1
    for i in range(num_qubits):
        control = (i + 1) % num_qubits
        target = (i + 2) % num_qubits
        qml.CRX(w_ent2[i], wires=[control, target])

    # Step C: Pauli-Z expectation measurement on all qubits (Passage [35])
    # Restricts output features to [-1, 1] for stable feature fusion.
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]


# ==========================================
# 2. CUSTOM QUANTUM CONVOLUTIONAL LAYER
# ==========================================
class QuantumConvLayer(nn.Module):
    """
    Applies a 2x2 quantum sliding window (stride=2) over the input image.
    Uses the Parameterized Quantum Circuit defined in Circuit 11.
    """
    def __init__(self):
        super(QuantumConvLayer, self).__init__()
        # Circuit 11 requires exactly 16 trainable parameters (Passage [57])
        self.weights = nn.Parameter(torch.randn(16))

    def forward(self, x):
        # Input shape: [Batch_size, 1, Height, Width]
        batch_size, channels, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, "Image height and width must be even for 2x2 stride-2 convolutions."
        
        out_h, out_w = h // 2, w // 2
        # Quantum branch outputs 4 channels corresponding to the 4 measured qubits (Passage [20])
        out = torch.zeros((batch_size, 4, out_h, out_w), device=x.device, dtype=x.dtype)
        
        # Slide 2x2 window across the image with stride 2
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    # Extract the 2x2 patch
                    patch = x[b, 0, i*2:(i*2)+2, j*2:(j*2)+2]
                    # Flatten the patch into a 4D vector and scale to [-pi, pi] for angle encoding (Passage [23])
                    flattened_patch = patch.flatten() * np.pi
                    
                    # Evaluate the QNode using PyTorch autograd
                    q_features = torch.stack(quantum_circuit(flattened_patch, self.weights))
                    out[b, :, i, j] = q_features
                    
        return out


# ==========================================
# 3. FULL QC-CNN-PARALLEL MODEL
# ==========================================
class QCCNNParallel(nn.Module):
    """
    The full QC-CNN-Parallel architecture (Passage [17, 20, 36, 37]).
    Features dual parallel branches (classical & quantum) and a 3-layer dense classification head.
    """
    def __init__(self, num_classes=10):
        super(QCCNNParallel, self).__init__()
        
        # Dual Parallel Branch 1: Classical Convolution (Passage [17, 18])
        # Kernel: 4x4, Stride: 2, Padding: 1 -> output spatial size is matches quantum branch (14x14)
        # Table 4 notes 136 parameters in the conv portion. Using 8 kernels of size 4x4:
        # 8 * (1 * 4 * 4) + 8 biases = 136 parameters exactly!
        self.classical_conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2, padding=1)
        
        # Dual Parallel Branch 2: Quantum Convolution (Passage [19, 20])
        # Kernel: 2x2, Stride: 2 -> output spatial size is (14x14), Channels: 4
        self.quantum_conv = QuantumConvLayer()
        
        # Feature Fusion & Dense Classification Head (Passage [36, 37])
        # Concatenated channels = 8 (classical) + 4 (quantum) = 12 channels.
        # Spatial size = 14x14. Flattened dimension = 12 * 14 * 14 = 2352.
        self.fc1 = nn.Linear(12 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Parallel Branch 1: Classical conv
        x_class = self.classical_conv(x)  # Shape: [Batch, 8, 14, 14]
        x_class = F.relu(x_class)
        
        # Parallel Branch 2: Quantum conv
        x_quant = self.quantum_conv(x)    # Shape: [Batch, 4, 14, 14]
        
        # Multichannel Feature Fusion: Concatenation along channel dimension (Passage [36])
        x_fused = torch.cat((x_class, x_quant), dim=1)  # Shape: [Batch, 12, 14, 14]
        
        # Flatten
        x_flat = x_fused.view(x_fused.size(0), -1)  # Shape: [Batch, 2352]
        
        # 3-Layer Fully Connected Classification Head (Passage [37])
        x_fc = F.relu(self.fc1(x_flat))
        x_fc = F.relu(self.fc2(x_fc))
        logits = self.fc3(x_fc)
        
        return logits


# ==========================================
# 4. TRAINING & DEMO PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Initializing QC-CNN-Parallel Architecture...")
    model = QCCNNParallel(num_classes=10)
    
    # Calculate parameter distribution
    conv_params = sum(p.numel() for p in model.classical_conv.parameters()) + sum(p.numel() for p in model.quantum_conv.parameters())
    print(f"-> Convolutional Parameters: {conv_params} (136 Classical + 16 Quantum)")
    print(f"-> Total Network Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Generate Dummy Grayscale Input Batch (similar to MNIST: 28x28 images)
    batch_size = 4
    dummy_images = torch.rand((batch_size, 1, 28, 28))
    dummy_labels = torch.randint(0, 10, (batch_size,))
    
    # Set optimizer & Loss (Passage [38, 43, 63])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    print("\nRunning a single forward and backward pass on dummy batch...")
    # Forward Pass
    logits = model(dummy_images)
    loss = loss_fn(logits, dummy_labels)
    
    print(f"-> Logits output shape: {logits.shape}")
    print(f"-> Initial cross-entropy loss: {loss.item():.4f}")
    
    # Backward Pass (Updates classical weights and quantum PQC parameters simultaneously)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("-> Optimization step completed successfully!")
