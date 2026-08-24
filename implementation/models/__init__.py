"""
__init__.py — models package
"""
# pyrefly: ignore [missing-import]
from .qc_cnn_parallel import QCCNNParallel, ClassicalCNN
from .quantum_circuit  import QuantumConvLayer, quantum_circuit, make_noisy_circuit
from .ablation_models  import ClassicalOnlyCNN, QuantumOnlyCNN, ClassicalExtendedCNN
from .scalable_quantum_circuit import ScalableQCCNNParallel, ScalableQuantumConvLayer, make_scalable_circuit, measure_gradient_variance

__all__ = [
    "QCCNNParallel",
    "ClassicalCNN",
    "QuantumConvLayer",
    "quantum_circuit",
    "make_noisy_circuit",
    "ClassicalOnlyCNN",
    "QuantumOnlyCNN",
    "ClassicalExtendedCNN",
    "ScalableQCCNNParallel",
    "ScalableQuantumConvLayer",
    "make_scalable_circuit",
    "measure_gradient_variance",
]

