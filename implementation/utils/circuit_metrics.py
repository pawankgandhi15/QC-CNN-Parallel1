"""
circuit_metrics.py
==================
Expressibility, Entangling capability, and Discreteness metrics for PQC
selection analysis (Experiment 1 in the paper).

Paper: "A Parallel Hybrid Quantum-Classical Convolutional Design Using
        Parameterized Quantum Circuits for Image Classification"
        Quantum Engineering (2026), article 6643049.

References
----------
- Section 3.2 / Equations 9-13 : Expressibility, MW entanglement, Discreteness
- Table 2 (page 9)              : Reference values for 11 circuits
- Section 4.3.1                 : Experiment 1 -- PQC selection study
- METHODOLOGY.md Section 13b   : Mathematical details

Paper Table 2 target values for Circuit 11 (the selected circuit)
-------------------------------------------------------------------
  Params        :  16
  Expressibility: 0.0071   (lower = better, closer to Haar measure)
  Entanglement  : 0.5463   (higher = more entanglement)
  Discreteness  : 0.0191   (new metric -- gradient heterogeneity)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from typing import Callable, List, Tuple


NUM_QUBITS = 4
NUM_SIMS   = 5000    # Paper: 5,000 numerical simulations (Section 4.3.1)


# ---------------------------------------------------------------------------
# Helper: sample random parameters uniformly from [0, 2pi]
# ---------------------------------------------------------------------------
def _sample_params(num_params: int, n_samples: int) -> np.ndarray:
    return np.random.uniform(0, 2 * np.pi, (n_samples, num_params))


# ---------------------------------------------------------------------------
# Helper: compute the output state vector for circuit_fn + weights
#   Uses qml.matrix() on a QuantumTape to get the unitary U, then U|0>.
#   This is the safest approach -- avoids nested-QNode and measurement
#   conflicts that arise when calling expval-returning circuit functions
#   inside another QNode.
# ---------------------------------------------------------------------------
def _circuit_to_state(circuit_fn: Callable,
                      weights: np.ndarray,
                      n_qubits: int = NUM_QUBITS) -> np.ndarray:
    """
    Compute the output state vector for a given circuit function and weights.
    Works for any circuit function regardless of what measurement it returns.
    """
    dim   = 2 ** n_qubits
    dummy = np.zeros(n_qubits)

    with qml.tape.QuantumTape() as tape:
        circuit_fn(dummy, weights)

    # qml.matrix() computes the unitary U for all the gate operations;
    # state = U|0...0>
    mat        = qml.matrix(tape, wire_order=list(range(n_qubits)))
    zero_state = np.zeros(dim, dtype=complex)
    zero_state[0] = 1.0
    return mat @ zero_state


# ---------------------------------------------------------------------------
# 1. Expressibility  (Equation 9, page 5)
#    Expr = D_KL( P_PQC(F; theta) || P_Haar(F) )
#    Lower value = higher expressibility = better Hilbert-space coverage.
# ---------------------------------------------------------------------------
def _state_fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """Compute |<psi1|psi2>|^2."""
    return float(abs(np.dot(psi1.conj(), psi2)) ** 2)


def _haar_fidelity_pdf(f_bins: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Theoretical Haar-random fidelity distribution:
    P_Haar(F) = (N-1)(1-F)^{N-2},  N = 2^n_qubits.
    """
    N = 2 ** n_qubits
    return (N - 1) * (1 - f_bins) ** (N - 2)


def expressibility(circuit_fn: Callable,
                   num_params: int,
                   n_qubits: int = NUM_QUBITS,
                   n_samples: int = NUM_SIMS,
                   n_bins: int = 75) -> float:
    """
    Estimate expressibility via KL divergence (Equation 9).

    Parameters
    ----------
    circuit_fn  : circuit function (inputs, weights) -> measurements
    num_params  : number of trainable parameters
    n_qubits    : number of qubits
    n_samples   : number of parameter-pair samples (paper uses 5,000)
    n_bins      : histogram bins for fidelity

    Returns
    -------
    float -- KL divergence; lower = more expressible
    """
    params1 = _sample_params(num_params, n_samples)
    params2 = _sample_params(num_params, n_samples)

    fidelities = []
    for p1, p2 in zip(params1, params2):
        sv1 = _circuit_to_state(circuit_fn, p1, n_qubits)
        sv2 = _circuit_to_state(circuit_fn, p2, n_qubits)
        fidelities.append(_state_fidelity(sv1, sv2))

    fidelities = np.array(fidelities)
    bins  = np.linspace(0, 1, n_bins + 1)
    mids  = 0.5 * (bins[:-1] + bins[1:])

    pqc_hist, _  = np.histogram(fidelities, bins=bins, density=True)
    haar_hist    = _haar_fidelity_pdf(mids, n_qubits)

    # Normalize
    pqc_hist  = pqc_hist  / (pqc_hist.sum()  + 1e-12)
    haar_hist = haar_hist / (haar_hist.sum() + 1e-12)

    # KL divergence (Eq. 9)
    mask = (pqc_hist > 0) & (haar_hist > 0)
    kl   = float(np.sum(pqc_hist[mask] * np.log(pqc_hist[mask] / haar_hist[mask])))
    return kl


# ---------------------------------------------------------------------------
# 2. Meyer-Wallach Entanglement Measure  (Equation 10, page 5)
#    Ent = (1/|S|) sum Q(|psi_theta>)
# ---------------------------------------------------------------------------
def _mw_entanglement(state_vec: np.ndarray, n_qubits: int) -> float:
    """
    Compute Meyer-Wallach entanglement measure Q for a pure state.
    Q = (4/n) sum_k  [1 - Tr(rho_k^2)]   where rho_k is the reduced density
    matrix of qubit k.
    """
    dim = 2 ** n_qubits
    assert len(state_vec) == dim

    Q = 0.0
    for k in range(n_qubits):
        # Reshape state vector to tensor [2, 2, ..., 2] (n_qubits axes)
        sv_tensor = state_vec.reshape([2] * n_qubits)

        # Move qubit k to axis 0, then reshape to [2, 2^(n-1)]
        # so rows correspond to |0> and |1> of qubit k
        axes = [k] + [i for i in range(n_qubits) if i != k]
        sv_perm = np.transpose(sv_tensor, axes=axes)
        sv_mat  = sv_perm.reshape(2, -1)          # shape (2, 2^(n-1))

        # Reduced density matrix: rho_k = sv_mat @ sv_mat^dagger
        rho_k  = sv_mat @ sv_mat.conj().T        # shape (2, 2)
        purity = float(np.real(np.trace(rho_k @ rho_k)))
        Q     += (1.0 - purity)

    return (4.0 / n_qubits) * Q


def entangling_capability(circuit_fn: Callable,
                          num_params: int,
                          n_qubits: int = NUM_QUBITS,
                          n_samples: int = NUM_SIMS) -> float:
    """
    Estimate Meyer-Wallach entangling capability (Equation 10).

    Parameters
    ----------
    circuit_fn  : circuit function (inputs, weights) -> measurements
    num_params  : number of PQC parameters
    n_samples   : sample size (paper uses 5,000)

    Returns
    -------
    float in [0, 1] -- higher = more entangled
    """
    params_set = _sample_params(num_params, n_samples)
    ent_vals   = []
    for p in params_set:
        sv = _circuit_to_state(circuit_fn, p, n_qubits)
        ent_vals.append(_mw_entanglement(sv, n_qubits))
    return float(np.mean(ent_vals))


# ---------------------------------------------------------------------------
# 3. Discreteness  (Equations 12-13, page 5-6)
#    New metric introduced by the paper to capture gradient heterogeneity.
#
#    Var(g_i) = (1/N) sum_j (g_i^(j) - g_i_bar)^2
#    Disc     = (1/M) sum_i Var(g_i)
#
#    where g_i^(j) = gradient of i-th parameter at j-th random init.
#    Computed via finite differences (parameter-shift rule, shift=pi/2).
# ---------------------------------------------------------------------------
def _param_shift_grad(circuit_fn: Callable,
                      weights: np.ndarray,
                      n_qubits: int) -> np.ndarray:
    """
    Compute gradients of sum(expval(PauliZ_i)) w.r.t. all weights using
    the parameter-shift rule (shift = pi/2).

    Returns a 1D numpy array of gradients, one per weight.
    """
    grads = np.zeros(len(weights))
    for i in range(len(weights)):
        w_plus  = weights.copy(); w_plus[i]  += np.pi / 2
        w_minus = weights.copy(); w_minus[i] -= np.pi / 2

        sv_plus  = _circuit_to_state(circuit_fn, w_plus,  n_qubits)
        sv_minus = _circuit_to_state(circuit_fn, w_minus, n_qubits)

        # Expectation value of sum(PauliZ) = sum of |coeff_0|^2 - |coeff_1|^2
        # for each qubit's reduced state.  Simpler: use |sv|^2 bit counting.
        def _z_expval_sum(sv: np.ndarray) -> float:
            probs = np.abs(sv) ** 2
            total = 0.0
            for k in range(n_qubits):
                # PauliZ eigenvalue for qubit k: +1 if k-th bit=0, -1 if k-th bit=1
                dim   = 2 ** n_qubits
                for idx in range(dim):
                    bit_k = (idx >> (n_qubits - 1 - k)) & 1
                    total += probs[idx] * (1 - 2 * bit_k)
            return float(total)

        grads[i] = (_z_expval_sum(sv_plus) - _z_expval_sum(sv_minus)) / 2
    return grads


def discreteness(circuit_fn: Callable,
                 num_params: int,
                 n_qubits: int = NUM_QUBITS,
                 n_samples: int = 100) -> float:
    """
    Estimate PQC Discreteness metric (Equations 12-13).

    Discreteness measures the variance of gradients across random parameter
    initialisations.  A higher value means the loss landscape is more
    heterogeneous (more 'discrete').

    Parameters
    ----------
    circuit_fn  : circuit function (inputs, weights) -> measurement
    num_params  : number of trainable PQC parameters
    n_qubits    : number of qubits
    n_samples   : N random initialisations (Eq. 12; paper uses 5,000)

    Returns
    -------
    float -- mean gradient variance across all parameters (Eq. 13)
    """
    all_grads: List[np.ndarray] = []
    for _ in range(n_samples):
        w     = np.random.uniform(0, 2 * np.pi, num_params)
        grads = _param_shift_grad(circuit_fn, w, n_qubits)
        all_grads.append(grads)

    if not all_grads:
        return 0.0

    G    = np.stack(all_grads)              # shape [N, M]
    mean = G.mean(axis=0, keepdims=True)    # g_bar_i  (Eq. 12)
    var  = ((G - mean) ** 2).mean(axis=0)  # Var(g_i) (Eq. 12)
    disc = float(var.mean())               # Disc (Eq. 13)
    return disc


# ---------------------------------------------------------------------------
# 4. Full analysis runner (reproduces Table 2 and Table 3)
# ---------------------------------------------------------------------------
def run_circuit_analysis(circuits: dict,
                         n_qubits: int = NUM_QUBITS,
                         n_sims: int = NUM_SIMS) -> dict:
    """
    Run all three metrics on a dict of circuits.

    Parameters
    ----------
    circuits : dict -- {name: {"fn": circuit_fn, "num_params": int}}
                       where circuit_fn(inputs, weights) -> measurements
    n_qubits : int
    n_sims   : int   (paper uses 5,000)

    Returns
    -------
    dict of results keyed by circuit name.
    """
    results = {}
    for name, cfg in circuits.items():
        fn         = cfg.get("fn", cfg.get("qnode"))
        num_params = cfg["num_params"]
        print(f"  Analyzing {name} ({num_params} params)...")
        expr = expressibility(fn,       num_params, n_qubits, n_sims)
        ent  = entangling_capability(fn, num_params, n_qubits, n_sims)
        # Section 4.3.1 specifies 5,000 simulations for all three metrics.
        # Callers can pass a smaller n_sims value for a quick development run.
        disc = discreteness(fn,          num_params, n_qubits, n_sims)
        results[name] = {
            "num_params"     : num_params,
            "expressibility" : round(expr, 4),
            "entanglement"   : round(ent,  4),
            "discreteness"   : round(disc, 4),
        }
        print(f"    Expr={expr:.4f}  Ent={ent:.4f}  Disc={disc:.4f}")
    return results
