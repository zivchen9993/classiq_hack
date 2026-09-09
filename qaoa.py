"""
qaoa.py -- QAOA for a diagonal Ising cost Hamiltonian, simulated exactly
on the statevector.  Dependency-free so it runs anywhere.

    |psi(g,b)> = prod_l  e^{-i b_l sum_i X_i}  e^{-i g_l H_C}  |+>^M

H_C is diagonal, so the cost layer is an elementwise phase multiply.
The mixer is a single-qubit RX(2b) on every wire.

>>> PORTING TO CLASSIQ <<<
The cost layer is  sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j , i.e.
    RZ(2*gamma*h_i) on qubit i
    RZZ(2*gamma*J_ij) on the pair (i,j)   [ CX - RZ - CX ]
and the mixer is RX(2*beta) on every qubit.  That is the whole circuit;
h and J come straight from ising.extract_ising().
"""

import numpy as np
from scipy.optimize import minimize


def apply_mixer(psi, beta, M):
    """RX(2*beta) on every qubit."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(M):
        psi = psi.reshape(-1, 2, 1 << q)
        a, b = psi[:, 0, :].copy(), psi[:, 1, :].copy()
        psi[:, 0, :] = c * a + s * b
        psi[:, 1, :] = s * a + c * b
        psi = psi.reshape(-1)
    return psi


def qaoa_state(gammas, betas, Ediag, M):
    n = 1 << M
    psi = np.full(n, 1.0 / np.sqrt(n), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * Ediag)      # cost layer (diagonal)
        psi = apply_mixer(psi, b, M)             # mixer layer
    return psi


def expectation(params, Ediag, M, p):
    psi = qaoa_state(params[:p], params[p:], Ediag, M)
    return float(np.real(np.abs(psi) ** 2 @ Ediag))


def run_qaoa(Ediag, M, p=2, n_restart=6, seed=0, maxiter=400):
    """
    Optimise (gamma, beta) with COBYLA from several random starts.
    Returns the best statevector, its parameters and the energy trace.
    """
    rng = np.random.default_rng(seed)
    Es = Ediag - Ediag.mean()
    scale = np.std(Es) + 1e-12
    Es = Es / scale                              # normalise so gamma ~ O(1)

    best = None
    for _ in range(n_restart):
        x0 = np.concatenate([rng.uniform(0, np.pi, p),
                             rng.uniform(0, np.pi / 2, p)])
        res = minimize(expectation, x0, args=(Es, M, p),
                       method="COBYLA", options=dict(maxiter=maxiter))
        if best is None or res.fun < best.fun:
            best = res

    psi = qaoa_state(best.x[:p], best.x[p:], Es, M)
    probs = np.abs(psi) ** 2
    return dict(probs=probs, params=best.x,
                exp_energy=float(best.fun * scale + Ediag.mean()))


def sample_best(probs, Etrue, n_shots=512, rng=None):
    """
    Measure the circuit, then evaluate the sampled bitstrings against the
    TRUE landscape.  This is how QAOA is used in practice: the circuit
    proposes candidates, the simulator scores them.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    idx = rng.choice(probs.size, size=n_shots, p=probs / probs.sum())
    uniq = np.unique(idx)
    best = uniq[np.argmin(Etrue[uniq])]
    return int(best), uniq
