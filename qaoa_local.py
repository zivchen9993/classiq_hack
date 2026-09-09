"""
Exact local statevector QAOA simulator for the tilt QUBO.

Purpose
-------
Classiq is a cloud platform: every synthesis/execution is a network round trip.
This module runs the IDENTICAL QAOA ansatz locally and exactly, which lets us
(a) develop and tune depth / optimizer settings without burning cloud time,
(b) validate that the Classiq results match theory, and
(c) still have a genuine quantum-algorithm result if the cloud is unavailable.

The ansatz is the standard QAOA:

    |psi(gamma, beta)> = prod_{l=1..p} [ U_M(beta_l) U_C(gamma_l) ] H^{(x)n} |0>

    U_C(gamma) = exp(-i * gamma * H_C)     H_C diagonal in the computational basis
    U_M(beta)  = exp(-i * beta * sum_k X_k)

Because the cost Hamiltonian of a QUBO is diagonal, U_C is an elementwise
phase multiply -- exact and cheap. The mixer is a product of single-qubit
RX rotations, applied with a reshape/axis trick. No approximation anywhere.

Feasible up to roughly 20-22 qubits on a laptop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Cost Hamiltonian diagonal
# ---------------------------------------------------------------------------

def cost_diagonal(qubo) -> np.ndarray:
    """Energy of every one of the 2^n computational basis states.

    Bit ordering: qubit k is bit (n-1-k) of the state index, i.e. qubit 0 is
    the most significant bit. Kept consistent with `bits_of_state`.
    """
    n = qubo.n_vars
    if n > 24:
        raise ValueError(f"{n} qubits is too many for exact statevector simulation")

    dim = 1 << n
    idx = np.arange(dim, dtype=np.uint32)
    # x[k] for every state, as float arrays (dim,) -- qubit 0 = most significant
    bits = [((idx >> (n - 1 - k)) & 1).astype(np.float64) for k in range(n)]

    energy = np.full(dim, float(qubo.offset))
    Q = qubo.Q
    nz_i, nz_j = np.nonzero(Q)
    for i, j in zip(nz_i, nz_j):
        q = Q[i, j]
        if i == j:
            energy += q * bits[i]
        else:
            energy += q * bits[i] * bits[j]
    return energy


def bits_of_state(state_index: int, n: int) -> np.ndarray:
    """Decode a basis-state index into its bit vector (qubit 0 = MSB)."""
    return np.array([(state_index >> (n - 1 - k)) & 1 for k in range(n)], dtype=int)


# ---------------------------------------------------------------------------
# QAOA circuit
# ---------------------------------------------------------------------------

def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply exp(-i * beta * sum_k X_k) = prod_k RX_k(2*beta).

    For each qubit: psi -> cos(beta) * psi - i sin(beta) * X_k psi
    """
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * n)
    for k in range(n):
        flipped = np.flip(psi, axis=k)
        psi = c * psi + s * flipped
    return psi.reshape(-1)


# ---------------------------------------------------------------------------
# XY / one-hot-preserving ansatz
# ---------------------------------------------------------------------------
#
# The standard X mixer wanders all over the 2^n Hilbert space, of which only
# n_tilts^n_sectors states are feasible one-hot encodings. On our instances
# that is 262,144 valid states inside a 2^36 space -- a vanishing fraction, so
# the penalty term has to be large, and it then dominates the spectrum and
# swamps the physics we actually want to optimize.
#
# The XY mixer (Hadfield et al., "Quantum Alternating Operator Ansatz") fixes
# this. exp(-i*beta*(XX+YY)/2) on a qubit pair only mixes |01> <-> |10>: it
# CONSERVES Hamming weight. Applying it around a ring within each sector's
# one-hot block, starting from a W state (uniform superposition of that
# block's one-hot states), keeps the state inside the feasible subspace for
# every parameter value. Consequences:
#   * the one-hot penalty term can be dropped entirely
#   * 100% of measured shots are feasible by construction
#   * all of the circuit's expressive power goes into the physics objective


def apply_two_qubit(psi: np.ndarray, n: int, a: int, b: int, U: np.ndarray) -> np.ndarray:
    """Apply a 4x4 gate U to qubits a, b of an n-qubit statevector."""
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(psi, (a, b), (0, 1))
    shape = psi.shape
    psi = (U @ psi.reshape(4, -1)).reshape(shape)
    psi = np.moveaxis(psi, (0, 1), (a, b))
    return psi.reshape(-1)


def xy_gate(beta: float) -> np.ndarray:
    """exp(-i * beta * (XX + YY) / 2) in the |00>,|01>,|10>,|11> basis.

    Acts as a rotation on the single-excitation subspace {|01>, |10>} and as
    the identity on |00> and |11> -- hence it preserves Hamming weight.
    """
    c, s = np.cos(beta), -1j * np.sin(beta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=np.complex128)


def feasible_state_indices(qubo) -> np.ndarray:
    """Basis-state indices of every valid one-hot configuration.

    There are n_tilts ** n_sectors of them inside a 2^n_vars space -- often
    well under 1% -- which is precisely why the constrained ansatz pays off.
    """
    n = qubo.n_vars
    m = qubo.n_tilts
    # index contribution of "sector i uses tilt t", with qubit 0 as the MSB
    weights = np.array([[1 << (n - 1 - (i * m + t)) for t in range(m)]
                        for i in range(qubo.n_sectors)], dtype=np.int64)

    indices = np.zeros(1, dtype=np.int64)
    for i in range(qubo.n_sectors):
        indices = (indices[:, None] + weights[i][None, :]).ravel()
    return indices


def w_state_init(qubo) -> np.ndarray:
    """Product over sectors of W states: within each sector's block, an equal
    superposition of that sector's one-hot states.

    This is the uniform superposition over ALL feasible configurations, and
    nothing else -- the correct QAOA starting point for a constrained problem.
    """
    psi = np.zeros(1 << qubo.n_vars, dtype=np.complex128)
    idx = feasible_state_indices(qubo)
    psi[idx] = 1.0 / np.sqrt(len(idx))
    return psi


def xy_mixer_pairs(qubo, topology: str = "chain"):
    """The qubit pairs the XY mixer couples, within each sector's block.

    "chain" (default): (0,1), (1,2), ... -- only ADJACENT qubits, which is
        what Classiq's RXX/RYY need, since a Qmod QArray slice must be
        contiguous. Keeping the local simulator on the same topology means it
        mirrors the synthesized circuit exactly.
    "ring": additionally couples (m-1, 0). Slightly faster mixing, but needs
        a non-contiguous two-qubit gate.
    """
    m = qubo.n_tilts
    pairs = []
    for i in range(qubo.n_sectors):
        base = i * m
        for k in range(m - 1):
            pairs.append((base + k, base + k + 1))
        if topology == "ring" and m > 2:
            pairs.append((base + m - 1, base))
    return pairs


def apply_xy_mixer(psi: np.ndarray, beta: float, qubo,
                   topology: str = "chain") -> np.ndarray:
    """XY mixer applied within each sector's one-hot block.

    Trotterized: the pair terms do not commute, so we apply them sequentially.
    This is the standard implementation, and crucially it preserves the
    feasible subspace exactly regardless of any Trotter error -- the
    feasibility guarantee does not depend on the approximation being good.
    """
    U = xy_gate(beta)
    for a, b in xy_mixer_pairs(qubo, topology):
        psi = apply_two_qubit(psi, qubo.n_vars, a, b, U)
    return psi


def qaoa_statevector(gammas, betas, diag: np.ndarray, n: int,
                     qubo=None, mixer: str = "x", psi0=None) -> np.ndarray:
    """Build the exact QAOA statevector.

    mixer="x"  : standard transverse-field mixer, |+>^n initial state
    mixer="xy" : one-hot-preserving XY ring mixer, W-state initial state
    """
    if mixer == "xy":
        if qubo is None:
            raise ValueError("the XY mixer needs the qubo for its block structure")
        psi = w_state_init(qubo) if psi0 is None else psi0.copy()
        for gamma, beta in zip(gammas, betas):
            psi = np.exp(-1j * gamma * diag) * psi
            psi = apply_xy_mixer(psi, beta, qubo)
        return psi

    dim = 1 << n
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)   # H^{on} |0>
    for gamma, beta in zip(gammas, betas):
        psi = np.exp(-1j * gamma * diag) * psi                    # cost layer
        psi = apply_mixer(psi, beta, n)                           # mixer layer
    return psi


# ---------------------------------------------------------------------------
# Objective used by the classical outer loop
# ---------------------------------------------------------------------------

def expectation(params, diag: np.ndarray, n: int, p: int,
                quantile: float = 1.0, qubo=None, mixer: str = "x",
                psi0=None) -> float:
    """<psi|H_C|psi>, or the CVaR (mean of the best `quantile` fraction of the
    measured energy distribution) when quantile < 1.

    CVaR is standard practice for QAOA on optimization problems: we only care
    about the best bitstrings we can sample, not the mean of the whole
    distribution, and it markedly improves the optimizer's behaviour.
    """
    gammas, betas = params[:p], params[p:]
    psi = qaoa_statevector(gammas, betas, diag, n, qubo=qubo, mixer=mixer, psi0=psi0)
    probs = np.abs(psi) ** 2

    if quantile >= 1.0:
        return float(probs @ diag)

    order = np.argsort(diag)
    sorted_probs = probs[order]
    sorted_diag = diag[order]
    cum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cum, quantile) + 1
    cutoff = min(cutoff, len(sorted_diag))
    w = sorted_probs[:cutoff].copy()
    total = w.sum()
    if total <= 0:
        return float(probs @ diag)
    return float((w @ sorted_diag[:cutoff]) / total)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class QAOAResult:
    name: str
    config: np.ndarray
    objective: float
    seconds: float
    evaluations: int
    n_qubits: int
    depth: int
    optimal_params: np.ndarray
    best_energy: float
    prob_of_optimum: float
    valid_prob: float
    extra: dict = field(default_factory=dict)

    def __str__(self):
        return (f"{self.name:<22} obj={self.objective:+.5f}  "
                f"{self.seconds * 1000:8.1f} ms  {self.n_qubits}q p={self.depth}  "
                f"tilts={self.config.tolist()}")


def solve_qaoa_local(qubo, p: int = 3, seed: int = 0, maxiter: int = 300,
                     quantile: float = 0.15, n_restarts: int = 3,
                     n_shots: int = 2000, mixer: str = "x") -> QAOAResult:
    """Run QAOA locally and return the best tilt configuration found.

    The classical outer loop (COBYLA) optimizes the 2p variational parameters;
    the final state is then sampled `n_shots` times and every sampled
    bitstring is decoded to a tilt configuration and scored with the SAME
    physics objective the classical baselines use.

    mixer="x"  : standard QAOA (penalty term enforces the one-hot encoding)
    mixer="xy" : one-hot-preserving XY ring mixer + W-state init
    """
    n = qubo.n_vars
    t0 = time.perf_counter()
    diag = cost_diagonal(qubo)
    rng = np.random.default_rng(seed)

    # the XY ansatz never leaves the feasible subspace, so build the initial
    # state once and reuse it across every optimizer evaluation
    psi0 = w_state_init(qubo) if mixer == "xy" else None

    evals = 0

    def objective(params):
        nonlocal evals
        evals += 1
        return expectation(params, diag, n, p, quantile,
                           qubo=qubo, mixer=mixer, psi0=psi0)

    best_res, best_val = None, np.inf
    for _ in range(n_restarts):
        # standard QAOA initialization: small gammas, betas spread over [0, pi/2]
        x0 = np.concatenate([
            rng.uniform(0.0, 0.6, size=p),
            rng.uniform(0.0, np.pi / 2, size=p),
        ])
        res = minimize(objective, x0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.3})
        if res.fun < best_val:
            best_val, best_res = res.fun, res

    params = best_res.x
    gammas, betas = params[:p], params[p:]
    psi = qaoa_statevector(gammas, betas, diag, n, qubo=qubo, mixer=mixer, psi0=psi0)
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()

    # --- sample the optimized state and decode every shot -----------------
    samples = rng.choice(len(probs), size=n_shots, p=probs)
    unique, counts = np.unique(samples, return_counts=True)

    best_cfg, best_obj = None, np.inf
    valid_shots = 0
    obj_sum = 0.0
    for state_idx, count in zip(unique, counts):
        x = bits_of_state(int(state_idx), n)
        if qubo.is_valid(x):
            valid_shots += count
        cfg = qubo.decode(x)          # invalid one-hot blocks repaired by argmax
        obj = qubo.objective_from_config(cfg)
        obj_sum += obj * count
        if obj < best_obj:
            best_obj, best_cfg = obj, cfg

    # --- diagnostics ------------------------------------------------------
    # The reference must be the best FEASIBLE state. Without the penalty term
    # the global minimum over all 2^n bitstrings is typically an infeasible
    # state (e.g. several tilts active at once), which no valid solution and
    # no XY-mixer state can ever reach -- comparing against it would be wrong.
    feasible_idx = feasible_state_indices(qubo)
    feasible_energies = diag[feasible_idx]
    ground_energy = float(feasible_energies.min())
    optimum_states = feasible_idx[np.isclose(feasible_energies, ground_energy, atol=1e-9)]
    prob_of_optimum = float(probs[optimum_states].sum())
    valid_prob = float(valid_shots / n_shots)
    mean_sampled_obj = float(obj_sum / n_shots)

    seconds = time.perf_counter() - t0
    label = "QAOA-XY" if mixer == "xy" else "QAOA-X"
    return QAOAResult(
        name=f"{label} local (p={p})", config=best_cfg, objective=best_obj,
        seconds=seconds, evaluations=evals, n_qubits=n, depth=p,
        optimal_params=params, best_energy=ground_energy,
        prob_of_optimum=prob_of_optimum, valid_prob=valid_prob,
        extra={"expectation": best_val, "n_shots": n_shots,
               "quantile": quantile, "mixer": mixer,
               "mean_sampled_objective": mean_sampled_obj},
    )


if __name__ == "__main__":
    from rf_model import make_network
    from qubo_builder import build_qubo
    from classical_baseline import brute_force

    # the flagship instance the benchmark and write-up use:
    # 2 towers = 6 sectors x 3 tilts = 18 qubits, exactly simulable
    net = make_network(n_towers=2, seed=27, tilt_levels_deg=[0.0, 4.0, 8.0])
    qubo_pen = build_qubo(net)                       # with one-hot penalty (X mixer)
    qubo_free = build_qubo(net, penalty_scale=0.0)   # no penalty (XY mixer)
    print(qubo_pen.summary())

    diag_pen = cost_diagonal(qubo_pen)
    diag_free = cost_diagonal(qubo_free)
    exact = brute_force(qubo_pen)

    print(f"\nfeasible configs: {qubo_pen.n_tilts ** qubo_pen.n_sectors:,} "
          f"out of 2^{qubo_pen.n_vars} = {1 << qubo_pen.n_vars:,} basis states "
          f"({(qubo_pen.n_tilts ** qubo_pen.n_sectors) / (1 << qubo_pen.n_vars):.2%})")
    print(f"brute-force optimum      : {exact.objective:+.5f} tilts={exact.config.tolist()}")
    print(f"Hamiltonian ground energy: {diag_pen.min():+.5f}  "
          f"(match: {np.isclose(diag_pen.min(), exact.objective, atol=1e-9)})")
    print(f"\nenergy spread WITH penalty : {diag_pen.min():+.3f} .. {diag_pen.max():+.3f}"
          f"   <- physics signal is a tiny fraction of this range")
    print(f"energy spread NO penalty   : {diag_free.min():+.3f} .. {diag_free.max():+.3f}")

    # Random-guess reference: with a uniform distribution over feasible states,
    # what fraction of the time would we land on the optimum?
    n_feasible = qubo_pen.n_tilts ** qubo_pen.n_sectors
    print(f"\nuniform-random reference: P(optimum) = 1/{n_feasible} = {1 / n_feasible:.2%}")

    print(f"\n{'ansatz':<18} {'p':>2} {'best obj':>10} {'gap':>9} "
          f"{'mean obj':>10} {'P(optimum)':>11} {'feasible':>9} {'time':>8}")
    print("-" * 82)
    for mixer, qubo in (("x", qubo_pen), ("xy", qubo_free)):
        for p in (1, 2, 3, 5):
            r = solve_qaoa_local(qubo, p=p, seed=1, n_restarts=3, mixer=mixer)
            gap = r.objective - exact.objective
            label = "QAOA-XY (ours)" if mixer == "xy" else "QAOA-X (std)"
            print(f"{label:<18} {p:>2} {r.objective:>+10.5f} {gap:>+9.5f} "
                  f"{r.extra['mean_sampled_objective']:>+10.5f} "
                  f"{r.prob_of_optimum:>10.2%} {r.valid_prob:>9.1%} {r.seconds:>7.1f}s")

