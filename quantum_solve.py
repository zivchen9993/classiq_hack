"""
QAOA for antenna tilt optimization, implemented on the Classiq platform.

Two ansaetze are built, synthesized and executed on Classiq:

  1. "x"  -- textbook QAOA: Hadamard init + cost layer + transverse-field
             mixer, with the one-hot constraint enforced by a penalty term
             added to the Hamiltonian.

  2. "xy" -- constrained QAOA (Quantum Alternating Operator Ansatz): a W state
             per sector + cost layer + XY mixer restricted to each sector's
             one-hot block. RXX(b)*RYY(b) = exp(-i*b*(XX+YY)/2) acts only on
             the {|01>,|10>} subspace of a pair, so it CONSERVES Hamming
             weight: the state can never leave the feasible subspace and the
             penalty term is dropped entirely.

The Hamiltonian handed to Classiq is exactly the Ising form of the QUBO from
`qubo_builder.py`, and every sampled bitstring is scored with exactly the
objective the classical baselines use -- so the comparison is apples-to-apples.

Requires Classiq authentication:
    python -c "import classiq; classiq.authenticate()"
"""

# NOTE: deliberately NO `from __future__ import annotations` here.
# That would turn the @qfunc parameter annotations into strings, and Classiq
# infers each Qmod function's signature by introspecting the real annotation
# objects -- with PEP 563 active it raises
# "TypeError: issubclass() arg 1 must be a class" during synthesis.

import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import minimize

from classiq import (
    CArray, CReal, Output, Pauli, PauliTerm, QArray, QBit,
    RXX, RYY, allocate, prepare_dicke_state, qaoa_cost_layer,
    qaoa_init, qaoa_mixer_layer, qfunc, synthesize,
)
from classiq.execution import ExecutionSession


# ---------------------------------------------------------------------------
# QUBO -> Ising Pauli terms
# ---------------------------------------------------------------------------

def ising_pauli_terms(qubo):
    """Convert the QUBO into Classiq PauliTerms plus a constant.

        H = const + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

    The constant is not put into the circuit (a global phase cannot change any
    measurement outcome) but is returned so energies can be reconstructed.
    """
    h, J, const = qubo.to_ising()
    n = qubo.n_vars
    terms = []

    for i in range(n):
        if abs(h[i]) > 1e-12:
            paulis = [Pauli.I] * n
            paulis[i] = Pauli.Z
            terms.append(PauliTerm(pauli=paulis, coefficient=float(h[i])))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                paulis = [Pauli.I] * n
                paulis[i] = Pauli.Z
                paulis[j] = Pauli.Z
                terms.append(PauliTerm(pauli=paulis, coefficient=float(J[i, j])))

    return terms, float(const)


# ---------------------------------------------------------------------------
# Ansatz construction
# ---------------------------------------------------------------------------

def build_qaoa_model(qubo, p: int, mixer: str = "xy"):
    """Return a Qmod `main` entry point for the requested ansatz.

    The Python loops are unrolled at model-construction time, so the
    Hamiltonian and the block structure are compile-time constants, which is
    what lets Classiq's synthesis engine optimize the whole circuit.
    """
    hamiltonian, const = ising_pauli_terms(qubo)
    n, m, n_sectors = qubo.n_vars, qubo.n_tilts, qubo.n_sectors

    if mixer == "x":
        @qfunc
        def main(params: CArray[CReal, 2 * p], v: Output[QArray[QBit, n]]):
            allocate(n, v)
            qaoa_init(v)                                    # |+>^n
            for layer in range(p):
                qaoa_cost_layer(params[2 * layer], hamiltonian, v)
                qaoa_mixer_layer(params[2 * layer + 1], v)

    elif mixer == "xy":
        @qfunc
        def main(params: CArray[CReal, 2 * p], v: Output[QArray[QBit, n]]):
            allocate(n, v)
            # Feasible initial state: Dicke(m, 1) on each sector's block is the
            # equal superposition of that block's m one-hot states, so the
            # register starts as the uniform superposition over every valid
            # tilt configuration and nothing else.
            for i in range(n_sectors):
                prepare_dicke_state(1, v[i * m:(i + 1) * m])

            for layer in range(p):
                qaoa_cost_layer(params[2 * layer], hamiltonian, v)
                beta = params[2 * layer + 1]
                # XY mixer on adjacent pairs within each block. Adjacent pairs
                # only, because a Qmod QArray slice must be contiguous -- the
                # local simulator uses the same chain topology so the two match.
                for i in range(n_sectors):
                    base = i * m
                    for k in range(m - 1):
                        RXX(beta, v[base + k:base + k + 2])
                        RYY(beta, v[base + k:base + k + 2])
    else:
        raise ValueError(f"unknown mixer {mixer!r}")

    return main, const


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ClassiqQAOAResult:
    name: str
    config: np.ndarray
    objective: float
    seconds: float
    n_qubits: int
    depth: int
    mixer: str
    circuit_depth: int
    gate_count: int
    prob_of_optimum: float
    valid_prob: float
    mean_sampled_objective: float
    optimal_params: np.ndarray
    evaluations: int = 0
    extra: dict = field(default_factory=dict)

    def __str__(self):
        return (f"{self.name:<26} obj={self.objective:+.5f}  {self.n_qubits}q "
                f"p={self.depth}  P(opt)={self.prob_of_optimum:6.2%}  "
                f"feasible={self.valid_prob:6.1%}  {self.seconds:.1f}s")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _score_samples(result, qubo, cache):
    """Score one Classiq SampleResult with the physics objective.

    Returns (best_obj, best_cfg, mean_obj, feasible_fraction, cvar, counts_by_cfg).
    """
    best_obj, best_cfg = np.inf, None
    total = valid = 0
    obj_sum = 0.0
    per_shot = []
    counts_by_cfg = {}

    for _, row in result.iterrows():
        bits = np.asarray(row["v"], dtype=int)
        c = int(row["counts"])
        total += c
        if qubo.is_valid(bits):
            valid += c
        cfg = qubo.decode(bits)
        key = tuple(int(t) for t in cfg)
        if key not in cache:
            cache[key] = qubo.objective_from_config(cfg)
        obj = cache[key]
        counts_by_cfg[key] = counts_by_cfg.get(key, 0) + c
        obj_sum += obj * c
        per_shot.append((obj, c))
        if obj < best_obj:
            best_obj, best_cfg = obj, cfg

    total = max(total, 1)
    return best_obj, best_cfg, obj_sum / total, valid / total, per_shot, counts_by_cfg


def _cvar(per_shot, quantile: float) -> float:
    """Mean objective of the best `quantile` fraction of shots."""
    if quantile >= 1.0:
        tot = sum(c for _, c in per_shot)
        return sum(o * c for o, c in per_shot) / max(tot, 1)
    per_shot = sorted(per_shot)
    n_total = sum(c for _, c in per_shot)
    target = max(int(np.ceil(quantile * n_total)), 1)
    taken, acc = 0, 0.0
    for obj, c in per_shot:
        take = min(c, target - taken)
        acc += obj * take
        taken += take
        if taken >= target:
            break
    return acc / max(taken, 1)


def run_classiq_qaoa(qubo, p: int = 2, mixer: str = "xy", n_shots: int = 1000,
                     maxiter: int = 40, seed: int = 0, quantile: float = 0.2,
                     backend: str = "simulator",
                     exact_objective: Optional[float] = None,
                     verbose: bool = True) -> ClassiqQAOAResult:
    """Build, synthesize and run the QAOA ansatz on Classiq end to end.

    A COBYLA outer loop drives the variational parameters; each iteration
    samples the synthesized circuit on Classiq's simulator and scores the shot
    distribution with the CVaR of the physics objective.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    main, const = build_qaoa_model(qubo, p, mixer)
    if verbose:
        print(f"  synthesizing {mixer.upper()} ansatz: {qubo.n_vars} qubits, p={p} ...")
    qprog = synthesize(main)

    circuit_depth = gate_count = 0
    try:
        from classiq import get_transpiled_circuit_metrics
        metrics = get_transpiled_circuit_metrics(qprog)
        circuit_depth = int(getattr(metrics, "depth", 0) or 0)
        ops = getattr(metrics, "count_ops", None) or {}
        gate_count = int(sum(ops.values())) if ops else 0
    except Exception:
        pass
    if verbose and circuit_depth:
        print(f"  synthesized: circuit depth {circuit_depth}, {gate_count} gates")

    cache: dict = {}
    evals = 0

    with ExecutionSession(qprog, backend=backend, num_shots=n_shots) as sess:

        def cost(params):
            nonlocal evals
            evals += 1
            res = sess.sample({"params": [float(v) for v in params]})
            _, _, _, _, per_shot, _ = _score_samples(res, qubo, cache)
            return _cvar(per_shot, quantile)

        x0 = np.concatenate([rng.uniform(0.0, 0.6, size=p),
                             rng.uniform(0.0, np.pi / 2, size=p)])
        if verbose:
            print(f"  optimizing {2 * p} parameters, <= {maxiter} circuit executions ...")
        res = minimize(cost, x0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.3})

        final = sess.sample({"params": [float(v) for v in res.x]})

    best_obj, best_cfg, mean_obj, valid_prob, _, counts_by_cfg = _score_samples(
        final, qubo, cache)

    # probability of sampling the true optimum
    if exact_objective is None:
        from classical_baseline import brute_force
        try:
            exact_objective = brute_force(qubo).objective
        except ValueError:
            exact_objective = None

    if exact_objective is None:
        prob_of_optimum = float("nan")
    else:
        total = sum(counts_by_cfg.values()) or 1
        hit = sum(c for cfg, c in counts_by_cfg.items()
                  if np.isclose(cache[cfg], exact_objective, atol=1e-9))
        prob_of_optimum = hit / total

    seconds = time.perf_counter() - t0
    label = "Classiq QAOA-XY" if mixer == "xy" else "Classiq QAOA-X"
    return ClassiqQAOAResult(
        name=f"{label} (p={p})", config=best_cfg, objective=best_obj,
        seconds=seconds, n_qubits=qubo.n_vars, depth=p, mixer=mixer,
        circuit_depth=circuit_depth, gate_count=gate_count,
        prob_of_optimum=prob_of_optimum, valid_prob=valid_prob,
        mean_sampled_objective=mean_obj, optimal_params=res.x,
        evaluations=evals,
        extra={"n_shots": n_shots, "quantile": quantile, "backend": backend},
    )


if __name__ == "__main__":
    from rf_model import make_network
    from qubo_builder import build_qubo
    from classical_baseline import brute_force

    # the same flagship instance the benchmark and write-up use
    net = make_network(n_towers=2, seed=27, tilt_levels_deg=[0.0, 4.0, 8.0])
    qubo_pen = build_qubo(net)
    qubo_free = build_qubo(net, penalty_scale=0.0)

    exact = brute_force(qubo_pen)
    print(qubo_pen.summary())
    print(f"brute-force optimum: {exact.objective:+.5f} tilts={exact.config.tolist()}")
    n_feasible = qubo_pen.n_tilts ** qubo_pen.n_sectors
    print(f"uniform-random reference: P(optimum) = 1/{n_feasible} = {1/n_feasible:.2%}\n")

    for mixer, qubo in (("xy", qubo_free), ("x", qubo_pen)):
        print(f"--- {mixer.upper()} ansatz on Classiq ---")
        r = run_classiq_qaoa(qubo, p=2, mixer=mixer, n_shots=1000, maxiter=30,
                             exact_objective=exact.objective)
        print(" ", r)
        print(f"  gap to optimum: {r.objective - exact.objective:+.5f}")
        print(f"  tilts: {r.config.tolist()}\n")

