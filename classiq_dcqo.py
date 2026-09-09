"""Gate-level Classiq implementation of constrained DCQO and BF-DCQO.

The Pauli algebra in this module is dependency-free and tested locally.
Classiq is imported only when a model is built or executed, so the rest of the
project remains runnable without cloud credentials.  The circuit uses one
qubit per one-hot variable, a weight-one Dicke state per block, an XY driver,
and Classiq ``suzuki_trotter`` evolution of the scheduled Hamiltonian.

This is separate from the ``T**S`` NumPy reference in :mod:`dcqo`: that
reference is useful for validation but is not a scalable circuit execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np

from dcqo import (AnnealingSchedule, BiasField, SubspaceOperators,
                  estimate_cd_coefficient)
from qaoa_subspace import subspace_cost


# A sparse Pauli string is a sorted tuple of ``(qubit, 'X'|'Y'|'Z')``.
_MUL = {
    ("I", "I"): (1, "I"), ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"), ("Y", "I"): (1, "Y"),
    ("Z", "I"): (1, "Z"), ("X", "X"): (1, "I"),
    ("Y", "Y"): (1, "I"), ("Z", "Z"): (1, "I"),
    ("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"), ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y"),
}


def _multiply_strings(left, right):
    a, b = dict(left), dict(right)
    phase, out = 1.0 + 0.0j, {}
    for q in sorted(set(a) | set(b)):
        local_phase, p = _MUL[(a.get(q, "I"), b.get(q, "I"))]
        phase *= local_phase
        if p != "I":
            out[q] = p
    return phase, tuple(sorted(out.items()))


def merge_pauli_terms(terms, tolerance: float = 1e-12):
    merged = {}
    for coefficient, paulis in terms:
        key = tuple(sorted(dict(paulis).items()))
        merged[key] = merged.get(key, 0.0 + 0.0j) + coefficient
    result = []
    for key, coefficient in sorted(merged.items()):
        if abs(coefficient) <= tolerance:
            continue
        coefficient = np.real_if_close(coefficient, tol=1000)
        if np.iscomplexobj(coefficient):
            raise ValueError(f"non-Hermitian Pauli coefficient {coefficient} for {key}")
        result.append((float(coefficient), key))
    return result


def pauli_commutator(left, right, tolerance: float = 1e-12):
    """Return ``i[left, right]`` as merged real Pauli terms."""
    terms = []
    for ca, pa in left:
        for cb, pb in right:
            phase_ab, product_ab = _multiply_strings(pa, pb)
            phase_ba, product_ba = _multiply_strings(pb, pa)
            if product_ab != product_ba:
                raise AssertionError("Pauli multiplication produced inconsistent support")
            terms.append((1j * ca * cb * (phase_ab - phase_ba), product_ab))
    return merge_pauli_terms(terms, tolerance)


def _constraint_free_qubo_matrix(qubo):
    """Remove one-hot penalties; the XY circuit never leaves their kernel."""
    Q = np.asarray(qubo.Q, dtype=float).copy()
    p, T = float(qubo.penalty), qubo.n_tilts
    if p == 0.0:
        return Q
    for block in range(qubo.n_sectors):
        base = block * T
        for k in range(T):
            Q[base + k, base + k] += p
        for k in range(T):
            for k2 in range(k + 1, T):
                Q[base + k, base + k2] -= 2.0 * p
    return Q


def cost_pauli_terms(qubo, tolerance: float = 1e-12):
    """Constraint-free Ising cost terms (the irrelevant constant is omitted)."""
    Q = _constraint_free_qubo_matrix(qubo)
    n = qubo.n_vars
    h = np.zeros(n)
    J = np.zeros((n, n))
    for i in range(n):
        h[i] -= Q[i, i] / 2.0
        for j in range(i + 1, n):
            q = Q[i, j] + Q[j, i]
            h[i] -= q / 4.0
            h[j] -= q / 4.0
            J[i, j] += q / 4.0
    terms = [(h[i], ((i, "Z"),)) for i in range(n)]
    terms.extend((J[i, j], ((i, "Z"), (j, "Z")))
                 for i in range(n) for j in range(i + 1, n))
    return merge_pauli_terms(terms, tolerance)


def xy_driver_pauli_terms(qubo, driver_g: float, topology: str = "ring"):
    """``-g/2 (XX+YY)`` per mixer edge, matching the NumPy hop matrix."""
    from dcqo import ring_edges

    terms = []
    for block in range(qubo.n_sectors):
        base = block * qubo.n_tilts
        for a, b in ring_edges(qubo.n_tilts, topology):
            qa, qb = base + a, base + b
            terms.append((-0.5 * driver_g, ((qa, "X"), (qb, "X"))))
            terms.append((-0.5 * driver_g, ((qa, "Y"), (qb, "Y"))))
    return merge_pauli_terms(terms)


def bias_pauli_terms(unary_bias, tolerance: float = 1e-12):
    """Map ``sum u[b,k] x[b,k]`` to Z terms, dropping its constant."""
    bias = np.asarray(unary_bias, dtype=float).ravel()
    return merge_pauli_terms([(-0.5 * u, ((q, "Z"),))
                              for q, u in enumerate(bias)], tolerance)


def dcqo_pauli_operators(qubo, *, unary_bias=None, topology="ring",
                         driver_strength="auto", cd_probes=6, cd_scale=1.0,
                         seed=0, tolerance=1e-12):
    """Generate driver, cost and ``i[H_d,H_f]`` terms plus alpha metadata."""
    cost = subspace_cost(qubo)
    bias = np.zeros((qubo.n_sectors, qubo.n_tilts)) if unary_bias is None \
        else np.asarray(unary_bias, dtype=float).reshape(qubo.n_sectors, qubo.n_tilts)
    ops = SubspaceOperators(cost, bias, topology=topology,
                            driver_strength=driver_strength)
    driver = xy_driver_pauli_terms(qubo, ops.g, topology)
    bias_terms = bias_pauli_terms(bias)
    initial = merge_pauli_terms(driver + bias_terms, tolerance)
    # Match the local reference exactly: the longitudinal BF term is present
    # in both endpoints. Scoring still always uses the original unbiased cost.
    final = merge_pauli_terms(cost_pauli_terms(qubo, tolerance) + bias_terms,
                              tolerance)
    # The diagonal parts commute with one another, so the commutator naturally
    # keeps only driver versus (cost + bias) contributions.
    cd = pauli_commutator(initial, final, tolerance)
    coefficient = estimate_cd_coefficient(
        ops, n_probes=cd_probes, seed=seed, scale=cd_scale)
    return {"initial": initial, "final": final, "cd": cd,
            "ops": ops, "coefficient": coefficient,
            "counts": {"initial": len(initial), "final": len(final),
                       "cd": len(cd),
                       "cd_by_weight": {str(w): sum(len(p) == w for _, p in cd)
                                        for w in range(1, 5)}}}


def scheduled_step_terms(operators, n_steps: int, time_scale: float,
                         mode="full", schedule_form="sin2",
                         tolerance=1e-12):
    """Materialize every digitized scheduled Hamiltonian as Pauli terms."""
    if mode not in ("full", "no_cd", "cd_only"):
        raise ValueError("mode must be full, no_cd or cd_only")
    spread = max(operators["ops"].cost_spread, 1e-12)
    total_time = float(time_scale) / spread
    schedule = AnnealingSchedule(n_steps, total_time, schedule_form)
    out = []
    for s, lam, lam_dot, dt in schedule.steps():
        terms = []
        if mode != "cd_only":
            terms.extend(((1.0 - lam) * c, p) for c, p in operators["initial"])
            terms.extend((lam * c, p) for c, p in operators["final"])
        if mode != "no_cd":
            alpha = operators["coefficient"].alpha(lam)
            terms.extend((lam_dot * alpha * c, p) for c, p in operators["cd"])
        out.append({"s": s, "lambda": lam, "lambda_dot": lam_dot, "dt": dt,
                    "terms": merge_pauli_terms(terms, tolerance)})
    return out


def _to_classiq_operator(terms):
    from classiq import Pauli

    total = None
    for coefficient, paulis in terms:
        term = float(coefficient)
        for q, name in paulis:
            term = term * getattr(Pauli, name)(q)
        total = term if total is None else total + term
    if total is None:
        total = 0.0 * Pauli.I(0)
    return total


def build_classiq_dcqo_model(qubo, *, n_steps=20, time_scale=80.0, mode="full",
                             trotter_order=1, trotter_repetitions=1,
                             topology="ring", unary_bias=None, cd_probes=6,
                             cd_scale=1.0, seed=0):
    """Return a static Classiq Qmod entry point and complete model metadata."""
    try:
        from classiq import Output, QArray, QBit, allocate, prepare_dicke_state
        from classiq import qfunc, suzuki_trotter
    except ImportError as exc:
        raise RuntimeError("Classiq is not installed; install requirements.txt and authenticate") from exc

    operators = dcqo_pauli_operators(
        qubo, unary_bias=unary_bias, topology=topology, cd_probes=cd_probes,
        cd_scale=cd_scale, seed=seed)
    steps = scheduled_step_terms(operators, n_steps, time_scale, mode)
    classiq_steps = [(_to_classiq_operator(step["terms"]), step["dt"]) for step in steps]
    n, T, blocks = qubo.n_vars, qubo.n_tilts, qubo.n_sectors

    @qfunc
    def main(v: Output[QArray[QBit, n]]):
        allocate(n, v)
        for block in range(blocks):
            prepare_dicke_state(1, v[block * T:(block + 1) * T])
        for hamiltonian, dt in classiq_steps:
            suzuki_trotter(hamiltonian, evolution_coefficient=dt,
                           order=trotter_order, repetitions=trotter_repetitions,
                           qbv=v)

    metadata = {
        "n_steps": n_steps, "time_scale": time_scale, "mode": mode,
        "trotter_order": trotter_order,
        "trotter_repetitions": trotter_repetitions, "topology": topology,
        "cd_probes": cd_probes, "cd_scale": cd_scale,
        "pauli_counts": operators["counts"],
        "schedule": [{k: v for k, v in step.items() if k != "terms"}
                     for step in steps],
        "step_pauli_counts": [len(step["terms"]) for step in steps],
        "alpha": {"numerator": operators["coefficient"].numerator.tolist(),
                  "denominator": operators["coefficient"].denominator.tolist(),
                  "scale": operators["coefficient"].scale,
                  "n_probes": operators["coefficient"].n_probes},
    }
    return main, metadata


@dataclass
class ClassiqDcqoResult:
    name: str
    config: np.ndarray
    objective: float
    seconds: float
    n_qubits: int
    depth: int
    prob_of_optimum: float
    mean_sampled_objective: float
    valid_prob: float
    circuit_depth: int = 0
    gate_count: int = 0
    extra: dict = field(default_factory=dict)


def _score_sample_frame(frame, qubo, exact_objective=None):
    total = valid = hits = 0
    weighted = 0.0
    best_obj, best_cfg = np.inf, None
    marginals = np.zeros((qubo.n_sectors, qubo.n_tilts), dtype=float)
    for _, row in frame.iterrows():
        bits = np.asarray(row["v"], dtype=int)
        count = int(row["counts"])
        total += count
        if not qubo.is_valid(bits):
            continue
        valid += count
        cfg = qubo.decode(bits)
        obj = qubo.objective_from_config(cfg)
        weighted += count * obj
        marginals[np.arange(qubo.n_sectors), cfg] += count
        if exact_objective is not None and np.isclose(obj, exact_objective, atol=1e-9):
            hits += count
        if obj < best_obj:
            best_obj, best_cfg = obj, cfg.copy()
    if valid == 0:
        raise RuntimeError("Classiq returned no feasible one-hot samples")
    marginals /= valid
    return {"config": best_cfg, "objective": float(best_obj),
            "mean": float(weighted / valid), "valid_prob": valid / max(total, 1),
            "p_opt": (float(hits / total) if exact_objective is not None else float("nan")),
            "marginals": marginals}


def run_classiq_dcqo(qubo, *, n_steps=20, n_shots=1000, time_scale=80.0,
                      mode="full", trotter_order=1, trotter_repetitions=1,
                      topology="ring", unary_bias=None, cd_probes=6,
                      cd_scale=1.0, seed=0, backend="simulator",
                      exact_objective=None, artifact_dir=None):
    """Synthesize and execute one DCQO schedule on Classiq."""
    try:
        from classiq import get_transpiled_circuit_metrics, synthesize
        from classiq.execution import ExecutionSession
    except ImportError as exc:
        raise RuntimeError("Classiq is not installed; install requirements.txt and authenticate") from exc

    t0 = time.perf_counter()
    main, metadata = build_classiq_dcqo_model(
        qubo, n_steps=n_steps, time_scale=time_scale, mode=mode,
        trotter_order=trotter_order, trotter_repetitions=trotter_repetitions,
        topology=topology, unary_bias=unary_bias, cd_probes=cd_probes,
        cd_scale=cd_scale, seed=seed)
    synthesis_start = time.perf_counter()
    qprog = synthesize(main)
    synthesis_seconds = time.perf_counter() - synthesis_start
    circuit_depth = gate_count = 0
    try:
        metrics = get_transpiled_circuit_metrics(qprog)
        circuit_depth = int(getattr(metrics, "depth", 0) or 0)
        counts = getattr(metrics, "count_ops", None) or {}
        gate_count = int(sum(counts.values())) if counts else 0
        metadata["gate_counts"] = counts
    except Exception as exc:  # metrics are useful, but not an execution blocker
        metadata["metrics_error"] = repr(exc)

    if artifact_dir is not None:
        out = Path(artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        program_text = qprog if isinstance(qprog, str) else json.dumps(qprog, default=str)
        (out / "classiq_program.json").write_text(program_text, encoding="utf-8")
        (out / "classiq_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")

    with ExecutionSession(qprog, backend=backend, num_shots=n_shots) as session:
        frame = session.sample({})
    scored = _score_sample_frame(frame, qubo, exact_objective)
    metadata.update({"backend": backend, "n_shots": n_shots,
                     "synthesis_seconds": synthesis_seconds,
                     "marginals": scored["marginals"]})
    return ClassiqDcqoResult(
        name=f"Classiq DCQO ({mode})", config=scored["config"],
        objective=scored["objective"], seconds=time.perf_counter() - t0,
        n_qubits=qubo.n_vars, depth=n_steps,
        prob_of_optimum=scored["p_opt"], mean_sampled_objective=scored["mean"],
        valid_prob=scored["valid_prob"], circuit_depth=circuit_depth,
        gate_count=gate_count, extra=metadata)


def run_classiq_bf_dcqo(qubo, *, n_iters=3, n_steps=20, total_shots=1000,
                         bias_strength=1.0, exact_objective=None, artifact_dir=None,
                         **kwargs):
    """BF-DCQO as repeated Classiq executions with marginal feedback."""
    shots = max(int(total_shots) // int(n_iters), 1)
    cost = subspace_cost(qubo)
    field_builder = BiasField(qubo.n_sectors, qubo.n_tilts,
                              float(cost.max() - cost.min()), bias_strength)
    bias = field_builder.zero()
    results = []
    t0 = time.perf_counter()
    for iteration in range(n_iters):
        iteration_dir = None if artifact_dir is None \
            else Path(artifact_dir) / f"bf_iteration_{iteration + 1}"
        result = run_classiq_dcqo(
            qubo, n_steps=n_steps, n_shots=shots, unary_bias=bias,
            exact_objective=exact_objective, artifact_dir=iteration_dir,
            seed=int(kwargs.get("seed", 0)) + iteration, **{k: v for k, v in kwargs.items()
                                                        if k != "seed"})
        results.append(result)
        bias = field_builder.from_marginals(result.extra["marginals"])
    best = min(results, key=lambda r: r.objective)
    best.name = "Classiq BF-DCQO"
    best.seconds = time.perf_counter() - t0
    best.prob_of_optimum = max(r.prob_of_optimum for r in results)
    best.extra.update({"n_iters": n_iters, "shots_per_iteration": shots,
                       "total_shots": shots * n_iters, "resynthesis_count": n_iters,
                       "history": [{"objective": r.objective,
                                    "prob_of_optimum": r.prob_of_optimum,
                                    "mean_sampled_objective": r.mean_sampled_objective,
                                    "seconds": r.seconds} for r in results]})
    return best
