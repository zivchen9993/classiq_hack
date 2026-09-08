"""
Classical baselines for antenna tilt optimization.

Three solvers, all minimizing the EXACT SAME objective the QAOA circuit
minimizes (`TiltQUBO.objective_from_config`), so the comparison is fair:

  1. brute_force        -- exhaustive search, the true optimum (small instances only)
  2. greedy_local_search -- "what operators do today": per-sector greedy sweeps
                            until no single-sector change helps (a local optimum)
  3. simulated_annealing -- a strong classical metaheuristic baseline

Each returns a SolverResult with the configuration, objective value, wall-clock
time and the number of objective evaluations used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
import numpy as np


@dataclass
class SolverResult:
    name: str
    config: np.ndarray
    objective: float
    seconds: float
    evaluations: int
    extra: dict = field(default_factory=dict)

    def __str__(self):
        return (f"{self.name:<22} obj={self.objective:+.5f}  "
                f"{self.seconds * 1000:8.1f} ms  {self.evaluations:>9,} evals  "
                f"tilts={self.config.tolist()}")


def brute_force(qubo, max_configs: int = 2_000_000) -> SolverResult:
    """Exhaustive search over all tilt configurations -> the true optimum.

    Only tractable for small instances; that is exactly the point we make in
    the scalability argument (n_tilts ** n_sectors grows explosively).
    """
    total = qubo.n_tilts ** qubo.n_sectors
    if total > max_configs:
        raise ValueError(f"brute force would need {total:,} evaluations "
                         f"(limit {max_configs:,}) -- instance too large")

    t0 = time.perf_counter()
    best_cfg, best_obj, evals = None, np.inf, 0
    for cfg in product(range(qubo.n_tilts), repeat=qubo.n_sectors):
        obj = qubo.objective_from_config(cfg)
        evals += 1
        if obj < best_obj:
            best_obj, best_cfg = obj, cfg
    seconds = time.perf_counter() - t0

    return SolverResult("brute force (exact)", np.array(best_cfg), best_obj, seconds, evals,
                        extra={"search_space": total})


def greedy_local_search(qubo, seed: int = 0, n_restarts: int = 1) -> SolverResult:
    """Per-sector greedy coordinate descent, repeated until no single-sector
    change improves the objective.

    This is the honest stand-in for classical practice: today's tilt tuning is
    local, sector-by-sector, and gets stuck in local optima precisely because
    sectors are strongly coupled.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    evals = 0
    best_cfg, best_obj = None, np.inf

    for _ in range(n_restarts):
        cfg = rng.integers(0, qubo.n_tilts, size=qubo.n_sectors)
        obj = qubo.objective_from_config(cfg)
        evals += 1
        improved = True
        while improved:
            improved = False
            for i in range(qubo.n_sectors):
                current_t = cfg[i]
                for t in range(qubo.n_tilts):
                    if t == current_t:
                        continue
                    cfg[i] = t
                    cand = qubo.objective_from_config(cfg)
                    evals += 1
                    if cand < obj:
                        obj, current_t, improved = cand, t, True
                    else:
                        cfg[i] = current_t
                cfg[i] = current_t
        if obj < best_obj:
            best_obj, best_cfg = obj, cfg.copy()

    seconds = time.perf_counter() - t0
    return SolverResult(f"greedy local search", best_cfg, best_obj, seconds, evals,
                        extra={"restarts": n_restarts})


def simulated_annealing(qubo, seed: int = 0, n_steps: int = 20_000,
                        t_start: float = 1.0, t_end: float = 1e-3) -> SolverResult:
    """Simulated annealing over single-sector tilt moves."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    cfg = rng.integers(0, qubo.n_tilts, size=qubo.n_sectors)
    obj = qubo.objective_from_config(cfg)
    evals = 1
    best_cfg, best_obj = cfg.copy(), obj

    for step in range(n_steps):
        temp = t_start * (t_end / t_start) ** (step / max(n_steps - 1, 1))
        i = rng.integers(0, qubo.n_sectors)
        old_t = cfg[i]
        new_t = rng.integers(0, qubo.n_tilts)
        if new_t == old_t:
            continue
        cfg[i] = new_t
        cand = qubo.objective_from_config(cfg)
        evals += 1
        delta = cand - obj
        if delta <= 0 or rng.random() < np.exp(-delta / temp):
            obj = cand
            if obj < best_obj:
                best_obj, best_cfg = obj, cfg.copy()
        else:
            cfg[i] = old_t

    seconds = time.perf_counter() - t0
    return SolverResult("simulated annealing", best_cfg, best_obj, seconds, evals,
                        extra={"steps": n_steps})


def uniform_baseline(qubo, tilt_index: int) -> SolverResult:
    """Every sector at the same tilt -- the 'do nothing clever' reference,
    which is how many networks are actually configured."""
    cfg = np.full(qubo.n_sectors, tilt_index, dtype=int)
    return SolverResult(f"uniform tilt (level {tilt_index})", cfg,
                        qubo.objective_from_config(cfg), 0.0, 1)


def best_uniform_baseline(qubo) -> SolverResult:
    """The best single tilt applied network-wide."""
    results = [uniform_baseline(qubo, t) for t in range(qubo.n_tilts)]
    best = min(results, key=lambda r: r.objective)
    return SolverResult("best uniform tilt", best.config, best.objective, 0.0,
                        qubo.n_tilts, extra={"tilt_index": int(best.config[0])})


if __name__ == "__main__":
    from rf_model import make_hex_like_network
    from qubo_builder import build_qubo

    net = make_hex_like_network(3)
    qubo = build_qubo(net)
    print(qubo.summary())
    print(f"search space: {qubo.n_tilts ** qubo.n_sectors:,} configurations\n")

    results = [
        best_uniform_baseline(qubo),
        greedy_local_search(qubo, n_restarts=1),
        greedy_local_search(qubo, n_restarts=20),
        simulated_annealing(qubo),
        brute_force(qubo),
    ]
    results[2].name = "greedy (20 restarts)"

    for r in results:
        print(r)

    exact = results[-1]
    print("\nExact-KPI validation of each solution (full joint SINR simulation):")
    for r in results:
        kpi = net.evaluate_config(r.config)
        gap = r.objective - exact.objective
        print(f"  {r.name:<22} mean SINR {kpi['mean_sinr_db']:6.2f} dB | "
              f"edge {kpi['edge_sinr_db']:6.2f} dB | outage {kpi['outage_pct']:5.1f}% | "
              f"spec.eff {kpi['mean_spectral_efficiency']:.3f} | obj gap {gap:+.5f}")

