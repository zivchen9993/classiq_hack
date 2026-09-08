"""
Diagnostic: is the tilt-optimization instance actually HARD?

The entire quantum-advantage argument rests on the landscape being frustrated
and multi-modal. If greedy local search finds the global optimum every time,
there is nothing for a quantum optimizer to contribute, and we should say so
rather than pretend otherwise.

This probe measures, over several random network instances:
  - how often greedy local search from a random start reaches the true optimum
  - how many DISTINCT local optima greedy falls into
  - the objective gap between the best uniform tilt and the true optimum
  - whether the optimum is non-uniform (i.e. per-sector tuning actually matters)
"""

from __future__ import annotations

import numpy as np

from rf_model import make_network
from qubo_builder import build_qubo
from classical_baseline import brute_force, greedy_local_search, best_uniform_baseline


def probe(seed: int, n_towers: int = 3, n_starts: int = 300, verbose: bool = True):
    net = make_network(n_towers, seed=seed)
    qubo = build_qubo(net)

    exact = brute_force(qubo)
    uniform = best_uniform_baseline(qubo)

    rng = np.random.default_rng(seed + 1000)
    local_optima, hits = {}, 0
    for s in range(n_starts):
        r = greedy_local_search(qubo, seed=int(rng.integers(0, 1 << 30)), n_restarts=1)
        key = tuple(int(v) for v in r.config)
        local_optima[key] = local_optima.get(key, 0) + 1
        if np.isclose(r.objective, exact.objective, atol=1e-9):
            hits += 1

    success_rate = 100.0 * hits / n_starts
    is_uniform = len(set(exact.config.tolist())) == 1

    if verbose:
        print(f"\n--- seed {seed} | {len(net.sectors)} sectors x {qubo.n_tilts} tilts "
              f"({qubo.n_vars} qubits, {qubo.n_tilts ** qubo.n_sectors:,} configs) ---")
        print(f"  optimum tilt config : {exact.config.tolist()}  "
              f"{'(UNIFORM - too easy)' if is_uniform else '(non-uniform)'}")
        print(f"  optimum objective   : {exact.objective:+.5f}")
        print(f"  best uniform tilt   : {uniform.objective:+.5f} "
              f"(gap {uniform.objective - exact.objective:+.5f})")
        print(f"  greedy hits optimum : {success_rate:.1f}% of {n_starts} random starts")
        print(f"  distinct local optima found by greedy: {len(local_optima)}")
        top = sorted(local_optima.items(), key=lambda kv: -kv[1])[:3]
        for cfg, count in top:
            obj = qubo.objective_from_config(np.array(cfg))
            print(f"      {list(cfg)}  hit {count:>3}x  obj={obj:+.5f} "
                  f"(gap {obj - exact.objective:+.5f})")

    return {
        "seed": seed,
        "success_rate": success_rate,
        "n_local_optima": len(local_optima),
        "uniform_gap": uniform.objective - exact.objective,
        "optimum_is_uniform": is_uniform,
        "optimum_config": exact.config,
        "net": net,
        "qubo": qubo,
    }


if __name__ == "__main__":
    print("Probing instance hardness across random network seeds")
    print("=" * 70)
    stats = [probe(seed) for seed in (7, 11, 23, 42, 99)]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'seed':>5} {'greedy hit-rate':>16} {'local optima':>13} "
          f"{'uniform gap':>12} {'non-uniform opt':>16}")
    for s in stats:
        print(f"{s['seed']:>5} {s['success_rate']:>15.1f}% {s['n_local_optima']:>13} "
              f"{s['uniform_gap']:>+12.5f} {str(not s['optimum_is_uniform']):>16}")

    mean_hit = np.mean([s["success_rate"] for s in stats])
    print(f"\nmean greedy hit-rate: {mean_hit:.1f}%")
    if mean_hit > 90:
        print("=> instance is TOO EASY: greedy nearly always finds the optimum.")
    else:
        print("=> instance is frustrated: greedy gets trapped in local optima, "
              "which is what a global (quantum) optimizer can exploit.")

