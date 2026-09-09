"""
End-to-end benchmark for the antenna tilt optimization proof-of-concept.

Sections:

  1. SURROGATE VALIDATION
     The QUBO is a unary+pairwise DECOMPOSITION of a fully coupled RF problem,
     so we measure how well each term tracks the KPI it is meant to control,
     using the exact joint-SINR simulator as ground truth.

  2. SOLVER COMPARISON + NETWORK KPIs
     Uniform tilt / greedy / simulated annealing / brute force / QAOA, all
     scored on the same objective and independently re-scored with the exact
     joint-SINR simulator.

  3. PARETO FRONTIER
     Coverage+interference and handover reliability genuinely compete. The
     objective weights are an operator policy dial, so we show the frontier
     rather than pretending a single point dominates.

  4. HARDNESS AT SCALE
     Where greedy local search actually breaks down, which is the honest basis
     for the quantum argument.

  5. SCALING
     How the classical search space and the qubit requirement grow.
"""

import time
import numpy as np
from scipy.stats import spearmanr

from rf_model import make_network
from qubo_builder import build_qubo
from classical_baseline import (
    brute_force, greedy_local_search, simulated_annealing, best_uniform_baseline,
)
from qaoa_local import solve_qaoa_local


# --- the flagship instance, small enough for exact QAOA statevector simulation
FLAGSHIP = dict(n_towers=2, seed=27, tilt_levels_deg=[0.0, 4.0, 8.0])
# --- a larger instance where classical local search genuinely struggles
HARD = dict(n_towers=3, seed=99, tilt_levels_deg=[0.0, 3.0, 6.0, 10.0])


# ---------------------------------------------------------------------------
# 1. Surrogate validation
# ---------------------------------------------------------------------------

def validate_surrogate(net, qubo, n_samples: int = 400, seed: int = 0):
    """Does minimizing the QUBO actually improve the real network?

    The QUBO decomposes a fully coupled problem into unary + pairwise terms,
    so this must be checked rather than assumed. Each term is validated
    against the KPI it is meant to control: the objective is deliberately
    multi-objective, so the total is NOT expected to track any single KPI.
    """
    rng = np.random.default_rng(seed)
    cfgs = [rng.integers(0, qubo.n_tilts, size=qubo.n_sectors) for _ in range(n_samples)]
    kpis = [net.evaluate_config(c) for c in cfgs]

    sinr = np.array([k["mean_sinr_db"] for k in kpis])
    eff = np.array([k["mean_spectral_efficiency"] for k in kpis])
    outage = np.array([k["outage_pct"] for k in kpis])
    ho_fail = np.array([k["handover_failure_pct"] for k in kpis])

    cov_part = np.array([sum(qubo.cov_norm[i, c[i]] for i in range(qubo.n_sectors))
                         for c in cfgs])
    int_part = np.array([sum(t[c[i], c[j]] for (i, j), t in qubo.int_norm.items())
                         for c in cfgs])
    ho_part = np.array([sum(t[c[i], c[j]] for (i, j), t in qubo.ho_norm.items())
                        for c in cfgs])
    quality = -qubo.w_cov * cov_part + qubo.w_int * int_part

    return {
        "n_samples": n_samples,
        "rows": [
            ("coverage term", "outage %", spearmanr(cov_part, outage).statistic, "negative"),
            ("interference term", "mean SINR", spearmanr(int_part, sinr).statistic, "negative"),
            ("coverage+interference", "mean SINR", spearmanr(quality, sinr).statistic, "negative"),
            ("coverage+interference", "spectral eff.", spearmanr(quality, eff).statistic, "negative"),
            ("handover term", "handover failure %", spearmanr(ho_part, ho_fail).statistic, "positive"),
        ],
        "sinr_range": (float(sinr.min()), float(sinr.max())),
        "ho_range": (float(ho_fail.min()), float(ho_fail.max())),
    }


# ---------------------------------------------------------------------------
# 2. Solver comparison
# ---------------------------------------------------------------------------

def compare_solvers(qubo, qubo_free, run_qaoa: bool = True, qaoa_p: int = 3,
                    seed: int = 0):
    results = [
        best_uniform_baseline(qubo),
        greedy_local_search(qubo, seed=seed, n_restarts=1),
    ]
    r = greedy_local_search(qubo, seed=seed, n_restarts=20)
    r.name = "greedy (20 restarts)"
    results.append(r)
    results.append(simulated_annealing(qubo, seed=seed))

    if run_qaoa:
        results.append(solve_qaoa_local(qubo_free, p=qaoa_p, seed=seed,
                                        mixer="xy", n_restarts=3))
    try:
        results.append(brute_force(qubo))
    except ValueError as e:
        print(f"  (brute force skipped: {e})")
    return results


def kpi_table(net, results, reference_name="best uniform tilt"):
    exact = next((r for r in results if "brute force" in r.name), None)

    print(f"\n{'solver':<24} {'objective':>11} {'gap':>9} {'time':>11} {'evals':>11}")
    print("-" * 71)
    for r in results:
        gap = (r.objective - exact.objective) if exact else float("nan")
        print(f"{r.name:<24} {r.objective:>+11.5f} {gap:>+9.5f} "
              f"{r.seconds * 1000:>9.1f}ms {getattr(r, 'evaluations', 0):>11,}")

    print(f"\n{'solver':<24} {'SINR':>9} {'edge':>9} {'sp.eff':>8} "
          f"{'outage':>8} {'HO fail':>9}")
    print("-" * 71)
    kpis = {}
    for r in results:
        k = net.evaluate_config(r.config)
        kpis[r.name] = k
        print(f"{r.name:<24} {k['mean_sinr_db']:>8.2f}dB {k['edge_sinr_db']:>8.2f}dB "
              f"{k['mean_spectral_efficiency']:>8.3f} {k['outage_pct']:>7.1f}% "
              f"{k['handover_failure_pct']:>8.1f}%")

    ref = kpis.get(reference_name)
    if ref and exact:
        best = kpis[exact.name]
        eff_gain = 100.0 * (best['mean_spectral_efficiency'] / ref['mean_spectral_efficiency'] - 1)
        print(f"\nOptimized per-sector tilts vs '{reference_name}' "
              f"(one network-wide tilt = today's practice):")
        print(f"  mean SINR            {ref['mean_sinr_db']:6.2f} dB -> "
              f"{best['mean_sinr_db']:6.2f} dB  ({best['mean_sinr_db'] - ref['mean_sinr_db']:+.2f} dB)")
        print(f"  cell-edge SINR       {ref['edge_sinr_db']:6.2f} dB -> "
              f"{best['edge_sinr_db']:6.2f} dB  ({best['edge_sinr_db'] - ref['edge_sinr_db']:+.2f} dB)")
        print(f"  spectral efficiency  {ref['mean_spectral_efficiency']:6.3f}    -> "
              f"{best['mean_spectral_efficiency']:6.3f}     ({eff_gain:+.1f}% throughput)")
        print(f"  outage               {ref['outage_pct']:6.1f} %  -> "
              f"{best['outage_pct']:6.1f} %   ({best['outage_pct'] - ref['outage_pct']:+.1f} pp)")
        print(f"  handover failures    {ref['handover_failure_pct']:6.1f} %  -> "
              f"{best['handover_failure_pct']:6.1f} %   "
              f"({best['handover_failure_pct'] - ref['handover_failure_pct']:+.1f} pp)")
    return kpis


# ---------------------------------------------------------------------------
# 3. Pareto frontier
# ---------------------------------------------------------------------------

def pareto_frontier(net, weight_sets=None):
    """Coverage/interference vs handover reliability is a real trade-off.

    The weights are an operator policy choice, not a fact of nature, so we
    report the frontier instead of a single cherry-picked point.
    """
    if weight_sets is None:
        weight_sets = [(0.40, 0.35, 0.25), (0.375, 0.325, 0.30),
                       (0.35, 0.30, 0.35), (0.325, 0.275, 0.40),
                       (0.30, 0.25, 0.45), (0.30, 0.20, 0.50)]

    uni_ref = None
    print(f"\n{'w_cov':>6} {'w_int':>6} {'w_ho':>6} | {'SINR':>8} {'sp.eff':>8} "
          f"{'outage':>8} {'HO fail':>9} | {'tilts':>20}")
    print("-" * 84)
    for (w_cov, w_int, w_ho) in weight_sets:
        qubo = build_qubo(net, w_cov=w_cov, w_int=w_int, w_ho=w_ho)
        opt = brute_force(qubo)
        k = net.evaluate_config(opt.config)
        if uni_ref is None:
            uni = best_uniform_baseline(qubo)
            uni_ref = net.evaluate_config(uni.config)
            print(f"{'uniform':>6} {'(baseline)':>13} | {uni_ref['mean_sinr_db']:>7.2f}dB "
                  f"{uni_ref['mean_spectral_efficiency']:>8.3f} {uni_ref['outage_pct']:>7.1f}% "
                  f"{uni_ref['handover_failure_pct']:>8.1f}% | "
                  f"{str(uni.config.tolist()):>20}")
            print("-" * 84)
        print(f"{w_cov:>6.3f} {w_int:>6.3f} {w_ho:>6.3f} | {k['mean_sinr_db']:>7.2f}dB "
              f"{k['mean_spectral_efficiency']:>8.3f} {k['outage_pct']:>7.1f}% "
              f"{k['handover_failure_pct']:>8.1f}% | {str(opt.config.tolist()):>20}")


# ---------------------------------------------------------------------------
# 4. Hardness at scale
# ---------------------------------------------------------------------------

def hardness(net, qubo, n_starts: int = 200, seed: int = 0):
    """How often does greedy local search miss the global optimum?

    This is the honest basis for the quantum argument: if greedy always won,
    there would be nothing for a global optimizer to contribute.
    """
    exact = brute_force(qubo)
    rng = np.random.default_rng(seed)
    local, hits = {}, 0
    for _ in range(n_starts):
        r = greedy_local_search(qubo, seed=int(rng.integers(0, 1 << 30)), n_restarts=1)
        key = tuple(int(v) for v in r.config)
        local[key] = local.get(key, 0) + 1
        if np.isclose(r.objective, exact.objective, atol=1e-9):
            hits += 1
    return {
        "hit_rate": 100.0 * hits / n_starts,
        "n_local_optima": len(local),
        "optimum": exact,
        "local_optima": local,
        "n_starts": n_starts,
    }


# ---------------------------------------------------------------------------
# 5. Scaling
# ---------------------------------------------------------------------------

def scaling_table(tilt_levels=4, max_towers=8, rate_per_sec=90_000.0):
    print(f"\n{'towers':>7} {'sectors':>8} {'configurations':>24} {'qubits':>8} "
          f"{'brute force':>16}")
    print("-" * 68)
    for n_towers in range(1, max_towers + 1):
        n_sectors = 3 * n_towers
        configs = tilt_levels ** n_sectors
        secs = configs / rate_per_sec
        if secs < 60:
            t = f"{secs:.1f} s"
        elif secs < 3600:
            t = f"{secs/60:.1f} min"
        elif secs < 86400 * 365:
            t = f"{secs/86400:.1f} days"
        else:
            t = f"{secs/(86400*365):.3g} years"
        print(f"{n_towers:>7} {n_sectors:>8} {configs:>24,} {n_sectors*tilt_levels:>8} {t:>16}")


# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("ANTENNA TILT OPTIMIZATION -- QUANTUM/CLASSICAL BENCHMARK")
    print("=" * 80)

    net = make_network(**FLAGSHIP)
    qubo = build_qubo(net)
    qubo_free = build_qubo(net, penalty_scale=0.0)

    print(f"\nFlagship instance: {len(net.sectors)} sectors, "
          f"{len(net.tilt_levels_deg)} tilt levels {net.tilt_levels_deg}")
    print(f"  coupled sector pairs : {len(net.neighbor_pairs)} "
          f"(interference {len(net.interference)}, handover {len(net.handover)})")
    print(f"  search space         : {qubo.n_tilts ** qubo.n_sectors:,} configurations")
    print(f"  qubits (one-hot)     : {qubo.n_vars}")

    print("\n" + "-" * 80)
    print("1. SURROGATE VALIDATION -- does the QUBO track the real network?")
    print("-" * 80)
    v = validate_surrogate(net, qubo)
    print(f"  {v['n_samples']} random configurations "
          f"(SINR spans {v['sinr_range'][0]:.2f}..{v['sinr_range'][1]:.2f} dB, "
          f"handover failure {v['ho_range'][0]:.1f}..{v['ho_range'][1]:.1f}%)\n")
    print(f"  {'QUBO term':<24} {'vs KPI':<20} {'rho':>8}   want")
    print("  " + "-" * 62)
    for term, kpi, rho, want in v["rows"]:
        flag = "OK " if ((rho < -0.2) if want == "negative" else (rho > 0.2)) else "weak"
        print(f"  {term:<24} {kpi:<20} {rho:>+8.3f}   {want:<9} {flag}")
    print("\n  The objective is multi-objective by construction: interference")
    print("  suppression and handover reliability genuinely oppose each other,")
    print("  so no single KPI should track the total. Each term tracks its own.")

    print("\n" + "-" * 80)
    print("2. SOLVER COMPARISON + NETWORK KPIs (exact joint-SINR simulation)")
    print("-" * 80)
    results = compare_solvers(qubo, qubo_free, run_qaoa=True, qaoa_p=3)
    kpi_table(net, results)

    print("\n" + "-" * 80)
    print("3. PARETO FRONTIER -- the weights are an operator policy dial")
    print("-" * 80)
    pareto_frontier(net)

    print("\n" + "-" * 80)
    print("4. HARDNESS -- where classical local search actually breaks down")
    print("-" * 80)
    for label, cfg in (("flagship (6 sectors, 3 tilts)", FLAGSHIP),
                       ("larger  (9 sectors, 4 tilts)", HARD)):
        n = make_network(**cfg)
        q = build_qubo(n)
        h = hardness(n, q)
        print(f"\n  {label}: {q.n_vars} qubits, "
              f"{q.n_tilts ** q.n_sectors:,} configurations")
        print(f"    greedy reaches the global optimum in {h['hit_rate']:.0f}% "
              f"of {h['n_starts']} random starts")
        print(f"    distinct local optima found: {h['n_local_optima']}")
        print(f"    optimum tilt indices: {h['optimum'].config.tolist()}")

    print("\n" + "-" * 80)
    print("5. SCALING -- why this needs more than brute force")
    print("-" * 80)
    scaling_table(tilt_levels=4)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

