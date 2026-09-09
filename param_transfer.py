"""
Does a trained QAOA schedule transfer to UNSEEN network instances?

Why this matters
----------------
Per instance and from cold, QAOA loses badly: finding the variational
parameters costs far more circuit executions than simply running a classical
heuristic. The only honest route to a favourable resource count is
AMORTIZATION -- train the schedule once, then reuse it on many instances,
paying only the (cheap) sampling cost each time.

For antenna tilt that is exactly the operational picture: the network is
re-optimized continuously as traffic shifts, so the same schedule would be
reused thousands of times.

But amortization is only valid if the parameters actually transfer. This
script measures it instead of assuming it:

  1. train a depth-p schedule on ONE instance
  2. apply it VERBATIM (no re-optimization) to fresh, unseen instances
  3. compare the resulting P(optimum) against
       - random sampling
       - a schedule trained on that instance directly (the ceiling)
"""

from __future__ import annotations

import numpy as np

from rf_model import make_network
from qubo_builder import build_qubo
from classical_baseline import brute_force
from qaoa_subspace import (
    subspace_cost, subspace_statevector, solve_qaoa_subspace,
    resource_to_solution, greedy_success_profile,
)

DEPTH = 8
TRAIN_SEED = 99
TEST_SEEDS = [11, 23, 42, 7, 55, 68]
TILTS = [0.0, 3.0, 6.0, 10.0]


def prob_of_optimum(qubo, params, p):
    """P(sampling the optimum) for a GIVEN parameter schedule -- no training."""
    S, T = qubo.n_sectors, qubo.n_tilts
    cost = subspace_cost(qubo)
    psi = subspace_statevector(params[:p], params[p:], cost, S, T)
    probs = np.abs(psi.ravel()) ** 2
    probs /= probs.sum()
    flat = cost.ravel()
    ground = float(flat.min())
    return float(probs[np.isclose(flat, ground, atol=1e-9)].sum()), ground


def main():
    print("=" * 80)
    print("QAOA PARAMETER TRANSFER ACROSS UNSEEN NETWORK INSTANCES")
    print("=" * 80)

    # ---- 1. train once on a single instance --------------------------------
    train_net = make_network(n_towers=3, seed=TRAIN_SEED, tilt_levels_deg=TILTS)
    train_qubo = build_qubo(train_net, penalty_scale=0.0)
    print(f"\nTraining a p={DEPTH} schedule on instance seed={TRAIN_SEED} "
          f"({train_qubo.n_vars} qubits) ...")
    trained = solve_qaoa_subspace(train_qubo, p=DEPTH, seed=1, n_restarts=4, interp=True)
    params = trained.optimal_params
    print(f"  trained P(optimum) on its own instance : {trained.prob_of_optimum:.3%}")
    print(f"  training cost                          : {trained.evaluations:,} "
          f"circuit evaluations")
    print(f"  gammas: {np.round(params[:DEPTH], 3).tolist()}")
    print(f"  betas : {np.round(params[DEPTH:], 3).tolist()}")

    # ---- 2. apply verbatim to unseen instances -----------------------------
    n_feasible = train_qubo.n_tilts ** train_qubo.n_sectors
    p_random = 1.0 / n_feasible

    print(f"\nApplying that schedule VERBATIM to unseen instances "
          f"(no re-optimization):\n")
    print(f"  {'instance':>10} {'transferred':>13} {'retrained':>11} "
          f"{'random':>10} {'transfer/random':>16} {'of ceiling':>12}")
    print("  " + "-" * 76)

    transferred, retrained_all = [], []
    for seed in TEST_SEEDS:
        net = make_network(n_towers=3, seed=seed, tilt_levels_deg=TILTS)
        qubo = build_qubo(net, penalty_scale=0.0)

        p_trans, _ = prob_of_optimum(qubo, params, DEPTH)
        ceiling = solve_qaoa_subspace(qubo, p=DEPTH, seed=1, n_restarts=4, interp=True)
        p_ceil = ceiling.prob_of_optimum

        transferred.append(p_trans)
        retrained_all.append(p_ceil)
        frac = (p_trans / p_ceil * 100) if p_ceil > 0 else float("nan")
        print(f"  {seed:>10} {p_trans:>12.3%} {p_ceil:>10.3%} {p_random:>9.4%} "
              f"{p_trans / p_random:>15.0f}x {frac:>11.0f}%")

    mean_trans = float(np.mean(transferred))
    mean_ceil = float(np.mean(retrained_all))
    print("  " + "-" * 76)
    print(f"  {'mean':>10} {mean_trans:>12.3%} {mean_ceil:>10.3%} {p_random:>9.4%} "
          f"{mean_trans / p_random:>15.0f}x "
          f"{mean_trans / mean_ceil * 100:>11.0f}%")

    # ---- 3. amortized resource accounting ----------------------------------
    print("\n" + "=" * 80)
    print("AMORTIZED RESOURCE-TO-SOLUTION")
    print("=" * 80)

    # classical reference measured on the same test instances
    g_probs, g_evals_list = [], []
    for seed in TEST_SEEDS:
        net = make_network(n_towers=3, seed=seed, tilt_levels_deg=TILTS)
        q = build_qubo(net, penalty_scale=0.0)
        gp, ge = greedy_success_profile(q, brute_force(q).objective, n_trials=120)
        g_probs.append(gp)
        g_evals_list.append(ge)
    g_prob, g_evals = float(np.mean(g_probs)), float(np.mean(g_evals_list))

    shots = resource_to_solution(mean_trans)
    g_reps = resource_to_solution(g_prob)
    g_total = g_reps * g_evals
    train_cost = trained.evaluations

    print(f"\n  greedy local search : P(hit)/restart {g_prob:.1%}, "
          f"{g_evals:.0f} evaluations/restart")
    print(f"                        -> {g_total:,.0f} objective evaluations per instance")
    print(f"  QAOA (transferred)  : P(hit)/shot {mean_trans:.3%} "
          f"-> {shots:,.0f} shots per instance")
    print(f"  one-off training    : {train_cost:,} circuit evaluations")

    if shots < g_total:
        breakeven = train_cost / (g_total - shots)
        print(f"\n  Per instance after training, QAOA needs {g_total/shots:.1f}x fewer")
        print(f"  repetitions. Training pays for itself after "
              f"{breakeven:.0f} instances.")
    else:
        print(f"\n  Even after training, QAOA needs {shots/g_total:.1f}x MORE "
              f"repetitions than greedy.")

    print("\n  Caveats, stated plainly:")
    print("   - a circuit shot is not equal in wall-clock cost to a classical")
    print("     objective evaluation; on today's hardware it is far more expensive")
    print("   - these are noiseless simulations")
    print("   - all test instances share the same generator, 9 sectors and 4 tilt")
    print("     levels; transfer across genuinely different network topologies")
    print("     is not established here")


if __name__ == "__main__":
    main()
