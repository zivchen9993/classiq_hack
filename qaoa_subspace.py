"""
Exact FEASIBLE-SUBSPACE simulation of the constrained (XY) QAOA.

The idea
--------
The XY ansatz provably never leaves the one-hot subspace: `prepare_dicke_state`
puts exactly one excitation in each sector's block, and RXX*RYY conserves
Hamming weight. So the state -- for ANY parameter values -- is supported
entirely on the T^S feasible configurations, never on the other
2^(S*T) - T^S basis states.

That means we can simulate it exactly in a space of dimension T^S instead of
2^(S*T):

    9 sectors x 4 tilts:   4^9  =    262,144 amplitudes
                    vs     2^36 = 68,719,476,736 amplitudes

a factor of 262,144 smaller. This lets us validate the algorithm on the 36-qubit
instance where classical greedy search genuinely fails -- which is out of reach
for any full statevector simulator.

How the mixer acts in the subspace
----------------------------------
Within one sector's block of m qubits, the single-excitation subspace is
m-dimensional and is spanned by the one-hot states, i.e. by the tilt levels.
The two-qubit gate exp(-i*b*(XX+YY)/2) on the adjacent pair (k, k+1) acts on
that subspace as a plain 2-level rotation between tilt k and tilt k+1:

    [[cos b, -i sin b],
     [-i sin b, cos b]]

So the whole XY chain on a block is an m x m unitary M(b) acting on that
sector's tilt index. The full mixer is M(b) applied independently along each
sector's axis of a rank-S tensor of shape (T, T, ..., T).

Note this is a statement about our VALIDATION being cheap, not a claim that the
quantum algorithm is unnecessary: the same reduction does not apply to the
unconstrained ansatz, and it does not make the underlying optimization easy
(the cost landscape over those T^S configurations is exactly the hard problem).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product as iproduct
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Cost landscape over the feasible subspace
# ---------------------------------------------------------------------------

def subspace_cost(qubo) -> np.ndarray:
    """Objective value of every feasible configuration, as a (T,)*S tensor.

    Uses the physics objective directly (no penalty term), which is exactly
    what the classical baselines minimize.
    """
    S, T = qubo.n_sectors, qubo.n_tilts
    cost = np.full((T,) * S, float(getattr(qubo, "objective_offset", 0.0)))

    # unary coverage term
    for i in range(S):
        shape = [1] * S
        shape[i] = T
        cost += (-qubo.w_cov * qubo.cov_norm[i]).reshape(shape)

    # pairwise terms. Pairs are always stored with i < j, so reshaping the
    # (T, T) table into a shape with T at axis i and T at axis j (and 1
    # everywhere else) places table[t_i, t_j] on exactly the right axes, and
    # broadcasting handles the rest.
    for weight, tables in ((qubo.w_int, qubo.int_norm), (qubo.w_ho, qubo.ho_norm)):
        for (i, j), table in tables.items():
            assert i < j, "pair keys are expected to be ordered"
            shape = [1] * S
            shape[i] = T
            shape[j] = T
            cost += weight * np.asarray(table).reshape(shape)

    # Formulation-specific terms are already weighted.  Direct temporal
    # QUBOs use this channel for cross-timestep switching edges, keeping those
    # coefficients separate from the RF handover weight.
    for (i, j), table in getattr(qubo, "extra_pairwise", {}).items():
        assert i < j, "pair keys are expected to be ordered"
        shape = [1] * S
        shape[i] = T
        shape[j] = T
        cost += np.asarray(table).reshape(shape)

    return cost


# ---------------------------------------------------------------------------
# The ansatz, restricted to the feasible subspace
# ---------------------------------------------------------------------------

def sector_mixer_matrix(beta: float, T: int, topology: str = "chain") -> np.ndarray:
    """The T x T unitary the XY chain induces on one sector's tilt index."""
    M = np.eye(T, dtype=np.complex128)
    c, s = np.cos(beta), -1j * np.sin(beta)

    pairs = [(k, k + 1) for k in range(T - 1)]
    if topology == "ring" and T > 2:
        pairs.append((T - 1, 0))

    for (a, b) in pairs:
        R = np.eye(T, dtype=np.complex128)
        R[a, a] = c
        R[b, b] = c
        R[a, b] = s
        R[b, a] = s
        M = R @ M            # applied in circuit order
    return M


def apply_subspace_mixer(psi: np.ndarray, M: np.ndarray, S: int) -> np.ndarray:
    """Apply the per-sector mixer M along every sector axis."""
    for i in range(S):
        psi = np.moveaxis(psi, i, 0)
        shape = psi.shape
        psi = (M @ psi.reshape(shape[0], -1)).reshape(shape)
        psi = np.moveaxis(psi, 0, i)
    return psi


def subspace_statevector(gammas, betas, cost: np.ndarray, S: int, T: int,
                         topology: str = "chain") -> np.ndarray:
    """Exact QAOA state, represented over the feasible subspace only."""
    psi = np.full(cost.shape, 1.0 / np.sqrt(cost.size), dtype=np.complex128)
    for gamma, beta in zip(gammas, betas):
        psi = np.exp(-1j * gamma * cost) * psi
        psi = apply_subspace_mixer(psi, sector_mixer_matrix(beta, T, topology), S)
    return psi


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class SubspaceResult:
    name: str
    config: np.ndarray
    objective: float
    seconds: float
    evaluations: int
    n_qubits: int
    depth: int
    prob_of_optimum: float
    mean_sampled_objective: float
    optimal_params: np.ndarray
    extra: dict = field(default_factory=dict)

    def __str__(self):
        return (f"{self.name:<24} obj={self.objective:+.5f}  {self.n_qubits}q "
                f"p={self.depth}  P(opt)={self.prob_of_optimum:7.3%}  {self.seconds:.1f}s")


def _cvar_from_probs(probs_flat, cost_flat, quantile):
    if quantile >= 1.0:
        return float(probs_flat @ cost_flat)
    order = np.argsort(cost_flat)
    sp, sc = probs_flat[order], cost_flat[order]
    cum = np.cumsum(sp)
    cut = min(int(np.searchsorted(cum, quantile)) + 1, len(sc))
    w = sp[:cut]
    tot = w.sum()
    return float((w @ sc[:cut]) / tot) if tot > 0 else float(probs_flat @ cost_flat)


def _interp_init(params_p, p_old: int) -> np.ndarray:
    """INTERP heuristic (Zhou et al., PRX 10, 021067): build a depth-(p+1)
    starting point by linearly interpolating an optimized depth-p schedule.

    QAOA's optimal (gamma, beta) schedules are smooth in the layer index, so a
    good depth-p solution is an excellent guide for depth p+1. This turns the
    outer loop from a blind non-convex search into a sequence of local
    refinements, and it is what makes deeper circuits actually pay off.
    """
    g_old, b_old = params_p[:p_old], params_p[p_old:]
    p_new = p_old + 1

    def interp(v):
        out = np.zeros(p_new)
        for i in range(1, p_new + 1):          # 1-indexed, as in the paper
            lo = v[i - 2] if 2 <= i else 0.0
            hi = v[i - 1] if i <= p_old else 0.0
            out[i - 1] = ((i - 1) / p_old) * lo + ((p_old - i + 1) / p_old) * hi
        return out

    return np.concatenate([interp(g_old), interp(b_old)])


def solve_qaoa_subspace(qubo, p: int = 3, seed: int = 0, maxiter: int = 250,
                        quantile: float = 0.15, n_restarts: int = 3,
                        n_shots: int = 2000, topology: str = "chain",
                        interp: bool = True, init_params=None) -> SubspaceResult:
    """Run the constrained QAOA exactly, in the feasible subspace.

    interp=True builds the depth-p schedule by optimizing depths 1..p in
    sequence, seeding each from the previous one (INTERP). This is markedly
    more reliable than optimizing 2p parameters from scratch.

    init_params gives QAOA the warm start the temporal problem makes available:
    pass the previous timestep's optimized schedule (length 2p) and the
    optimizer refines from there instead of searching from cold. It overrides
    `interp` and `n_restarts`, since both exist only to find a starting point.
    This is QAOA's counterpart to DCQO's bias field, and `dcqo_benchmark.py`
    compares the two warm-start mechanisms directly.
    """
    S, T = qubo.n_sectors, qubo.n_tilts
    t0 = time.perf_counter()
    cost = subspace_cost(qubo)
    cost_flat = cost.ravel()
    rng = np.random.default_rng(seed)

    evals = 0

    def make_objective(depth):
        def objective(params):
            nonlocal evals
            evals += 1
            psi = subspace_statevector(params[:depth], params[depth:], cost, S, T, topology)
            probs = np.abs(psi.ravel()) ** 2
            return _cvar_from_probs(probs, cost_flat, quantile)
        return objective

    if init_params is not None:
        x0 = np.asarray(init_params, dtype=float).ravel()
        if x0.size != 2 * p:
            raise ValueError(f"init_params must have length 2p={2 * p}, got {x0.size}")
        res = minimize(make_objective(p), x0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.05})
        params = res.x
    elif interp:
        # --- optimize depth 1 from random restarts, then climb with INTERP ---
        obj1 = make_objective(1)
        best_params, best_val = None, np.inf
        for _ in range(max(n_restarts, 4)):
            x0 = np.concatenate([rng.uniform(0.0, 0.6, size=1),
                                 rng.uniform(0.0, np.pi / 2, size=1)])
            res = minimize(obj1, x0, method="COBYLA",
                           options={"maxiter": maxiter, "rhobeg": 0.3})
            if res.fun < best_val:
                best_val, best_params = res.fun, res.x

        for depth in range(2, p + 1):
            x0 = _interp_init(best_params, depth - 1)
            res = minimize(make_objective(depth), x0, method="COBYLA",
                           options={"maxiter": maxiter, "rhobeg": 0.15})
            best_params, best_val = res.x, res.fun
        params = best_params
    else:
        best_res, best_val = None, np.inf
        objective = make_objective(p)
        for _ in range(n_restarts):
            x0 = np.concatenate([rng.uniform(0.0, 0.6, size=p),
                                 rng.uniform(0.0, np.pi / 2, size=p)])
            res = minimize(objective, x0, method="COBYLA",
                           options={"maxiter": maxiter, "rhobeg": 0.3})
            if res.fun < best_val:
                best_val, best_res = res.fun, res
        params = best_res.x
    psi = subspace_statevector(params[:p], params[p:], cost, S, T, topology)
    probs = np.abs(psi.ravel()) ** 2
    probs /= probs.sum()

    ground = float(cost_flat.min())
    prob_of_optimum = float(probs[np.isclose(cost_flat, ground, atol=1e-9)].sum())

    samples = rng.choice(len(probs), size=n_shots, p=probs)
    uniq, counts = np.unique(samples, return_counts=True)
    best_idx = uniq[np.argmin(cost_flat[uniq])]
    best_cfg = np.array(np.unravel_index(int(best_idx), cost.shape))
    best_obj = float(cost_flat[best_idx])
    mean_obj = float(np.sum(cost_flat[uniq] * counts) / n_shots)

    return SubspaceResult(
        name=f"QAOA-XY subspace (p={p})", config=best_cfg, objective=best_obj,
        seconds=time.perf_counter() - t0, evaluations=evals,
        n_qubits=qubo.n_vars, depth=p, prob_of_optimum=prob_of_optimum,
        mean_sampled_objective=mean_obj, optimal_params=params,
        extra={"ground_energy": ground, "n_feasible": int(cost.size),
               "quantile": quantile, "n_shots": n_shots},
    )


# ---------------------------------------------------------------------------
# Resource-to-solution: the only rigorous way to compare these solvers
# ---------------------------------------------------------------------------

def resource_to_solution(success_prob: float, confidence: float = 0.95) -> float:
    """Repetitions needed to hit the optimum with `confidence` probability.

    R = ln(1 - confidence) / ln(1 - p_success)

    This is the standard time-to-solution construction. It lets us compare a
    quantum sampler (repetitions = circuit shots) with a randomized classical
    heuristic (repetitions = restarts) on equal footing -- as long as we are
    explicit about what one repetition costs on each side.
    """
    if success_prob <= 0:
        return float("inf")
    if success_prob >= 1:
        return 1.0
    return float(np.log(1 - confidence) / np.log(1 - success_prob))


def greedy_success_profile(qubo, exact_objective, n_trials: int = 200, seed: int = 7):
    """Measure greedy's per-restart success probability AND its cost in
    objective evaluations, so the comparison is like-for-like."""
    from classical_baseline import greedy_local_search
    rng = np.random.default_rng(seed)
    hits, total_evals = 0, 0
    for _ in range(n_trials):
        r = greedy_local_search(qubo, seed=int(rng.integers(0, 1 << 30)), n_restarts=1)
        total_evals += r.evaluations
        if np.isclose(r.objective, exact_objective, atol=1e-9):
            hits += 1
    return hits / n_trials, total_evals / n_trials


if __name__ == "__main__":
    from rf_model import make_network
    from qubo_builder import build_qubo
    from classical_baseline import brute_force, greedy_local_search

    print("=" * 78)
    print("CONSTRAINED QAOA ON THE HARD INSTANCE (36 qubits)")
    print("=" * 78)

    net = make_network(n_towers=3, seed=99, tilt_levels_deg=[0.0, 3.0, 6.0, 10.0])
    qubo = build_qubo(net, penalty_scale=0.0)
    S, T = qubo.n_sectors, qubo.n_tilts

    print(f"\ninstance: {S} sectors x {T} tilts = {qubo.n_vars} qubits")
    print(f"  feasible configurations : {T ** S:,}")
    print(f"  full Hilbert space      : 2^{qubo.n_vars} = {1 << qubo.n_vars:,}")
    print(f"  subspace simulation is  : {(1 << qubo.n_vars) / T ** S:,.0f}x smaller")

    # --- correctness: the subspace cost tensor must match the QUBO objective
    rng = np.random.default_rng(0)
    cost = subspace_cost(qubo)
    err = 0.0
    for _ in range(300):
        cfg = rng.integers(0, T, size=S)
        err = max(err, abs(cost[tuple(cfg)] - qubo.objective_from_config(cfg)))
    print(f"\n  max |subspace cost - QUBO objective| over 300 configs: {err:.3e}")

    exact = brute_force(qubo)
    print(f"  brute-force optimum: {exact.objective:+.5f} tilts={exact.config.tolist()}")

    g_prob, g_evals = greedy_success_profile(qubo, exact.objective)
    p_random = 1.0 / T ** S
    print(f"  greedy local search finds it in {g_prob:.0%} of 200 random starts "
          f"({g_evals:.0f} objective evaluations per restart)")
    print(f"  uniform random guess          : {p_random:.6%}")

    print(f"\n{'ansatz':<26} {'gap':>10} {'P(optimum)':>12} {'vs random':>11} {'time':>9}")
    print("-" * 74)
    results = []
    for p in (1, 2, 3, 5, 8, 12):
        r = solve_qaoa_subspace(qubo, p=p, seed=1, n_restarts=4, interp=True)
        results.append(r)
        gap = r.objective - exact.objective
        print(f"{r.name:<26} {gap:>+10.5f} {r.prob_of_optimum:>11.3%} "
              f"{r.prob_of_optimum / p_random:>10.0f}x {r.seconds:>8.1f}s")

    # ---- resource-to-solution ------------------------------------------
    print("\n" + "=" * 78)
    print("RESOURCE-TO-SOLUTION (repetitions to find the optimum with 95% confidence)")
    print("=" * 78)
    best = max(results, key=lambda r: r.prob_of_optimum)

    r_qaoa = resource_to_solution(best.prob_of_optimum)
    r_greedy = resource_to_solution(g_prob)
    r_random = resource_to_solution(p_random)

    print(f"\n  {'method':<34} {'per-rep P(hit)':>15} {'repetitions':>13} {'unit':>22}")
    print("  " + "-" * 86)
    print(f"  {'random sampling':<34} {p_random:>14.6%} {r_random:>13,.0f} "
          f"{'objective evaluations':>22}")
    print(f"  {'greedy local search (1 restart)':<34} {g_prob:>14.2%} {r_greedy:>13,.1f} "
          f"{'restarts':>22}")
    print(f"  {'':<34} {'':>14} {r_greedy * g_evals:>13,.0f} "
          f"{'objective evaluations':>22}")
    print(f"  {best.name:<34} {best.prob_of_optimum:>14.3%} {r_qaoa:>13,.0f} "
          f"{'circuit shots':>22}")

    g_total = r_greedy * g_evals
    print(f"\n  Sampling phase only: QAOA {r_qaoa:,.0f} shots vs greedy "
          f"{g_total:,.0f} evaluations", end="")
    ratio = g_total / r_qaoa
    print(f"  ({ratio:.1f}x fewer)" if ratio > 1 else f"  ({1/ratio:.1f}x more)")

    # --- the honest accounting: training is not free ---------------------
    train = best.evaluations
    print(f"\n  But the {r_qaoa:,.0f} shots above count only the SAMPLING phase.")
    print(f"  Finding the parameters took {train:,} additional circuit evaluations.")
    print(f"  Total for QAOA, one instance from cold: "
          f"{train + r_qaoa:,.0f} circuit executions vs {g_total:,.0f} for greedy "
          f"-- greedy wins by {(train + r_qaoa) / g_total:.0f}x.")
    print("\n  The only way the QAOA column wins is if the trained parameters are")
    print("  AMORTIZED across many instances. That is a real effect for this use")
    print("  case -- a network is re-optimized continuously as traffic shifts, and")
    print("  QAOA parameters are known to transfer between similar instances -- but")
    print("  we did NOT measure parameter transfer here, so treat it as a hypothesis.")
    print("\n  IMPORTANT: on today's hardware a circuit shot also costs far more")
    print("  wall-clock time than one classical objective evaluation. This is a")
    print("  resource-COUNT comparison, NOT a demonstration of quantum advantage.")
