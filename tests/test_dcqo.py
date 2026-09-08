"""
Tests for the DCQO / BF-DCQO solvers.

Runnable either way:

    pytest -q tests/test_dcqo.py
    python tests/test_dcqo.py          # no pytest needed, ~60 s

Everything here is either a numerical identity (operator algebra, schedule,
trace estimates, encoding) or a stated inequality that must hold on the fixed
12-qubit instance. Performance numbers that depend on the instance live in the
benchmark, not here.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rf_model import make_network                                   # noqa: E402
from qubo_builder import build_qubo                                  # noqa: E402
from qaoa_subspace import subspace_cost                              # noqa: E402
from classical_baseline import brute_force                           # noqa: E402
from dcqo import (AnnealingSchedule, SubspaceOperators, CdCoefficient,   # noqa: E402
                  BiasField, DcqoSolver, estimate_cd_coefficient,
                  ring_edges, hop_matrix, solve_dcqo, solve_bf_dcqo,
                  _dense_operators)
from dcqo_ising import (full_space_cost, feasible_mask, cd_pauli_terms,  # noqa: E402
                        solve_dcqo_ising, _verify_pauli_terms)


# ---------------------------------------------------------------------------
# Shared fixtures, built once (module-level: cheap and deterministic)
# ---------------------------------------------------------------------------

NET = make_network(1, seed=7)                    # 3 sectors x 4 tilts = 12 qubits
QUBO_FREE = build_qubo(NET, penalty_scale=0.0)   # subspace solvers need no penalty
QUBO_PEN = build_qubo(NET)                       # full-space solver needs the penalty
COST = subspace_cost(QUBO_FREE)
S, T = COST.ndim, COST.shape[0]
EXACT = brute_force(QUBO_FREE)
P_RANDOM = 1.0 / COST.size


# ---------------------------------------------------------------------------
# 1. Driver graph and the subspace reduction
# ---------------------------------------------------------------------------

def test_ring_edges_and_hop_matrix():
    assert ring_edges(4, "ring") == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert ring_edges(4, "chain") == [(0, 1), (1, 2), (2, 3)]
    assert ring_edges(2, "ring") == [(0, 1)], "a 2-node ring must not double-count"
    B = hop_matrix(4, "ring")
    assert np.allclose(B, B.T)
    assert np.allclose(np.diag(B), 0.0)
    # a 2-level 'axis' is exactly sigma_x -- this is what makes dcqo_ising work
    assert np.allclose(hop_matrix(2, "chain"), [[0, 1], [1, 0]])


def test_subspace_cost_matches_qubo_objective():
    rng = np.random.default_rng(0)
    for _ in range(200):
        cfg = rng.integers(0, T, size=S)
        assert abs(COST[tuple(cfg)] - QUBO_FREE.objective_from_config(cfg)) < 1e-12


def test_driver_autoscale_matches_cost_halfwidth():
    """'auto' must put the driver on the same energy scale as the cost, or one
    of the two terms silently dominates the whole schedule."""
    ops = SubspaceOperators(COST, None)
    b_max = float(np.abs(np.linalg.eigvalsh(ops.B)).max())
    driver_halfwidth = ops.g * S * b_max
    assert abs(driver_halfwidth - ops.cost_spread / 2.0) < 1e-9 * ops.cost_spread


# ---------------------------------------------------------------------------
# 2. Operator algebra, against dense matrices
# ---------------------------------------------------------------------------

def test_apply_cd_equals_commutator():
    rng = np.random.default_rng(1)
    ops = SubspaceOperators(COST, rng.normal(0, 0.05, size=(S, T)))
    _, _, X_dense, _ = _dense_operators(ops)
    assert np.abs(X_dense - X_dense.conj().T).max() < 1e-12, "X must be Hermitian"
    v = rng.normal(size=COST.shape) + 1j * rng.normal(size=COST.shape)
    got = ops.apply_cd(v).ravel()
    assert np.abs(got - X_dense @ v.ravel()).max() < 1e-10


def test_apply_h_init_and_h_cost_equal_dense():
    rng = np.random.default_rng(2)
    ops = SubspaceOperators(COST, rng.normal(0, 0.05, size=(S, T)))
    H_i, H_f, _, _ = _dense_operators(ops)
    v = rng.normal(size=COST.shape) + 1j * rng.normal(size=COST.shape)
    assert np.abs(ops.apply_h_init(v).ravel() - H_i @ v.ravel()).max() < 1e-10
    assert np.abs(ops.apply_h_cost(v).ravel() - H_f @ v.ravel()).max() < 1e-10
    assert np.abs(ops.apply_dlam(v).ravel() - (H_f - H_i) @ v.ravel()).max() < 1e-10


def test_exp_h_init_matches_dense_expm():
    ops = SubspaceOperators(COST, None)
    H_i, _, _, _ = _dense_operators(ops)
    rng = np.random.default_rng(3)
    v = rng.normal(size=COST.shape) + 1j * rng.normal(size=COST.shape)
    for theta in (0.11, 1.9, 7.3):
        got = ops.exp_h_init(v, theta).ravel()
        want = expm(-1j * theta * H_i) @ v.ravel()
        assert np.abs(got - want).max() < 1e-9, f"theta={theta}"


def test_all_step_factors_are_unitary():
    """Norm must be preserved exactly by every factor, including the
    Trotterized CD product -- each edge rotation is exact, so the only error
    a step may carry is ordering, never leakage of probability."""
    ops = SubspaceOperators(COST, None)
    rng = np.random.default_rng(4)
    v = rng.normal(size=COST.shape) + 1j * rng.normal(size=COST.shape)
    n0 = np.linalg.norm(v)
    for theta in (0.03, 0.7, 5.0):
        for out in (ops.exp_h_init(v, theta), ops.exp_h_cost(v, theta),
                    ops.exp_cd(v, theta), ops.exp_cd(v, theta, reverse=True)):
            assert abs(np.linalg.norm(out) - n0) < 1e-10


def test_exp_cd_first_order_expansion():
    """exp(-i theta X) v = v - i theta X v + O(theta^2)."""
    ops = SubspaceOperators(COST, None)
    rng = np.random.default_rng(5)
    v = rng.normal(size=COST.shape) + 1j * rng.normal(size=COST.shape)
    Xv = ops.apply_cd(v)
    errs = []
    for theta in (1e-2, 5e-3):
        u = ops.exp_cd(v, theta)
        errs.append(np.abs(u - v + 1j * theta * Xv).max() / theta ** 2)
    # residual/theta^2 must stay bounded as theta shrinks -> the error is
    # genuinely second order, not first
    assert errs[1] < 2.0 * errs[0]


# ---------------------------------------------------------------------------
# 3. Initial states
# ---------------------------------------------------------------------------

def test_cold_ground_state_is_uniform():
    ops = SubspaceOperators(COST, None)
    psi = ops.ground_state()
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12
    assert np.abs(np.abs(psi) - 1 / np.sqrt(COST.size)).max() < 1e-12


def test_biased_ground_state_is_the_ground_state_of_h_init():
    rng = np.random.default_rng(6)
    ops = SubspaceOperators(COST, rng.normal(0, 0.1, size=(S, T)))
    H_i, _, _, _ = _dense_operators(ops)
    psi = ops.ground_state().ravel()
    got = float(np.real(np.vdot(psi, H_i @ psi)))
    assert abs(got - float(np.linalg.eigvalsh(H_i)[0])) < 1e-10


def test_marginals_are_normalized_probabilities():
    ops = SubspaceOperators(COST, None)
    rng = np.random.default_rng(7)
    psi = rng.normal(size=COST.shape) + 1j * rng.normal(size=COST.shape)
    m = ops.marginals(psi)
    assert m.shape == (S, T)
    assert np.all(m >= 0.0)
    assert np.abs(m.sum(axis=1) - 1.0).max() < 1e-12


# ---------------------------------------------------------------------------
# 4. Schedule
# ---------------------------------------------------------------------------

def test_schedule_endpoints_and_derivative():
    sched = AnnealingSchedule(n_steps=32, total_time=4.0)
    assert abs(float(sched.lam(0.0))) < 1e-15
    assert abs(float(sched.lam(1.0)) - 1.0) < 1e-15
    assert abs(float(sched.dlam_ds(0.0))) < 1e-12
    assert abs(float(sched.dlam_ds(1.0))) < 1e-12
    for s in (0.2, 0.5, 0.8):
        num = (sched.lam(s + 1e-6) - sched.lam(s - 1e-6)) / 2e-6
        assert abs(float(sched.dlam_ds(s)) - float(num)) < 1e-6


def test_schedule_steps_are_consistent():
    sched = AnnealingSchedule(n_steps=7, total_time=2.5)
    steps = list(sched.steps())
    assert len(steps) == 7
    for (s, lam, lam_dot, dt) in steps:
        assert abs(dt - 2.5 / 7) < 1e-12
        assert abs(lam_dot - float(sched.dlam_ds(s)) / 2.5) < 1e-12
        assert 0.0 <= lam <= 1.0


def test_diagonal_commuting_trap_is_rejected():
    """A schedule with no driver amplitude anywhere makes H(s) diagonal for
    every s; the evolution could then never move amplitude between
    configurations. It must raise, not silently return the initial state."""
    class Flat(AnnealingSchedule):
        def lam(self, s):
            return np.ones_like(np.asarray(s, dtype=float))

    raised = False
    try:
        Flat(10, 1.0)
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# 5. The counterdiabatic coefficient
# ---------------------------------------------------------------------------

def test_alpha_sign_matches_exact_two_level_gauge_potential():
    """The one thing that cannot be checked by inspection.

    For H = h_x sx + lam sz the exact adiabatic gauge potential is
    A = -h_x sy / (2 (h_x^2 + lam^2)). With the ansatz X = sy the variational
    formula must reproduce it, sign included -- with the opposite sign the CD
    term drives transitions instead of suppressing them, and DCQO performs
    worse than uniform random sampling.
    """
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    hx = 0.37
    for lam in (0.0, 0.3, 1.0, 1.4):
        H = hx * sx + lam * sz
        Y = 1j * (sy @ H - H @ sy)
        alpha = -np.real(np.trace(sz @ Y)) / np.real(np.trace(Y @ Y))
        assert abs(alpha - (-hx / (2 * (hx ** 2 + lam ** 2)))) < 1e-12


def test_hutchinson_traces_converge_to_exact():
    rng = np.random.default_rng(8)
    ops = SubspaceOperators(COST, rng.normal(0, 0.05, size=(S, T)))
    H_i, H_f, X, _ = _dense_operators(ops)
    P = 1j * (X @ H_i - H_i @ X)
    Q = 1j * (X @ H_f - H_f @ X)
    D = H_f - H_i
    exact = CdCoefficient(a=float(np.real(np.trace(D @ P))),
                          b=float(np.real(np.trace(D @ Q))),
                          c=float(np.real(np.trace(P @ P))),
                          d=float(np.real(np.trace(P @ Q))),
                          e=float(np.real(np.trace(Q @ Q))), n_probes=0)
    coarse = estimate_cd_coefficient(ops, n_probes=4, seed=1)
    fine = estimate_cd_coefficient(ops, n_probes=256, seed=1)

    def rel(est):
        return abs(est.alpha(0.5) - exact.alpha(0.5)) / abs(exact.alpha(0.5))

    assert rel(fine) < 0.05, f"256 probes should be within 5%, got {rel(fine):.1%}"
    assert rel(fine) <= rel(coarse) + 0.02, "more probes must not be worse"
    assert exact.alpha(0.5) < 0.0, "alpha must be negative for this ansatz"


def test_cd_scale_zero_reproduces_no_cd():
    a = solve_dcqo(QUBO_FREE, n_steps=16, mode="full", cd_scale=0.0, seed=3)
    b = solve_dcqo(QUBO_FREE, n_steps=16, mode="no_cd", seed=3)
    assert abs(a.prob_of_optimum - b.prob_of_optimum) < 1e-12


# ---------------------------------------------------------------------------
# 6. The solver
# ---------------------------------------------------------------------------

def test_dcqo_finds_the_optimum_and_beats_random():
    r = solve_dcqo(QUBO_FREE, n_steps=40, mode="full", time_scale=320.0,
                   seed=3, n_shots=4000)
    assert np.isclose(r.objective, EXACT.objective, atol=1e-9)
    assert r.prob_of_optimum > 10.0 * P_RANDOM
    assert r.evaluations == 0, "DCQO is non-variational: no parameters are fitted"
    assert r.depth == 40


def test_counterdiabatic_term_helps_at_short_annealing_time():
    """The claim the CD term has to earn: at short annealing time (i.e.
    shallow circuits) it must beat plain digitized annealing."""
    with_cd = solve_dcqo(QUBO_FREE, n_steps=10, mode="full", time_scale=20.0, seed=3)
    without = solve_dcqo(QUBO_FREE, n_steps=10, mode="no_cd", time_scale=20.0, seed=3)
    assert with_cd.prob_of_optimum > 1.5 * without.prob_of_optimum


def test_bias_toward_the_optimum_raises_success_probability():
    ops = SubspaceOperators(COST, None)
    bf = BiasField(S, T, ops.cost_spread, strength=1.0)
    cold = solve_dcqo(QUBO_FREE, n_steps=20, mode="full", time_scale=80.0, seed=3)
    warm = solve_dcqo(QUBO_FREE, n_steps=20, mode="full", time_scale=80.0, seed=3,
                      bias=bf.from_config(EXACT.config, confidence=0.7))
    assert warm.prob_of_optimum > cold.prob_of_optimum


def test_biased_runs_are_still_scored_on_the_true_objective():
    """The bias changes the Hamiltonian. If it also changed the scoring, a
    biased run could report an objective that does not exist on the real
    problem -- which would invalidate every comparison in the benchmark."""
    ops = SubspaceOperators(COST, None)
    bf = BiasField(S, T, ops.cost_spread, strength=4.0)     # deliberately huge
    bad = np.array(np.unravel_index(int(np.argmax(COST)), COST.shape))
    r = solve_dcqo(QUBO_FREE, n_steps=20, seed=3, bias=bf.from_config(bad, 0.95))
    assert np.any(np.isclose(COST.ravel(), r.objective, atol=1e-9)), \
        "reported objective must be a value of the UNBIASED cost"
    assert np.isclose(r.objective, QUBO_FREE.objective_from_config(r.config), atol=1e-9)
    assert r.objective >= EXACT.objective - 1e-12


def test_bias_from_config_and_marginals_agree():
    bf = BiasField(S, T, 2.0, strength=1.0)
    cfg = np.array([2, 0, 1])
    p = np.zeros((S, T))
    p[np.arange(S), cfg] = 1.0
    assert np.allclose(bf.from_config(cfg, confidence=1.0), bf.from_marginals(p))
    assert np.allclose(bf.zero(), 0.0)
    # a bias that reinforces a level must LOWER that level's energy
    u = bf.from_config(cfg, confidence=1.0)
    for i in range(S):
        assert u[i, cfg[i]] == u[i].min()


def test_bf_dcqo_never_reports_worse_than_its_first_iteration():
    for source in ("marginals", "best_sample"):
        r = solve_bf_dcqo(QUBO_FREE, n_iters=3, n_steps=20, time_scale=80.0,
                          seed=3, bias_source=source, verbose=False)
        hist = r.extra["history"]
        assert len(hist) == 3
        assert r.objective <= hist[0]["objective"] + 1e-12
        assert r.prob_of_optimum >= hist[0]["prob_of_optimum"] - 1e-12
        assert r.evaluations == 0
        assert r.extra["total_shots"] == 3 * 2000


def test_solver_rejects_bad_arguments():
    for kwargs in ({"mode": "nonsense"}, {"trotter_order": 3}):
        raised = False
        try:
            DcqoSolver(qubo=QUBO_FREE, **kwargs)
        except ValueError:
            raised = True
        assert raised, kwargs
    raised = False
    try:
        DcqoSolver(qubo=QUBO_FREE, cost=COST)      # both given
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# 7. Full-space (unconstrained) DCQO
# ---------------------------------------------------------------------------

def test_full_space_cost_matches_qubo_energy():
    cost = full_space_cost(QUBO_PEN)
    assert cost.shape == (2,) * QUBO_PEN.n_vars
    rng = np.random.default_rng(9)
    for _ in range(300):
        x = rng.integers(0, 2, size=QUBO_PEN.n_vars)
        assert abs(cost[tuple(x)] - QUBO_PEN.energy(x)) < 1e-10


def test_full_space_cost_equals_physics_objective_on_feasible_bitstrings():
    cost = full_space_cost(QUBO_PEN)
    rng = np.random.default_rng(10)
    for _ in range(100):
        cfg = rng.integers(0, QUBO_PEN.n_tilts, size=QUBO_PEN.n_sectors)
        x = QUBO_PEN.encode(cfg)
        assert abs(cost[tuple(x)] - QUBO_PEN.objective_from_config(cfg)) < 1e-10


def test_feasible_mask_counts_the_one_hot_subspace():
    ok = feasible_mask(QUBO_PEN)
    assert int(ok.sum()) == QUBO_PEN.n_tilts ** QUBO_PEN.n_sectors


def test_cd_pauli_terms_reproduce_the_numeric_operator():
    """Independent check of the whole CD implementation: the symbolic
    first-order operator -2 sum h_k Y_k - 2 sum J_kj (Y_k Z_j + Z_k Y_j) must
    equal what `apply_cd` applies numerically."""
    assert _verify_pauli_terms(QUBO_PEN) < 1e-9
    _, counts = cd_pauli_terms(QUBO_PEN)
    h, J, _ = QUBO_PEN.to_ising()
    assert counts["one_body_Y"] == int(np.count_nonzero(h))
    assert counts["two_body_YZ"] == 2 * int(np.count_nonzero(np.triu(J, 1)))


def test_full_space_dcqo_reports_a_feasible_fraction_below_one():
    """The unconstrained solver can leave the feasible set; the subspace one
    cannot. That difference is the point of keeping both."""
    r = solve_dcqo_ising(QUBO_PEN, n_steps=20, seed=3, n_shots=4000)
    assert 0.0 < r.extra["feasible_fraction"] <= 1.0
    assert r.n_qubits == QUBO_PEN.n_vars
    assert r.extra["best_feasible_objective"] >= EXACT.objective - 1e-9


def test_full_space_refuses_oversized_instances():
    raised = False
    try:
        full_space_cost(QUBO_PEN, max_qubits=8)
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"running {len(tests)} tests on a {QUBO_FREE.n_vars}-qubit instance "
          f"({COST.size} feasible configurations)\n")
    failed = 0
    for i, (name, fn) in enumerate(tests, 1):
        try:
            fn()
            print(f"  [{i:>2}/{len(tests)}] PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  [{i:>2}/{len(tests)}] FAIL  {name}: {exc}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1
            print(f"  [{i:>2}/{len(tests)}] ERROR {name}: "
                  f"{type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
