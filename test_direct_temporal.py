"""Regression tests for the direct spatiotemporal formulation and schema."""

from __future__ import annotations

from itertools import product
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical_baseline import brute_force, exact_milp  # noqa: E402
from direct_temporal import (build_direct_temporal_problem,
                             build_direct_temporal_qubo,
                             best_static_trajectory,
                             myopic_hysteresis_trajectory,
                             snapshot_chasing_trajectory)  # noqa: E402
from mobility import make_commuter_mobility, evolve_network  # noqa: E402
from rf_model import make_network  # noqa: E402
from result_schema import (method_record, new_run_record, save_run_record,
                           validate_run_record)  # noqa: E402
from qaoa_subspace import subspace_cost  # noqa: E402


NET = make_network(1, tilt_levels_deg=[0.0, 5.0, 10.0], seed=17,
                   grid_points_per_axis=17)
MOB = make_commuter_mobility(NET, n_steps=2, seed=4)
SNAPS = evolve_network(NET, MOB, n_steps=2)
PROBLEM = build_direct_temporal_problem(SNAPS, lam=0.07)
QUBO = build_direct_temporal_qubo(PROBLEM)


def test_direct_dimensions_and_interaction_graph():
    assert QUBO.n_sectors == PROBLEM.horizon * PROBLEM.n_sectors
    assert QUBO.n_tilts == PROBLEM.n_tilts
    assert QUBO.n_vars == PROBLEM.n_vars
    assert len(QUBO.extra_pairwise) == (PROBLEM.horizon - 1) * PROBLEM.n_sectors
    assert all(a // PROBLEM.n_sectors != b // PROBLEM.n_sectors
               for a, b in QUBO.extra_pairwise)


def test_direct_qubo_matches_problem_exhaustively():
    max_error = 0.0
    for flat in product(range(PROBLEM.n_tilts), repeat=QUBO.n_sectors):
        traj = PROBLEM.reshape(flat)
        x = QUBO.encode(flat)
        values = (PROBLEM.objective(traj), QUBO.objective_from_config(flat),
                  QUBO.energy(x), QUBO.ising_energy(x))
        max_error = max(max_error, np.ptp(values))
    assert max_error < 1e-11


def test_reduced_quantum_cost_includes_temporal_edges():
    reduced = subspace_cost(QUBO)
    for flat in product(range(PROBLEM.n_tilts), repeat=QUBO.n_sectors):
        assert abs(reduced[flat] - PROBLEM.objective(PROBLEM.reshape(flat))) < 1e-11


def test_switch_modes_have_expected_cost():
    traj = np.array([[0, 1, 2], [1, 1, 0]])
    assert PROBLEM.switch_cost(traj) == 2.0
    degree_problem = build_direct_temporal_problem(SNAPS, lam=0.07, cost_mode="degrees")
    assert degree_problem.switch_cost(traj) == 15.0


def test_zero_switch_weight_factorizes_into_snapshot_optima():
    problem = build_direct_temporal_problem(SNAPS, lam=0.0)
    qubo = build_direct_temporal_qubo(problem)
    chase = snapshot_chasing_trajectory(problem)
    exact = brute_force(qubo)
    assert abs(problem.objective(chase) - exact.objective) < 1e-11


def test_exact_milp_matches_brute_force():
    brute = brute_force(QUBO)
    milp = exact_milp(QUBO, time_limit=30.0)
    assert milp.extra["success"]
    assert abs(milp.objective - brute.objective) < 1e-9
    assert QUBO.is_valid(QUBO.encode(milp.config))


def test_static_baseline_is_constant_and_correctly_scored():
    trajectory, objective = best_static_trajectory(PROBLEM)
    assert np.all(trajectory == trajectory[0])
    assert abs(objective - PROBLEM.objective(trajectory)) < 1e-12


def test_kpi_report_keeps_surrogate_and_operational_metrics_together():
    trajectory, _ = best_static_trajectory(PROBLEM)
    report = PROBLEM.evaluate_trajectory(trajectory)
    for key in ("objective", "mean_sinr_db", "edge_sinr_db", "outage_pct",
                "handover_failure_pct", "n_switches", "switch_cost"):
        assert key in report


def test_result_schema_round_trip(tmp_path):
    record = new_run_record(
        "tiny-correctness", {"S": 3, "T": 3, "H": 2},
        {"shots": 100}, seeds={"instance": 17, "sampling": 2})
    record["methods"].append(method_record(
        "brute force", "classical", -1.25, best_known_objective=-1.25,
        classical_objective_evaluations=729, trajectory=np.zeros((2, 3), int)))
    validate_run_record(record)
    path = save_run_record(record, tmp_path / "result.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_run_record(loaded)
    assert loaded["methods"][0]["resources"]["final_sampling_shots"] == 0


def test_myopic_hysteresis_is_scored_on_direct_objective():
    trajectory = myopic_hysteresis_trajectory(PROBLEM)
    assert trajectory.shape == (PROBLEM.horizon, PROBLEM.n_sectors)
    # With a positive switch cost it cannot be better than the global optimum,
    # but it must be a valid trajectory and its score must be self-consistent.
    qubo = build_direct_temporal_qubo(PROBLEM)
    exact = brute_force(qubo)
    value = PROBLEM.objective(trajectory)
    assert value >= exact.objective - 1e-10
    assert abs(value - qubo.objective_from_config(PROBLEM.flatten(trajectory))) < 1e-11
