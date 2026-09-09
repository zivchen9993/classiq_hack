"""Canonical benchmark for the direct spatiotemporal antenna formulation.

The quick tier is deliberately small enough for exact enumeration and the
local feasible-subspace quantum references.  Larger instances can be run with
``--classical-only``; they retain the MILP bound and heuristic comparisons but
are never mislabeled as measured quantum results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from classical_baseline import (brute_force, exact_milp, greedy_local_search,
                                simulated_annealing)
from dcqo import solve_bf_dcqo, solve_dcqo
from direct_temporal import (best_static_trajectory,
                             build_direct_temporal_problem,
                             build_direct_temporal_qubo,
                             snapshot_chasing_trajectory)
from mobility import evolve_network, make_commuter_mobility
from qaoa_subspace import solve_qaoa_subspace
from result_schema import method_record, new_run_record, save_run_record
from rf_model import make_network


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y_%m_%d__%H_%M_%S_utc")


def _method_from_solver(name, family, result, problem, best, *, shots=0,
                        quantum_executions=0, train_circuits=0, algorithm=None,
                        circuit=None, synthesis_seconds=0.0, resynthesis_count=0):
    trajectory = problem.reshape(result.config)
    return method_record(
        name, family, result.objective, best_known_objective=best,
        probability_of_optimum=getattr(result, "prob_of_optimum", None),
        trajectory=trajectory, kpis=problem.evaluate_trajectory(trajectory),
        classical_objective_evaluations=(result.evaluations if family == "classical" else 0),
        parameter_search_circuit_executions=train_circuits,
        final_sampling_shots=shots, quantum_executions=quantum_executions,
        wall_time_seconds=result.seconds, circuit=circuit,
        synthesis_time_seconds=synthesis_seconds,
        resynthesis_count=resynthesis_count,
        algorithm={**(algorithm or {}), "raw_extra": getattr(result, "extra", {})},
    )


def run(*, n_towers=1, horizon=3, n_tilts=3, seed=0, mobility_seed=0,
        speed_frac=0.8, lam=0.07, cost_mode="switches", shots=600, qaoa_depth=1,
        qaoa_maxiter=40, dcqo_steps=8, bf_iters=3, sa_steps=5000,
        milp_time_limit=60.0, run_quantum=True, execute_classiq=False,
        classiq_backend="simulator", out_root="results"):
    levels = np.linspace(0.0, 10.0, n_tilts).tolist()
    net = make_network(n_towers, tilt_levels_deg=levels, seed=seed)
    mobility = make_commuter_mobility(net, n_steps=horizon, seed=mobility_seed,
                                      speed_frac=speed_frac)
    snapshots = evolve_network(net, mobility, n_steps=horizon)
    problem = build_direct_temporal_problem(snapshots, lam=lam, cost_mode=cost_mode)
    qubo = build_direct_temporal_qubo(problem)

    instance = {
        "formulation": "direct-spatiotemporal-one-hot",
        "n_towers": n_towers, "S": problem.n_sectors, "T": problem.n_tilts,
        "H": horizon, "qubits": qubo.n_vars,
        "feasible_trajectories": int(problem.n_tilts ** (problem.n_sectors * horizon)),
        "spatial_interaction_pairs": len(qubo.int_norm) + len(qubo.ho_norm),
        "temporal_interaction_pairs": len(qubo.extra_pairwise),
        "tilt_levels_deg": levels, "switching_weight": lam,
        "switching_mode": cost_mode, "traffic_speed_fraction": speed_frac,
    }
    config = {"shots_per_quantum_method": shots, "qaoa_depth": qaoa_depth,
              "qaoa_maxiter": qaoa_maxiter, "dcqo_steps": dcqo_steps,
              "bf_iters": bf_iters, "sa_steps": sa_steps,
              "milp_time_limit_seconds": milp_time_limit,
              "local_quantum_simulation": bool(run_quantum),
              "classiq_execution": bool(execute_classiq),
              "classiq_backend": classiq_backend}
    record = new_run_record(
        "tiny-correctness" if instance["feasible_trajectories"] <= 2_000_000
        else "intermediate-comparison",
        instance, config,
        backend=(f"local-numpy-feasible-subspace+classiq-{classiq_backend}"
                 if execute_classiq else "local-numpy-feasible-subspace"),
        seeds={"network": seed, "mobility": mobility_seed, "qaoa": 101,
               "dcqo": 102, "bf_dcqo": 103, "greedy": 104,
               "simulated_annealing": 105})
    out_dir = Path(out_root) / _stamp()
    result_path = out_dir / "direct_temporal_results.json"
    record["artifacts"].append({"kind": "canonical-results", "path": str(result_path)})

    def checkpoint(method=None):
        if method is not None:
            record["methods"].append(method)
            print(f"  {method['name']:<28} J={method['solution']['objective']:+.8f} "
                  f"gap={method['solution']['gap_to_best_known']:+.3e}")
        save_run_record(record, result_path)

    print("DIRECT SPATIOTEMPORAL ANTENNA BENCHMARK")
    print(f"  {problem.n_sectors} sectors x {problem.n_tilts} tilts x H={horizon}")
    print(f"  {qubo.n_vars} qubits; {instance['feasible_trajectories']:,} feasible trajectories")
    print(f"  output: {result_path}")

    # Establish the common best-known reference before calculating gaps.
    print("\nStrong classical reference")
    milp = exact_milp(qubo, time_limit=milp_time_limit)
    best = milp.objective
    checkpoint(_method_from_solver("MILP exact/best-known", "classical", milp,
                                   problem, best,
                                   algorithm={"solver": "scipy.optimize.milp"}))

    if instance["feasible_trajectories"] <= 2_000_000:
        exact = brute_force(qubo)
        if abs(exact.objective - best) > 1e-8:
            raise RuntimeError("MILP and brute force disagree on the common objective")
        checkpoint(_method_from_solver("brute force (exact)", "classical", exact,
                                       problem, best))

    static_traj, static_obj = best_static_trajectory(problem)
    static = type("Result", (), {"config": problem.flatten(static_traj),
                                  "objective": static_obj, "seconds": 0.0,
                                  "evaluations": problem.n_tilts ** problem.n_sectors,
                                  "extra": {"policy": "best constant with hindsight"}})()
    checkpoint(_method_from_solver("best static trajectory", "classical", static,
                                   problem, best))

    chase_traj = snapshot_chasing_trajectory(problem)
    chase = type("Result", (), {"config": problem.flatten(chase_traj),
                                 "objective": problem.objective(chase_traj),
                                 "seconds": 0.0,
                                 "evaluations": horizon * problem.n_tilts ** problem.n_sectors,
                                 "extra": {"policy": "independent snapshot optima"}})()
    checkpoint(_method_from_solver("snapshot chasing", "classical", chase,
                                   problem, best))

    for label, result in (
            ("greedy (single start)", greedy_local_search(qubo, seed=104, n_restarts=1)),
            ("greedy (32 starts)", greedy_local_search(qubo, seed=104, n_restarts=32)),
            ("simulated annealing", simulated_annealing(qubo, seed=105,
                                                        n_steps=sa_steps))):
        checkpoint(_method_from_solver(label, "classical", result, problem, best))

    if run_quantum:
        print("\nLocal quantum-reference methods (same QUBO and total shot budget)")
        qaoa = solve_qaoa_subspace(
            qubo, p=qaoa_depth, seed=101, maxiter=qaoa_maxiter,
            n_restarts=3, n_shots=shots)
        checkpoint(_method_from_solver(
            "QAOA-XY", "quantum-reference", qaoa, problem, best, shots=shots,
            quantum_executions=qaoa.evaluations + 1, train_circuits=qaoa.evaluations,
            circuit={"width": qubo.n_vars, "depth": qaoa_depth,
                     "gate_count": None, "two_qubit_gates": None,
                     "three_qubit_gates": None, "pauli_terms": None},
            algorithm={"optimizer": "COBYLA", "quantile": 0.15,
                       "simulation_note": "reduced T^(S*H) NumPy statevector; not scalable"}))

        for label, mode, run_seed in (("annealing (no CD)", "no_cd", 102),
                                      ("DCQO", "full", 102)):
            result = solve_dcqo(qubo, n_steps=dcqo_steps, n_shots=shots, mode=mode,
                                seed=run_seed, cd_probes=4, time_scale=40.0)
            checkpoint(_method_from_solver(
                label, "quantum-reference", result, problem, best, shots=shots,
                quantum_executions=1,
                synthesis_seconds=result.extra["synthesis_seconds"],
                resynthesis_count=1,
                circuit={"width": qubo.n_vars, "depth": dcqo_steps,
                         "gate_count": None, "two_qubit_gates": None,
                         "three_qubit_gates": None, "pauli_terms": None},
                algorithm={"mode": mode, "steps": dcqo_steps,
                           "time_scale": 40.0, "trotter_order": 2,
                           "cd_probes": 4,
                           "simulation_note": "reduced T^(S*H) NumPy statevector; not scalable"}))

        # BF feedback executions share the same TOTAL budget as every other
        # quantum method; integer division is recorded rather than concealed.
        shots_per_bf_iteration = max(shots // bf_iters, 1)
        bf = solve_bf_dcqo(
            qubo, n_iters=bf_iters, n_steps=dcqo_steps,
            n_shots=shots_per_bf_iteration, seed=103, cd_probes=4,
            time_scale=40.0, bias_source="marginals", verbose=False)
        bf_total_shots = shots_per_bf_iteration * bf_iters
        checkpoint(_method_from_solver(
            "BF-DCQO", "quantum-reference", bf, problem, best,
            shots=bf_total_shots, quantum_executions=bf_iters,
            circuit={"width": qubo.n_vars, "depth": dcqo_steps,
                     "gate_count": None, "two_qubit_gates": None,
                     "three_qubit_gates": None, "pauli_terms": None},
            algorithm={"steps": dcqo_steps, "bf_iterations": bf_iters,
                       "shots_per_iteration": shots_per_bf_iteration,
                       "bias_source": "marginals", "time_scale": 40.0,
                       "trotter_order": 2, "cd_probes": 4,
                       "simulation_note": "reduced T^(S*H) NumPy statevector; not scalable"}))

    if execute_classiq:
        print("\nGate-level Classiq execution")
        from classiq_dcqo import run_classiq_bf_dcqo, run_classiq_dcqo
        from quantum_solve import run_classiq_qaoa

        # The constrained circuit does not need a one-hot penalty. Removing it
        # reduces Pauli terms without changing a single feasible energy.
        circuit_qubo = build_direct_temporal_qubo(problem, penalty_scale=0.0)
        classiq_root = out_dir / "classiq"
        classiq_qaoa = run_classiq_qaoa(
            circuit_qubo, p=qaoa_depth, mixer="xy", n_shots=shots,
            maxiter=qaoa_maxiter, seed=101, backend=classiq_backend,
            exact_objective=best)
        checkpoint(_method_from_solver(
            "Classiq QAOA-XY", "quantum-classiq", classiq_qaoa, problem, best,
            shots=shots, quantum_executions=classiq_qaoa.evaluations + 1,
            train_circuits=classiq_qaoa.evaluations,
            circuit={"width": classiq_qaoa.n_qubits,
                     "depth": classiq_qaoa.circuit_depth,
                     "gate_count": classiq_qaoa.gate_count,
                     "two_qubit_gates": None, "three_qubit_gates": None,
                     "pauli_terms": None},
            algorithm={"optimizer": "COBYLA", "backend": classiq_backend}))

        for label, mode, artifact_name in (
                ("Classiq annealing (no CD)", "no_cd", "annealing_no_cd"),
                ("Classiq DCQO", "full", "dcqo")):
            result = run_classiq_dcqo(
                circuit_qubo, n_steps=dcqo_steps, n_shots=shots,
                time_scale=40.0, mode=mode, trotter_order=1,
                trotter_repetitions=1, cd_probes=4, seed=102,
                backend=classiq_backend, exact_objective=best,
                artifact_dir=classiq_root / artifact_name)
            checkpoint(_method_from_solver(
                label, "quantum-classiq", result, problem, best, shots=shots,
                quantum_executions=1,
                circuit={"width": result.n_qubits, "depth": result.circuit_depth,
                         "gate_count": result.gate_count,
                         "two_qubit_gates": result.extra.get("gate_counts", {}).get("cx"),
                         "three_qubit_gates": None,
                         "pauli_terms": result.extra["pauli_counts"]},
                algorithm={"mode": mode, "backend": classiq_backend,
                           "steps": dcqo_steps, "time_scale": 40.0,
                           "trotter_order": 1, "trotter_repetitions": 1,
                           "synthesis_seconds": result.extra["synthesis_seconds"]}))

        classiq_bf = run_classiq_bf_dcqo(
            circuit_qubo, n_iters=bf_iters, n_steps=dcqo_steps,
            total_shots=shots, time_scale=40.0, trotter_order=1,
            trotter_repetitions=1, cd_probes=4, seed=103,
            backend=classiq_backend, exact_objective=best,
            artifact_dir=classiq_root / "bf_dcqo")
        checkpoint(_method_from_solver(
            "Classiq BF-DCQO", "quantum-classiq", classiq_bf, problem, best,
            shots=classiq_bf.extra["total_shots"], quantum_executions=bf_iters,
            synthesis_seconds=classiq_bf.extra["total_synthesis_seconds"],
            resynthesis_count=classiq_bf.extra["resynthesis_count"],
            circuit={"width": classiq_bf.n_qubits,
                     "depth": classiq_bf.circuit_depth,
                     "gate_count": classiq_bf.gate_count,
                     "two_qubit_gates": classiq_bf.extra.get("gate_counts", {}).get("cx"),
                     "three_qubit_gates": None,
                     "pauli_terms": classiq_bf.extra["pauli_counts"]},
            algorithm={"backend": classiq_backend, "steps": dcqo_steps,
                       "bf_iterations": bf_iters, "bias_source": "marginals",
                       "resynthesis_count": classiq_bf.extra["resynthesis_count"]}))

    record["run"]["completed_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    record["run"]["status"] = "complete"
    checkpoint()
    print(f"\nSaved canonical record: {result_path}")
    return record, result_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--classiq", action="store_true",
                        help="also synthesize and execute gate-level Classiq methods")
    parser.add_argument("--classiq-backend", default="simulator")
    parser.add_argument("--n-towers", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--n-tilts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mobility-seed", type=int, default=0)
    parser.add_argument("--speed-frac", type=float, default=0.8)
    parser.add_argument("--shots", type=int, default=600)
    parser.add_argument("--qaoa-depth", type=int, default=1)
    parser.add_argument("--qaoa-maxiter", type=int, default=40)
    parser.add_argument("--dcqo-steps", type=int, default=8)
    parser.add_argument("--bf-iters", type=int, default=3)
    parser.add_argument("--sa-steps", type=int, default=5000)
    parser.add_argument("--milp-time-limit", type=float, default=60.0)
    parser.add_argument("--out-root", default="results")
    args = parser.parse_args()
    run(n_towers=args.n_towers, horizon=args.horizon, n_tilts=args.n_tilts,
        seed=args.seed, mobility_seed=args.mobility_seed,
        speed_frac=args.speed_frac,
        shots=args.shots, qaoa_depth=args.qaoa_depth,
        qaoa_maxiter=args.qaoa_maxiter, dcqo_steps=args.dcqo_steps,
        bf_iters=args.bf_iters, sa_steps=args.sa_steps,
        milp_time_limit=args.milp_time_limit,
        run_quantum=not args.classical_only, execute_classiq=args.classiq,
        classiq_backend=args.classiq_backend, out_root=args.out_root)


if __name__ == "__main__":
    main()
