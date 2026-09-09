"""Measured encoding/Pauli scaling for the direct spatiotemporal formulation.

This script does not pretend to synthesize Classiq circuits when Classiq is not
available.  It measures instance size, exact Pauli term counts, MILP/heuristic
performance, and tiny local quantum-reference quality.  Any Classiq circuit
metrics are kept in a separate measured-artifact section.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from classical_baseline import exact_milp, greedy_local_search, simulated_annealing
from classiq_dcqo import dcqo_pauli_operators
from dcqo import solve_bf_dcqo, solve_dcqo
from direct_temporal import build_direct_temporal_problem, build_direct_temporal_qubo
from mobility import evolve_network, make_commuter_mobility
from qaoa_subspace import solve_qaoa_subspace
from rf_model import make_network


def stamp():
    return datetime.now(timezone.utc).strftime("%Y_%m_%d__%H_%M_%S_utc")


def load_classiq_artifact(root: Path):
    summary = root / "results/classiq_tiny_validation_2026_09_09/three_step/validation_summary.json"
    metadata = root / "results/classiq_tiny_validation_2026_09_09/three_step/classiq_metadata.json"
    program = root / "results/classiq_tiny_validation_2026_09_09/three_step/classiq_program.json"
    if not (summary.exists() and metadata.exists() and program.exists()):
        return None
    s, m, p = json.loads(summary.read_text()), json.loads(metadata.read_text()), json.loads(program.read_text())
    return {
        "status": "measured_saved_artifact",
        "instance": s["instance"],
        "n_steps": m["n_steps"],
        "depth": s["classiq"]["depth"],
        "gate_count": s["classiq"]["gate_count"],
        "gate_counts": m.get("gate_counts", {}),
        "pauli_counts": m.get("pauli_counts", {}),
        "program_id": p.get("program_id"),
        "p_opt": s["classiq"]["probability_of_optimum"],
        "local_p_opt": s["local_reference"]["probability_of_optimum"],
        "expected_objective": s["classiq"]["mean_sampled_objective"],
        "local_expected_objective": s["local_reference"]["expected_objective"],
        "shots": s["shots"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-horizon", type=int, default=6)
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()

    rows = []
    for H in range(1, args.max_horizon + 1):
        net = make_network(1, tilt_levels_deg=[0.0, 5.0, 10.0], seed=7)
        mob = make_commuter_mobility(net, n_steps=H, seed=77, speed_frac=0.8)
        snaps = evolve_network(net, mob, n_steps=H)
        problem = build_direct_temporal_problem(snaps, lam=0.07)
        qubo = build_direct_temporal_qubo(problem, penalty_scale=0.0)
        t0 = time.perf_counter()
        operators = dcqo_pauli_operators(qubo, topology="ring", cd_probes=2, seed=1)
        pauli_seconds = time.perf_counter() - t0
        milp = exact_milp(qubo, time_limit=10.0)
        greedy = greedy_local_search(qubo, seed=1, n_restarts=32)
        sa = simulated_annealing(qubo, seed=2, n_steps=3000)
        row = {
            "H": H, "S": problem.n_sectors, "T": problem.n_tilts,
            "qubits": qubo.n_vars,
            "feasible_trajectories": int(problem.n_tilts ** (problem.n_sectors * H)),
            "spatial_pairs": len(qubo.int_norm) + len(qubo.ho_norm),
            "temporal_pairs": len(qubo.extra_pairwise),
            "pauli_initial": operators["counts"]["initial"],
            "pauli_final": operators["counts"]["final"],
            "pauli_cd": operators["counts"]["cd"],
            "pauli_cd_by_weight": operators["counts"]["cd_by_weight"],
            "pauli_build_seconds": pauli_seconds,
            "milp_objective": milp.objective,
            "milp_seconds": milp.seconds,
            "milp_success": bool(milp.extra.get("success", False)),
            "greedy_gap": float(greedy.objective - milp.objective),
            "greedy_seconds": greedy.seconds,
            "sa_gap": float(sa.objective - milp.objective),
            "sa_seconds": sa.seconds,
        }
        # Local quantum-reference quality is measured only where the reduced
        # state is deliberately small.  Larger rows stay resource-only.
        if H <= 3:
            qaoa = solve_qaoa_subspace(qubo, p=1, seed=3, maxiter=20, n_restarts=2, n_shots=300)
            dcqo = solve_dcqo(qubo, n_steps=8, n_shots=300, seed=4, cd_probes=2, time_scale=40.0)
            bf = solve_bf_dcqo(qubo, n_iters=3, n_steps=8, n_shots=100, seed=5,
                               cd_probes=2, time_scale=40.0, bias_source="marginals", verbose=False)
            row["quantum_reference"] = {
                "QAOA-XY": {"gap": float(qaoa.objective - milp.objective), "p_opt": qaoa.prob_of_optimum,
                            "seconds": qaoa.seconds},
                "DCQO": {"gap": float(dcqo.objective - milp.objective), "p_opt": dcqo.prob_of_optimum,
                         "seconds": dcqo.seconds},
                "BF-DCQO": {"gap": float(bf.objective - milp.objective), "p_opt": bf.prob_of_optimum,
                            "seconds": bf.seconds},
            }
        rows.append(row)
        print(f"H={H}: {qubo.n_vars}q, final={row['pauli_final']} Pauli, CD={row['pauli_cd']} Pauli, MILP={milp.seconds:.3f}s")

    root = Path(__file__).resolve().parent
    out = Path(args.out_root) / f"scaling_{stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "kind": "direct-temporal-scaling",
        "measured": rows,
        "classiq_measured_artifact": load_classiq_artifact(root),
        "notes": [
            "Qubit and Pauli counts are measured from the exact direct QUBO/Pauli construction.",
            "Local quantum-reference rows use the reduced feasible-subspace simulator and are not scalable circuit executions.",
            "Classiq synthesized depth is included only where a saved Classiq artifact exists; no missing points are extrapolated as measured data.",
        ],
    }
    path = out / "scaling_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
