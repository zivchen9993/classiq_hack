"""Reproducible Classiq-vs-local validation for temporal antenna DCQO.

Run this only in an authenticated Classiq environment.  The script deliberately
uses a 12-qubit direct temporal antenna instance so that an independent full
2**n local Trotter reference is still practical.  It writes one directory per
mode and a single comparison summary; no number in the report is hardcoded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from classiq_dcqo import (dcqo_pauli_operators, local_trotter_reference,
                          run_classiq_bf_dcqo, run_classiq_dcqo,
                          scheduled_step_terms)
from classical_baseline import brute_force
from direct_temporal import build_direct_temporal_problem, build_direct_temporal_qubo
from mobility import evolve_network, make_commuter_mobility
from rf_model import make_network


def build_instance(seed=0, mobility_seed=0, speed_frac=0.8, lam=0.07):
    net = make_network(1, tilt_levels_deg=[0.0, 10.0], seed=seed)
    mob = make_commuter_mobility(net, n_steps=2, seed=mobility_seed, speed_frac=speed_frac)
    snaps = evolve_network(net, mob, n_steps=2)
    problem = build_direct_temporal_problem(snaps, lam=lam)
    # Constraint-preserving Dicke/XY evolution never needs the one-hot penalty.
    qubo = build_direct_temporal_qubo(problem, penalty_scale=0.0)
    return problem, qubo


def local_reference(qubo, *, mode, n_steps, time_scale, trotter_order,
                    trotter_repetitions, cd_probes, seed, exact_objective):
    operators = dcqo_pauli_operators(qubo, cd_probes=cd_probes, seed=seed)
    steps = scheduled_step_terms(operators, n_steps=n_steps, time_scale=time_scale, mode=mode)
    ref = local_trotter_reference(qubo, steps, repetitions=trotter_repetitions,
                                  order=trotter_order, exact_objective=exact_objective)
    return ref, operators


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--time-scale", type=float, default=2.0)
    ap.add_argument("--trotter-order", type=int, default=1)
    ap.add_argument("--trotter-repetitions", type=int, default=1)
    ap.add_argument("--cd-probes", type=int, default=4)
    ap.add_argument("--bf-iters", type=int, default=3)
    ap.add_argument("--backend", default="simulator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mobility-seed", type=int, default=0)
    ap.add_argument("--out", default="results/classiq_validation_rerun")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problem, qubo = build_instance(args.seed, args.mobility_seed)
    exact = brute_force(qubo)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instance": {"S": problem.n_sectors, "T": problem.n_tilts,
                     "H": problem.horizon, "qubits": qubo.n_vars,
                     "network_seed": args.seed, "mobility_seed": args.mobility_seed},
        "config": vars(args),
        "exact_objective": float(exact.objective),
        "modes": {},
    }

    for mode in ("no_cd", "full"):
        local, operators = local_reference(
            qubo, mode=mode, n_steps=args.steps, time_scale=args.time_scale,
            trotter_order=args.trotter_order,
            trotter_repetitions=args.trotter_repetitions,
            cd_probes=args.cd_probes, seed=102, exact_objective=exact.objective)
        result = run_classiq_dcqo(
            qubo, n_steps=args.steps, n_shots=args.shots,
            time_scale=args.time_scale, mode=mode,
            trotter_order=args.trotter_order,
            trotter_repetitions=args.trotter_repetitions,
            cd_probes=args.cd_probes, seed=102, backend=args.backend,
            exact_objective=exact.objective, artifact_dir=out / mode)
        delta_p = float(result.prob_of_optimum - local["probability_of_optimum"])
        summary["modes"][mode] = {
            "local": local,
            "classiq": {
                "best_objective": float(result.objective),
                "mean_sampled_objective": float(result.mean_sampled_objective),
                "probability_of_optimum": float(result.prob_of_optimum),
                "feasible_probability": float(result.valid_prob),
                "circuit_depth": int(result.circuit_depth),
                "gate_count": int(result.gate_count),
                "seconds": float(result.seconds),
                "synthesis_seconds": float(result.extra.get("synthesis_seconds", 0.0)),
                "gate_counts": result.extra.get("gate_counts", {}),
            },
            "difference": {
                "p_opt_classiq_minus_local": delta_p,
                "mean_objective_classiq_minus_local": float(result.mean_sampled_objective - local["expected_objective"]),
                "feasible_probability_classiq_minus_local": float(result.valid_prob - local["feasible_probability"]),
            },
            "pauli_counts": operators["counts"],
        }

    bf = run_classiq_bf_dcqo(
        qubo, n_iters=args.bf_iters, n_steps=args.steps,
        total_shots=args.shots, time_scale=args.time_scale,
        trotter_order=args.trotter_order, trotter_repetitions=args.trotter_repetitions,
        cd_probes=args.cd_probes, seed=103, backend=args.backend,
        exact_objective=exact.objective, artifact_dir=out / "bf_dcqo")
    summary["bf_dcqo"] = {
        "best_objective": float(bf.objective),
        "probability_of_optimum_best_iteration": float(bf.prob_of_optimum),
        "feasible_probability": float(bf.valid_prob),
        "total_shots": int(bf.extra["total_shots"]),
        "resynthesis_count": int(bf.extra["resynthesis_count"]),
        "total_synthesis_seconds": float(bf.extra["total_synthesis_seconds"]),
        "history": bf.extra["history"],
    }

    # Sampling equivalence tolerance is statistical.  A 4-sigma binomial band
    # is recorded rather than inventing a fixed arbitrary percentage tolerance.
    p = summary["modes"]["full"]["local"]["probability_of_optimum"]
    sigma = float(np.sqrt(max(p * (1-p), 1e-15) / args.shots))
    observed = abs(summary["modes"]["full"]["difference"]["p_opt_classiq_minus_local"])
    summary["validation"] = {
        "p_opt_binomial_sigma_at_local_p": sigma,
        "p_opt_4sigma_tolerance": 4.0 * sigma,
        "p_opt_difference_abs": observed,
        "p_opt_passes_4sigma_sampling_tolerance": bool(observed <= 4.0 * sigma),
        "note": "A statevector-level Classiq comparison is still preferable if the backend exposes exact amplitudes; this run validates sampled distributions, energy, feasibility and best solution.",
    }

    path = out / "validation_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary["validation"], indent=2))
    print(path)


if __name__ == "__main__":
    main()
