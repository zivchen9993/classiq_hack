"""Fixed-vs-dynamic serving-cell sensitivity for temporal antenna tilt.

This is validation only.  The direct QUBO keeps service areas fixed so that the
snapshot objective remains unary + pairwise.  On the small canonical instance
we can enumerate every snapshot configuration and ask whether recomputing the
serving sector materially changes the ranking of antenna tilt settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from direct_temporal import build_direct_temporal_problem
from mobility import evolve_network, make_commuter_mobility
from rf_model import make_network


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rank_corr(a, b):
    r = spearmanr(np.asarray(a, float), np.asarray(b, float)).statistic
    return None if np.isnan(r) else float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True, help="canonical direct-temporal JSON")
    ap.add_argument("--out", default="results/assignment_sensitivity_2026_09_09.json")
    args = ap.parse_args()

    record = load(args.direct)
    inst, run = record["instance"], record["run"]
    seeds = run["provenance"]["seeds"]
    net = make_network(inst["n_towers"], tilt_levels_deg=inst["tilt_levels_deg"],
                       seed=seeds["network"])
    mobility = make_commuter_mobility(net, n_steps=inst["H"], seed=seeds["mobility"],
                                      speed_frac=inst["traffic_speed_fraction"])
    snaps = evolve_network(net, mobility, n_steps=inst["H"])
    problem = build_direct_temporal_problem(snaps, lam=inst["switching_weight"],
                                            cost_mode=inst["switching_mode"])

    configs = [np.asarray(c, dtype=int) for c in product(range(problem.n_tilts),
                                                         repeat=problem.n_sectors)]
    if len(configs) > 200000:
        raise RuntimeError("sensitivity enumeration is intentionally limited to small snapshots")

    by_step = []
    for t, (snap, qubo) in enumerate(zip(snaps, problem.snapshot_qubos)):
        surrogate, fixed_sinr, dynamic_sinr = [], [], []
        fixed_se, dynamic_se, server_change = [], [], []
        for cfg in configs:
            surrogate.append(qubo.objective_from_config(cfg))
            fixed = snap.evaluate_config(cfg)
            dynamic = snap.evaluate_config_dynamic_service(cfg)
            fixed_sinr.append(fixed["mean_sinr_db"])
            dynamic_sinr.append(dynamic["mean_sinr_db"])
            fixed_se.append(fixed["mean_spectral_efficiency"])
            dynamic_se.append(dynamic["mean_spectral_efficiency"])
            server_change.append(dynamic["server_change_pct_vs_reference"])

        i_sur = int(np.argmin(surrogate))
        i_dyn = int(np.argmax(dynamic_se))
        by_step.append({
            "timestep": t,
            "n_configurations": len(configs),
            "spearman_fixed_vs_dynamic_mean_sinr": rank_corr(fixed_sinr, dynamic_sinr),
            "spearman_fixed_vs_dynamic_spectral_efficiency": rank_corr(fixed_se, dynamic_se),
            "spearman_surrogate_cost_vs_negative_dynamic_spectral_efficiency": rank_corr(surrogate, -np.asarray(dynamic_se)),
            "surrogate_optimum_config": configs[i_sur].tolist(),
            "dynamic_spectral_efficiency_optimum_config": configs[i_dyn].tolist(),
            "same_selected_config": bool(i_sur == i_dyn),
            "surrogate_optimum": {
                "surrogate_cost": float(surrogate[i_sur]),
                "fixed_mean_sinr_db": float(fixed_sinr[i_sur]),
                "dynamic_mean_sinr_db": float(dynamic_sinr[i_sur]),
                "fixed_mean_spectral_efficiency": float(fixed_se[i_sur]),
                "dynamic_mean_spectral_efficiency": float(dynamic_se[i_sur]),
                "server_change_pct_vs_reference": float(server_change[i_sur]),
            },
            "dynamic_kpi_optimum": {
                "surrogate_cost": float(surrogate[i_dyn]),
                "dynamic_mean_sinr_db": float(dynamic_sinr[i_dyn]),
                "dynamic_mean_spectral_efficiency": float(dynamic_se[i_dyn]),
                "server_change_pct_vs_reference": float(server_change[i_dyn]),
            },
            "server_change_pct_range": [float(min(server_change)), float(max(server_change))],
        })

    result = {
        "analysis": "fixed-vs-dynamic-serving-cell sensitivity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_result": args.direct,
        "instance": inst,
        "method": "enumerate all T^S snapshot tilt configurations; dynamic association uses strongest chosen-tilt received power",
        "caveat": "dynamic association comparison does not redefine handover edge sets/targets; it isolates serving-cell assignment sensitivity",
        "timesteps": by_step,
        "summary": {
            "min_fixed_dynamic_sinr_rank_correlation": min(x["spearman_fixed_vs_dynamic_mean_sinr"] for x in by_step),
            "min_fixed_dynamic_spectral_efficiency_rank_correlation": min(x["spearman_fixed_vs_dynamic_spectral_efficiency"] for x in by_step),
            "surrogate_and_dynamic_kpi_optimum_match_fraction": float(np.mean([x["same_selected_config"] for x in by_step])),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(out)


if __name__ == "__main__":
    main()
