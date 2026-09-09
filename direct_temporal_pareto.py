"""Operational Pareto/guardrail analysis for the canonical temporal antenna case.

The direct QUBO surrogate is optimized jointly with switching cost, but every
trajectory is also scored with exact joint-SINR KPIs.  On the tiny correctness
instance all snapshot configurations can be cached, making an exhaustive
trajectory Pareto analysis cheap and reproducible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from direct_temporal import build_direct_temporal_problem
from mobility import evolve_network, make_commuter_mobility
from rf_model import make_network


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reconstruct(record):
    inst, run = record["instance"], record["run"]
    seeds = run["provenance"]["seeds"]
    net = make_network(inst["n_towers"], tilt_levels_deg=inst["tilt_levels_deg"], seed=seeds["network"])
    mob = make_commuter_mobility(net, n_steps=inst["H"], seed=seeds["mobility"], speed_frac=inst["traffic_speed_fraction"])
    snaps = evolve_network(net, mob, n_steps=inst["H"])
    return build_direct_temporal_problem(snaps, lam=inst["switching_weight"], cost_mode=inst["switching_mode"])


def pareto_mask(objective, handover):
    order = np.argsort(objective)
    keep = np.zeros(len(objective), dtype=bool)
    best_ho = np.inf
    for idx in order:
        if handover[idx] < best_ho - 1e-12:
            keep[idx] = True
            best_ho = handover[idx]
    return keep


def plot_saved(path, figure):
    d = load(path)
    rows = d["trajectories"]
    obj = np.asarray([r["objective"] for r in rows])
    ho = np.asarray([r["handover_failure_pct"] for r in rows])
    se = np.asarray([r["mean_spectral_efficiency"] for r in rows])
    pareto = np.asarray([r["pareto"] for r in rows], bool)
    best = d["reference"]["unconstrained_optimum"]
    static = d["reference"]["best_static"]
    guard = d["reference"]["guardrail_optimum"]

    gap = obj - best["objective"]
    # The full 19,683-trajectory cloud is saved in JSON.  The deck figure zooms
    # into the operationally relevant near-optimal region so the trade-off is
    # readable rather than burying the frontier under thousands of bad points.
    visible = gap <= 2.0
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    sc = ax.scatter(ho[visible], gap[visible], c=se[visible], s=22, alpha=0.32)
    shown_pareto = pareto & visible
    ax.scatter(ho[shown_pareto], gap[shown_pareto], s=52, facecolors="none", edgecolors="black", label="Pareto frontier")
    ax.scatter([best["handover_failure_pct"]], [0.0], marker="*", s=190, label="surrogate optimum")
    ax.scatter([static["handover_failure_pct"]], [static["objective"]-best["objective"]], marker="s", s=90, label="best static")
    ax.scatter([guard["handover_failure_pct"]], [guard["objective"]-best["objective"]], marker="D", s=90, label="guardrail optimum")
    ax.axvline(static["handover_failure_pct"], linestyle="--", linewidth=1, label="static HO guardrail")
    ax.set_xlabel("Mean handover failure (%)")
    ax.set_ylabel("Surrogate objective gap to unconstrained optimum")
    ax.set_title("Near-optimal temporal antenna trade-off: objective vs handover reliability")
    fig.colorbar(sc, ax=ax, label="Mean spectral efficiency")
    ax.legend(fontsize=8, loc="best")
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True)
    ap.add_argument("--out", default="results/direct_temporal_pareto_2026_09_09.json")
    ap.add_argument("--figure", default="figures/final_temporal/direct_temporal_pareto.png")
    args = ap.parse_args()

    record = load(args.direct)
    problem = reconstruct(record)
    H, S, T = problem.horizon, problem.n_sectors, problem.n_tilts
    configs = [np.asarray(c, int) for c in product(range(T), repeat=S)]
    if len(configs) ** H > 2_000_000:
        raise RuntimeError("exhaustive Pareto analysis is intentionally a tiny-tier validation")

    # Cache exact KPI evaluations: H*T^S RF evaluations instead of T^(S*H)*H.
    kpi_cache = [[snap.evaluate_config(cfg) for cfg in configs] for snap in problem.snapshots]
    snapshot_cost = np.asarray([[q.objective_from_config(cfg) for cfg in configs]
                                for q in problem.snapshot_qubos])
    levels = np.asarray(problem.snapshots[0].tilt_levels_deg, float)

    rows = []
    for choice in product(range(len(configs)), repeat=H):
        traj = np.stack([configs[i] for i in choice])
        raw_switch = (np.count_nonzero(np.diff(traj, axis=0)) if problem.cost_mode == "switches"
                      else np.abs(np.diff(levels[traj], axis=0)).sum())
        objective = float(sum(snapshot_cost[t, choice[t]] for t in range(H)) + problem.lam * raw_switch)
        step = [kpi_cache[t][choice[t]] for t in range(H)]
        rows.append({
            "trajectory": traj.tolist(),
            "objective": objective,
            "mean_sinr_db": float(np.mean([x["mean_sinr_db"] for x in step])),
            "edge_sinr_db": float(np.mean([x["edge_sinr_db"] for x in step])),
            "mean_spectral_efficiency": float(np.mean([x["mean_spectral_efficiency"] for x in step])),
            "outage_pct": float(np.mean([x["outage_pct"] for x in step])),
            "handover_failure_pct": float(np.mean([x["handover_failure_pct"] for x in step])),
            "n_switches": int(np.count_nonzero(np.diff(traj, axis=0))),
            "switch_cost": float(raw_switch),
        })

    obj = np.asarray([r["objective"] for r in rows])
    ho = np.asarray([r["handover_failure_pct"] for r in rows])
    outage = np.asarray([r["outage_pct"] for r in rows])
    p = pareto_mask(obj, ho)
    for row, flag in zip(rows, p):
        row["pareto"] = bool(flag)

    methods = {m["name"]: m for m in record["methods"]}
    static_k = methods["best static trajectory"]["solution"]["kpis"]
    static_traj = methods["best static trajectory"]["solution"]["trajectory"]
    static_obj = methods["best static trajectory"]["solution"]["objective"]
    exact = int(np.argmin(obj))
    guard_mask = ((ho <= float(static_k["handover_failure_pct"]) + 1e-12) &
                  (outage <= float(static_k["outage_pct"]) + 1e-12))
    guard_idx = int(np.argmin(np.where(guard_mask, obj, np.inf)))

    result = {
        "analysis": "temporal antenna operational Pareto and guardrail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_result": args.direct,
        "guardrail": {
            "handover_failure_pct_max": float(static_k["handover_failure_pct"]),
            "outage_pct_max": float(static_k["outage_pct"]),
            "definition": "do not worsen mean handover failure or mean outage relative to the best static trajectory",
        },
        "reference": {
            "unconstrained_optimum": rows[exact],
            "best_static": {
                "trajectory": static_traj,
                "objective": float(static_obj),
                "handover_failure_pct": float(static_k["handover_failure_pct"]),
                "outage_pct": float(static_k["outage_pct"]),
                "mean_spectral_efficiency": float(static_k["mean_spectral_efficiency"]),
                "n_switches": int(static_k["n_switches"]),
            },
            "guardrail_optimum": rows[guard_idx],
        },
        "summary": {
            "n_trajectories": len(rows),
            "n_pareto": int(p.sum()),
            "n_guardrail_feasible": int(guard_mask.sum()),
            "guardrail_objective_penalty": float(rows[guard_idx]["objective"] - rows[exact]["objective"]),
            "guardrail_throughput_delta_vs_static": float(rows[guard_idx]["mean_spectral_efficiency"] - static_k["mean_spectral_efficiency"]),
        },
        "trajectories": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fig = Path(args.figure)
    fig.parent.mkdir(parents=True, exist_ok=True)
    plot_saved(out, fig)
    print(json.dumps(result["summary"], indent=2))
    print("guardrail optimum", result["reference"]["guardrail_optimum"])
    print(out)
    print(fig)


if __name__ == "__main__":
    main()
