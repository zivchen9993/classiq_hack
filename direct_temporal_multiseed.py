"""Run the direct temporal benchmark across independent network/mobility seeds.

Each individual run remains a canonical result record.  This wrapper creates an
aggregate JSON/CSV so stochastic quantum-reference results are reported as a
distribution rather than a single favorable seed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from direct_temporal_benchmark import run


def stamp():
    return datetime.now(timezone.utc).strftime("%Y_%m_%d__%H_%M_%S_utc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--shots", type=int, default=300)
    ap.add_argument("--qaoa-maxiter", type=int, default=24)
    ap.add_argument("--dcqo-steps", type=int, default=8)
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    source_files = []
    for seed in seeds:
        record, path = run(
            n_towers=1, horizon=3, n_tilts=3, seed=seed,
            mobility_seed=100 + seed, speed_frac=0.8, lam=0.07,
            shots=args.shots, qaoa_depth=1, qaoa_maxiter=args.qaoa_maxiter,
            dcqo_steps=args.dcqo_steps, bf_iters=3, sa_steps=3000,
            milp_time_limit=30.0, run_quantum=True, execute_classiq=False,
            out_root=args.out_root,
        )
        source_files.append(str(path))
        for method in record["methods"]:
            rows.append({
                "seed": seed,
                "mobility_seed": 100 + seed,
                "method": method["name"],
                "family": method["family"],
                "objective": method["solution"]["objective"],
                "gap": method["solution"]["gap_to_best_known"],
                "p_opt": method["solution"]["probability_of_optimum"],
                "wall_time_seconds": method["resources"]["wall_time_seconds"],
                "shots": method["resources"]["final_sampling_shots"],
                "quantum_executions": method["resources"]["quantum_executions"],
                "n_switches": method["solution"]["kpis"].get("n_switches"),
                "mean_sinr_db": method["solution"]["kpis"].get("mean_sinr_db"),
                "edge_sinr_db": method["solution"]["kpis"].get("edge_sinr_db"),
                "mean_spectral_efficiency": method["solution"]["kpis"].get("mean_spectral_efficiency"),
                "handover_failure_pct": method["solution"]["kpis"].get("handover_failure_pct"),
            })

    out = Path(args.out_root) / f"multiseed_{stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "multiseed_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        entry = {"n": len(subset)}
        for key in ("objective", "gap", "wall_time_seconds", "n_switches",
                    "mean_sinr_db", "edge_sinr_db", "mean_spectral_efficiency",
                    "handover_failure_pct"):
            vals = np.asarray([r[key] for r in subset if r[key] is not None], float)
            entry[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                          "min": float(vals.min()), "max": float(vals.max())}
        popt = np.asarray([r["p_opt"] for r in subset if r["p_opt"] is not None], float)
        if len(popt):
            entry["p_opt"] = {"mean": float(popt.mean()), "std": float(popt.std(ddof=1)) if len(popt)>1 else 0.0,
                              "min": float(popt.min()), "max": float(popt.max())}
        entry["exact_hit_fraction"] = float(np.mean([abs(r["gap"]) < 1e-9 for r in subset]))
        aggregate[method] = entry

    summary = {
        "kind": "direct-temporal-multiseed",
        "seeds": seeds,
        "shots_per_quantum_method": args.shots,
        "source_records": source_files,
        "aggregate": aggregate,
    }
    summary_path = out / "multiseed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    print(csv_path)


if __name__ == "__main__":
    main()
