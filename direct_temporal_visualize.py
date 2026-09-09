"""Figure generator for the direct temporal antenna submission.

Every numerical plot reads saved JSON/CSV artifacts.  The moving-demand figure
reconstructs the exact saved instance from its recorded configuration/seeds.
No benchmark values are embedded in the source.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from direct_temporal import build_direct_temporal_problem
from mobility import evolve_network, make_commuter_mobility
from rf_model import make_network


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def method_map(record):
    return {m["name"]: m for m in record["methods"]}


def save(fig, out, name):
    p = out / name
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_snapshot_validation(snapshot_path, out):
    d = load(snapshot_path)
    methods = d["methods"]
    names = [m["name"].replace(" + one-hot penalty", "\n+ penalty").replace(" + Dicke/W state", "\n+ Dicke/W") for m in methods]
    feasible = [100.0 * m["feasible_shot_fraction"] for m in methods]
    popt = [100.0 * m["probability_of_optimum"] for m in methods]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x - width/2, feasible, width, label="feasible shots")
    ax.bar(x + width/2, popt, width, label="P(optimum)")
    ax.set_xticks(x, names)
    ax.set_ylabel("probability (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"Snapshot antenna encoding check ({d['instance']['qubits']} qubits, p={d['config']['p']})")
    ax.legend()
    for xx, val in zip(x - width/2, feasible):
        ax.text(xx, val + 2, f"{val:.1f}%", ha="center", fontsize=9)
    for xx, val in zip(x + width/2, popt):
        ax.text(xx, val + 2, f"{val:.1f}%", ha="center", fontsize=9)
    return save(fig, out, "snapshot_xy_feasibility.png")


def plot_assignment_sensitivity(path, out):
    d = load(path)
    rows = d["timesteps"]
    t = [r["timestep"] for r in rows]
    sinr = [r["spearman_fixed_vs_dynamic_mean_sinr"] for r in rows]
    se = [r["spearman_fixed_vs_dynamic_spectral_efficiency"] for r in rows]
    surrogate = [r["spearman_surrogate_cost_vs_negative_dynamic_spectral_efficiency"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(t, sinr, marker="o", label="fixed vs dynamic mean SINR ranking")
    ax.plot(t, se, marker="s", label="fixed vs dynamic spectral-efficiency ranking")
    ax.plot(t, surrogate, marker="^", label="surrogate vs dynamic spectral-efficiency ranking")
    ax.set_xticks(t)
    ax.set_ylim(0.7, 1.01)
    ax.set_xlabel("timestep")
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Fixed serving areas preserve broad ranking, but not the exact KPI optimum")
    ax.legend(fontsize=8)
    return save(fig, out, "assignment_sensitivity.png")

def plot_moving_demand(record, out):
    inst, run = record["instance"], record["run"]
    seeds = run["provenance"]["seeds"]
    net = make_network(inst["n_towers"], tilt_levels_deg=inst["tilt_levels_deg"], seed=seeds["network"])
    mob = make_commuter_mobility(net, n_steps=inst["H"], seed=seeds["mobility"],
                                  speed_frac=inst["traffic_speed_fraction"])
    snaps = evolve_network(net, mob, n_steps=inst["H"])
    m = method_map(record)
    best = m["MILP exact/best-known"]["solution"]["trajectory"]

    fig, axes = plt.subplots(1, inst["H"], figsize=(4.3 * inst["H"], 3.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    vmax = max(float(s.demand.max()) for s in snaps)
    vmin = min(float(s.demand.min()) for s in snaps)
    for t, (ax, snap) in enumerate(zip(axes, snaps)):
        n = int(np.sqrt(len(snap.demand)))
        im = ax.imshow(snap.demand.reshape(n, n), origin="lower", vmin=vmin, vmax=vmax,
                       extent=[-snap.grid_extent_m, snap.grid_extent_m, -snap.grid_extent_m, snap.grid_extent_m])
        ax.scatter([s.tower_xy[0] for s in snap.sectors], [s.tower_xy[1] for s in snap.sectors], marker="^", s=45)
        levels = np.asarray(inst["tilt_levels_deg"], float)
        tilt_deg = [float(levels[int(k)]) for k in best[t]]
        ax.set_title(f"t={t}\noptimal tilts={tilt_deg} deg", fontsize=10)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.78, label="relative user demand")
    fig.suptitle("Moving traffic demand and the direct-temporal optimum", fontsize=14)
    return save(fig, out, "direct_moving_demand.png")


def plot_trajectory(record, out):
    m = method_map(record)
    order = [name for name in ["best static trajectory", "snapshot chasing", "myopic hysteresis",
                               "MILP exact/best-known", "QAOA-XY", "DCQO", "BF-DCQO"] if name in m]
    arrays = [np.asarray(m[name]["solution"]["trajectory"], int) for name in order]
    H, S = arrays[0].shape
    fig, axes = plt.subplots(len(order), 1, figsize=(9, 1.15 * len(order) + 1.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name, arr in zip(axes, order, arrays):
        ax.imshow(arr.T, aspect="auto", interpolation="nearest", vmin=0,
                  vmax=record["instance"]["T"] - 1)
        ax.set_yticks(range(S))
        ax.set_ylabel(name.replace(" trajectory", ""), rotation=0, labelpad=58, va="center")
    axes[-1].set_xticks(range(H))
    axes[-1].set_xlabel("timestep")
    fig.suptitle("Tilt trajectories on the same direct spatiotemporal objective")
    return save(fig, out, "direct_trajectories.png")


def plot_objective_and_popt(record, out):
    m = method_map(record)
    names = [n for n in ["QAOA-XY", "annealing (no CD)", "DCQO", "BF-DCQO"] if n in m]
    probs = [100.0 * float(m[n]["solution"]["probability_of_optimum"]) for n in names]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(names))
    bars = ax.bar(x, probs)
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("P(optimum), % (log scale)")
    ax.set_title("Same best sampled trajectory, very different probability mass")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, p * 1.10, f"{p:.3f}%", ha="center", fontsize=9)
    ax.margins(y=0.16)
    return save(fig, out, "direct_objective_gap.png")

def plot_kpis(record, out):
    m = method_map(record)
    names = [n for n in ["best static trajectory", "MILP exact/best-known", "QAOA-XY", "DCQO", "BF-DCQO"] if n in m]
    metrics = [
        ("mean_sinr_db", "Mean SINR (dB)"),
        ("edge_sinr_db", "Edge SINR (dB)"),
        ("mean_spectral_efficiency", "Mean spectral efficiency"),
        ("handover_failure_pct", "Handover failure (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    x = np.arange(len(names))
    for ax, (key, label) in zip(axes.ravel(), metrics):
        vals = [m[n]["solution"]["kpis"][key] for n in names]
        ax.bar(x, vals)
        ax.set_xticks(x, [n.replace(" trajectory", "") for n in names], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(label)
    fig.suptitle("Exact RF KPIs: surrogate optimum and operational trade-offs")
    return save(fig, out, "direct_kpis.png")


def plot_multiseed(summary_path, csv_path, out):
    summary = load(summary_path)
    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
    methods = [m for m in ["QAOA-XY", "annealing (no CD)", "DCQO", "BF-DCQO"]
               if m in summary["aggregate"]]
    data = []
    hit = []
    for method in methods:
        vals = [float(r["p_opt"]) for r in rows if r["method"] == method and r["p_opt"] not in ("", "None")]
        data.append(vals)
        hit.append(summary["aggregate"][method]["exact_hit_fraction"])
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = [f"{method}\nexact sampled {100*h:.0f}%" for method, h in zip(methods, hit)]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("exact state probability P(optimum)")
    ax.set_title(f"Quantum-reference robustness across {len(summary['seeds'])} antenna/mobility seeds")
    ymax = max(max(v) for v in data)
    ax.set_ylim(bottom=-0.01 * max(ymax, 1e-3), top=ymax * 1.08)
    return save(fig, out, "direct_multiseed_popt.png")

def plot_scaling(scaling_path, out):
    d = load(scaling_path)
    rows = d["measured"]
    H = [r["H"] for r in rows]
    qubits = [r["qubits"] for r in rows]
    final = [r["pauli_final"] for r in rows]
    cd = [r["pauli_cd"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
    ax1.plot(H, qubits, marker="o", label="qubits")
    ax1.set_xlabel("horizon H (S=3, T=3)")
    ax1.set_ylabel("qubits")
    ax2 = ax1.twinx()
    ax2.plot(H, final, marker="s", label="final-H Pauli terms")
    ax2.plot(H, cd, marker="^", label="CD Pauli terms")
    ax2.set_ylabel("Pauli terms")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ax1.set_title("Measured encoding scaling (no Classiq depth extrapolation)")
    return save(fig, out, "direct_scaling.png")


def plot_classiq_resources(classiq_root, out):
    summary = load(Path(classiq_root) / "three_step/validation_summary.json")
    meta = load(Path(classiq_root) / "three_step/classiq_metadata.json")
    counts = meta["gate_counts"]
    labels = list(counts)
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(labels, values)
    ax.set_ylabel("gate count")
    ax.set_title(f"Classiq-synthesized 12-qubit DCQO: depth {summary['classiq']['depth']}, {summary['classiq']['gate_count']} gates")
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    return save(fig, out, "classiq_circuit_resources.png")


def plot_classiq_validation(classiq_root, out):
    s = load(Path(classiq_root) / "three_step/validation_summary.json")
    local_e = s["local_reference"]["expected_objective"]
    classiq_e = s["classiq"]["mean_sampled_objective"]
    local_p = s["local_reference"]["probability_of_optimum"]
    classiq_p = s["classiq"]["probability_of_optimum"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    axes[0].bar(["local", "Classiq"], [local_e, classiq_e])
    axes[0].set_ylabel("expected objective")
    axes[0].set_title("Mean energy")
    for i, v in enumerate([local_e, classiq_e]):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="top" if v < 0 else "bottom", fontsize=9)
    axes[1].bar(["local", "Classiq"], [100*local_p, 100*classiq_p])
    axes[1].set_ylabel("P(optimum), %")
    axes[1].set_title("Optimum probability")
    for i, v in enumerate([100*local_p, 100*classiq_p]):
        axes[1].text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Saved 12-qubit Classiq DCQO validation (5000 shots): P(opt) mismatch remains")
    return save(fig, out, "classiq_validation.png")

def plot_classiq_logical(classiq_root, out):
    meta = load(Path(classiq_root) / "three_step/classiq_metadata.json")
    summary = load(Path(classiq_root) / "three_step/validation_summary.json")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, 6.6)
    ax.axis("off")
    block_labels = ["(t0,s0)", "(t0,s1)", "(t0,s2)", "(t1,s0)", "(t1,s1)", "(t1,s2)"]
    ys = [5.7 - i for i in range(6)]
    for y, label in zip(ys, block_labels):
        ax.hlines(y, 0.7, 11.4, linewidth=1)
        ax.text(0.4, y, label, ha="right", va="center", fontsize=9)
        ax.add_patch(plt.Rectangle((0.9, y-0.23), 1.25, 0.46, fill=False))
        ax.text(1.525, y, "Dicke W(1,2)", ha="center", va="center", fontsize=7.5)
    for j in range(meta["n_steps"]):
        x = 2.8 + j * 2.35
        ax.add_patch(plt.Rectangle((x, 0.45), 1.75, 5.65, fill=False, linewidth=1.5))
        ax.text(x+0.875, 3.3, f"Suzuki-Trotter\nstep {j+1}", ha="center", va="center", fontsize=9)
        ax.text(x+0.875, 1.05, "global spatial +\ntemporal couplings", ha="center", va="center", fontsize=7.5)
    ax.add_patch(plt.Rectangle((10.25, 0.45), 0.95, 5.65, fill=False))
    ax.text(10.725, 3.3, "measure", ha="center", va="center", rotation=90, fontsize=9)
    ax.text(6.0, 6.35, "Logical structure of the saved Classiq DCQO program", ha="center", fontsize=14)
    ax.text(6.0, 0.05, f"12 qubits | 6 one-hot blocks | 3 scheduled Hamiltonians | synthesized depth={summary['classiq']['depth']} | gates={summary['classiq']['gate_count']}", ha="center", fontsize=9)
    return save(fig, out, "classiq_logical_circuit.png")

def plot_limitations(record, scaling_path, classiq_root, out):
    scaling = load(scaling_path)
    cv = load(Path(classiq_root) / "three_step/validation_summary.json")
    m = method_map(record)
    lines = [
        "MEASURED",
        f"Direct objective: S={record['instance']['S']}, T={record['instance']['T']}, H={record['instance']['H']} ({record['instance']['qubits']} one-hot qubits).",
        "Strong classical methods solve the demo instance exactly in milliseconds.",
        f"QAOA-XY P(opt)={100*m['QAOA-XY']['solution']['probability_of_optimum']:.2f}%, DCQO={100*m['DCQO']['solution']['probability_of_optimum']:.2f}%, BF-DCQO={100*m['BF-DCQO']['solution']['probability_of_optimum']:.2f}% on the selected run.",
        f"Saved Classiq DCQO: 12 qubits, depth {cv['classiq']['depth']}, {cv['classiq']['gate_count']} gates, 5000 shots.",
        f"Classiq P(opt) differs from the local Trotter reference by {abs(cv['classiq']['probability_of_optimum']-cv['local_reference']['probability_of_optimum']):.3f}; validation is not yet within a tight tolerance.",
        "",
        "NOT DEMONSTRATED",
        "No present quantum advantage; no end-to-end wall-clock win over MILP/heuristics.",
        "No Classiq circuit-depth scaling sweep in this environment (SDK/authentication unavailable).",
        "No evidence that BF-DCQO is consistently better than cold DCQO across seeds.",
        "Full-scale behavior remains a conditional future hypothesis, not measured data.",
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    ax.text(0.02, 0.98, "Evidence boundary", va="top", fontsize=17, weight="bold")
    ax.text(0.02, 0.90, "\n".join(lines), va="top", fontsize=11, linespacing=1.45)
    return save(fig, out, "limitations_panel.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True)
    ap.add_argument("--multiseed-summary", required=True)
    ap.add_argument("--multiseed-csv", required=True)
    ap.add_argument("--scaling", required=True)
    ap.add_argument("--classiq-root", default="results/classiq_tiny_validation_2026_09_09")
    ap.add_argument("--snapshot", default="results/snapshot_validation_2026_09_09.json")
    ap.add_argument("--assignment", default="results/assignment_sensitivity_2026_09_09.json")
    ap.add_argument("--pareto-figure", default="figures/final_temporal/direct_temporal_pareto.png")
    ap.add_argument("--out", default="figures/final_temporal")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = load(args.direct)
    paths = [
        plot_snapshot_validation(args.snapshot, out),
        plot_moving_demand(record, out), plot_trajectory(record, out),
        plot_objective_and_popt(record, out), plot_kpis(record, out),
        plot_multiseed(args.multiseed_summary, args.multiseed_csv, out),
        plot_scaling(args.scaling, out), plot_classiq_resources(args.classiq_root, out),
        plot_classiq_validation(args.classiq_root, out), plot_classiq_logical(args.classiq_root, out),
        plot_assignment_sensitivity(args.assignment, out),
        plot_limitations(record, args.scaling, args.classiq_root, out),
    ]
    pareto = Path(args.pareto_figure)
    if pareto.exists():
        paths.append(pareto)
    manifest = out / "FIGURE_MANIFEST.json"
    manifest.write_text(json.dumps({"sources": vars(args), "figures": [str(p) for p in paths]}, indent=2)+"\n")
    print("\n".join(map(str, paths)))
    print(manifest)


if __name__ == "__main__":
    main()
