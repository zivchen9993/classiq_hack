"""
Figures for the temporal (dynamic) optimization results.

Reads the JSON written by `temporal_benchmark.py` -- so figures can be
regenerated without re-running the ~12 minute QAOA -- and produces, into
./figures:

  1. temporal_regime.png    -- the switching-weight sweep: where planning a
                               trajectory actually beats both "never move"
                               and "always chase the optimum"
  2. temporal_ceiling.png   -- the surrogate ceiling: objective gain that
                               does NOT survive into network KPIs
  3. temporal_coherence.png -- is the premise true? demand drift vs how much
                               the optimum actually moves
  4. temporal_timeline.png  -- what each policy does over the horizon, and
                               when it moves antennas

Usage:
    python temporal_visualize.py                  # newest results dir
    python temporal_visualize.py results/2026_09_08__17_38_03
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# same palette as visualize.py so the deck reads as one system
BLUE, RED, GREEN, GREY = "#4c72b0", "#c44e52", "#55a868", "#8c8c8c"


def load_results(path=None) -> dict:
    """Load a results json; defaults to the most recent results directory."""
    if path:
        p = Path(path)
        if p.is_dir():
            p = p / "temporal_results.json"
    else:
        runs = sorted(Path("results").glob("*/temporal_results.json"))
        if not runs:
            raise SystemExit("no results found -- run temporal_benchmark.py first")
        p = runs[-1]
    print(f"  reading {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _save(fig, name):
    path = FIGDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# 1. The operating regime
# ---------------------------------------------------------------------------

def fig_regime(rec):
    sweep = rec["lam_sweep"]
    lam = [s["lam"] for s in sweep]
    dp = [s["dp"] for s in sweep]
    static = [s["static"] for s in sweep]
    greedy = [s["greedy"] for s in sweep]
    moves = [s["dp_moves"] for s in sweep]
    strict = [s["strict"] for s in sweep]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.1]})

    # shade the regime where planning strictly beats both baselines
    win = [l for l, s in zip(lam, strict) if s]
    if win:
        ax.axvspan(min(win), max(win), color=GREEN, alpha=0.14, zorder=0)
        ax2.axvspan(min(win), max(win), color=GREEN, alpha=0.14, zorder=0)
        # label sits just under the shaded band's top, clear of the legend
        ax.annotate("planning strictly wins",
                    xy=((min(win) + max(win)) / 2, max(static) * 1.55),
                    ha="center", fontsize=9, color="#2f6b45", fontweight="bold")

    ax.plot(lam, greedy, "o-", color=RED, lw=2, label="always chase the optimum")
    ax.plot(lam, static, "s-", color=GREY, lw=2, label="never retune")
    ax.plot(lam, dp, "D-", color=BLUE, lw=2.5, markersize=7,
            label="planned trajectory (exact DP)")

    ax.set_ylabel("total cost J  (lower is better)")
    ax.set_title("Planning a trajectory only pays inside a window\n"
                 "(outside it, the optimum degenerates to a baseline)",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    # the interesting structure lives at small lam; the tail is flat, so a
    # log x-axis stops one far point from squeezing the whole story
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.5)
    ax2.set_xscale("symlog", linthresh=0.01, linscale=0.5)
    ax2.set_xticks(lam)
    ax2.set_xticklabels([f"{l:g}" for l in lam], fontsize=8)

    ax2.plot(lam, moves, "D-", color=BLUE, lw=2)
    ax2.set_xlabel("switching weight  lam   (cost of moving an antenna)")
    ax2.set_ylabel("antenna\nmoves", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-0.4, max(moves) + 0.6)
    ax2.annotate("chase", (lam[0], moves[0]), textcoords="offset points",
                 xytext=(6, 4), fontsize=8, color=RED)
    ax2.annotate("freeze", (lam[-1], moves[-1]), textcoords="offset points",
                 xytext=(-8, 6), fontsize=8, color=GREY, ha="right")

    _save(fig, "temporal_regime.png")


# ---------------------------------------------------------------------------
# 2. The surrogate ceiling -- the honest headline
# ---------------------------------------------------------------------------

def fig_ceiling(rec):
    sc = rec["surrogate_ceiling"]
    kpis = rec["kpis"]
    dp_name = [p["name"] for p in rec["policies"] if "DP" in p["name"]][0]
    st_name = [p["name"] for p in rec["policies"] if "static" in p["name"]][0]
    k_dp, k_st = kpis[dp_name], kpis[st_name]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.7))

    # (title, static value, dp value, lower_is_better)
    data = [
        ("surrogate objective J",
         [p for p in rec["policies"] if p["name"] == st_name][0]["objective"],
         [p for p in rec["policies"] if p["name"] == dp_name][0]["objective"], True),
        ("mean SINR (dB)", k_st["sinr"], k_dp["sinr"], False),
        ("handover failure (%)", k_st["ho"], k_dp["ho"], True),
        ("antenna moves", k_st["moves"], k_dp["moves"], True),
    ]

    for ax, (title, v_st, v_dp, lower_better) in zip(axes, data):
        bars = ax.bar(["never\nretune", "planned\n(DP)"], [v_st, v_dp],
                      color=[GREY, BLUE], width=0.62)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")
        ax.set_axisbelow(True)

        improved = (v_dp < v_st) if lower_better else (v_dp > v_st)
        span = max(abs(v_st), abs(v_dp), 1e-9)
        meaningful = abs(v_dp - v_st) > 0.02 * span

        for b, v in zip(bars, [v_st, v_dp]):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

        if improved and meaningful:
            ax.set_xlabel("improved", color="#2f6b45", fontsize=9, fontweight="bold")
        elif meaningful:
            ax.set_xlabel("worse", color=RED, fontsize=9, fontweight="bold")
        else:
            ax.set_xlabel("no real change", color=GREY, fontsize=9)
        lo = min(0, v_st, v_dp)
        ax.set_ylim(lo, max(v_st, v_dp) * 1.22 + 1e-9)

    fig.suptitle(
        f"The {sc['objective_gain_pct']:.0f}% objective gain does not reach the network\n"
        "Optimizing the surrogate better stops paying once surrogate error "
        "exceeds the gap between solutions",
        fontweight="bold", fontsize=11.5, y=1.04)
    fig.tight_layout()
    _save(fig, "temporal_ceiling.png")


# ---------------------------------------------------------------------------
# 3. Is the temporal-coherence premise true?
# ---------------------------------------------------------------------------

def fig_coherence(rec):
    coh = rec["coherence"]
    drift = np.asarray(coh["drift"])
    moves = np.asarray(coh["opt_moves"])
    stay = np.asarray(coh["stay_cost"])
    n_sec = coh["n_sectors"]
    steps = np.arange(len(drift))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.bar(steps - 0.19, drift / drift.max(), width=0.36, color=GREY,
           label=f"demand drift (normalized, max {drift.max():.3f})")
    ax.bar(steps + 0.19, moves / n_sec, width=0.36, color=BLUE,
           label=f"fraction of the {n_sec} sectors whose optimum moves")

    ax.set_xlabel("timestep")
    ax.set_ylabel("normalized")
    ax.set_xticks(steps)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.32)          # headroom so the legend clears the bars

    ax2 = ax.twinx()
    ax2.plot(steps, stay, "o--", color=RED, lw=2, label="cost of NOT retuning")
    ax2.set_ylabel("regret from holding the previous config", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, max(stay.max() * 1.35, 1e-6))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right")

    ax.set_title(
        "The premise holds: demand moves steadily, the optimum barely does\n"
        f"(mean {coh['mean_opt_moves']:.2f} of {n_sec} sectors change per step)",
        fontweight="bold")
    _save(fig, "temporal_coherence.png")


# ---------------------------------------------------------------------------
# 4. What each policy actually does over the horizon
# ---------------------------------------------------------------------------

def fig_timeline(rec):
    policies = rec["policies"]
    H = len(policies[0]["trajectory"])
    S = len(policies[0]["trajectory"][0])

    fig, axes = plt.subplots(len(policies), 1, figsize=(9.0, 1.55 * len(policies)),
                             sharex=True)
    if len(policies) == 1:
        axes = [axes]

    vmax = max(max(max(row) for row in p["trajectory"]) for p in policies)

    for ax, p in zip(axes, policies):
        traj = np.asarray(p["trajectory"]).T          # (S, H)
        ax.imshow(traj, aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                  interpolation="nearest")

        # a reconfiguration EVENT is a timestep where anything changed; one
        # event can move several sectors at once, so the two counts differ
        events = 0
        for t in range(1, H):
            if not np.array_equal(traj[:, t], traj[:, t - 1]):
                events += 1
                ax.axvline(t - 0.5, color="white", lw=2.5)
                ax.axvline(t - 0.5, color=RED, lw=1.4, ls="--")

        ax.set_ylabel(p["name"].split("(")[0].strip().replace(" ", "\n"),
                      fontsize=8, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.text(1.012, 0.5, f"{p['moves']} sector-moves\nin {events} event(s)",
                transform=ax.transAxes, va="center", fontsize=8,
                color=BLUE if p["moves"] else GREY, fontweight="bold")

    axes[-1].set_xlabel("timestep")
    axes[-1].set_xticks(range(H))
    axes[0].set_title(
        f"Tilt trajectories over the horizon ({S} sectors, colour = tilt level)\n"
        "red lines mark reconfiguration events -- planning batches its moves "
        "into fewer of them", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    _save(fig, "temporal_timeline.png")


# ---------------------------------------------------------------------------

def main(path=None):
    print("Generating temporal figures ...")
    rec = load_results(path)
    fig_regime(rec)
    fig_ceiling(rec)
    fig_coherence(rec)
    fig_timeline(rec)
    print("done.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
