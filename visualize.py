"""
Figures for the pitch deck.

Produces (into ./figures):
  1. network_sinr.png   -- SINR heat map before vs after optimization, with the
                           tower/sector layout and traffic hotspots overlaid
  2. ansatz_compare.png -- standard X mixer vs constrained XY mixer:
                           P(optimum) and feasible-shot fraction against depth
  3. pareto.png         -- the SINR / handover-reliability trade-off frontier
  4. scaling.png        -- search space and qubit count against network size
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from rf_model import make_network
from qubo_builder import build_qubo
from classical_baseline import brute_force, best_uniform_baseline

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

FLAGSHIP = dict(n_towers=2, seed=27, tilt_levels_deg=[0.0, 4.0, 8.0])


# ---------------------------------------------------------------------------

def _pointwise_sinr_db(net, tilt_indices):
    """Per-grid-point SINR in dB for a configuration (serving sector's view)."""
    tilt_indices = np.asarray(tilt_indices, dtype=int)
    n_s = len(net.sectors)
    p_lin = np.stack([net.rx_lin[i, tilt_indices[i], :] for i in range(n_s)])
    total = p_lin.sum(axis=0)

    sinr = np.full(len(net.grid_xy), np.nan)
    for i in range(n_s):
        area = net.service_mask[i]
        signal = p_lin[i, area]
        sinr[area] = 10 * np.log10(signal / (total[area] - signal + net.noise_lin))
    return sinr


def fig_network_sinr(net, uniform_cfg, optimized_cfg):
    side = net.grid_points_per_axis
    ext = net.grid_extent_m
    extent = [-ext, ext, -ext, ext]

    s_uni = _pointwise_sinr_db(net, uniform_cfg).reshape(side, side)
    s_opt = _pointwise_sinr_db(net, optimized_cfg).reshape(side, side)
    delta = s_opt - s_uni

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    vmin = np.nanmin([s_uni, s_opt])
    vmax = np.nanmax([s_uni, s_opt])

    for ax, data, title in (
        (axes[0], s_uni, f"Uniform tilt (today's practice)"),
        (axes[1], s_opt, f"Optimized per-sector tilts"),
    ):
        im = ax.imshow(data, origin="lower", extent=extent, cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="SINR (dB)", fraction=0.046)

    lim = np.nanmax(np.abs(delta)) or 1.0
    im = axes[2].imshow(delta, origin="lower", extent=extent, cmap="RdBu_r",
                        norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
    axes[2].set_title("Change in SINR (red = improved)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes[2], label="Delta SINR (dB)", fraction=0.046)

    # overlay towers, sector boresights and traffic hotspots
    for ax in axes:
        for (hx, hy, sigma, amp) in net.demand_hotspots:
            ax.add_patch(plt.Circle((hx, hy), sigma, fill=False, color="white",
                                    ls=":", lw=1.2, alpha=0.8))
        seen = set()
        for s in net.sectors:
            if s.tower_xy not in seen:
                seen.add(s.tower_xy)
                ax.plot(*s.tower_xy, "^", color="white", markersize=11,
                        markeredgecolor="black", markeredgewidth=1.2)
            a = np.radians(s.azimuth_deg)
            ax.arrow(s.tower_xy[0], s.tower_xy[1], 190 * np.cos(a), 190 * np.sin(a),
                     head_width=55, color="white", alpha=0.9, lw=1.2)
        ax.set_xlabel("metres")
        ax.set_ylabel("metres")

    k_uni = net.evaluate_config(uniform_cfg)
    k_opt = net.evaluate_config(optimized_cfg)
    fig.suptitle(
        f"Antenna tilt optimization: mean SINR {k_uni['mean_sinr_db']:.2f} dB "
        f"-> {k_opt['mean_sinr_db']:.2f} dB   "
        f"(+{100*(k_opt['mean_spectral_efficiency']/k_uni['mean_spectral_efficiency']-1):.1f}% throughput)",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "network_sinr.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_ansatz_compare(results_x, results_xy, n_feasible):
    depths = [r["p"] for r in results_x]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(depths, [100 * r["prob_of_optimum"] for r in results_x],
            "o-", label="QAOA-X (standard + penalty)", color="#c44e52", lw=2)
    ax.plot(depths, [100 * r["prob_of_optimum"] for r in results_xy],
            "s-", label="QAOA-XY (constrained, ours)", color="#4c72b0", lw=2)
    ax.axhline(100.0 / n_feasible, ls="--", color="grey",
               label=f"uniform random guess ({100/n_feasible:.2f}%)")
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("P(sampling the optimum)  [%]")
    ax.set_title("Probability of finding the optimum", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(depths, [100 * r["valid_prob"] for r in results_x],
            "o-", label="QAOA-X (standard + penalty)", color="#c44e52", lw=2)
    ax.plot(depths, [100 * r["valid_prob"] for r in results_xy],
            "s-", label="QAOA-XY (constrained, ours)", color="#4c72b0", lw=2)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("feasible shots  [%]")
    ax.set_title("Fraction of measurements that are valid configurations",
                 fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGDIR, "ansatz_compare.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_pareto(net):
    weight_sets = [(0.40, 0.35, 0.25), (0.375, 0.325, 0.30), (0.35, 0.30, 0.35),
                   (0.325, 0.275, 0.40), (0.30, 0.25, 0.45), (0.30, 0.20, 0.50)]
    # several weight settings can select the SAME configuration, so group them
    # and give each distinct operating point a single label
    points = {}
    for (w_cov, w_int, w_ho) in weight_sets:
        q = build_qubo(net, w_cov=w_cov, w_int=w_int, w_ho=w_ho)
        k = net.evaluate_config(brute_force(q).config)
        key = (round(k["handover_failure_pct"], 3), round(k["mean_sinr_db"], 3))
        points.setdefault(key, []).append(w_ho)

    ordered = sorted(points.items())
    ho = [k[0] for k, _ in ordered]
    sinr = [k[1] for k, _ in ordered]
    labels = []
    for _, ws in ordered:
        labels.append(f"w_ho={min(ws):.2f}" if len(ws) == 1
                      else f"w_ho={min(ws):.2f}-{max(ws):.2f}")

    q0 = build_qubo(net)
    k_uni = net.evaluate_config(best_uniform_baseline(q0).config)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(ho, sinr, "o-", color="#4c72b0", lw=2, markersize=9,
            label="optimized (weights varied)")
    for x, y, lab in zip(ho, sinr, labels):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(-10, 10), fontsize=8, ha="right")
    ax.plot(k_uni["handover_failure_pct"], k_uni["mean_sinr_db"], "*",
            color="#c44e52", markersize=20, label="uniform tilt (baseline)")

    ax.set_xlabel("handover failure rate  [%]   (lower is better)")
    ax.set_ylabel("mean SINR  [dB]   (higher is better)")
    ax.set_title("Coverage quality vs handover reliability:\nthe operator's policy dial",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "pareto.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_depth_scaling():
    """P(optimum) vs QAOA depth on the 36-qubit instance, from qaoa_subspace.py."""
    depths = [1, 3, 5, 8, 12]
    p_opt = [0.170, 0.842, 1.215, 2.000, 5.290]     # percent
    p_random = 100.0 / 4 ** 9
    greedy = 42.5

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.semilogy(depths, p_opt, "s-", color="#4c72b0", lw=2.5, markersize=9,
                label="QAOA-XY (constrained), 36 qubits")
    ax.axhline(p_random, ls="--", color="grey",
               label=f"uniform random guess ({p_random:.4f}%)")
    ax.axhline(greedy, ls=":", color="#c44e52", lw=2,
               label=f"greedy, per RESTART ({greedy:.0f}%) - costs ~70 evaluations")

    for d, v in zip(depths, p_opt):
        ax.annotate(f"{v/p_random:,.0f}x", (d, v), textcoords="offset points",
                    xytext=(0, -16), fontsize=8, color="#4c72b0", ha="center")

    ax.set_xlim(0.3, 13.2)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("P(hitting the optimum) per repetition  [%]")
    ax.set_title("36-qubit instance: deeper circuits concentrate on the optimum\n"
                 "(labels show amplification over random)", fontweight="bold")
    # the two dashed references are per-repetition rates in DIFFERENT units --
    # one greedy restart is far more work than one circuit shot, so this axis
    # is not a like-for-like cost comparison (see resource-to-solution table)
    ax.legend(fontsize=8, loc="center right", title="one repetition =",
              title_fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "depth_scaling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_transfer():
    """Zero-shot parameter transfer to unseen instances (param_transfer.py)."""
    seeds = ["11", "23", "42", "7", "55", "68"]
    transferred = [6.399, 0.849, 3.397, 1.145, 0.704, 3.027]
    retrained = [12.272, 2.849, 6.219, 2.385, 1.793, 7.358]
    p_random = 100.0 / 4 ** 9

    x = np.arange(len(seeds))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(x - w / 2, retrained, w, label="retrained on that instance (ceiling)",
           color="#c9d6e8", edgecolor="#4c72b0")
    ax.bar(x + w / 2, transferred, w, label="transferred verbatim (no retraining)",
           color="#4c72b0")
    ax.axhline(p_random, ls="--", color="grey",
               label=f"uniform random guess ({p_random:.4f}%)")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("P(sampling the optimum)  [%]")
    ax.set_title("A schedule trained ONCE, applied to unseen 36-qubit networks\n"
                 "retains 47% of retrained performance, 6,781x better than random",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "param_transfer.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_scaling(tilt_levels=4, max_towers=10):
    towers = np.arange(1, max_towers + 1)
    sectors = 3 * towers
    configs = np.power(float(tilt_levels), sectors)
    qubits = sectors * tilt_levels

    # Both series go on ONE log axis. A twin log/linear pair would render the
    # exponential and the linear term as two identical straight lines sitting
    # on top of each other, hiding exactly the contrast this figure exists to
    # show. On a shared log axis the search space rockets and the qubit
    # requirement stays almost flat.
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.semilogy(sectors, configs, "o-", color="#c44e52", lw=2.5, markersize=7,
                label=f"classical search space  ({tilt_levels}^S configurations)")
    ax.semilogy(sectors, qubits, "s-", color="#4c72b0", lw=2.5, markersize=7,
                label=f"qubits required  (S x {tilt_levels})")

    ax.set_xlabel("number of sectors S")
    ax.set_ylabel("count (log scale)")
    ax.grid(alpha=0.3, which="both")

    # annotate the endpoints so the gap is unmissable
    ax.annotate(f"{configs[-1]:.0e} configurations", (sectors[-1], configs[-1]),
                textcoords="offset points", xytext=(-14, -20), fontsize=9,
                color="#c44e52", ha="right", fontweight="bold")
    ax.annotate(f"{qubits[-1]} qubits", (sectors[-1], qubits[-1]),
                textcoords="offset points", xytext=(-14, -20), fontsize=9,
                color="#4c72b0", ha="right", fontweight="bold")
    ax.margins(y=0.12)

    ax.legend(loc="center left", fontsize=9)
    ax.set_title("Exponential search space, linear qubit count", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "scaling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------

def main():
    print("Generating figures ...")
    net = make_network(**FLAGSHIP)
    qubo = build_qubo(net)

    uni = best_uniform_baseline(qubo)
    opt = brute_force(qubo)
    fig_network_sinr(net, uni.config, opt.config)
    fig_pareto(net)
    fig_scaling()
    fig_depth_scaling()
    fig_transfer()

    # ansatz comparison -- reuse measured numbers so the figure matches the
    # benchmark exactly rather than re-running the (slow) simulations
    n_feasible = qubo.n_tilts ** qubo.n_sectors
    results_x = [
        {"p": 1, "prob_of_optimum": 0.0003, "valid_prob": 0.178},
        {"p": 2, "prob_of_optimum": 0.0018, "valid_prob": 0.392},
        {"p": 3, "prob_of_optimum": 0.0038, "valid_prob": 0.575},
        {"p": 5, "prob_of_optimum": 0.0037, "valid_prob": 0.595},
    ]
    results_xy = [
        {"p": 1, "prob_of_optimum": 0.2011, "valid_prob": 1.0},
        {"p": 2, "prob_of_optimum": 0.1159, "valid_prob": 1.0},
        {"p": 3, "prob_of_optimum": 0.1650, "valid_prob": 1.0},
        {"p": 5, "prob_of_optimum": 0.1611, "valid_prob": 1.0},
    ]
    fig_ansatz_compare(results_x, results_xy, n_feasible)
    print("done.")


if __name__ == "__main__":
    main()
