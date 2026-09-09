"""plots.py -- figures for the deck, from landscape.npz + results.json."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ising import popcount

d = np.load("landscape.npz")
res = json.load(open("results.json"))

plt.rcParams.update({"font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 140})

# ----------------------------------------------------------------- fig 1
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
for a, key, ttl in zip(ax,
                       ("rho_now", "rho_pred", "rho_true"),
                       (r"demand $\rho(t)$",
                        r"LBM prediction $\hat\rho(t+\Delta t)$",
                        r"true $\rho(t+\Delta t)$")):
    im = a.imshow(d[key], origin="lower", cmap="magma")
    a.set_title(ttl)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.colorbar(im, ax=ax, fraction=0.025, label="user density")
fig.suptitle("Stage 1 — demand field evolved by lattice Boltzmann", y=1.02)
fig.savefig("fig1_demand.png", bbox_inches="tight")

# ----------------------------------------------------------------- fig 2
M = int(np.log2(d["E"].size))
Ehat = d["Ehat"]
idx = np.arange(1, 1 << M)
deg = np.array([popcount(int(k)) for k in idx])
p = Ehat[idx] ** 2
frac = np.array([p[deg == k].sum() for k in range(1, deg.max() + 1)]) / p.sum()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].bar(np.arange(1, len(frac) + 1), 100 * frac, color="#3b6ea5")
ax[0].set_xlabel("Walsh degree $|S|$")
ax[0].set_ylabel("% of decision-relevant energy")
ax[0].set_title(rf"Spectrum concentration  ($\kappa$={res['walsh']['kappa']:.3f})")
ax[0].set_yscale("log")

J = np.array(res["walsh"]["J"])
J = J + J.T
im = ax[1].imshow(J, cmap="coolwarm",
                  vmin=-np.abs(J).max(), vmax=np.abs(J).max())
ax[1].set_title("Ising couplings $J_{ij}$ between sectors")
ax[1].set_xlabel("sector"); ax[1].set_ylabel("sector"); ax[1].grid(False)
fig.colorbar(im, ax=ax[1], fraction=0.046)
fig.suptitle("Stage 3 — Walsh–Hadamard → Ising Hamiltonian", y=1.03)
fig.savefig("fig2_walsh.png", bbox_inches="tight")

# ----------------------------------------------------------------- fig 3
E, E2 = d["E"], d["E2"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].scatter(E, E2, s=4, alpha=0.35, color="#3b6ea5")
lim = [min(E.min(), E2.min()), max(E.max(), E2.max())]
ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("true energy $E(z)$")
ax[0].set_ylabel("2-local $E^{(2)}(z)$")
ax[0].set_title(f"Truncation fidelity  (rank ρ={res['walsh']['rank']:.3f})")

ps = sorted(int(k) for k in res["qaoa"])
ar = [res["qaoa"][str(k)]["ar"] for k in ps]
ax[1].plot(ps, ar, "o-", color="#c0504d")
ax[1].set_xlabel("QAOA depth $p$")
ax[1].set_ylabel("approximation ratio")
ax[1].set_title("QAOA convergence")
ax[1].set_xticks(ps)
ax[1].set_ylim(0, 1)
fig.suptitle("Stage 4 — quantum optimisation", y=1.03)
fig.savefig("fig3_qaoa.png", bbox_inches="tight")

# ----------------------------------------------------------------- fig 4
h = res["headline"]
names = ["static\ntilts", "reactive\n(demand now)",
         "proactive\n(LBM predicted)", "oracle\n(true future)"]
vals = [h["static"], h["reactive"], h["proactive"], h["oracle"]]
cols = ["#999999", "#c0504d", "#3b6ea5", "#4f8a4f"]

fig, ax = plt.subplots(figsize=(5.6, 3.4))
b = ax.bar(names, vals, color=cols)
ax.bar_label(b, fmt="%.3f", padding=2)
ax.set_ylabel("network utility $F_{safe}$")
ax.set_ylim(0, max(vals) * 1.18)
ax.set_title(f"Proactive tilting closes {100*h['recovered']:.0f}% "
             f"of the gap to oracle")
fig.savefig("fig4_headline.png", bbox_inches="tight")

print("wrote fig1_demand.png fig2_walsh.png fig3_qaoa.png fig4_headline.png")
