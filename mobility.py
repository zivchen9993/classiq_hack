"""
Time-varying traffic demand: the classical half of the dynamic problem.

The snapshot optimizer answers "what is the best tilt configuration for this
user distribution?". To ask the temporal question we need the user
distribution to *move*, so this module evolves the demand field over a
horizon and produces one network snapshot per timestep.

What moves and what does not
----------------------------
Towers do not move, so the whole link budget (`rx_dbm`, `rx_lin`) and the
fixed service-area assignment are computed ONCE and shared by every
timestep. Only the demand field changes, which changes:

    coverage[i, t]        -- demand-weighted mean SNR
    interference[(i,j)]   -- demand-weighted ISR
    handover[(i,j)]       -- demand-weighted edge-failure risk

i.e. exactly the three tables the QUBO consumes. That is why a snapshot is
cheap: it is a demand re-weighting of precomputed per-point quantities, not
a new propagation solve.

Mobility model
--------------
Demand is a uniform floor plus Gaussian hotspots (as in `rf_model`). Each
hotspot is given a velocity and a dispersal rate, so it translates and
spreads:

    x(t) = x0 + vx * t
    sigma(t) = sigma0 * (1 + dispersal * t)

This is the "commuters move from the residential cluster to the business
district" demo. It is deliberately smooth: the premise under test is that a
small demand change produces a small change in the optimal configuration,
and a smooth field is the most favourable case for that premise. If the
premise fails even here, it fails generally -- see `temporal.coherence_report`.

Amplitude is conserved per hotspot (no users are created or destroyed);
`Network._build_demand` normalizes the field to unit mean anyway, so only
the *shape* of the demand matters to the objective.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Hotspot trajectories
# ---------------------------------------------------------------------------

@dataclass
class Hotspot:
    """One moving cluster of traffic demand."""
    x: float
    y: float
    sigma: float
    amp: float
    vx: float = 0.0
    vy: float = 0.0
    dispersal: float = 0.0        # fractional sigma growth per timestep

    def at(self, t: float) -> tuple[float, float, float, float]:
        """(x, y, sigma, amp) at time t, in the tuple form Network expects."""
        return (
            self.x + self.vx * t,
            self.y + self.vy * t,
            max(self.sigma * (1.0 + self.dispersal * t), 1.0),
            self.amp,
        )


@dataclass
class MobilityModel:
    """A set of moving hotspots defining demand over a horizon."""
    hotspots: list[Hotspot] = field(default_factory=list)

    def hotspots_at(self, t: float) -> list[tuple[float, float, float, float]]:
        return [h.at(t) for h in self.hotspots]

    def __len__(self) -> int:
        return len(self.hotspots)


def make_commuter_mobility(net, n_steps: int, seed: int = 0,
                           speed_frac: float = 0.55,
                           dispersal: float = 0.05) -> MobilityModel:
    """Build a mobility model from a network's existing static hotspots.

    Each static hotspot is given a velocity aimed at a randomly chosen
    destination inside the grid, scaled so it traverses `speed_frac` of that
    distance over the whole horizon. `speed_frac` is the single knob that
    controls how hard the temporal problem is: 0 reproduces the static
    problem, large values make consecutive timesteps unrelated.
    """
    rng = np.random.default_rng(seed)
    extent = net.grid_extent_m
    horizon = max(n_steps - 1, 1)

    hotspots = []
    for (hx, hy, sigma, amp) in net.demand_hotspots:
        dest = rng.uniform(-extent * 0.7, extent * 0.7, size=2)
        vx = speed_frac * (dest[0] - hx) / horizon
        vy = speed_frac * (dest[1] - hy) / horizon
        hotspots.append(Hotspot(x=float(hx), y=float(hy), sigma=float(sigma),
                                amp=float(amp), vx=float(vx), vy=float(vy),
                                dispersal=dispersal))
    return MobilityModel(hotspots=hotspots)


# ---------------------------------------------------------------------------
# Snapshot sequence
# ---------------------------------------------------------------------------

def snapshot_network(base, hotspots):
    """A network identical to `base` but with a different demand field.

    Shallow-copies so the expensive, demand-independent arrays (`rx_dbm`,
    `rx_lin`, `service_mask`, `edge_mask`, `second_best`, `grid_xy`) are
    SHARED, not duplicated. `_build_demand` and `_build_tables` rebind
    `demand` / `coverage` / `interference` / `handover` to fresh objects
    rather than mutating in place, so the shared arrays are never touched.
    """
    snap = copy.copy(base)
    snap.demand_hotspots = list(hotspots)
    snap._build_demand()
    snap._build_tables()
    return snap


def evolve_network(base, mobility: MobilityModel, n_steps: int) -> list:
    """The sequence of network snapshots over the horizon."""
    return [snapshot_network(base, mobility.hotspots_at(t)) for t in range(n_steps)]


def demand_drift(snapshots) -> np.ndarray:
    """Normalized L1 distance between consecutive demand fields.

    Both fields have unit mean (Network normalizes), so this is directly
    comparable across instances: 0 means nothing moved, and 2 would mean the
    distributions have disjoint support.
    """
    drift = np.zeros(len(snapshots) - 1)
    for t in range(len(snapshots) - 1):
        a, b = snapshots[t].demand, snapshots[t + 1].demand
        drift[t] = float(np.abs(a - b).mean())
    return drift


if __name__ == "__main__":
    from rf_model import make_network

    net = make_network(3, seed=7)
    n_steps = 8
    mob = make_commuter_mobility(net, n_steps, seed=3)
    snaps = evolve_network(net, mob, n_steps)

    print(f"{len(mob)} hotspots over {n_steps} timesteps")
    print(f"snapshots share rx_dbm: {snaps[0].rx_dbm is net.rx_dbm}")
    print(f"snapshots have distinct demand: "
          f"{snaps[0].demand is not snaps[1].demand}")

    drift = demand_drift(snaps)
    print(f"\nper-step demand drift (mean |rho_t+1 - rho_t|):")
    print("  " + "  ".join(f"{d:.4f}" for d in drift))

    print("\nbest single tilt per snapshot (uniform config KPIs):")
    for t, s in enumerate(snaps):
        kpis = [s.evaluate_config([k] * len(s.sectors))["mean_sinr_db"]
                for k in range(len(s.tilt_levels_deg))]
        best = int(np.argmax(kpis))
        print(f"  t={t}: best uniform tilt {s.tilt_levels_deg[best]:+5.1f} deg "
              f"-> SINR {kpis[best]:6.2f} dB")
