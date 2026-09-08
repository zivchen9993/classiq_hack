"""
Simplified RF / coverage model for the antenna tilt optimization proof-of-concept.

Models
------
  - 3GPP-style vertical (elevation) and horizontal antenna gain patterns.
  - Log-distance path loss.
  - Irregular, heterogeneous network geometry (jittered tower positions,
    varying tower heights and azimuths) -- real networks are not symmetric.
  - Non-uniform traffic demand (Gaussian hotspots) weighting every grid point.
  - A FIXED nominal service area per sector.
  - Three decomposed cost terms, matching the challenge brief's KPIs:
        unary    : coverage quality      (demand-weighted mean SNR)
        pairwise : inter-cell interference (demand-weighted interference-to-signal ratio)
        pairwise : handover failure risk  (cell-edge points with no viable handover target)
  - An exact joint SINR evaluator used for ground-truth KPI reporting.

Why this produces a HARD optimization problem
---------------------------------------------
The interference term pushes sectors to tilt DOWN (shrink the footprint, stop
overshooting into the neighbour). The handover term pushes them to tilt UP
(keep enough overlap at the cell edge so a handover target exists). The
coverage term wants the tilt that best matches each sector's own demand
distribution. These three pull in different directions, and because the
interference and handover terms are pairwise, the sectors are strongly
coupled: the best tilt for one sector depends on what its neighbours do.
That is textbook frustration -- exactly the structure that makes greedy /
local search get stuck, and exactly what a QUBO/Ising formulation captures.

The service area S_i is FIXED (it does not move with the tilt), which is what
makes the coverage term genuinely unary and the other two genuinely pairwise
-- i.e. representable as a true QUBO with no higher-order terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Antenna patterns (3GPP TR 36.814 / TR 38.901 style)
# ---------------------------------------------------------------------------

def vertical_gain_db(theta_deg, tilt_deg: float, beamwidth_deg: float = 10.0,
                     max_gain_db: float = 18.0, sla_v_db: float = 20.0):
    """Vertical pattern: A(theta) = -min(12*((theta - tilt)/theta_3dB)^2, SLA_v).

    theta_deg: depression angle from horizontal down to the point.
    tilt_deg:  electrical downtilt (positive = tilted down), +/-10 deg per the brief.
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    atten = np.minimum(12.0 * ((theta_deg - tilt_deg) / beamwidth_deg) ** 2, sla_v_db)
    return max_gain_db - atten


def horizontal_gain_db(rel_bearing_deg, beamwidth_deg: float = 65.0,
                       front_back_db: float = 25.0):
    """Horizontal pattern: -min(12*(phi/phi_3dB)^2, A_m)."""
    rel_bearing_deg = np.asarray(rel_bearing_deg, dtype=float)
    return -np.minimum(12.0 * (rel_bearing_deg / beamwidth_deg) ** 2, front_back_db)


def path_loss_db(distance_m, pl_exponent: float = 3.5,
                 ref_distance_m: float = 1.0, ref_loss_db: float = 40.0):
    """Log-distance path loss: PL(d) = PL(d0) + 10*n*log10(d/d0)."""
    distance_m = np.maximum(np.asarray(distance_m, dtype=float), ref_distance_m)
    return ref_loss_db + 10.0 * pl_exponent * np.log10(distance_m / ref_distance_m)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

@dataclass
class Sector:
    id: int
    tower_xy: tuple[float, float]
    azimuth_deg: float
    tower_height_m: float = 30.0
    tx_power_dbm: float = 43.0


@dataclass
class Network:
    sectors: list[Sector]
    tilt_levels_deg: list[float] = field(default_factory=lambda: [0.0, 3.0, 6.0, 10.0])
    grid_extent_m: float = 1400.0
    grid_points_per_axis: int = 45
    noise_dbm: float = -104.0
    demand_hotspots: list[tuple[float, float, float, float]] = field(default_factory=list)
    handover_margin_db: float = 9.0     # a neighbour within this margin is a viable HO target
    edge_fraction: float = 0.30         # weakest x% of a service area counts as "cell edge"
    coupling_prune_ratio: float = 0.02
    seed: int = 0

    def __post_init__(self):
        self._build_grid()
        self._build_demand()
        self._precompute_rx()
        self._assign_service_areas()
        self._build_tables()

    # -- grid & demand -----------------------------------------------------

    def _build_grid(self):
        xs = np.linspace(-self.grid_extent_m, self.grid_extent_m, self.grid_points_per_axis)
        ys = np.linspace(-self.grid_extent_m, self.grid_extent_m, self.grid_points_per_axis)
        gx, gy = np.meshgrid(xs, ys)
        self.grid_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)

    def _build_demand(self):
        """Traffic demand weight per grid point: a uniform floor plus Gaussian
        hotspots (dense urban clusters). Non-uniform demand is a major source
        of asymmetry -- different sectors want different tilts."""
        w = np.full(len(self.grid_xy), 0.25)
        for (hx, hy, sigma, amp) in self.demand_hotspots:
            d2 = (self.grid_xy[:, 0] - hx) ** 2 + (self.grid_xy[:, 1] - hy) ** 2
            w += amp * np.exp(-d2 / (2.0 * sigma ** 2))
        self.demand = w / w.mean()

    # -- link budget -------------------------------------------------------

    def _geometry(self, sector: Sector):
        dx = self.grid_xy[:, 0] - sector.tower_xy[0]
        dy = self.grid_xy[:, 1] - sector.tower_xy[1]
        horiz_dist = np.maximum(np.hypot(dx, dy), 10.0)
        depression_deg = np.degrees(np.arctan2(sector.tower_height_m, horiz_dist))
        slant_dist = np.hypot(horiz_dist, sector.tower_height_m)
        bearing = np.degrees(np.arctan2(dy, dx))
        rel_bearing = (bearing - sector.azimuth_deg + 180.0) % 360.0 - 180.0
        return depression_deg, slant_dist, rel_bearing

    def _precompute_rx(self):
        """rx_dbm[i, t, p] = power at grid point p from sector i at tilt level t."""
        n_s, n_t, n_p = len(self.sectors), len(self.tilt_levels_deg), len(self.grid_xy)
        self.rx_dbm = np.zeros((n_s, n_t, n_p))

        for i, sector in enumerate(self.sectors):
            depression_deg, dist_m, rel_bearing = self._geometry(sector)
            h_gain = horizontal_gain_db(rel_bearing)
            pl = path_loss_db(dist_m)
            for t, tilt in enumerate(self.tilt_levels_deg):
                v_gain = vertical_gain_db(depression_deg, tilt)
                self.rx_dbm[i, t, :] = sector.tx_power_dbm + v_gain + h_gain - pl

        self.rx_lin = 10 ** (self.rx_dbm / 10.0)
        self.noise_lin = 10 ** (self.noise_dbm / 10.0)

    # -- fixed service areas ----------------------------------------------

    def _assign_service_areas(self):
        """Assign each grid point to one serving sector using a tilt-independent
        reference (all sectors at mid tilt). Also identify, for each sector,
        its cell-edge points and its dominant neighbour at those points."""
        ref_t = len(self.tilt_levels_deg) // 2
        ref_rx = self.rx_dbm[:, ref_t, :]
        best_sector = np.argmax(ref_rx, axis=0)
        best_power = np.max(ref_rx, axis=0)
        served = best_power > self.noise_dbm

        n_s = len(self.sectors)
        self.service_mask = np.zeros((n_s, len(self.grid_xy)), dtype=bool)
        for i in range(n_s):
            self.service_mask[i] = served & (best_sector == i)

        empty = [i for i in range(n_s) if not self.service_mask[i].any()]
        if empty:
            raise ValueError(f"sectors {empty} have an empty service area; check geometry")

        # cell-edge points: the weakest `edge_fraction` of each service area
        # (at the reference tilt), where handover happens
        self.edge_mask = np.zeros_like(self.service_mask)
        for i in range(n_s):
            area_idx = np.flatnonzero(self.service_mask[i])
            powers = ref_rx[i, area_idx]
            k = max(int(np.ceil(self.edge_fraction * area_idx.size)), 1)
            weakest = area_idx[np.argsort(powers)[:k]]
            self.edge_mask[i, weakest] = True

        # for each sector, which neighbour is the best handover candidate at
        # each of its edge points (2nd strongest at reference tilt)
        ref_sorted = np.argsort(-ref_rx, axis=0)
        self.second_best = np.where(ref_sorted[0] == best_sector, ref_sorted[1], ref_sorted[0])

    # -- decomposed cost tables (what the QUBO consumes) -------------------

    def _build_tables(self):
        n_s, n_t = len(self.sectors), len(self.tilt_levels_deg)

        # ---- UNARY: demand-weighted mean SNR over the fixed service area ----
        self.coverage = np.zeros((n_s, n_t))
        for i in range(n_s):
            area = self.service_mask[i]
            w = self.demand[area]
            for t in range(n_t):
                snr_db = self.rx_dbm[i, t, area] - self.noise_dbm
                self.coverage[i, t] = float(np.average(snr_db, weights=w))

        # ---- PAIRWISE 1: pairwise SINR degradation -------------------------
        # For sector i's service area, considering ONLY j as an interferer:
        #     SINR_ij = P_i(ti) / (P_j(tj) + N)
        # The penalty is the negated demand-weighted mean of that in dB,
        # symmetrised over both directions. Pushes sectors to tilt DOWN.
        #
        # Working in dB (not as a linear power ratio) matters: the linear
        # ratio P_j/P_i is heavy-tailed and its mean is dominated by a handful
        # of deeply faded points, which decorrelates it from the dB-domain
        # SINR the network is actually judged on. Measured rank correlation
        # against the exact joint-SINR simulator improved from -0.10 (linear
        # ratio) to -0.68 (this form) -- see `benchmark.py` part 1.
        full_int = {}
        for i in range(n_s):
            for j in range(i + 1, n_s):
                table = np.zeros((n_t, n_t))
                area_i, area_j = self.service_mask[i], self.service_mask[j]
                wi, wj = self.demand[area_i], self.demand[area_j]
                for ti in range(n_t):
                    for tj in range(n_t):
                        sinr_i = 10 * np.log10(
                            self.rx_lin[i, ti, area_i] / (self.rx_lin[j, tj, area_i] + self.noise_lin))
                        sinr_j = 10 * np.log10(
                            self.rx_lin[j, tj, area_j] / (self.rx_lin[i, ti, area_j] + self.noise_lin))
                        table[ti, tj] = -(np.average(sinr_i, weights=wi)
                                          + np.average(sinr_j, weights=wj))
                # centre the table: only the tilt-DEPENDENT variation matters,
                # since exactly one (ti,tj) entry is selected per pair, so a
                # constant offset just shifts every configuration equally.
                # Removing it keeps the QUBO coefficients well conditioned.
                full_int[(i, j)] = table - table.mean()

        # ---- PAIRWISE 2: handover failure risk -----------------------------
        # At sector i's cell-edge points whose natural handover target is j,
        # a handover needs P_j within `handover_margin_db` of P_i. If sector j
        # tilts too far down it vanishes from i's edge and the handover fails.
        # Pushes sectors to tilt UP -- directly opposing the interference term.
        full_ho = {}
        for i in range(n_s):
            for j in range(i + 1, n_s):
                table = np.zeros((n_t, n_t))
                # edge points of i whose handover target is j, and vice versa
                edge_ij = np.flatnonzero(self.edge_mask[i] & (self.second_best == j))
                edge_ji = np.flatnonzero(self.edge_mask[j] & (self.second_best == i))
                if edge_ij.size == 0 and edge_ji.size == 0:
                    continue
                for ti in range(n_t):
                    for tj in range(n_t):
                        risk = 0.0
                        if edge_ij.size:
                            margin = self.rx_dbm[j, tj, edge_ij] - self.rx_dbm[i, ti, edge_ij]
                            fail = margin < -self.handover_margin_db
                            risk += float(np.average(fail, weights=self.demand[edge_ij]))
                        if edge_ji.size:
                            margin = self.rx_dbm[i, ti, edge_ji] - self.rx_dbm[j, tj, edge_ji]
                            fail = margin < -self.handover_margin_db
                            risk += float(np.average(fail, weights=self.demand[edge_ji]))
                        table[ti, tj] = risk
                full_ho[(i, j)] = table - table.mean()

        # ---- prune negligible couplings to keep the QUBO sparse -------------
        # Tables are centred, so "strength" is the spread of a coupling, i.e.
        # how much the pair's tilt choice actually changes the objective.
        def spread(t):
            return float(np.max(t) - np.min(t))

        int_strongest = max((spread(t) for t in full_int.values()), default=1.0)
        int_thresh = self.coupling_prune_ratio * int_strongest
        self.interference = {p: t for p, t in full_int.items() if spread(t) >= int_thresh}
        self.handover = {p: t for p, t in full_ho.items() if spread(t) > 1e-12}

        self.neighbor_pairs = sorted(set(self.interference) | set(self.handover))

    # -- exact joint evaluation (ground-truth KPIs) ------------------------

    def evaluate_config(self, tilt_indices) -> dict:
        """Exact joint evaluation of a full tilt configuration.

        Every sector transmits simultaneously at its chosen tilt. For each grid
        point, the serving sector's power is signal and all others are
        interference. Demand-weighted. This does NOT use the unary/pairwise
        decomposition, so it independently validates the optimizers' output.
        """
        tilt_indices = np.asarray(tilt_indices, dtype=int)
        n_s = len(self.sectors)

        p_lin = np.stack([self.rx_lin[i, tilt_indices[i], :] for i in range(n_s)])
        p_db = np.stack([self.rx_dbm[i, tilt_indices[i], :] for i in range(n_s)])
        total_lin = p_lin.sum(axis=0)

        sinr_parts, weight_parts, ho_fail, ho_weight = [], [], [], []
        for i in range(n_s):
            area = self.service_mask[i]
            signal = p_lin[i, area]
            interference = total_lin[area] - signal
            sinr_parts.append(signal / (interference + self.noise_lin))
            weight_parts.append(self.demand[area])

            # handover: at i's edge points, is the natural target still reachable?
            edge_idx = np.flatnonzero(self.edge_mask[i])
            if edge_idx.size:
                targets = self.second_best[edge_idx]
                margin = p_db[targets, edge_idx] - p_db[i, edge_idx]
                ho_fail.append(margin < -self.handover_margin_db)
                ho_weight.append(self.demand[edge_idx])

        sinr_lin = np.concatenate(sinr_parts)
        weights = np.concatenate(weight_parts)
        sinr_db = 10.0 * np.log10(sinr_lin)
        spectral_eff = np.minimum(np.log2(1.0 + sinr_lin), 7.0)

        ho_fail = np.concatenate(ho_fail) if ho_fail else np.zeros(1, dtype=bool)
        ho_weight = np.concatenate(ho_weight) if ho_weight else np.ones(1)

        outage_threshold_db = 0.0
        covered = sinr_db >= outage_threshold_db

        return {
            "mean_sinr_db": float(np.average(sinr_db, weights=weights)),
            "edge_sinr_db": float(np.percentile(sinr_db, 5.0)),
            "mean_spectral_efficiency": float(np.average(spectral_eff, weights=weights)),
            "outage_pct": float(100.0 * np.average(~covered, weights=weights)),
            "coverage_pct": float(100.0 * np.average(covered, weights=weights)),
            "handover_failure_pct": float(100.0 * np.average(ho_fail, weights=ho_weight)),
            "n_points": int(sinr_db.size),
        }


# ---------------------------------------------------------------------------
# Network builders
# ---------------------------------------------------------------------------

def make_network(n_towers: int, tilt_levels_deg=None, seed: int = 0,
                 tower_spacing_m: float = 700.0, n_hotspots: int = 3,
                 **kwargs) -> Network:
    """Build an irregular, heterogeneous synthetic network.

    n_towers towers roughly on a ring but with position jitter, varying
    heights, varying azimuth orientation and varying transmit power, plus
    randomly placed traffic hotspots. Each tower carries 3 sectors.
    """
    if tilt_levels_deg is None:
        tilt_levels_deg = [0.0, 3.0, 6.0, 10.0]
    rng = np.random.default_rng(seed)

    if n_towers == 1:
        centers = [(0.0, 0.0)]
    else:
        radius = tower_spacing_m / (2 * np.sin(np.pi / n_towers)) if n_towers > 2 \
            else tower_spacing_m / 2
        centers = []
        for k in range(n_towers):
            angle = 2 * np.pi * k / n_towers
            jitter_r = rng.uniform(-0.22, 0.22) * tower_spacing_m
            jitter_a = rng.uniform(-0.18, 0.18)
            r = radius + jitter_r
            centers.append((r * np.cos(angle + jitter_a), r * np.sin(angle + jitter_a)))

    sectors, sid = [], 0
    for (tx, ty) in centers:
        height = float(rng.uniform(22.0, 45.0))
        az_offset = float(rng.uniform(0.0, 120.0))
        power = float(rng.uniform(40.0, 46.0))
        for k in range(3):
            sectors.append(Sector(
                id=sid, tower_xy=(float(tx), float(ty)),
                azimuth_deg=(az_offset + 120.0 * k) % 360.0,
                tower_height_m=height, tx_power_dbm=power,
            ))
            sid += 1

    extent = kwargs.pop("grid_extent_m", max(1400.0, tower_spacing_m * 2.0))
    hotspots = []
    for _ in range(n_hotspots):
        hx, hy = rng.uniform(-extent * 0.7, extent * 0.7, size=2)
        sigma = float(rng.uniform(180.0, 400.0))
        amp = float(rng.uniform(1.5, 4.0))
        hotspots.append((float(hx), float(hy), sigma, amp))

    return Network(sectors=sectors, tilt_levels_deg=tilt_levels_deg,
                   grid_extent_m=extent, demand_hotspots=hotspots, seed=seed, **kwargs)


# backwards-compatible alias
make_hex_like_network = make_network


if __name__ == "__main__":
    net = make_network(3, seed=7)
    print(f"Sectors: {len(net.sectors)}   tilt levels: {net.tilt_levels_deg}")
    print(f"Coupled pairs: {len(net.neighbor_pairs)} "
          f"(interference: {len(net.interference)}, handover: {len(net.handover)})")

    print("\nCoverage table (demand-weighted mean SNR dB, sector x tilt):")
    print(np.round(net.coverage, 2))
    print("argmax tilt per sector:", np.argmax(net.coverage, axis=1))

    pair = sorted(net.handover)[0]
    print(f"\nInterference table {pair} (ISR):")
    print(np.round(net.interference[pair], 4))
    print(f"Handover-risk table {pair}:")
    print(np.round(net.handover[pair], 3))

    print("\nExact KPIs for uniform configurations:")
    for t, tilt in enumerate(net.tilt_levels_deg):
        k = net.evaluate_config([t] * len(net.sectors))
        print(f"  all at {tilt:+5.1f} deg: SINR {k['mean_sinr_db']:6.2f} dB | "
              f"edge {k['edge_sinr_db']:6.2f} | outage {k['outage_pct']:5.1f}% | "
              f"HO fail {k['handover_failure_pct']:5.1f}% | "
              f"spec.eff {k['mean_spectral_efficiency']:.3f}")
