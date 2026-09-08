"""
netsim.py -- Multi-cell RAN simulator for antenna tilt optimisation.

Model follows 3GPP TR 38.901 (antenna pattern, UMa pathloss) and
TS 38.215 (SINR).  9 controllable sectors = 9 binary tilt decisions.

Everything is vectorised over (sector, user, time, fading-scenario).
"""

import numpy as np

# ----------------------------------------------------------------------
# Physical / system constants
# ----------------------------------------------------------------------
FC_GHZ      = 3.5           # carrier frequency
BW_HZ       = 20e6          # per-cell bandwidth
PTX_DBM     = 46.0          # per-sector transmit power
NF_DB       = 7.0           # UE noise figure
H_BS        = 25.0          # base-station height (m)
H_UT        = 1.5           # user height (m)

# TR 38.901 Table 7.3-1 antenna element pattern
G_MAX_DBI   = 8.0
THETA_3DB   = 65.0
PHI_3DB     = 65.0
SLA_V       = 30.0
A_MAX       = 30.0

SIGMA_SF_DB = 6.0           # shadow fading std (UMa NLOS)

NOISE_DBM   = -174.0 + 10 * np.log10(BW_HZ) + NF_DB

# KPI thresholds
GAMMA_MIN_DB   = 0.0       # coverage threshold
GAMMA_DROP_DB  = -6.0       # radio-link failure threshold
N_DROP_STEPS   = 3          # consecutive bad samples -> drop (proxy for N310/T310)
SE_MAX         = 7.4        # spectral-efficiency cap (256QAM r=0.93)
A3_OFFSET_DB   = 2.0
A3_HYST_DB     = 1.0
TTT_STEPS      = 2          # time-to-trigger, in samples
HOF_SINR_DB    = -6.0       # SINR below this during execution -> HO failure


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def build_network(isd=500.0, n_sites=3):
    """n_sites on a ring, 3 sectors each -> M = 3*n_sites tilt variables."""
    r = isd / np.sqrt(3.0) * (1.0 if n_sites <= 3 else n_sites / 3.0) ** 0.5
    ang = np.pi / 2 + np.arange(n_sites) * 2 * np.pi / n_sites
    site_xy = np.array([[r * np.cos(a), r * np.sin(a)] for a in ang])
    xy, bearing, site_of = [], [], []
    for s in range(n_sites):
        for k in range(3):
            xy.append(site_xy[s])
            bearing.append(120.0 * k)          # degrees, clockwise from +y
            site_of.append(s)
    return (np.array(xy), np.array(bearing),
            np.array(site_of), site_xy)


# ----------------------------------------------------------------------
# 3GPP TR 38.901 antenna element pattern (Table 7.3-1)
# ----------------------------------------------------------------------
N_V      = 8      # vertical elements per panel
D_V_LAM  = 0.5    # vertical element spacing / lambda


def antenna_gain_db(theta_deg, phi_deg, tilt_deg):
    """
    Composite panel gain = element pattern (TR 38.901 Table 7.3-1)
    x vertical array factor (TR 38.901 eq. 7.3-1, complex weights).

    Electrical downtilt is produced by the progressive phase across the
    N_V elements -- this is what narrows the vertical beam from the
    element's 65 deg to roughly 102/N_V deg, and is why a +-2 deg tilt
    step is operationally meaningful.

    theta_deg : zenith angle BS->UE, 90 = horizon, >90 = below horizon
    phi_deg   : azimuth relative to boresight, wrapped to [-180,180]
    tilt_deg  : electrical downtilt (positive = beam pointing down)
    """
    a_v = -np.minimum(12.0 * ((theta_deg - 90.0 - tilt_deg) / THETA_3DB) ** 2,
                      SLA_V)
    a_h = -np.minimum(12.0 * (phi_deg / PHI_3DB) ** 2, A_MAX)
    elem = G_MAX_DBI - np.minimum(-(a_v + a_h), A_MAX)

    # --- vertical array factor -----------------------------------------
    th = np.radians(theta_deg)
    th_t = np.radians(90.0 + tilt_deg)
    n = np.arange(N_V).reshape((-1,) + (1,) * np.ndim(theta_deg))
    psi = 2 * np.pi * D_V_LAM * n * (np.cos(th)[None] - np.cos(th_t))
    af = np.abs(np.exp(1j * psi).sum(axis=0)) ** 2 / N_V
    af_db = 10.0 * np.log10(np.maximum(af, 1e-9))

    return elem + np.maximum(af_db, -SLA_V)


def uma_nlos_pathloss_db(d3d_m):
    """3GPP TR 38.901 UMa NLOS (simplified, hUT-corrected)."""
    d3d = np.maximum(d3d_m, 10.0)
    return (13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(FC_GHZ)
            - 0.6 * (H_UT - 1.5))


# ----------------------------------------------------------------------
# Scenario: users, trajectories, fading
# ----------------------------------------------------------------------
class Scenario:
    """
    Holds everything that does NOT depend on the tilt decision, plus the
    per-tilt-option antenna gains.  Built once, reused for all 2^M configs.
    """

    def __init__(self, n_users=48, n_time=16, n_fade=12,
                 tilt_base=8.0, tilt_delta=2.0, speed_mps=8.0,
                 density=None, area_m=700.0, rng=None, n_sites=3,
                 isd=500.0):
        self.rng = np.random.default_rng(0 if rng is None else rng)
        self.U, self.T, self.R = n_users, n_time, n_fade
        self.tilt_base, self.tilt_delta = tilt_base, tilt_delta

        xy, bearing, site_of, site_xy = build_network(isd=isd,
                                                       n_sites=n_sites)
        self.M = len(xy)
        self.sec_xy, self.bearing, self.site_of = xy, bearing, site_of
        self.site_xy = site_xy

        self._place_users(density, area_m, speed_mps)
        self._precompute_gains()
        self._precompute_largescale()
        self._precompute_fading()

    # -- users -----------------------------------------------------------
    def _place_users(self, density, area_m, speed_mps):
        """Users drawn from a demand density field (or uniform), then moved."""
        U, T = self.U, self.T
        if density is None:
            p0 = self.rng.uniform(-area_m / 2, area_m / 2, size=(U, 2))
        else:
            p0 = sample_from_density(density, area_m, U, self.rng)

        # linear trajectories with random heading
        head = self.rng.uniform(0, 2 * np.pi, size=U)
        step = speed_mps * 1.0                      # 1 s per sample
        dxy = np.stack([np.cos(head), np.sin(head)], axis=1) * step
        t = np.arange(T)[None, :, None]
        self.pos = p0[:, None, :] + dxy[:, None, :] * t      # (U,T,2)

    # -- antenna gain for both tilt options -------------------------------
    def _precompute_gains(self):
        """gain[m,u,t,opt] -- opt 0 = tilt-up (-D), opt 1 = tilt-down (+D)."""
        M, U, T = self.M, self.U, self.T
        d = self.pos[None, :, :, :] - self.sec_xy[:, None, None, :]   # (M,U,T,2)
        d2d = np.linalg.norm(d, axis=-1)
        self.d2d = np.maximum(d2d, 10.0)
        self.d3d = np.sqrt(self.d2d ** 2 + (H_BS - H_UT) ** 2)

        # zenith angle: 90 deg = horizon, larger = below horizon
        theta = 90.0 + np.degrees(np.arctan2(H_BS - H_UT, self.d2d))

        az = np.degrees(np.arctan2(d[..., 0], d[..., 1]))     # from +y axis
        phi = (az - self.bearing[:, None, None] + 180.0) % 360.0 - 180.0

        self.gain = np.empty((M, U, T, 2))
        for opt, sgn in enumerate((-1.0, +1.0)):
            tilt = self.tilt_base + sgn * self.tilt_delta
            self.gain[..., opt] = antenna_gain_db(theta, phi, tilt)

    # -- pathloss + shadowing ---------------------------------------------
    def _precompute_largescale(self):
        pl = uma_nlos_pathloss_db(self.d3d)                    # (M,U,T)
        # shadow fading: per (sector,user), constant over the short window
        sf = self.rng.normal(0.0, SIGMA_SF_DB, size=(self.M, self.U, 1))
        self.base_db = PTX_DBM - pl + sf                       # (M,U,T)

    # -- time-correlated Rayleigh fading ----------------------------------
    def _precompute_fading(self):
        M, U, T, R = self.M, self.U, self.T, self.R
        alpha = 0.9                                            # AR(1) coeff
        g = np.empty((R, M, U, T), dtype=complex)
        w = (self.rng.normal(size=(R, M, U)) +
             1j * self.rng.normal(size=(R, M, U))) / np.sqrt(2)
        g[..., 0] = w
        for t in range(1, T):
            n = (self.rng.normal(size=(R, M, U)) +
                 1j * self.rng.normal(size=(R, M, U))) / np.sqrt(2)
            g[..., t] = alpha * g[..., t - 1] + np.sqrt(1 - alpha ** 2) * n
        self.fade_db = 10.0 * np.log10(np.maximum(np.abs(g) ** 2, 1e-12))

    # ------------------------------------------------------------------
    # Evaluate one tilt configuration
    # ------------------------------------------------------------------
    def evaluate(self, z):
        """
        z : array of +-1, length M.  +1 = tilt DOWN by delta, -1 = tilt UP.
        Returns dict of the four raw KPIs (higher = better for all four).
        """
        opt = ((np.asarray(z) + 1) // 2).astype(int)            # -1->0, +1->1
        g = np.take_along_axis(self.gain,
                               opt[:, None, None, None].repeat(self.U, 1)
                                  .repeat(self.T, 2), axis=3)[..., 0]  # (M,U,T)

        rsrp_db = self.base_db + g                              # (M,U,T) no fading
        rx_db = rsrp_db[None] + self.fade_db                    # (R,M,U,T)

        rx_lin = 10 ** (rx_db / 10.0)
        noise = 10 ** (NOISE_DBM / 10.0)

        # sticky attachment driven by Event A3 (TS 38.331 5.5.4.4)
        serv, ho_att = self._attach(rsrp_db)                    # (U,T), (U,T)

        total = rx_lin.sum(axis=1)                              # (R,U,T)
        sidx = serv[None, None, :, :]
        sig = np.take_along_axis(rx_lin, sidx, axis=1)[:, 0]    # (R,U,T)
        sinr = sig / (noise + total - sig)
        sinr_db = 10 * np.log10(np.maximum(sinr, 1e-12))

        # ---- KPI A: coverage probability -------------------------------
        k_cov = float(np.mean(sinr_db >= GAMMA_MIN_DB))

        # ---- KPI B: 5th-percentile user throughput ---------------------
        # equal share of cell bandwidth among users attached to that sector
        load = np.zeros((self.M, self.T))
        for m in range(self.M):
            load[m] = np.maximum((serv == m).sum(axis=0), 1)
        share = BW_HZ / load[serv, np.arange(self.T)[None, :]]  # (U,T)
        se = np.minimum(np.log2(1.0 + sinr), SE_MAX)            # (R,U,T)
        rate = se * share[None]                                 # bit/s
        user_rate = rate.mean(axis=2)                           # (R,U) time-avg
        k_thr = float(np.percentile(user_rate, 5) / 1e6)        # Mbit/s

        # ---- KPI C: drop / RLF rate ------------------------------------
        bad = sinr_db < GAMMA_DROP_DB                           # (R,U,T)
        run = np.zeros_like(bad, dtype=int)
        run[..., 0] = bad[..., 0]
        for t in range(1, self.T):
            run[..., t] = (run[..., t - 1] + 1) * bad[..., t]
        dropped = (run >= N_DROP_STEPS).any(axis=-1)            # (R,U)
        k_drop = 1.0 - float(dropped.mean())                    # success rate

        # ---- KPI D: handover success -----------------------------------
        k_ho = self._handover_kpi(sinr_db, ho_att)

        return dict(cov=k_cov, thr=k_thr, drop=k_drop, ho=k_ho,
                    sinr_db=sinr_db, serv=serv, ho_att=ho_att)

    def _attach(self, rsrp_db):
        """
        Sticky serving-cell selection.  The UE camps on its current cell and
        only hands over once Event A3 (neighbour offset-better than serving)
        has held for time-to-trigger -- TS 38.331 5.5.4.4.  Without this the
        serving cell is always the strongest and A3 can never fire.
        """
        U, T = self.U, self.T
        serv = np.empty((U, T), dtype=int)
        att = np.zeros((U, T), dtype=bool)
        cur = np.argmax(rsrp_db[:, :, 0], axis=0)               # initial camp
        run = np.zeros(U, dtype=int)

        for t in range(T):
            r_t = rsrp_db[:, :, t]                              # (M,U)
            s_r = r_t[cur, np.arange(U)]
            nb = r_t.copy()
            nb[cur, np.arange(U)] = -np.inf
            best = np.argmax(nb, axis=0)
            n_r = nb[best, np.arange(U)]

            a3 = n_r - A3_HYST_DB > s_r + A3_OFFSET_DB
            run = (run + 1) * a3
            fire = run >= TTT_STEPS
            cur = np.where(fire, best, cur)
            run = np.where(fire, 0, run)

            serv[:, t] = cur
            att[:, t] = fire
        return serv, att

    def _handover_kpi(self, sinr_db, ho_att):
        """Success rate over A3-triggered attempts; failure = poor SINR."""
        n_att = int(ho_att.sum())
        if n_att == 0:
            return 1.0
        fail = (sinr_db < HOF_SINR_DB) & ho_att[None]
        n_fail = float(fail.mean(axis=0).sum())                 # avg over fading
        return float(1.0 - n_fail / n_att)


# ----------------------------------------------------------------------
# Demand-density helper (links the fluid stage to the RAN stage)
# ----------------------------------------------------------------------
def sample_from_density(rho, area_m, n, rng):
    """Draw n user positions with probability proportional to rho (2-D grid)."""
    p = np.maximum(rho, 0).ravel()
    p = p / p.sum()
    idx = rng.choice(p.size, size=n, p=p)
    ny, nx = rho.shape
    iy, ix = np.divmod(idx, nx)
    cell = area_m / nx
    x = (ix + rng.random(n)) * cell - area_m / 2
    y = (iy + rng.random(n)) * cell - area_m / 2
    return np.stack([x, y], axis=1)
