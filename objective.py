"""
objective.py -- combine the four network KPIs into one scalar utility,
then into the energy function whose ground state we search for.

    raw KPIs -> min-max normalised scores in [0,1]
             -> weighted sum  F
             -> guardrail penalties            F_safe
             -> energy  E = -F_safe

The normalisation bounds and the equal weights are OUR design choices,
not standardised quantities.  Say so in the write-up.
"""

import numpy as np

KPIS = ("cov", "thr", "drop", "ho")
WEIGHTS = {k: 0.25 for k in KPIS}          # equal weight after normalisation
TAU = {k: 0.15 for k in KPIS}              # guardrail floor on each score
LAMBDA = {k: 2.0 for k in KPIS}            # guardrail penalty strength


class Objective:
    """Calibrates normalisation bounds, then scores any tilt configuration."""

    def __init__(self, scenario, n_calib=64, seed=7):
        self.sc = scenario
        rng = np.random.default_rng(seed)
        raw = []
        for _ in range(n_calib):
            z = rng.choice([-1, 1], scenario.M)
            r = scenario.evaluate(z)
            raw.append([r[k] for k in KPIS])
        raw = np.array(raw)
        # 5th / 95th percentile of a random-configuration sample.
        # Declared design choice: keeps the mapping well-defined at M=16+
        # where exhaustive enumeration is impossible.
        self.lo = np.percentile(raw, 5, axis=0)
        self.hi = np.percentile(raw, 95, axis=0)
        self.hi = np.where(self.hi - self.lo < 1e-9, self.lo + 1e-9, self.hi)

    def scores(self, raw_kpis):
        v = np.array([raw_kpis[k] for k in KPIS])
        return np.clip((v - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def utility(self, z, detail=False):
        r = self.sc.evaluate(z)
        s = self.scores(r)
        w = np.array([WEIGHTS[k] for k in KPIS])
        tau = np.array([TAU[k] for k in KPIS])
        lam = np.array([LAMBDA[k] for k in KPIS])

        F = float(w @ s)
        pen = float(np.sum(lam * np.maximum(0.0, tau - s) ** 2))
        F_safe = F - pen
        if detail:
            return F_safe, dict(raw=r, scores=s, F=F, penalty=pen)
        return F_safe

    def energy(self, z):
        """QAOA minimises, so energy is the negated utility."""
        return -self.utility(z)


# ----------------------------------------------------------------------
def index_to_spins(k, M):
    """basis index k -> spin vector z in {+1,-1}^M, with z_i = (-1)^bit_i."""
    bits = (k >> np.arange(M)) & 1
    return 1 - 2 * bits


def enumerate_energies(obj, M):
    """Full 2^M energy landscape.  Only feasible for small M (<= ~14)."""
    n = 1 << M
    E = np.empty(n)
    for k in range(n):
        E[k] = obj.energy(index_to_spins(k, M))
    return E
