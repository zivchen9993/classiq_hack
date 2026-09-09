"""
QUBO / Ising construction for antenna tilt optimization.

Takes the unary (coverage) and pairwise (interference) tables produced by
`rf_model.Network` and turns them into a Quadratic Unconstrained Binary
Optimization problem that a QAOA circuit can minimize.

Encoding
--------
One-hot per sector: binary variable x[i, t] = 1 iff sector i uses tilt level t.
  * number of qubits = n_sectors * n_tilt_levels
  * the pairwise interference term becomes a plain quadratic term
    x[i,ti] * x[j,tj]  -- no higher-order terms, so this is a true QUBO
  * validity (exactly one tilt per sector) is enforced by a penalty term
    P * (sum_t x[i,t] - 1)^2

Objective (minimized)
---------------------
    C(x) = -w_cov * sum_i  cov_i(t_i)          / cov_scale     (coverage, unary)
           +w_int * sum_ij ISR_ij(t_i, t_j)    / isr_scale     (interference, pairwise)
           +w_ho  * sum_ij HO_ij(t_i, t_j)     / ho_scale      (handover risk, pairwise)
           + P    * sum_i (sum_t x[i,t] - 1)^2                 (encoding constraint)

The first three terms are the physics; the last is the encoding constraint.
The interference term pushes sectors to tilt down, the handover term pushes
them to tilt up, and the coverage term wants whatever best serves each
sector's own demand -- the competition between them is what makes the
landscape frustrated and multi-modal.

All physics terms are normalized to O(1) per sector so the weights are
interpretable and the penalty P can be set safely above them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass
class TiltQUBO:
    """A QUBO instance for one network, plus the encode/decode/evaluate logic."""

    n_sectors: int
    n_tilts: int
    Q: np.ndarray                 # (n_vars, n_vars) QUBO matrix
    offset: float                 # constant energy offset
    penalty: float
    w_cov: float
    w_int: float
    w_ho: float
    cov_norm: np.ndarray          # normalized coverage table (n_sectors, n_tilts)
    int_norm: dict                # normalized interference tables, keyed by (i, j)
    ho_norm: dict                 # normalized handover-risk tables, keyed by (i, j)
    # Optional already-weighted pair tables used by formulations that have
    # interactions beyond the snapshot coverage/interference/handover split
    # (for example temporal switching edges).  Keeping them explicit avoids
    # hiding temporal coefficients inside one of the RF weights.
    extra_pairwise: dict = None
    objective_offset: float = 0.0

    def __post_init__(self):
        if self.extra_pairwise is None:
            self.extra_pairwise = {}

    # -- variable indexing -------------------------------------------------

    @property
    def n_vars(self) -> int:
        return self.n_sectors * self.n_tilts

    def var_index(self, sector: int, tilt: int) -> int:
        return sector * self.n_tilts + tilt

    # -- encode / decode ---------------------------------------------------

    def encode(self, tilt_indices) -> np.ndarray:
        """Tilt configuration -> one-hot bit vector."""
        x = np.zeros(self.n_vars, dtype=int)
        for i, t in enumerate(tilt_indices):
            x[self.var_index(i, int(t))] = 1
        return x

    def decode(self, x) -> np.ndarray:
        """Bit vector -> tilt configuration.

        Invalid one-hot blocks are repaired by argmax (a block of all zeros
        falls back to tilt 0). Repair is reported separately so we never
        silently hide constraint violations from the benchmark.
        """
        x = np.asarray(x, dtype=int).reshape(self.n_sectors, self.n_tilts)
        return np.argmax(x, axis=1)

    def is_valid(self, x) -> bool:
        x = np.asarray(x, dtype=int).reshape(self.n_sectors, self.n_tilts)
        return bool(np.all(x.sum(axis=1) == 1))

    # -- objective evaluation ----------------------------------------------

    def objective_from_config(self, tilt_indices) -> float:
        """The physics objective (no penalty term) for a tilt configuration.

        This is THE function every solver -- classical and quantum -- is
        compared on, so the comparison is apples-to-apples.
        """
        tilt_indices = np.asarray(tilt_indices, dtype=int)
        cov = sum(self.cov_norm[i, tilt_indices[i]] for i in range(self.n_sectors))
        isr = sum(table[tilt_indices[i], tilt_indices[j]]
                  for (i, j), table in self.int_norm.items())
        ho = sum(table[tilt_indices[i], tilt_indices[j]]
                 for (i, j), table in self.ho_norm.items())
        extra = sum(table[tilt_indices[i], tilt_indices[j]]
                    for (i, j), table in self.extra_pairwise.items())
        return float(self.objective_offset - self.w_cov * cov
                     + self.w_int * isr + self.w_ho * ho + extra)

    def energy(self, x) -> float:
        """Full QUBO energy of a bit vector, including the penalty term."""
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def penalty_violation(self, x) -> int:
        """How many sectors violate the one-hot constraint."""
        x = np.asarray(x, dtype=int).reshape(self.n_sectors, self.n_tilts)
        return int(np.sum(x.sum(axis=1) != 1))

    # -- export formats ----------------------------------------------------

    def to_ising(self):
        """Convert the QUBO to Ising form.

            H = const + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

        using the standard quantum convention x_i = (1 - z_i) / 2, i.e.
        bit 0 <-> |0> <-> z = +1 and bit 1 <-> |1> <-> z = -1.

        Substituting:
            x_i          = 1/2 - z_i/2
            x_i * x_j    = 1/4 - z_i/4 - z_j/4 + z_i z_j / 4

        Returns (h, J, const) with J strictly upper-triangular.
        """
        Q = self.Q
        n = self.n_vars
        J = np.zeros((n, n))
        h = np.zeros(n)
        const = float(self.offset)

        for i in range(n):
            for j in range(n):
                q = float(Q[i, j])
                if q == 0.0:
                    continue
                if i == j:
                    # q * x_i = q/2 - (q/2) z_i
                    const += q / 2.0
                    h[i] += -q / 2.0
                else:
                    # q * x_i x_j = q/4 - (q/4) z_i - (q/4) z_j + (q/4) z_i z_j
                    a, b = (i, j) if i < j else (j, i)
                    const += q / 4.0
                    h[i] += -q / 4.0
                    h[j] += -q / 4.0
                    J[a, b] += q / 4.0
        return h, J, const

    def ising_energy(self, x) -> float:
        """Energy computed via the Ising form -- used to verify to_ising()."""
        x = np.asarray(x, dtype=int)
        z = 1 - 2 * x                     # bit 0 -> +1, bit 1 -> -1
        h, J, const = self.to_ising()
        return float(const + h @ z + z @ np.triu(J, 1) @ z)

    def summary(self) -> str:
        nnz = int(np.count_nonzero(self.Q))
        return (f"TiltQUBO: {self.n_sectors} sectors x {self.n_tilts} tilts "
                f"= {self.n_vars} qubits | {nnz} non-zero QUBO entries | "
                f"penalty={self.penalty:.2f} "
                f"w_cov={self.w_cov} w_int={self.w_int} w_ho={self.w_ho}")


def _normalize_pair_tables(tables, n_sectors):
    """Scale a dict of pairwise tables to O(1) per sector.

    The tables arrive centred (zero mean), so they carry signed values and the
    meaningful magnitude is the largest absolute deviation, not the largest
    entry. Divides by that, then rescales by n_sectors / n_pairs so a denser
    network does not let the pairwise terms drown out the unary coverage term.
    """
    tables = {p: np.asarray(t, dtype=float) for p, t in tables.items()}
    if not tables:
        return {}
    scale = float(max(np.max(np.abs(t)) for t in tables.values())) or 1.0
    factor = n_sectors / max(len(tables), 1)
    return {p: (t / scale) * factor for p, t in tables.items()}


def build_qubo(network, w_cov: float = 0.4, w_int: float = 0.35, w_ho: float = 0.25,
               penalty_scale: float = 2.0) -> TiltQUBO:
    """Build the QUBO for a `rf_model.Network`.

    w_cov / w_int / w_ho trade coverage quality against interference
    suppression against handover reliability. penalty_scale sets the one-hot
    penalty as a multiple of the largest physics coefficient, so valid
    solutions are always energetically preferred over invalid ones.
    """
    n_sectors = len(network.sectors)
    n_tilts = len(network.tilt_levels_deg)

    # ---- normalize all physics terms to comparable O(1) scales -----------
    cov = np.asarray(network.coverage, dtype=float)
    # center per sector so the term rewards the *relative* gain of a tilt
    # choice rather than a constant offset that no choice can influence
    cov_centered = cov - cov.mean(axis=1, keepdims=True)
    cov_scale = float(np.max(np.abs(cov_centered))) or 1.0
    cov_norm = cov_centered / cov_scale

    int_norm = _normalize_pair_tables(network.interference, n_sectors)
    ho_norm = _normalize_pair_tables(network.handover, n_sectors)

    # ---- assemble the QUBO matrix ----------------------------------------
    n_vars = n_sectors * n_tilts
    Q = np.zeros((n_vars, n_vars))

    def idx(i, t):
        return i * n_tilts + t

    # unary coverage term: -w_cov * cov_norm[i, t] * x[i, t]
    for i in range(n_sectors):
        for t in range(n_tilts):
            Q[idx(i, t), idx(i, t)] += -w_cov * cov_norm[i, t]

    # pairwise terms: +w * table_ij(ti, tj) * x[i,ti] * x[j,tj]
    for weight, tables in ((w_int, int_norm), (w_ho, ho_norm)):
        for (i, j), table in tables.items():
            for ti in range(n_tilts):
                for tj in range(n_tilts):
                    Q[idx(i, ti), idx(j, tj)] += weight * table[ti, tj]

    physics_max = float(np.max(np.abs(Q))) or 1.0
    penalty = penalty_scale * max(physics_max * n_tilts, 1.0)

    # one-hot penalty: P * (sum_t x[i,t] - 1)^2
    #   = P * ( sum_t x[i,t]  +  2*sum_{t<t'} x[i,t]x[i,t']  - 2*sum_t x[i,t] + 1 )
    #   = P * ( -sum_t x[i,t] + 2*sum_{t<t'} x[i,t]x[i,t'] ) + P
    offset = 0.0
    for i in range(n_sectors):
        for t in range(n_tilts):
            Q[idx(i, t), idx(i, t)] += -penalty
        for t in range(n_tilts):
            for t2 in range(t + 1, n_tilts):
                Q[idx(i, t), idx(i, t2)] += 2.0 * penalty
        offset += penalty

    return TiltQUBO(
        n_sectors=n_sectors, n_tilts=n_tilts, Q=Q, offset=offset,
        penalty=penalty, w_cov=w_cov, w_int=w_int, w_ho=w_ho,
        cov_norm=cov_norm, int_norm=int_norm, ho_norm=ho_norm,
    )


if __name__ == "__main__":
    from rf_model import make_network

    net = make_network(3, seed=7)
    qubo = build_qubo(net)
    print(qubo.summary())

    # --- self-check: QUBO energy of a valid one-hot vector must equal the
    # --- physics objective (up to the constant penalty offset)
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        cfg = rng.integers(0, qubo.n_tilts, size=qubo.n_sectors)
        x = qubo.encode(cfg)
        err = abs(qubo.energy(x) - qubo.objective_from_config(cfg))
        max_err = max(max_err, err)
    print(f"max |QUBO energy - physics objective| over 200 valid configs: {max_err:.3e}")

    # --- self-check: the Ising form must reproduce the QUBO energy exactly,
    # --- including on INVALID bit vectors (the penalty term must survive
    # --- the transformation, otherwise QAOA optimizes the wrong Hamiltonian)
    max_ising_err = 0.0
    for _ in range(200):
        x = rng.integers(0, 2, size=qubo.n_vars)   # arbitrary, mostly invalid
        max_ising_err = max(max_ising_err, abs(qubo.energy(x) - qubo.ising_energy(x)))
    print(f"max |QUBO energy - Ising energy| over 200 arbitrary bitstrings: {max_ising_err:.3e}")

    # --- self-check: invalid vectors must be energetically penalized
    x_bad = qubo.encode(rng.integers(0, qubo.n_tilts, size=qubo.n_sectors))
    x_bad[0] = 1
    x_bad[1] = 1   # two tilts active on sector 0
    print(f"valid-config energy   : {qubo.energy(qubo.encode([1] * qubo.n_sectors)):.3f}")
    print(f"invalid-config energy : {qubo.energy(x_bad):.3f}  "
          f"(violations: {qubo.penalty_violation(x_bad)})")
