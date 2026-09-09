"""
Digitized counterdiabatic quantum optimization (DCQO / BF-DCQO), in the
feasible one-hot subspace.

Why a second quantum solver
---------------------------
`qaoa_subspace.py` is variational: almost all of its cost is the classical
outer loop (COBYLA) searching for good (gamma, beta). On the 48-qubit
temporal instance that loop took 388 s and still missed the optimum, while
exact DP took 0.2 ms. DCQO removes the outer loop entirely -- the circuit
parameters come from a fixed annealing schedule plus a variationally
determined counterdiabatic coefficient computed in closed form from five
traces. Nothing is fitted, so the resource count is just
`n_steps` circuit layers x `n_shots` measurements.

That is the comparison this module exists to make. It is not a claim that
DCQO beats DP on a chain-structured problem -- it does not, and
`dcqo_benchmark.py` says so in the same words `temporal_benchmark.py` uses.

The physics
-----------
Interpolate between an easy driver and the cost Hamiltonian,

    H_ad(lam) = (1 - lam) * H_i  +  lam * H_f ,      lam: 0 -> 1

and add the first-order counterdiabatic (CD) term that suppresses the
diabatic transitions a fast sweep would otherwise cause:

    H_cd(t) = H_ad(lam(t))  +  lam_dot(t) * A_lam ,
    A_lam   = alpha(lam) * X ,      X = i [H_d, H_f] .

Two facts make this cheap here:

1. For a *two-point* interpolation, `[H_ad, d_lam H_ad] = [H_i, H_f]`
   identically -- the commutator does not depend on lam. So `X` is built
   once. (Derivation: with `H_ad = (1-lam)H_i + lam*H_f` and
   `d_lam H_ad = H_f - H_i`, the cross terms give
   `(1-lam)[H_i,H_f] + lam[H_i,H_f]`.)
2. `Y(lam) = i[X, H_ad] = (1-lam) P + lam Q` is *linear* in lam, so the
   variational coefficient

       alpha(lam) = -Tr(D Y) / Tr(Y^2),    D = d_lam H_ad

   collapses onto five lam-independent traces (see `CdCoefficient`). They
   are estimated once, by Hutchinson probes, and reused at every step.

The subspace
------------
Exactly the reduction `qaoa_subspace.py` uses. The state lives on the `T^S`
one-hot-feasible configurations rather than `2^(S*T)` bitstrings, because
every operator below conserves the excitation number in each sector's block:

  * `H_f` is diagonal -- it is `subspace_cost(qubo)`.
  * The driver is XY *ring* hopping inside each block. On a block's
    single-excitation subspace, `sum_k (X_k X_k+1 + Y_k Y_k+1)/2` acts as the
    T x T hopping matrix `B` on the tilt index, so `H_d = -g * sum_i B^(i)`.
    A ring (not a chain) is used because then the uniform superposition is
    *exactly* the driver ground state -- the same initial state QAOA-XY
    starts from, which keeps the two solvers comparable.
  * `X = i[H_d, H_f]` inherits that structure: it connects two
    configurations only if they differ in one sector by one ring step, and
    its matrix element is `i * (cost difference)`. Restricted to one such
    pair it is `Delta * sigma_y`, so `exp(-i theta X_edge)` is an exact real
    2 x 2 rotation -- see `SubspaceOperators.exp_cd`.

Bias field (the BF in BF-DCQO)
------------------------------
After a run, measure the one-hot marginals `p_i(k)` (the probability that
sector i sits at tilt k) and fold them back in as a longitudinal bias

    u_i(k) = -strength * scale * (p_i(k) - 1/T)

added to the diagonal of both `H_i` and `H_f`. This is the one-hot analogue
of BF-DCQO's `h_b ~ <Z_j>` update. Two consequences worth being explicit
about:

  * The initial state is then the *biased* driver ground state, computed
    exactly: with a unary bias, `H_i = sum_i (-gB + diag(u_i))^(i)` is a sum
    of commuting single-axis operators, so its ground state is a product of
    per-axis ground vectors.
  * The bias changes the Hamiltonian being annealed, so every sampled
    configuration is scored on the ORIGINAL, unbiased objective. Never the
    biased one. `test_dcqo.py` asserts this.

For the temporal problem the same machinery warm-starts across *physical*
time: `bias_from_config` turns the previous timestep's solution into `u`, at
zero extra qubits, which is design decision 4 of
`background/NEXT_STEPS.md`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import expm, eigh

from qaoa_subspace import subspace_cost


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------

def _apply_axis_matrix(psi: np.ndarray, M: np.ndarray, axis: int) -> np.ndarray:
    """Apply the T x T matrix `M` along one axis of the configuration tensor."""
    psi = np.moveaxis(psi, axis, 0)
    shape = psi.shape
    out = (M @ psi.reshape(shape[0], -1)).reshape(shape)
    return np.moveaxis(out, 0, axis)


def ring_edges(T: int, topology: str = "ring") -> list[tuple[int, int]]:
    """Hopping edges of one sector's tilt index.

    'ring' closes the cycle, which is what makes the uniform superposition an
    exact driver eigenstate; 'chain' is offered for ablation. For T <= 2 the
    two coincide (a ring on 2 nodes would double-count the single edge).
    """
    if T < 2:
        return []
    edges = [(k, k + 1) for k in range(T - 1)]
    if topology == "ring" and T > 2:
        edges.append((T - 1, 0))
    elif topology not in ("ring", "chain"):
        raise ValueError(f"unknown topology {topology!r}")
    return edges


def hop_matrix(T: int, topology: str = "ring") -> np.ndarray:
    """The T x T adjacency (hopping) matrix `B` induced on the tilt index."""
    B = np.zeros((T, T))
    for (a, b) in ring_edges(T, topology):
        B[a, b] = B[b, a] = 1.0
    return B


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

class AnnealingSchedule:
    """lam(s) and its derivative, plus the digitized step list.

    Default `lam(s) = sin^2[(pi/2) sin^2(pi s / 2)]`, the smooth schedule used
    throughout the DCQO literature: `lam(0)=0`, `lam(1)=1` and `lam'` vanishes
    at both ends, so no CD term is required at the boundaries.

    Guards the diagonal-commuting trap flagged in
    `background/NEXT_STEPS.md`: if the driver amplitude `1 - lam(s)` were zero
    across the whole interior, `H(s)` would be diagonal for every s, its
    eigenvectors would be computational basis states, and the evolution could
    not move amplitude between configurations at all. `assert_has_driver`
    refuses such a schedule instead of silently producing a flat result.
    """

    def __init__(self, n_steps: int, total_time: float, form: str = "sin2"):
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if total_time <= 0:
            raise ValueError("total_time must be > 0")
        self.n_steps = int(n_steps)
        self.total_time = float(total_time)
        self.form = form
        self.assert_has_driver()

    # -- lam and its derivative -------------------------------------------

    def lam(self, s):
        s = np.asarray(s, dtype=float)
        if self.form == "sin2":
            return np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * s) ** 2) ** 2
        if self.form == "linear":
            return s
        raise ValueError(f"unknown schedule form {self.form!r}")

    def dlam_ds(self, s):
        s = np.asarray(s, dtype=float)
        if self.form == "sin2":
            u = 0.5 * np.pi * np.sin(0.5 * np.pi * s) ** 2
            return (np.pi ** 2 / 4.0) * np.sin(np.pi * s) * np.sin(2.0 * u)
        if self.form == "linear":
            return np.ones_like(s)
        raise ValueError(f"unknown schedule form {self.form!r}")

    # -- the digitized steps ----------------------------------------------

    def midpoints(self) -> np.ndarray:
        """Midpoint of each of the `n_steps` equal intervals in s."""
        return (np.arange(self.n_steps) + 0.5) / self.n_steps

    def steps(self):
        """Yield (s, lam, lam_dot, dt) per digitized step.

        `lam_dot = lam'(s) / total_time` because s = t / total_time. Note the
        CD *angle* per step, `dt * lam_dot * alpha = lam'(s) * alpha / n_steps`,
        is independent of `total_time`: the total counterdiabatic rotation is
        `integral alpha dlam`, a fixed quantity. The adiabatic angles, by
        contrast, grow with `total_time` -- which is the precise sense in
        which the CD term becomes negligible as `total_time -> infinity`.
        """
        dt = self.total_time / self.n_steps
        for s in self.midpoints():
            yield float(s), float(self.lam(s)), float(self.dlam_ds(s) / self.total_time), dt

    def assert_has_driver(self, tol: float = 1e-9) -> None:
        s = self.midpoints()
        amp = 1.0 - self.lam(s)
        if float(np.max(amp)) <= tol:
            raise ValueError(
                "schedule has no driver amplitude anywhere in the interior: "
                "H(s) would be diagonal for all s and the evolution could not "
                "move amplitude between configurations (the diagonal-commuting "
                "trap). Check lam(s).")

    def summary(self) -> str:
        return (f"AnnealingSchedule({self.form}): n_steps={self.n_steps} "
                f"total_time={self.total_time:.4g} "
                f"lam(0)={float(self.lam(0.0)):.3f} lam(1)={float(self.lam(1.0)):.3f}")


# ---------------------------------------------------------------------------
# Operators on the feasible subspace
# ---------------------------------------------------------------------------

class SubspaceOperators:
    """`H_i`, `H_f`, `X = i[H_d,H_f]` and their exponentials, on the `T^S` subspace.

    All of them are stored implicitly and applied as tensor operations; no
    matrix of dimension `T^S` is ever formed.

    Parameters
    ----------
    cost : (T,)*S ndarray
        The *unbiased* cost tensor, i.e. `subspace_cost(qubo)`.
    unary_bias : (S, T) ndarray or None
        Longitudinal bias field `u_i(k)`. Added to the diagonal of both `H_i`
        and `H_f`. None means a cold start.
    driver_strength : float or 'auto'
        `g` in `H_d = -g * sum_i B^(i)`. 'auto' matches the driver's spectral
        half-width (`2 g S` for a ring) to the cost's, so neither term
        dominates the crossover on an arbitrary instance.
    """

    MAX_CACHE_BYTES = 1 << 30          # 1 GiB ceiling on the Delta cache

    def __init__(self, cost: np.ndarray, unary_bias=None, topology: str = "ring",
                 driver_strength="auto", cache_deltas: bool = True):
        self.cost = np.asarray(cost, dtype=float)
        self.S = self.cost.ndim
        self.T = self.cost.shape[0]
        if any(d != self.T for d in self.cost.shape):
            raise ValueError("cost tensor must be hypercubic, shape (T,)*S")
        self.topology = topology
        self.edges = ring_edges(self.T, topology)
        self.B = hop_matrix(self.T, topology)

        spread = float(self.cost.max() - self.cost.min())
        self.cost_spread = spread
        if driver_strength == "auto":
            # match the driver's spectral half-width to the cost's:
            # `-g sum_i B^(i)` spans +-g*S*max|eig(B)|, so g = spread/(2 S max|eig B|)
            b_max = float(np.abs(np.linalg.eigvalsh(self.B)).max()) if self.T > 1 else 1.0
            denom = self.S * max(b_max, 1e-12)
            self.g = max((spread / 2.0) / denom, 1e-12) if spread > 0 else 1.0
        else:
            self.g = float(driver_strength)

        self.set_bias(unary_bias)

        self._cache_deltas = bool(cache_deltas)
        self._delta_cache: dict = {}
        self._axis_ops_cache: dict = {}

    # -- bias --------------------------------------------------------------

    def set_bias(self, unary_bias) -> None:
        """Install a unary bias field and refresh the biased cost tensor."""
        if unary_bias is None:
            self.unary_bias = np.zeros((self.S, self.T))
        else:
            self.unary_bias = np.asarray(unary_bias, dtype=float).reshape(self.S, self.T)
        self.bias_diag = np.zeros_like(self.cost)
        for i in range(self.S):
            shape = [1] * self.S
            shape[i] = self.T
            self.bias_diag = self.bias_diag + self.unary_bias[i].reshape(shape)
        self.cost_biased = self.cost + self.bias_diag
        self._delta_cache = {}
        self._axis_ops_cache = {}

    def axis_operator(self, i: int) -> np.ndarray:
        """`A_i = -g B + diag(u_i)`, the single-axis block of `H_i`.

        `H_i = sum_i A_i^(i)` is a sum of operators acting on distinct axes,
        so they commute -- which is why the initial ground state is an exact
        product state and `exp(-i theta H_i)` factorizes exactly.
        """
        return -self.g * self.B + np.diag(self.unary_bias[i])

    # -- bare operator applications (used for the trace estimates) ---------

    def apply_h_init(self, psi: np.ndarray) -> np.ndarray:
        """`H_i psi`."""
        out = np.zeros_like(psi)
        for i in range(self.S):
            out += _apply_axis_matrix(psi, self.axis_operator(i).astype(psi.dtype), i)
        return out

    def apply_h_cost(self, psi: np.ndarray) -> np.ndarray:
        """`H_f psi` (diagonal, biased)."""
        return self.cost_biased * psi

    def apply_dlam(self, psi: np.ndarray) -> np.ndarray:
        """`D psi` with `D = d_lam H_ad = H_f - H_i`."""
        return self.apply_h_cost(psi) - self.apply_h_init(psi)

    def _delta(self, i: int, a: int, b: int) -> np.ndarray:
        """`cost_biased(...b...) - cost_biased(...a...)` along axis i."""
        key = (i, a, b)
        cached = self._delta_cache.get(key)
        if cached is not None:
            return cached
        d = np.take(self.cost_biased, b, axis=i) - np.take(self.cost_biased, a, axis=i)
        if self._cache_deltas:
            n_entries = self.S * len(self.edges)
            if d.nbytes * n_entries <= self.MAX_CACHE_BYTES:
                self._delta_cache[key] = d
        return d

    def apply_cd(self, psi: np.ndarray) -> np.ndarray:
        """`X psi` with `X = i[H_d, H_f]`.

        Matrix elements: `H_d` connects configurations differing in one sector
        by one ring step, with weight `-g`, and `H_f` is diagonal, so

            X[c', c] = i * H_d[c',c] * (f(c) - f(c'))
                     = i * (-g) * (f(c) - f(c'))  =  i * g * Delta ,

        with `Delta = f(c') - f(c)`. Hermitian, as `i` times a commutator of
        Hermitian operators must be.
        """
        out = np.zeros_like(psi)
        for i in range(self.S):
            for (a, b) in self.edges:
                d = self._delta(i, a, b)                 # f(b) - f(a) along axis i
                sl_a = np.take(psi, a, axis=i)
                sl_b = np.take(psi, b, axis=i)
                idx_a = [slice(None)] * self.S
                idx_b = [slice(None)] * self.S
                idx_a[i], idx_b[i] = a, b
                out[tuple(idx_b)] += 1j * self.g * d * sl_a
                out[tuple(idx_a)] += -1j * self.g * d * sl_b
        return out

    # -- exponentials ------------------------------------------------------

    def exp_h_init(self, psi: np.ndarray, theta: float) -> np.ndarray:
        """`exp(-i theta H_i) psi`, exactly (the axis blocks commute)."""
        if theta == 0.0:
            return psi
        key = round(theta, 15)
        ops = self._axis_ops_cache.get(key)
        if ops is None:
            ops = [expm(-1j * theta * self.axis_operator(i)) for i in range(self.S)]
            if len(self._axis_ops_cache) < 512:
                self._axis_ops_cache[key] = ops
        for i in range(self.S):
            psi = _apply_axis_matrix(psi, ops[i], i)
        return psi

    def exp_h_cost(self, psi: np.ndarray, theta: float) -> np.ndarray:
        """`exp(-i theta H_f) psi`, exactly (diagonal)."""
        if theta == 0.0:
            return psi
        return np.exp(-1j * theta * self.cost_biased) * psi

    def exp_cd(self, psi: np.ndarray, theta: float, reverse: bool = False) -> np.ndarray:
        """`exp(-i theta X) psi`, Trotterized over the hopping edges.

        Restricted to one edge (a,b) of axis i, with the other coordinates
        held fixed, `X` is the 2 x 2 matrix `[[0, -i g Delta], [i g Delta, 0]]`
        `= g Delta sigma_y`, so

            exp(-i theta g Delta sigma_y) = [[cos phi, -sin phi],
                                             [sin phi,  cos phi]],
            phi = theta * g * Delta,

        a real rotation applied elementwise over the remaining coordinates.
        Each edge factor is therefore EXACT; the only approximation is the
        ordering of the `S * len(edges)` factors, which do not commute.
        `reverse=True` applies them in the opposite order, which is what the
        second-order Suzuki splitting needs.
        """
        if theta == 0.0:
            return psi
        psi = psi.copy()
        axes = range(self.S - 1, -1, -1) if reverse else range(self.S)
        for i in axes:
            edge_list = self.edges[::-1] if reverse else self.edges
            for (a, b) in edge_list:
                phi = (theta * self.g) * self._delta(i, a, b)
                c, s = np.cos(phi), np.sin(phi)
                # basic integer indexing gives VIEWS into psi, so no copies:
                # both new slices are formed before either is written back
                sl_a = psi[(slice(None),) * i + (a,)]
                sl_b = psi[(slice(None),) * i + (b,)]
                new_a = c * sl_a - s * sl_b
                new_b = s * sl_a + c * sl_b
                sl_a[...] = new_a
                sl_b[...] = new_b
        return psi

    # -- states ------------------------------------------------------------

    def ground_state(self) -> np.ndarray:
        """Exact ground state of `H_i`, as a product over sectors.

        With no bias this is the uniform superposition over all `T^S` feasible
        configurations -- identical to the QAOA-XY initial state, so the two
        solvers start from the same place.
        """
        vecs = []
        for i in range(self.S):
            w, v = eigh(self.axis_operator(i))
            vec = v[:, 0]
            if vec.sum() < 0:
                vec = -vec
            vecs.append(vec.astype(np.complex128))
        psi = vecs[0]
        for vec in vecs[1:]:
            psi = np.multiply.outer(psi, vec)
        return psi.reshape(self.cost.shape)

    def marginals(self, psi: np.ndarray) -> np.ndarray:
        """(S, T) per-sector tilt-level probabilities of a state."""
        probs = np.abs(psi) ** 2
        probs = probs / probs.sum()
        out = np.zeros((self.S, self.T))
        for i in range(self.S):
            axes = tuple(j for j in range(self.S) if j != i)
            out[i] = probs.sum(axis=axes)
        return out

    def summary(self) -> str:
        return (f"SubspaceOperators: {self.S} blocks x {self.T} levels = "
                f"{self.cost.size:,} amplitudes | driver g={self.g:.4g} "
                f"({self.topology}, {len(self.edges)} edges/block) | "
                f"cost spread={self.cost_spread:.4g} | "
                f"bias {'on' if np.any(self.unary_bias) else 'off'}")


# ---------------------------------------------------------------------------
# The counterdiabatic coefficient
# ---------------------------------------------------------------------------

@dataclass
class CdCoefficient:
    """Variational first-order CD coefficient `alpha(lam)`.

    The exact adiabatic gauge potential is the operator that annihilates the
    off-diagonal part of `G_lam = d_lam H + i[A_lam, H]`. Minimizing
    `Tr(G_lam^2)` (Sels & Polkovnikov, PNAS 114 E3909) over the one-parameter
    ansatz `A = alpha X` gives

        alpha = -Tr(D Y) / Tr(Y^2) ,     Y = i[X, H_ad] ,  D = d_lam H_ad .

    The minus sign is not cosmetic -- with the opposite sign the term drives
    diabatic transitions instead of suppressing them, and DCQO comes out
    *worse* than uniform random sampling. It is pinned down by a two-level
    check in `tests/test_dcqo.py`: for `H = h_x sigma_x + lam sigma_z` and
    ansatz `X = sigma_y` the formula must reproduce the exact AGP
    `A_lam = -h_x sigma_y / (2(h_x^2 + lam^2))`.

    Because `H_ad` is a two-point interpolation, `Y(lam) = (1-lam)P + lam Q`
    with `P = i[X,H_i]`, `Q = i[X,H_f]`, so

        alpha(lam) = -[ (1-lam) a + lam b ]
                     / [ (1-lam)^2 c + 2 lam(1-lam) d + lam^2 e ]

        a = Tr(D P)   b = Tr(D Q)   c = Tr(P^2)   d = Tr(P Q)   e = Tr(Q^2)

    Five lam-independent scalars: estimate once, reuse at every step. They are
    estimated with Hutchinson probes (`Tr(MN) ~ mean_v (Mv)^dag (Nv)` for
    Hermitian M, N and Rademacher v), which needs only matrix-vector products
    -- no `T^S x T^S` matrix is ever built. `n_probes` trades accuracy for
    time; alpha is a variational amplitude, so a few percent of error makes
    the CD term slightly suboptimal rather than wrong.
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    n_probes: int
    scale: float = 1.0            # user multiplier, for ablation
    seconds: float = 0.0

    def alpha(self, lam: float) -> float:
        num = (1.0 - lam) * self.a + lam * self.b
        den = ((1.0 - lam) ** 2 * self.c + 2.0 * lam * (1.0 - lam) * self.d
               + lam ** 2 * self.e)
        if den <= 0.0:
            return 0.0
        return -self.scale * num / den

    def summary(self) -> str:
        return (f"CdCoefficient: alpha(0)={self.alpha(0.0):+.4g} "
                f"alpha(0.5)={self.alpha(0.5):+.4g} alpha(1)={self.alpha(1.0):+.4g} "
                f"({self.n_probes} probes, {self.seconds:.2f} s, scale={self.scale})")


def estimate_cd_coefficient(ops: SubspaceOperators, n_probes: int = 6,
                            seed: int = 0, scale: float = 1.0,
                            verbose: bool = False) -> CdCoefficient:
    """Hutchinson estimate of the five traces `alpha(lam)` needs."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    acc = np.zeros(5)

    for k in range(n_probes):
        v = rng.choice(np.array([-1.0, 1.0]), size=ops.cost.shape).astype(np.complex128)
        Xv = ops.apply_cd(v)
        # P v = i (X H_i v - H_i X v);  Q v = i (X H_f v - H_f X v)
        Pv = 1j * (ops.apply_cd(ops.apply_h_init(v)) - ops.apply_h_init(Xv))
        Qv = 1j * (ops.apply_cd(ops.apply_h_cost(v)) - ops.apply_h_cost(Xv))
        Dv = ops.apply_dlam(v)
        acc += [
            float(np.real(np.vdot(Dv, Pv))),
            float(np.real(np.vdot(Dv, Qv))),
            float(np.real(np.vdot(Pv, Pv))),
            float(np.real(np.vdot(Pv, Qv))),
            float(np.real(np.vdot(Qv, Qv))),
        ]
        if verbose:
            print(f"      probe {k + 1}/{n_probes} done")

    acc /= n_probes
    return CdCoefficient(*acc, n_probes=n_probes, scale=scale,
                         seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class DcqoResult:
    """Deliberately field-compatible with `qaoa_subspace.SubspaceResult` so the
    benchmark can print and compare the two solvers with the same code."""

    name: str
    config: np.ndarray                 # best sampled configuration
    objective: float                   # its value on the TRUE (unbiased) objective
    seconds: float
    evaluations: int                   # classical objective evals spent on PARAMETERS
    n_qubits: int
    depth: int                         # digitized steps (circuit layers)
    prob_of_optimum: float             # w.r.t. the true objective's optimum
    mean_sampled_objective: float
    extra: dict = field(default_factory=dict)

    def __str__(self):
        return (f"{self.name:<28} obj={self.objective:+.5f}  {self.n_qubits}q "
                f"steps={self.depth}  P(opt)={self.prob_of_optimum:7.3%}  "
                f"{self.seconds:.1f}s")


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class DcqoSolver:
    """Digitized counterdiabatic evolution on the feasible subspace.

    Non-variational: there is no classical outer loop, so
    `DcqoResult.evaluations` is 0 by construction. The only knobs are the
    schedule (`n_steps`, `time_scale`) and the mode.

    modes
    -----
    'full'     H_ad + CD, the actual DCQO algorithm.
    'no_cd'    H_ad only -- plain digitized *adiabatic* optimization. The
               ablation that shows whether the CD term earns its depth.
    'cd_only'  CD term only, the impulse regime used for the shallowest
               DCQO circuits in the literature.
    """

    def __init__(self, qubo=None, cost=None, n_steps: int = 40,
                 time_scale: float = 320.0, mode: str = "full",
                 trotter_order: int = 2, topology: str = "ring",
                 driver_strength="auto", cd_probes: int = 6, cd_scale: float = 1.0,
                 schedule_form: str = "sin2", seed: int = 0, verbose: bool = False,
                 n_qubits: int = None):
        if (qubo is None) == (cost is None):
            raise ValueError("pass exactly one of `qubo` or `cost`")
        self.true_cost = subspace_cost(qubo) if cost is None else np.asarray(cost, float)
        # one-hot encoding: S blocks of T qubits. `dcqo_ising.py` passes a
        # (2,)*n tensor of full-space energies instead, where n_qubits = n.
        self.n_vars = int(n_qubits) if n_qubits is not None \
            else int(self.true_cost.ndim * self.true_cost.shape[0])
        if mode not in ("full", "no_cd", "cd_only"):
            raise ValueError(f"unknown mode {mode!r}")
        if trotter_order not in (1, 2):
            raise ValueError("trotter_order must be 1 or 2")

        self.n_steps = int(n_steps)
        self.time_scale = float(time_scale)
        self.mode = mode
        self.trotter_order = int(trotter_order)
        self.topology = topology
        self.driver_strength = driver_strength
        self.cd_probes = int(cd_probes)
        self.cd_scale = float(cd_scale)
        self.schedule_form = schedule_form
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.ops = SubspaceOperators(self.true_cost, None, topology=topology,
                                     driver_strength=driver_strength)
        # total_time in units of the inverse cost spread, so defaults transfer
        # between instances without retuning
        spread = max(self.ops.cost_spread, 1e-12)
        self.total_time = self.time_scale / spread
        self.schedule = AnnealingSchedule(self.n_steps, self.total_time, schedule_form)
        self._cd = None

    # -- bias --------------------------------------------------------------

    def set_bias(self, unary_bias) -> None:
        """Install a bias field. Invalidates the cached CD coefficient, since
        `alpha` depends on the biased Hamiltonian."""
        self.ops.set_bias(unary_bias)
        self._cd = None

    @property
    def cd(self) -> CdCoefficient:
        if self._cd is None:
            if self.verbose:
                print(f"    estimating CD coefficient ({self.cd_probes} probes) ...")
            self._cd = estimate_cd_coefficient(self.ops, n_probes=self.cd_probes,
                                               seed=self.seed, scale=self.cd_scale,
                                               verbose=False)
            if self.verbose:
                print(f"    {self._cd.summary()}")
        return self._cd

    # -- evolution ---------------------------------------------------------

    def evolve(self) -> np.ndarray:
        """Run the digitized schedule and return the final state.

        One step, second order (`trotter_order=2`):

            exp(-i dt/2 (1-lam) H_i)  exp(-i dt/2 lam H_f)  exp(-i dt lam_dot alpha X)
            exp(-i dt/2 lam H_f)      exp(-i dt/2 (1-lam) H_i)

        first order (`trotter_order=1`) drops the symmetrization. The CD edge
        factors are themselves ordered, and are applied in reverse on the
        second half of a symmetric step.
        """
        self.schedule.assert_has_driver()
        if float(np.abs(self.ops.B).sum()) <= 0.0:
            raise ValueError("driver has no off-diagonal weight -- the whole "
                             "Hamiltonian would be diagonal and the state could "
                             "never leave its initial configuration")

        psi = self.ops.ground_state()
        cd = self.cd if self.mode != "no_cd" else None
        report_every = max(self.n_steps // 5, 1)

        for k, (s, lam, lam_dot, dt) in enumerate(self.schedule.steps()):
            th_i = dt * (1.0 - lam)
            th_f = dt * lam
            th_cd = dt * lam_dot * (cd.alpha(lam) if cd is not None else 0.0)

            if self.mode == "cd_only":
                psi = self.ops.exp_cd(psi, th_cd)
            elif self.trotter_order == 1:
                psi = self.ops.exp_h_init(psi, th_i)
                psi = self.ops.exp_h_cost(psi, th_f)
                if cd is not None:
                    psi = self.ops.exp_cd(psi, th_cd)
            else:
                psi = self.ops.exp_h_init(psi, 0.5 * th_i)
                psi = self.ops.exp_h_cost(psi, 0.5 * th_f)
                if cd is not None:
                    psi = self.ops.exp_cd(psi, 0.5 * th_cd)
                    psi = self.ops.exp_cd(psi, 0.5 * th_cd, reverse=True)
                psi = self.ops.exp_h_cost(psi, 0.5 * th_f)
                psi = self.ops.exp_h_init(psi, 0.5 * th_i)

            if self.verbose and (k + 1) % report_every == 0:
                print(f"    step {k + 1}/{self.n_steps}  s={s:.3f} lam={lam:.3f}")

        return psi

    # -- measure and score -------------------------------------------------

    def solve(self, n_shots: int = 2000, name: str = None) -> DcqoResult:
        """Evolve, sample, and score the samples on the TRUE objective."""
        t0 = time.perf_counter()
        return self.measure(self.evolve(), n_shots=n_shots, name=name, t0=t0)

    def measure(self, psi: np.ndarray, n_shots: int = 2000, name: str = None,
                t0: float = None) -> DcqoResult:
        """Sample an evolved state and score the samples on the TRUE objective.

        Split out from `solve` so a caller that needs the state itself (e.g.
        `dcqo_ising.solve_dcqo_ising`, which also wants the feasible fraction)
        does not have to evolve twice.

        The bias field (if any) changes the Hamiltonian, never the scoring:
        `objective` and `prob_of_optimum` are always computed against
        `self.true_cost`.
        """
        t0 = time.perf_counter() if t0 is None else t0

        probs = np.abs(psi.ravel()) ** 2
        probs = probs / probs.sum()
        cost_flat = self.true_cost.ravel()
        ground = float(cost_flat.min())
        p_opt = float(probs[np.isclose(cost_flat, ground, atol=1e-9)].sum())

        rng = np.random.default_rng(self.seed + 1)
        samples = rng.choice(len(probs), size=n_shots, p=probs)
        uniq, counts = np.unique(samples, return_counts=True)
        best_idx = int(uniq[np.argmin(cost_flat[uniq])])
        best_cfg = np.array(np.unravel_index(best_idx, self.true_cost.shape))
        mean_obj = float(np.sum(cost_flat[uniq] * counts) / n_shots)
        self.last_samples = (uniq, counts)

        label = name or (f"DCQO ({self.mode}, {self.n_steps} steps)")
        return DcqoResult(
            name=label, config=best_cfg, objective=float(cost_flat[best_idx]),
            seconds=time.perf_counter() - t0, evaluations=0,
            n_qubits=self.n_vars, depth=self.n_steps, prob_of_optimum=p_opt,
            mean_sampled_objective=mean_obj,
            extra={"ground_energy": ground, "n_feasible": int(cost_flat.size),
                   "n_shots": n_shots, "mode": self.mode,
                   "total_time": self.total_time, "driver_g": self.ops.g,
                   "trotter_order": self.trotter_order,
                   "cd_alpha_mid": (self.cd.alpha(0.5) if self.mode != "no_cd" else 0.0),
                   "marginals": self.ops.marginals(psi),
                   "biased": bool(np.any(self.ops.unary_bias))},
        )


# ---------------------------------------------------------------------------
# Bias fields
# ---------------------------------------------------------------------------

class BiasField:
    """Construct the longitudinal bias `u_i(k)` fed back into the Hamiltonian.

    Three sources, matching design decision 4 of `background/NEXT_STEPS.md`:

      * `from_marginals`  -- the BF-DCQO update: reinforce what the previous
        DCQO iteration measured. The one-hot analogue of `h_b ~ <Z_j>`.
      * `from_config`     -- warm start from a known configuration, e.g. the
        previous *physical* timestep's solution in the time march. This is
        the temporal coupling, at zero extra qubits.
      * `zero`            -- cold start.

    `scale` is the per-sector share of the cost spread,
    `(max - min) / (2 S)`, so `strength = 1` produces a bias comparable to
    what one sector's tilt choice is worth. Without that normalization the
    same `strength` would mean wildly different things on instances of
    different size.
    """

    def __init__(self, n_sectors: int, n_tilts: int, cost_spread: float,
                 strength: float = 1.0):
        self.S = int(n_sectors)
        self.T = int(n_tilts)
        self.scale = float(cost_spread) / (2.0 * max(self.S, 1))
        self.strength = float(strength)

    def zero(self) -> np.ndarray:
        return np.zeros((self.S, self.T))

    def from_marginals(self, marginals) -> np.ndarray:
        """`u_i(k) = -strength * scale * (p_i(k) - 1/T)`.

        Negative where the measured probability is above uniform, so those
        levels are energetically favoured on the next iteration and the
        biased driver ground state already leans toward them.
        """
        p = np.asarray(marginals, dtype=float).reshape(self.S, self.T)
        return -self.strength * self.scale * (p - 1.0 / self.T)

    def from_config(self, config, confidence: float = 1.0) -> np.ndarray:
        """Bias toward a single configuration, as if it had been measured with
        probability `confidence` (the rest spread uniformly)."""
        cfg = np.asarray(config, dtype=int).reshape(self.S)
        p = np.full((self.S, self.T), (1.0 - confidence) / max(self.T - 1, 1))
        p[np.arange(self.S), cfg] = confidence
        return self.from_marginals(p)


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def solve_dcqo(qubo, n_steps: int = 40, n_shots: int = 2000, mode: str = "full",
               time_scale: float = 320.0, trotter_order: int = 2, seed: int = 0,
               cd_probes: int = 6, cd_scale: float = 1.0, topology: str = "ring",
               bias=None, verbose: bool = False, name: str = None) -> DcqoResult:
    """One DCQO run. `bias` may be None (cold) or an (S, T) array."""
    solver = DcqoSolver(qubo=qubo, n_steps=n_steps, mode=mode, time_scale=time_scale,
                        trotter_order=trotter_order, seed=seed, cd_probes=cd_probes,
                        cd_scale=cd_scale, topology=topology, verbose=verbose)
    if bias is not None:
        solver.set_bias(bias)
    return solver.solve(n_shots=n_shots, name=name)


def solve_bf_dcqo(qubo, n_iters: int = 3, n_steps: int = 40, n_shots: int = 2000,
                  bias_strength: float = 1.0, time_scale: float = 320.0,
                  mode: str = "full", trotter_order: int = 2, seed: int = 0,
                  cd_probes: int = 6, topology: str = "ring", init_bias=None,
                  bias_source: str = "best_sample", bias_damping: float = 0.5,
                  verbose: bool = True, name: str = None) -> DcqoResult:
    """Bias-field DCQO: iterate DCQO, feeding the measurement back as a bias.

    Iteration 1 is a cold DCQO run (unless `init_bias` is given, which is how
    the time march injects the previous timestep's solution). Each subsequent
    iteration re-anneals with a bias built from what has been measured so far.

    Still non-variational -- nothing is fitted, the bias is a measurement --
    so `evaluations` stays 0. The cost is `n_iters` circuits instead of one;
    `extra['history']` and `extra['total_shots']` record that so the benchmark
    can charge for it honestly.

    bias_source
    -----------
    'marginals'    the published BF-DCQO update: bias toward the measured
                   per-sector marginals `p_i(k)` (`h_b ~ <Z_j>` at T=2).
                   This is a MEAN-FIELD quantity, and when the final
                   distribution straddles several near-degenerate solutions
                   the average need not resemble any of them -- on the
                   temporal chain instance it actively hurts. Measured, not
                   assumed: `dcqo_benchmark.py` reports both.
    'best_sample'  bias toward the best configuration measured so far. More
                   robust for exactly that reason, and it is what the
                   temporal time march uses anyway (the previous timestep's
                   solution is a configuration, not a distribution).

    Safeguards, applied to both sources: the bias is always rebuilt from the
    BEST iteration rather than the last one, so errors cannot compound, and
    `bias_strength` is multiplied by `bias_damping` after any iteration that
    fails to improve the mean sampled objective.
    """
    if bias_source not in ("marginals", "best_sample"):
        raise ValueError(f"unknown bias_source {bias_source!r}")

    t0 = time.perf_counter()
    solver = DcqoSolver(qubo=qubo, n_steps=n_steps, mode=mode, time_scale=time_scale,
                        trotter_order=trotter_order, seed=seed, cd_probes=cd_probes,
                        topology=topology, verbose=False)
    S, T = solver.true_cost.ndim, solver.true_cost.shape[0]
    field_builder = BiasField(S, T, solver.ops.cost_spread, strength=bias_strength)

    bias = field_builder.zero() if init_bias is None else np.asarray(init_bias, float)
    best, best_mean, history = None, np.inf, []

    for it in range(n_iters):
        solver.set_bias(bias)
        res = solver.solve(n_shots=n_shots, name=f"BF-DCQO it{it + 1}")
        history.append({"iteration": it + 1, "objective": res.objective,
                        "prob_of_optimum": res.prob_of_optimum,
                        "mean_sampled_objective": res.mean_sampled_objective,
                        "bias_max": float(np.abs(bias).max()),
                        "bias_strength": field_builder.strength,
                        "seconds": res.seconds})
        if verbose:
            print(f"    it {it + 1}/{n_iters}: obj={res.objective:+.5f} "
                  f"P(opt)={res.prob_of_optimum:7.3%} "
                  f"|bias|max={np.abs(bias).max():.4g} "
                  f"strength={field_builder.strength:.3g}  {res.seconds:.1f}s")

        improved = res.mean_sampled_objective < best_mean - 1e-12
        if best is None or res.objective < best.objective - 1e-12:
            best = res
        if improved:
            best_mean, best_res_for_bias = res.mean_sampled_objective, res
        else:
            field_builder.strength *= bias_damping
        if it == 0:
            best_res_for_bias = res

        if bias_source == "marginals":
            bias = field_builder.from_marginals(best_res_for_bias.extra["marginals"])
        else:
            bias = field_builder.from_config(best.config, confidence=0.9)

    # report the best iteration, but charge for all of them
    out = DcqoResult(
        name=name or f"BF-DCQO ({n_iters} iters, {n_steps} steps)",
        config=best.config, objective=best.objective,
        seconds=time.perf_counter() - t0, evaluations=0,
        n_qubits=best.n_qubits, depth=n_steps,
        prob_of_optimum=max(h["prob_of_optimum"] for h in history),
        mean_sampled_objective=best.mean_sampled_objective,
        extra=dict(best.extra),
    )
    out.extra.update({"history": history, "n_iters": n_iters,
                      "bias_source": bias_source,
                      "bias_strength": bias_strength,
                      "total_circuits": n_iters,
                      "total_shots": n_iters * n_shots})
    return out


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

def _dense_operators(ops: SubspaceOperators):
    """Dense matrices of H_i, H_f and X -- for verification at tiny size only."""
    n = ops.cost.size
    shape = ops.cost.shape
    H_i = np.zeros((n, n), dtype=complex)
    H_f = np.diag(ops.cost_biased.ravel()).astype(complex)
    basis = np.eye(n, dtype=complex)
    for col in range(n):
        v = basis[col].reshape(shape)
        H_i[:, col] = ops.apply_h_init(v).ravel()
    H_d = H_i - np.diag(ops.bias_diag.ravel()).astype(complex)
    X = 1j * (H_d @ H_f - H_f @ H_d)
    return H_i, H_f, X, H_d


if __name__ == "__main__":
    from rf_model import make_network
    from qubo_builder import build_qubo
    from classical_baseline import brute_force, greedy_local_search

    print("=" * 74)
    print("DCQO / BF-DCQO SELF-CHECKS")
    print("=" * 74)
    print("  estimated runtime: ~20 s (CPU, numpy only -- no GPU used)")

    # ---- 1. tiny instance: verify every operator against dense algebra ----
    print("\n--- 1. operator algebra on a tiny instance (3 sectors x 4 tilts) ---")
    net = make_network(1, seed=7)
    qubo = build_qubo(net, penalty_scale=0.0)
    cost = subspace_cost(qubo)
    S, T = cost.ndim, cost.shape[0]
    print(f"  {S} blocks x {T} levels = {cost.size} feasible configurations")

    rng = np.random.default_rng(0)
    bias = rng.normal(0, 0.05, size=(S, T))
    ops = SubspaceOperators(cost, bias)
    print(f"  {ops.summary()}")

    H_i, H_f, X_dense, H_d = _dense_operators(ops)

    # cost tensor must equal the QUBO objective
    err = max(abs(cost[tuple(c)] - qubo.objective_from_config(c))
              for c in [rng.integers(0, T, size=S) for _ in range(200)])
    print(f"  max |subspace cost - QUBO objective|        : {err:.3e}")

    # X must be Hermitian and must equal i[H_d, H_f]
    v = rng.normal(size=cost.shape) + 1j * rng.normal(size=cost.shape)
    err_x = np.abs(ops.apply_cd(v).ravel() - X_dense @ v.ravel()).max()
    print(f"  max |apply_cd(v) - i[H_d,H_f] v|            : {err_x:.3e}")
    print(f"  max |X - X^dagger|                          : "
          f"{np.abs(X_dense - X_dense.conj().T).max():.3e}")

    # exponentials must be unitary and match dense expm
    for theta in (0.13, 1.7):
        got = ops.exp_h_init(v, theta).ravel()
        want = expm(-1j * theta * H_i) @ v.ravel()
        print(f"  max |exp(-i{theta}H_i)v - dense|             : "
              f"{np.abs(got - want).max():.3e}")

    for theta in (0.05, 0.4):
        u = ops.exp_cd(v, theta)
        norm_err = abs(np.linalg.norm(u) - np.linalg.norm(v))
        # first-order agreement in theta is all a Trotterized factor promises
        lin = np.abs((u.ravel() - v.ravel()) / theta
                     - (-1j * X_dense @ v.ravel())).max()
        print(f"  exp_cd(theta={theta}): |norm change|={norm_err:.3e}  "
              f"first-order residual={lin:.3e}")

    # cold start must be the uniform superposition
    cold = SubspaceOperators(cost, None)
    psi0 = cold.ground_state()
    print(f"  cold ground state uniform? max deviation    : "
          f"{np.abs(np.abs(psi0) - 1 / np.sqrt(cost.size)).max():.3e}")
    w = np.linalg.eigvalsh(_dense_operators(cold)[3])
    got = float(np.real(np.vdot(psi0.ravel(), _dense_operators(cold)[3] @ psi0.ravel())))
    print(f"  cold <H_d> = {got:+.6f} vs exact minimum {w[0]:+.6f}")

    # biased start must be the ground state of the biased H_i
    psi_b = ops.ground_state()
    w_b = np.linalg.eigvalsh(H_i)
    got_b = float(np.real(np.vdot(psi_b.ravel(), H_i @ psi_b.ravel())))
    print(f"  biased <H_i> = {got_b:+.6f} vs exact minimum {w_b[0]:+.6f}")

    # ---- 2. schedule --------------------------------------------------------
    print("\n--- 2. schedule ---")
    sched = AnnealingSchedule(n_steps=20, total_time=3.0)
    print(f"  {sched.summary()}")
    print(f"  lam(0)={float(sched.lam(0)):.6f}  lam(1)={float(sched.lam(1)):.6f}  "
          f"lam'(0)={float(sched.dlam_ds(0)):.3e}  lam'(1)={float(sched.dlam_ds(1)):.3e}")
    num = (sched.lam(0.5 + 1e-6) - sched.lam(0.5 - 1e-6)) / 2e-6
    print(f"  lam'(0.5): analytic {float(sched.dlam_ds(0.5)):.6f} vs "
          f"numeric {float(num):.6f}")
    try:
        class _Flat(AnnealingSchedule):
            def lam(self, s):
                return np.ones_like(np.asarray(s, dtype=float))
        _Flat(10, 1.0)
        print("  !! diagonal-commuting trap NOT caught")
    except ValueError as exc:
        print(f"  diagonal-commuting trap caught: {str(exc)[:52]}...")

    # ---- 3. CD coefficient --------------------------------------------------
    print("\n--- 3. CD coefficient (Hutchinson vs exact traces) ---")
    P = 1j * (X_dense @ H_i - H_i @ X_dense)
    Qd = 1j * (X_dense @ H_f - H_f @ X_dense)
    D = H_f - H_i
    exact = CdCoefficient(a=float(np.real(np.trace(D @ P))),
                          b=float(np.real(np.trace(D @ Qd))),
                          c=float(np.real(np.trace(P @ P))),
                          d=float(np.real(np.trace(P @ Qd))),
                          e=float(np.real(np.trace(Qd @ Qd))), n_probes=0)
    for n_probes in (4, 16, 64):
        est = estimate_cd_coefficient(ops, n_probes=n_probes, seed=1)
        rel = abs(est.alpha(0.5) - exact.alpha(0.5)) / max(abs(exact.alpha(0.5)), 1e-12)
        print(f"  {n_probes:>3} probes: alpha(0.5)={est.alpha(0.5):+.5f} "
              f"(exact {exact.alpha(0.5):+.5f}, rel err {rel:.2%})")

    # the sign of alpha is the one thing that cannot be checked by inspection:
    # pin it against the analytically known two-level gauge potential
    print("\n  sign check against the exact two-level AGP")
    print("    H = h_x sx + lam sz, ansatz X = sy, exact A = -h_x sy/(2(h_x^2+lam^2))")
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    hx = 0.37
    for lam2 in (0.0, 0.3, 1.4):
        Htl = hx * sx + lam2 * sz
        Ytl = 1j * (sy @ Htl - Htl @ sy)
        got = -np.real(np.trace(sz @ Ytl)) / np.real(np.trace(Ytl @ Ytl))
        want = -hx / (2 * (hx ** 2 + lam2 ** 2))
        print(f"    lam={lam2:<4} formula {got:+.6f}  exact {want:+.6f}  "
              f"|diff| {abs(got - want):.2e}")

    # ---- 4. does DCQO actually solve it? -----------------------------------
    print("\n--- 4. DCQO vs the exact optimum on the tiny instance ---")
    exact_sol = brute_force(qubo)
    p_random = 1.0 / cost.size
    print(f"  brute-force optimum {exact_sol.objective:+.5f} "
          f"tilts={exact_sol.config.tolist()}")
    print(f"  uniform random P(opt) = {p_random:.4%}")
    print("  time_scale is the annealing time in units of 1/(cost spread), so")
    print("  the same value means the same thing on any instance.\n")
    print(f"  {'mode':<10} {'time_scale':>11} {'steps':>6} {'obj':>11} "
          f"{'P(opt)':>9} {'vs random':>10}")
    for mode in ("no_cd", "cd_only", "full"):
        for ts in (20.0, 80.0, 320.0):
            for n_steps in (10, 40):
                r = solve_dcqo(qubo, n_steps=n_steps, mode=mode, seed=3,
                               time_scale=ts, n_shots=4000)
                print(f"  {mode:<10} {ts:>11.0f} {n_steps:>6} {r.objective:>+11.5f} "
                      f"{r.prob_of_optimum:>8.2%} {r.prob_of_optimum / p_random:>9.1f}x")
    print("\n  Read the 'full' vs 'no_cd' rows at the SAME time_scale and steps:")
    print("  that difference is what the counterdiabatic term buys. It is large")
    print("  at short annealing time (shallow circuits) and shrinks as the")
    print("  schedule lengthens, which is exactly what the theory predicts.")

    # ---- 5. the bias field must help ---------------------------------------
    print("\n--- 5. bias field: does warm starting raise P(optimum)? ---")
    bf = BiasField(S, T, ops.cost_spread, strength=1.0)
    cold_r = solve_dcqo(qubo, n_steps=20, mode="full", seed=3)
    warm_r = solve_dcqo(qubo, n_steps=20, mode="full", seed=3,
                        bias=bf.from_config(exact_sol.config, confidence=0.7))
    print(f"  cold start                P(opt) = {cold_r.prob_of_optimum:7.3%}")
    print(f"  bias toward the optimum   P(opt) = {warm_r.prob_of_optimum:7.3%}  "
          f"({warm_r.prob_of_optimum / max(cold_r.prob_of_optimum, 1e-12):.1f}x)")
    print(f"  best objective is still scored on the TRUE cost: "
          f"{warm_r.objective:+.5f} (optimum {exact_sol.objective:+.5f})")

    print("\n--- 6. BF-DCQO iterations ---")
    bfres = solve_bf_dcqo(qubo, n_iters=4, n_steps=20, seed=3, verbose=True)
    print(f"  {bfres}")
    print(f"  found optimum: "
          f"{bool(np.isclose(bfres.objective, exact_sol.objective, atol=1e-9))}")

    g = greedy_local_search(qubo, n_restarts=1)
    print(f"\n  reference: greedy local search obj={g.objective:+.5f} "
          f"({g.evaluations} objective evaluations)")
    print("\n  All checks above are numerical identities except 4-6, which are")
    print("  performance and depend on the instance. Run tests/test_dcqo.py for")
    print("  the assertion-based versions.")
