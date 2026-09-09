"""Direct spatiotemporal antenna-tilt optimization.

Unlike :mod:`temporal`, which first reduces each timestep to a small shared
candidate pool, this module keeps every sector/tilt decision at every time:

    x[t, s, k] = 1  iff sector s uses tilt k at timestep t.

The objective is the sum of the validated snapshot objectives and temporal
switching costs.  Its interaction graph therefore contains both the spatial
RF edges within a timestep and temporal edges for each sector.  One-hot blocks
are ordered ``(timestep, sector)`` so the existing constrained QAOA and DCQO
implementations can consume the returned :class:`TiltQUBO` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from qubo_builder import TiltQUBO, build_qubo


@dataclass
class DirectTemporalProblem:
    """The unreduced spatiotemporal problem and its scoring contract."""

    snapshots: list
    snapshot_qubos: list[TiltQUBO]
    lam: float
    cost_mode: str = "switches"

    def __post_init__(self):
        if not self.snapshots or len(self.snapshots) != len(self.snapshot_qubos):
            raise ValueError("snapshots and snapshot_qubos must be non-empty and aligned")
        shape = {(q.n_sectors, q.n_tilts) for q in self.snapshot_qubos}
        if len(shape) != 1:
            raise ValueError("all snapshots must use the same sectors and tilt levels")
        if self.cost_mode not in ("switches", "degrees"):
            raise ValueError("cost_mode must be 'switches' or 'degrees'")
        if self.lam < 0:
            raise ValueError("lam must be non-negative")

    @property
    def horizon(self) -> int:
        return len(self.snapshots)

    @property
    def n_sectors(self) -> int:
        return self.snapshot_qubos[0].n_sectors

    @property
    def n_tilts(self) -> int:
        return self.snapshot_qubos[0].n_tilts

    @property
    def n_vars(self) -> int:
        return self.horizon * self.n_sectors * self.n_tilts

    def flatten(self, trajectory) -> np.ndarray:
        return np.asarray(trajectory, dtype=int).reshape(self.horizon * self.n_sectors)

    def reshape(self, configuration) -> np.ndarray:
        return np.asarray(configuration, dtype=int).reshape(self.horizon, self.n_sectors)

    def switch_cost(self, trajectory) -> float:
        traj = self.reshape(trajectory)
        if self.horizon <= 1:
            return 0.0
        if self.cost_mode == "switches":
            return float(np.count_nonzero(np.diff(traj, axis=0)))
        levels = np.asarray(self.snapshots[0].tilt_levels_deg, dtype=float)
        return float(np.abs(np.diff(levels[traj], axis=0)).sum())

    def objective(self, trajectory) -> float:
        traj = self.reshape(trajectory)
        snapshot = sum(q.objective_from_config(traj[t])
                       for t, q in enumerate(self.snapshot_qubos))
        return float(snapshot + self.lam * self.switch_cost(traj))

    def breakdown(self, trajectory) -> dict:
        traj = self.reshape(trajectory)
        snapshot_costs = [float(q.objective_from_config(traj[t]))
                          for t, q in enumerate(self.snapshot_qubos)]
        raw_switch_cost = self.switch_cost(traj)
        return {
            "snapshot_costs": snapshot_costs,
            "snapshot_total": float(sum(snapshot_costs)),
            "switch_cost": raw_switch_cost,
            "weighted_switch_cost": float(self.lam * raw_switch_cost),
            "objective": float(sum(snapshot_costs) + self.lam * raw_switch_cost),
            "n_switches": int(np.count_nonzero(np.diff(traj, axis=0))),
        }

    def evaluate_trajectory(self, trajectory) -> dict:
        """Evaluate the trajectory with the exact joint-SINR simulator."""
        traj = self.reshape(trajectory)
        per_step = [self.snapshots[t].evaluate_config(traj[t])
                    for t in range(self.horizon)]
        keys = ("mean_sinr_db", "edge_sinr_db", "mean_spectral_efficiency",
                "outage_pct", "coverage_pct", "handover_failure_pct")
        out = {key: float(np.mean([row[key] for row in per_step])) for key in keys}
        out.update(self.breakdown(traj))
        out["per_step"] = per_step
        return out


def build_direct_temporal_problem(
        snapshots, lam: float = 0.15, cost_mode: str = "switches",
        w_cov: float = 0.4, w_int: float = 0.35, w_ho: float = 0.25,
) -> DirectTemporalProblem:
    """Build snapshot objectives once, with no one-hot penalty in their data."""
    snapshots = list(snapshots)
    qubos = [build_qubo(net, w_cov=w_cov, w_int=w_int, w_ho=w_ho,
                        penalty_scale=0.0) for net in snapshots]
    return DirectTemporalProblem(snapshots, qubos, float(lam), cost_mode)


def _switch_table(problem: DirectTemporalProblem) -> np.ndarray:
    if problem.cost_mode == "switches":
        return 1.0 - np.eye(problem.n_tilts)
    levels = np.asarray(problem.snapshots[0].tilt_levels_deg, dtype=float)
    return np.abs(levels[:, None] - levels[None, :])


def build_direct_temporal_qubo(problem: DirectTemporalProblem,
                               penalty_scale: float = 2.0) -> TiltQUBO:
    """Encode the direct objective as one QUBO over ``H*S*T`` variables.

    Spatial RF coefficients are copied without renormalizing them across the
    horizon.  Temporal edges carry ``lam * switch(k, k')``.  Consequently a
    feasible configuration has exactly the same value under
    ``problem.objective``, ``qubo.objective_from_config`` and ``qubo.energy``.
    """
    H, S, T = problem.horizon, problem.n_sectors, problem.n_tilts
    B, n_vars = H * S, H * S * T

    def block(t, s):
        return t * S + s

    def var(b, k):
        return b * T + k

    cov = np.zeros((B, T), dtype=float)
    spatial_int, spatial_ho, temporal = {}, {}, {}
    Q = np.zeros((n_vars, n_vars), dtype=float)

    for t, snap_q in enumerate(problem.snapshot_qubos):
        for s in range(S):
            b = block(t, s)
            cov[b] = snap_q.cov_norm[s]
            for k in range(T):
                Q[var(b, k), var(b, k)] += -snap_q.w_cov * snap_q.cov_norm[s, k]

        for source, target, weight in (
                (snap_q.int_norm, spatial_int, snap_q.w_int),
                (snap_q.ho_norm, spatial_ho, snap_q.w_ho)):
            for (s1, s2), table in source.items():
                b1, b2 = block(t, s1), block(t, s2)
                target[(b1, b2)] = np.asarray(table, dtype=float).copy()
                Q[np.ix_(range(var(b1, 0), var(b1, T)),
                         range(var(b2, 0), var(b2, T)))] += weight * table

    switch = problem.lam * _switch_table(problem)
    for t in range(H - 1):
        for s in range(S):
            b1, b2 = block(t, s), block(t + 1, s)
            temporal[(b1, b2)] = switch.copy()
            Q[np.ix_(range(var(b1, 0), var(b1, T)),
                     range(var(b2, 0), var(b2, T)))] += switch

    physics_max = float(np.max(np.abs(Q))) or 1.0
    penalty = float(penalty_scale) * max(physics_max * T, 1.0)
    offset = 0.0
    for b in range(B):
        for k in range(T):
            Q[var(b, k), var(b, k)] -= penalty
        for k in range(T):
            for k2 in range(k + 1, T):
                Q[var(b, k), var(b, k2)] += 2.0 * penalty
        offset += penalty

    # All snapshot QUBOs share these weights by construction.
    q0 = problem.snapshot_qubos[0]
    return TiltQUBO(
        n_sectors=B, n_tilts=T, Q=Q, offset=offset, penalty=penalty,
        w_cov=q0.w_cov, w_int=q0.w_int, w_ho=q0.w_ho,
        cov_norm=cov, int_norm=spatial_int, ho_norm=spatial_ho,
        extra_pairwise=temporal,
    )


def best_static_trajectory(problem: DirectTemporalProblem):
    """Exact best constant trajectory when ``T**S`` is enumerable."""
    from itertools import product

    best_obj, best_cfg = np.inf, None
    for cfg in product(range(problem.n_tilts), repeat=problem.n_sectors):
        traj = np.tile(cfg, (problem.horizon, 1))
        obj = problem.objective(traj)
        if obj < best_obj:
            best_obj, best_cfg = obj, np.asarray(cfg, dtype=int)
    return np.tile(best_cfg, (problem.horizon, 1)), float(best_obj)


def snapshot_chasing_trajectory(problem: DirectTemporalProblem):
    """Optimize each snapshot independently, ignoring future switch cost."""
    from classical_baseline import brute_force

    return np.stack([brute_force(q).config for q in problem.snapshot_qubos])


def myopic_hysteresis_trajectory(problem: DirectTemporalProblem, max_enumerated=2_000_000):
    """Switch-aware one-step lookahead baseline.

    At t=0 this chooses the exact snapshot optimum.  At each later timestep it
    chooses the configuration minimizing the current snapshot objective plus
    the immediate switching penalty from the previous chosen configuration.
    This is a strong, transparent rolling-horizon/hysteresis baseline for the
    small and intermediate instances used here; it deliberately ignores future
    timesteps, unlike the joint MILP/direct-temporal optimum.
    """
    from itertools import product

    S, T, H = problem.n_sectors, problem.n_tilts, problem.horizon
    n_configs = T ** S
    if n_configs > max_enumerated:
        raise ValueError(
            f"myopic hysteresis enumerates T**S configurations per step; "
            f"got {n_configs:,} > {max_enumerated:,}")
    configs = [np.asarray(cfg, dtype=int) for cfg in product(range(T), repeat=S)]
    trajectory = []
    previous = None
    levels = np.asarray(problem.snapshots[0].tilt_levels_deg, dtype=float)
    for t in range(H):
        best_value, best_cfg = np.inf, None
        for cfg in configs:
            value = problem.snapshot_qubos[t].objective_from_config(cfg)
            if previous is not None:
                if problem.cost_mode == "switches":
                    switch = float(np.count_nonzero(cfg != previous))
                else:
                    switch = float(np.abs(levels[cfg] - levels[previous]).sum())
                value += problem.lam * switch
            if value < best_value:
                best_value, best_cfg = value, cfg
        trajectory.append(best_cfg.copy())
        previous = best_cfg
    return np.stack(trajectory)
