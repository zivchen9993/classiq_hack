"""
DCQO in the FULL Hilbert space: the textbook X-driver / Ising-cost version.

`dcqo.py` runs DCQO inside the feasible one-hot subspace, which is what lets
it reach the same 48-qubit instance the constrained QAOA reaches. This module
runs the *unconstrained* version -- all `2^n` bitstrings, one-hot validity
enforced only by the QUBO penalty term -- so we can say what the constraint
preservation is actually worth.

It is the DCQO counterpart of the QAOA-X vs QAOA-XY comparison that is
already the headline of `README.md`: same instance, same objective, the only
difference being whether the algorithm can leave the feasible set.

Why this needs almost no new code
---------------------------------
The full-space problem is the *same* object with `T = 2`:

  * a bitstring is a point of a `(2,)*n` tensor,
  * the hopping matrix on a 2-level axis is `B = [[0,1],[1,0]] = sigma_x`,
    so the driver `H_d = -g sum_i B^(i)` is exactly the standard transverse
    field `-g sum_i X_i`, whose ground state is `|+>^n`,
  * `X = i[H_d, H_f]` connects bitstrings differing in one bit, which is
    exactly the first-order counterdiabatic operator for an Ising cost.

So `SubspaceOperators(cost_tensor_of_shape_(2,)*n, topology='chain')` *is*
the full-space DCQO engine, and every self-check in `dcqo.py` covers it too.

What the CD term is, in Pauli words
-----------------------------------
With `H_d = -sum_k X_k` and `H_f = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j`,
using `[X_k, Z_k] = -2i Y_k`:

    X = i[H_d, H_f] = -2 sum_k h_k Y_k
                      -2 sum_{k<j} J_kj (Y_k Z_j + Z_k Y_j)

i.e. exactly the `Y_i` and `Y_i Z_j + Z_i Y_j` first-order nested-commutator
terms named in `background/NEXT_STEPS.md` P4. `cd_pauli_terms` returns them
explicitly, and `_verify_pauli_terms` checks that summing them reproduces the
operator `dcqo.SubspaceOperators.apply_cd` applies numerically. That is the
bridge to a gate-level implementation: each `Y_k` is one rotation and each
`Y_k Z_j` is a two-qubit rotation, so the first-order CD layer costs
`n + 2|J|` rotations per Trotter step.

BF-DCQO reduces to the published form here
------------------------------------------
`dcqo.BiasField` builds `u_j(k) = -strength * scale * (p_j(k) - 1/T)`. At
`T = 2` that is

    u_j = -c (p_j(0) - 1/2) |0><0| - c (p_j(1) - 1/2) |1><1|
        = -(c/2) <Z_j> Z_j ,

i.e. precisely BF-DCQO's longitudinal bias `h_b ~ <Z_j>`. The one-hot
generalization in `dcqo.py` is therefore a strict extension of the published
algorithm, not a different one.

Cost
----
`2^n` amplitudes, so this is for SMALL instances: 12 qubits (3 sectors x 4
tilts) is instant, 24 qubits (6 sectors) needs 268 MB and is the practical
ceiling. `max_qubits` refuses anything larger rather than exhausting memory.
"""

from __future__ import annotations

import numpy as np

from dcqo import DcqoSolver, SubspaceOperators, BiasField, DcqoResult


# ---------------------------------------------------------------------------
# The full-space cost tensor
# ---------------------------------------------------------------------------

def full_space_cost(qubo, include_penalty: bool = True,
                    max_qubits: int = 24) -> np.ndarray:
    """QUBO energy of every bitstring, as a `(2,)*n` tensor.

    `include_penalty=True` uses the full QUBO energy `x'Qx + offset`, i.e. the
    physics objective *plus* the one-hot penalty -- which is the Hamiltonian an
    unconstrained solver actually has to minimize, and the whole reason the
    unconstrained approach struggles. `include_penalty=False` strips the
    penalty and is only meaningful for verification.
    """
    n = qubo.n_vars
    if n > max_qubits:
        raise ValueError(f"full-space DCQO needs 2^{n} amplitudes "
                         f"({2 ** n:,}); limit is {max_qubits} qubits. Use the "
                         f"subspace solver in dcqo.py for larger instances.")

    Q = np.asarray(qubo.Q, dtype=float)
    diag = np.diag(Q).copy()
    sym = Q + Q.T
    np.fill_diagonal(sym, 0.0)

    cost = np.zeros((2,) * n)
    for i in range(n):
        shape = [1] * n
        shape[i] = 2
        cost = cost + (diag[i] * np.array([0.0, 1.0])).reshape(shape)
    for i in range(n):
        for j in range(i + 1, n):
            if sym[i, j] == 0.0:
                continue
            shape = [1] * n
            shape[i] = shape[j] = 2
            block = sym[i, j] * np.outer([0.0, 1.0], [0.0, 1.0]).reshape(shape)
            cost = cost + block
    cost = cost + float(qubo.offset)

    if not include_penalty:
        # subtract the one-hot penalty contribution, term by term
        pen = np.zeros_like(cost)
        for s in range(qubo.n_sectors):
            block = np.zeros_like(cost)
            for t in range(qubo.n_tilts):
                shape = [1] * n
                shape[qubo.var_index(s, t)] = 2
                block = block + np.array([0.0, 1.0]).reshape(shape)
            pen = pen + qubo.penalty * (block - 1.0) ** 2
        cost = cost - pen
    return cost


def feasible_mask(qubo) -> np.ndarray:
    """`(2,)*n` boolean tensor: does this bitstring satisfy every one-hot block?"""
    n = qubo.n_vars
    ok = np.ones((2,) * n, dtype=bool)
    for s in range(qubo.n_sectors):
        block = np.zeros((2,) * n)
        for t in range(qubo.n_tilts):
            shape = [1] * n
            shape[qubo.var_index(s, t)] = 2
            block = block + np.array([0.0, 1.0]).reshape(shape)
        ok &= (block == 1.0)
    return ok


# ---------------------------------------------------------------------------
# The CD term as Pauli strings
# ---------------------------------------------------------------------------

def cd_pauli_terms(qubo, driver_g: float = 1.0):
    """First-order CD operator `X = i[H_d, H_f]` as an explicit Pauli list.

    Returns `(terms, counts)` where each term is `(coefficient, {qubit: pauli})`
    and `counts` reports how many one- and two-body rotations a gate-level
    implementation of one CD layer would need.
    """
    h, J, _ = qubo.to_ising()
    n = qubo.n_vars
    terms = []
    for k in range(n):
        if h[k] != 0.0:
            terms.append((-2.0 * driver_g * float(h[k]), {k: "Y"}))
    n_two = 0
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] == 0.0:
                continue
            c = -2.0 * driver_g * float(J[i, j])
            terms.append((c, {i: "Y", j: "Z"}))
            terms.append((c, {i: "Z", j: "Y"}))
            n_two += 2
    counts = {"one_body_Y": sum(1 for _, p in terms if len(p) == 1),
              "two_body_YZ": n_two,
              "rotations_per_layer": len(terms)}
    return terms, counts


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _apply_pauli_term(psi: np.ndarray, coeff: float, paulis: dict) -> np.ndarray:
    """Apply one Pauli-string term to a `(2,)*n` state tensor."""
    out = psi
    for q, p in paulis.items():
        out = np.moveaxis(out, q, 0)
        shape = out.shape
        out = (_PAULI[p] @ out.reshape(2, -1)).reshape(shape)
        out = np.moveaxis(out, 0, q)
    return coeff * out


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_dcqo_ising(qubo, n_steps: int = 40, n_shots: int = 2000,
                     mode: str = "full", time_scale: float = 320.0,
                     trotter_order: int = 2, seed: int = 0, cd_probes: int = 6,
                     bias=None, include_penalty: bool = True,
                     name: str = None, verbose: bool = False) -> DcqoResult:
    """DCQO over all `2^n` bitstrings, with an X driver.

    `extra` gains `feasible_fraction` -- the probability that a shot is a valid
    one-hot configuration at all. For the subspace solver that number is 1 by
    construction; here it is the thing the penalty term has to earn.
    """
    cost = full_space_cost(qubo, include_penalty=include_penalty)
    n = qubo.n_vars
    solver = DcqoSolver(cost=cost, n_steps=n_steps, mode=mode, time_scale=time_scale,
                        trotter_order=trotter_order, seed=seed, cd_probes=cd_probes,
                        topology="chain", verbose=verbose, n_qubits=n)
    if bias is not None:
        solver.set_bias(bias)

    psi = solver.evolve()
    res = solver.measure(psi, n_shots=n_shots,
                         name=name or f"DCQO-X full space ({mode}, {n_steps} steps)")

    # how much of the final distribution is even feasible?
    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    ok = feasible_mask(qubo)
    res.extra["feasible_fraction"] = float(probs[ok].sum())

    # the best FEASIBLE configuration actually SAMPLED -- not the best feasible
    # configuration in existence, which the solver has no way to know
    uniq, counts = solver.last_samples
    ok_flat = ok.ravel()
    feas = uniq[ok_flat[uniq]]
    res.extra["feasible_shots_frac"] = float(
        counts[ok_flat[uniq]].sum() / max(counts.sum(), 1))
    if len(feas):
        cost_flat = cost.ravel()
        best_flat = int(feas[np.argmin(cost_flat[feas])])
        bits = np.array(np.unravel_index(best_flat, cost.shape))
        res.extra["best_feasible_objective"] = float(
            qubo.objective_from_config(qubo.decode(bits)))
        res.extra["best_feasible_tilts"] = qubo.decode(bits).tolist()
    else:
        res.extra["best_feasible_objective"] = float("inf")
        res.extra["best_feasible_tilts"] = None
    res.extra["one_hot_violations"] = int(qubo.penalty_violation(res.config))
    _, cd_counts = cd_pauli_terms(qubo, driver_g=solver.ops.g)
    res.extra["cd_pauli_counts"] = cd_counts
    return res


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

def _verify_pauli_terms(qubo, tol: float = 1e-9) -> float:
    """max |sum of CD Pauli terms applied - SubspaceOperators.apply_cd|."""
    cost = full_space_cost(qubo)
    ops = SubspaceOperators(cost, None, topology="chain")
    terms, _ = cd_pauli_terms(qubo, driver_g=ops.g)

    rng = np.random.default_rng(0)
    v = (rng.normal(size=cost.shape) + 1j * rng.normal(size=cost.shape))
    numeric = ops.apply_cd(v)
    symbolic = np.zeros_like(v)
    for coeff, paulis in terms:
        symbolic = symbolic + _apply_pauli_term(v, coeff, paulis)
    return float(np.abs(numeric - symbolic).max())


if __name__ == "__main__":
    from rf_model import make_network
    from qubo_builder import build_qubo
    from classical_baseline import brute_force
    from dcqo import solve_dcqo, solve_bf_dcqo
    from qaoa_subspace import subspace_cost

    print("=" * 74)
    print("FULL-HILBERT-SPACE DCQO (X driver + penalty) SELF-CHECKS")
    print("=" * 74)
    print("  estimated runtime: ~25 s (CPU, numpy only -- no GPU used)")

    net = make_network(1, seed=7)
    qubo = build_qubo(net)                        # WITH the one-hot penalty
    n = qubo.n_vars
    print(f"\n  instance: {qubo.n_sectors} sectors x {qubo.n_tilts} tilts "
          f"= {n} qubits")
    print(f"  full Hilbert space : 2^{n} = {2 ** n:,}")
    print(f"  feasible subspace  : {qubo.n_tilts}^{qubo.n_sectors} = "
          f"{qubo.n_tilts ** qubo.n_sectors:,} "
          f"({100 * qubo.n_tilts ** qubo.n_sectors / 2 ** n:.2f}% of it)")

    # ---- 1. the cost tensor must equal TiltQUBO.energy -------------------
    print("\n--- 1. full-space cost tensor vs TiltQUBO.energy ---")
    cost = full_space_cost(qubo)
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(400):
        x = rng.integers(0, 2, size=n)
        err = max(err, abs(cost[tuple(x)] - qubo.energy(x)))
    print(f"  max |cost tensor - qubo.energy| over 400 bitstrings : {err:.3e}")

    ok = feasible_mask(qubo)
    print(f"  feasible bitstrings counted: {int(ok.sum()):,} "
          f"(expected {qubo.n_tilts ** qubo.n_sectors:,})")

    # on feasible bitstrings the penalised energy must equal the physics objective
    err_f = 0.0
    for _ in range(200):
        cfg = rng.integers(0, qubo.n_tilts, size=qubo.n_sectors)
        x = qubo.encode(cfg)
        err_f = max(err_f, abs(cost[tuple(x)] - qubo.objective_from_config(cfg)))
    print(f"  max |cost - physics objective| on feasible bitstrings: {err_f:.3e}")

    # ---- 2. the CD term as Pauli strings --------------------------------
    print("\n--- 2. CD operator: Pauli decomposition vs numerics ---")
    terms, counts = cd_pauli_terms(qubo)
    print(f"  first-order CD terms: {counts['one_body_Y']} x Y_k, "
          f"{counts['two_body_YZ']} x (Y_k Z_j / Z_k Y_j)")
    print(f"  rotations per CD layer: {counts['rotations_per_layer']} "
          f"(n + 2|J| with n={n})")
    print(f"  max |sum of Pauli terms - apply_cd(v)| : "
          f"{_verify_pauli_terms(qubo):.3e}")

    # ---- 3. constrained vs unconstrained --------------------------------
    print("\n--- 3. does constraint preservation matter? ---")
    qubo_free = build_qubo(net, penalty_scale=0.0)     # subspace solver needs no penalty
    exact = brute_force(qubo_free)
    sub_cost = subspace_cost(qubo_free)
    p_rand_full = 1.0 / 2 ** n
    p_rand_sub = 1.0 / sub_cost.size
    print(f"  exact optimum {exact.objective:+.5f}  tilts={exact.config.tolist()}")
    print(f"  random over 2^{n} bitstrings  : P(opt) = {p_rand_full:.6%}")
    print(f"  random over feasible configs  : P(opt) = {p_rand_sub:.6%}\n")

    print(f"  {'solver':<34} {'P(opt)':>9} {'feasible':>10} {'obj':>11} {'s':>6}")
    r_full = solve_dcqo_ising(qubo, n_steps=40, mode="full", seed=3, n_shots=4000)
    print(f"  {'DCQO-X (2^n space, penalty)':<34} {r_full.prob_of_optimum:>8.3%} "
          f"{r_full.extra['feasible_fraction']:>9.1%} "
          f"{r_full.objective:>+11.5f} {r_full.seconds:>6.1f}")

    r_sub = solve_dcqo(qubo_free, n_steps=40, mode="full", seed=3, n_shots=4000)
    print(f"  {'DCQO-XY (subspace, no penalty)':<34} {r_sub.prob_of_optimum:>8.3%} "
          f"{1.0:>9.1%} {r_sub.objective:>+11.5f} {r_sub.seconds:>6.1f}")

    r_bf = solve_bf_dcqo(qubo_free, n_iters=4, n_steps=40, seed=3, verbose=False)
    print(f"  {'BF-DCQO-XY (subspace)':<34} {r_bf.prob_of_optimum:>8.3%} "
          f"{1.0:>9.1%} {r_bf.objective:>+11.5f} {r_bf.seconds:>6.1f}")

    print(f"\n  Note the P(opt) columns are not directly comparable: the full-space")
    print(f"  solver must find the optimum among {2 ** n:,} bitstrings, the subspace")
    print(f"  solver among {sub_cost.size:,}. The like-for-like numbers are the")
    print(f"  enhancement over each solver's own random baseline:")
    print(f"    DCQO-X  : {r_full.prob_of_optimum / p_rand_full:>8.0f}x random")
    print(f"    DCQO-XY : {r_sub.prob_of_optimum / p_rand_sub:>8.0f}x random")
    print(f"    BF-DCQO : {r_bf.prob_of_optimum / p_rand_sub:>8.0f}x random")
    print(f"  and the feasible fraction, which is {r_full.extra['feasible_fraction']:.1%}")
    print(f"  for DCQO-X and 100% for the subspace solvers by construction.")

    # ---- 4. BF bias at T=2 IS the published <Z> bias ---------------------
    print("\n--- 4. BF bias at T=2 reduces to the published h_b ~ <Z_j> form ---")
    bf = BiasField(n_sectors=n, n_tilts=2, cost_spread=1.0, strength=1.0)
    p = np.array([[0.8, 0.2]] * n)
    u = bf.from_marginals(p)
    z_exp = p[:, 0] - p[:, 1]
    implied = -0.5 * bf.scale * bf.strength * z_exp           # coefficient of Z_j
    print(f"  <Z_j> = {z_exp[0]:+.3f}  ->  u_j = ({u[0, 0]:+.5f}, {u[0, 1]:+.5f})")
    print(f"  u_j as a field on Z_j: {implied[0]:+.5f} * Z_j  "
          f"(check: {abs(u[0, 0] - implied[0]):.3e})")
