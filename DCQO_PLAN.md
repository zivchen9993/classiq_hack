# DCQO / BF-DCQO implementation plan

Second quantum solver alongside the existing constrained QAOA, run on the
**same** dynamic (temporal) formulation so the two are directly comparable.
Written before the code, per `background/implementation_details.md` rule 13;
the "Progress" table at the bottom is updated as work lands, so an
interrupted session can be resumed without re-deriving anything.

---

## Why DCQO here

`background/NEXT_STEPS.md` locks in DCQO for three reasons that survive
contact with this codebase:

1. **Non-variational.** No classical outer loop. The existing QAOA result on
   the 48-qubit temporal instance cost 388 s, almost all of it COBYLA
   iterations. DCQO's cost is `n_steps` circuit layers and `n_shots`
   measurements -- nothing else. That is the honest resource comparison and
   it is the whole point of adding this solver.
2. **The switching cost is already a bias field.** The temporal objective's
   linear terms are exactly the longitudinal bias BF-DCQO wants, so warm
   starting across physical time costs zero extra qubits.
3. It ingests the cost Hamiltonian as-is, with no parameter training to
   transfer between instances.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Space | **Feasible one-hot subspace** (`T^S` amplitudes), same reduction as `qaoa_subspace.py` | Lets DCQO run on the identical 48-qubit temporal instance QAOA runs on. Without it the comparison would be at different sizes and worthless. |
| Driver | XY **ring** hopping per one-hot block, `H_d = -sum_i B_ring^(i)` | Conserves Hamming weight (so the subspace reduction is exact), and its ground state is exactly the uniform superposition -- the same initial state QAOA uses. A chain driver's ground state is *not* uniform, hence ring. |
| CD term | 1st-order variational nested commutator, `A = alpha(lam) * i[H_d, H_f]` | For a two-point interpolation `H(lam) = (1-lam)H_i + lam*H_f`, `[H, d_lam H] = [H_i, H_f]` exactly -- lam-independent, so the operator is built once. |
| `alpha(lam)` | Variational, `alpha = Tr(D Y)/Tr(Y^2)`, with the 5 required traces estimated once by Hutchinson probes | `Y(lam) = (1-lam)P + lam*Q` is linear in lam, so all lam-dependence collapses into 5 lam-independent scalars. Toggleable to a fixed coefficient. |
| Digitization | Per step: driver exp (per-axis `T x T` matrix), cost phase (diagonal), CD (exact 2-level `sigma_y` rotations per ring edge) | Every factor is applied exactly; the only approximation is the Trotter ordering, which is logged. |
| Bias field | Measured one-hot marginals `p_i(k)`, folded into the diagonal as `u_i(k) = -strength * scale * (p_i(k) - 1/T)`, and into the initial state as the exact product ground state of `-B + diag(u)` | The one-hot analogue of BF-DCQO's `h_b ~ <Z_j>`. Samples are ALWAYS scored on the true objective, never the biased one. |
| Quadratization | none | The temporal chain QUBO is already quadratic; no HUBO reduction happens anywhere, so nothing to log. |

### Traps guarded against in code

- **Diagonal-commuting trap** (`NEXT_STEPS.md` §1): `AnnealingSchedule`
  asserts the driver amplitude `1 - lam(s)` is non-zero in the interior, and
  `DcqoSolver` asserts the driver operator has non-zero off-diagonal weight.
- **Biased-objective leakage**: `DcqoResult.objective` is computed from
  `subspace_cost` of the UNBIASED problem for every sampled configuration.
  Asserted in the self-checks.
- **Silent parameter fitting**: `DcqoResult.evaluations` is the number of
  classical objective evaluations used to choose circuit parameters. It is 0
  for DCQO by construction, and the benchmark prints it next to QAOA's.

---

## Files

| File | Contents | Status |
|---|---|---|
| `dcqo.py` | Core engine. `AnnealingSchedule`, `SubspaceOperators`, `DcqoSolver`, `BiasField`, `solve_dcqo`, `solve_bf_dcqo`. Self-checks in `__main__`. | done |
| `dcqo_ising.py` | Full-Hilbert-space (`2^n`) DCQO from `TiltQUBO.to_ising()`: X driver, first-order Y / (YZ+ZY) CD terms. Small instances only (<= ~20 qubits). Independently validates the subspace engine and gives the penalty-vs-constraint-preserving comparison. | done |
| `dcqo_benchmark.py` | The head-to-head. Same instance, same objective, DCQO / BF-DCQO / QAOA-XY / DP / classical. | done |
| `dcqo_visualize.py` | Figures from the benchmark JSON. | done |
| `tests/test_dcqo.py` | Correctness tests (pytest, also runnable as a script). | done |

## Benchmark sections

0. Instance: `make_network(3)` + commuter mobility, horizon 8 -> the same
   `TemporalProblem` `temporal_benchmark.py` uses.
1. Correctness gate: subspace cost == QUBO objective; CD operator Hermitian
   and equal to `i[H_d,H_f]` by dense comparison at small size; schedule
   endpoints; unitarity of one digitized step.
2. **Snapshot** head-to-head (36 qubits): DCQO, BF-DCQO, QAOA-XY, greedy, SA,
   brute force.
3. **Temporal chain** head-to-head (48 qubits): DCQO, BF-DCQO, QAOA-XY,
   against the exact DP optimum.
4. Step sweep: `P(optimum)` and wall clock vs `n_steps` for DCQO, vs `p` for
   QAOA -- the depth/resource curve.
5. **Time-marched warm start**: for each physical timestep `t`, re-solve the
   snapshot with DCQO warm-started by a bias field from `theta*_{t-1}`, vs
   cold DCQO, vs QAOA with transferred parameters. This is the dynamic
   result: warm starting should cut the steps needed.
6. Resource accounting and honest summary.

## Runtime estimates

Everything here is **numpy statevector on CPU** -- no GPU is used and none
would help, because the subspace is only 1.7M amplitudes and the work is
memory-bound `T x T` matmuls. Stated for a single modern CPU core:

| Stage | Estimate |
|---|---|
| `dcqo.py` self-checks | ~20 s |
| `dcqo_ising.py` self-checks (12 qubits) | ~15 s |
| Benchmark §1-2 (36 qubits, DCQO+BF+QAOA p=3) | ~3 min |
| Benchmark §3 (48 qubits, DCQO n_steps sweep) | ~2 min |
| Benchmark §4 QAOA p=4 on 48 qubits (the slow arm) | ~6 min |
| Benchmark §5 time march (8 timesteps x 2 arms) | ~4 min |
| **full `dcqo_benchmark.py`** | **~15 min**; `--quick` ~90 s |

On an A100 the numbers are the same: none of this is GPU work.

## How to run

```powershell
$PY = "C:\Users\shiranev\AppData\Local\Programs\Python\Python312\python.exe"
cd "c:\Users\shiranev\Hackathon"

& $PY -u dcqo.py                  # engine self-checks,          ~20 s
& $PY -u dcqo_ising.py            # full-space DCQO self-checks,  ~15 s
& $PY -u tests\test_dcqo.py       # test suite,                   ~60 s
& $PY -u dcqo_benchmark.py        # full head-to-head,           ~15 min
& $PY -u dcqo_visualize.py        # figures from the newest json, ~20 s
```

`--quick` on the benchmark drops the horizon to 5, the pool to 5 and QAOA to
p=2 (~90 s) for a smoke test. `--no-qaoa` runs only the DCQO arms.

---

## Progress

| Step | State |
|---|---|
| Plan written | done |
| `dcqo.py` core engine | done |
| `dcqo.py` self-checks pass | done |
| `dcqo_ising.py` full-space engine | done |
| `tests/test_dcqo.py` | done, 14/14 pass |
| `dcqo_benchmark.py` | done |
| Benchmark full run | done, `results/2026_09_08__23_52_45` |
| `dcqo_visualize.py` | done, 5 figures |
| `STATUS.md` / `README.md` updated | done |
