# Project status

Updated as work proceeds, so a interrupted run can be resumed without
re-deriving context. See `background/implementation_details.md` for the
working conventions this file exists to satisfy.

## 2026-09-09 continuation checkpoint

The continuation brief is now being implemented with the temporal problem as
the primary target. New work is intentionally not folded into the older
snapshot headline tables until a canonical Classiq rerun exists.

- `result_schema.py` defines the canonical provenance and solver-resource
  schema. Unlike resources are stored in separate fields.
- `background/AUDIT_2026_09_09.md` freezes contradictions in the old docs,
  terminal transcripts and hardcoded figures. Old outputs are historical.
- `direct_temporal.py` implements the unreduced `x[t,s,k]` QUBO with spatial
  and temporal edges. Exhaustive tests verify direct objective = QUBO = Ising
  = reduced-subspace energy.
- `classical_baseline.exact_milp` supplies an exact/bounded MILP oracle using a
  verified product linearization.
- `classiq_dcqo.py` programmatically derives the constrained XY driver and
  first-order `i[H_d,H_f]` Pauli strings, including three-body strings, and
  builds DCQO/BF-DCQO Suzuki-Trotter circuits. Its Pauli Hamiltonians match the
  independent NumPy operators on tiny cases. Actual Classiq synthesis and
  execution remain pending because Classiq is not installed/authenticated in
  the current environment.
- `direct_temporal_benchmark.py` checkpoints every solver into the canonical
  schema. The corrected tiny run is
  `results/2026_09_09__06_23_54_utc/direct_temporal_results.json`.
- Current suite: 46 tests passing. The original pre-change run (30 passing) is
  saved as `results/baseline_2026_09_09/pytest_before.xml`.

The corrected 27-qubit run is a correctness tier only. QAOA-XY, DCQO and
BF-DCQO sampled the exact MILP trajectory; a matched five-step no-CD anneal did
not. Strong classical heuristics also reached the exact answer in milliseconds,
so this is not evidence of quantum advantage.

---

## Where things stand

### Done and verified

| Area | Files | State |
|---|---|---|
| RF / coverage model | `rf_model.py` | Working. 3GPP-style patterns, log-distance path loss, fixed service areas, exact joint-SINR evaluator. |
| QUBO / Ising | `qubo_builder.py` | Working. One-hot per sector, verified to 1e-14 against the physics objective; Ising form verified to 1e-13. |
| Classical baselines | `classical_baseline.py` | Working. Brute force, greedy local search, simulated annealing, uniform baselines. |
| Constrained QAOA (XY mixer) | `qaoa_local.py`, `qaoa_subspace.py` | Working. Feasible-subspace simulation reaches 36 qubits. |
| Classiq execution | `quantum_solve.py` | Working. |
| Parameter transfer | `param_transfer.py` | Working. |
| Figures | `visualize.py` -> `figures/` | 6 figures generated. |
| **Temporal: mobility** | `mobility.py` | **Working.** Moving-hotspot demand; snapshots share the link budget. |
| **Temporal: core** | `temporal.py` | **Working.** Shared candidate pool, exact DP, budget DP, policies, temporal QUBO. QUBO reproduces DP to 6.7e-15. |
| **Temporal: benchmark** | `temporal_benchmark.py` | **Done.** Full run completed; results below. |
| **Temporal: figures** | `temporal_visualize.py` -> `figures/temporal_*.png` | **Done.** 4 figures, generated from saved JSON (no QAOA re-run needed). |

### Temporal results (run `2026_09_08__17_38_03`, 9 sectors x 4 tilts, H=8)

1. **The coherence premise holds.** Demand drifts steadily but only
   **0.86 of 9 sectors** change optimum per step, and the cost of not
   retuning collapses to ~0 after step 2. The previous solution IS a good
   warm start.
2. **Planning has a real operating window.** For switching weight
   lam in **[0.01, 0.06]** the planned trajectory strictly beats both
   "never retune" and "chase the optimum". Outside it the optimum
   degenerates to one of those baselines -- shown, not hidden.
3. **At lam=0.04:** DP J=0.1806 vs static 0.2408, greedy 0.2400,
   hysteresis 0.2147. DP uses 3 sector-moves in **1 event**; greedy uses 6
   moves across 4 events.
4. **The objective gain does NOT reach the network.** 25% better surrogate
   objective produced **-0.03 dB SINR** and **+0.3pp handover failure** --
   i.e. within noise of never retuning, at the cost of 3 antenna moves.
   This is the surrogate ceiling, measured rather than assumed, and it
   bounds what ANY optimizer can deliver here, quantum or classical.
5. **QAOA lost to DP outright.** On the 48-qubit chain encoding, p=4 did
   **not** reach the optimum (0.19518 vs 0.18061) after **388 s**; DP is
   exact in **0.2 ms**. On the budget encoding QAOA did find the optimum
   (P(opt) 0.473%) in 311 s -- still versus 1 ms for DP.
6. Per-sector switch counts on the DP trajectory max out at **1**, so a
   per-sector switch limit would not even bind on this instance. The
   Level 3 hardness is real in principle but **not exercised here**, and
   the benchmark says so explicitly.

### Known issues, not yet fixed

1. **`WRITEUP.md` and `README.md` contradict each other** on the same Classiq
   experiment: P(optimum) 0.00% vs 1.00%, feasible 22.8% vs 8.3%,
   QAOA-XY P(optimum) 0.20% vs 8.70%. Must be reconciled before submission.
2. `WRITEUP.md:157` / `README.md:35` claim 36 qubits is "out of reach for any
   full statevector simulator" -- false as written (that is ~1.1 TB; published
   simulations reach 45+ qubits). Should say "beyond commodity hardware".
3. The XY-mixer ansatz is labelled "(ours)" with no citation. It is prior art
   (Hadfield et al. 2019; Wang et al. 2020) and must be cited.
4. "855x better than standard QAOA" and "worse than random guessing" both rest
   on a random baseline that is defined two different ways in the same
   document. Needs one definition, applied consistently.
5. Amortization arithmetic subtracts three different units (objective
   evaluations, circuit shots, circuit evaluations). The "repaid after 9
   instances" headline depends on it.
6. Simulated annealing is advertised in the file table but appears in no
   results table.
7. GitHub push to `zivchen9993/classiq_hack` is **not done** -- local commit
   `b1a8c8a` exists, device-code auth expired before it was entered.
   `antenna_quantum_advantage_chat.md`, `background/`, `*.pdf` and `.claude/`
   are gitignored.

---

## How to run things

Each of these is a complete command. Python lives at
`C:\Users\shiranev\AppData\Local\Programs\Python\Python312\python.exe`
(referred to as `$PY` below; `python` alone is the Store stub and does not work).

```powershell
$PY = "C:\Users\shiranev\AppData\Local\Programs\Python\Python312\python.exe"
cd "c:\Users\shiranev\Hackathon"

& $PY -u rf_model.py              # RF model self-check, ~2 s
& $PY -u qubo_builder.py          # QUBO/Ising numerical verification, ~2 s
& $PY -u mobility.py              # mobility model self-check, ~5 s
& $PY -u temporal.py              # temporal module self-check, ~15 s
& $PY -u temporal_benchmark.py    # FULL temporal benchmark, ~3-6 min
& $PY -u benchmark.py             # snapshot benchmark, ~2 min
& $PY -u visualize.py             # regenerate all figures, ~40 s
```

Run them in that order; each depends only on the ones above it.

---

## Temporal optimization: what it is

The snapshot problem asks "best configuration for the current user
distribution?". The temporal problem asks "what SEQUENCE of configurations
keeps the network near-optimal as demand moves, without reconfiguring it
constantly?"

    J = sum_t regret(theta_t) + lam * sum_t switching_cost(theta_t, theta_t-1)

Each timestep is a one-hot block over M candidate configurations -- structurally
identical to a sector being a one-hot block over T tilts -- so the existing XY
ansatz, subspace simulator and Classiq export all apply unchanged.

**The honest framing, which the benchmark leads with:**

| Level | Structure | Exact classical cost | Verdict |
|---|---|---|---|
| 1. chain (regret + adjacent switching) | line graph | DP, O(H*M^2), sub-ms | Quantum offers nothing. Said plainly. |
| 2. + cumulative deviation budget | complete graph | DP with budget in the state, O(H*M^2*D) | Harder to ENCODE, not asymptotically harder. |
| 3. + per-sector switch limits | needs one counter per sector | state (r+1)^S, exponential | Genuinely breaks exact DP. Quartic -> HUBO, not encoded. |

Reporting Level 1 as if it were hard is the fastest way to lose a technical
judge, so DP is the headline and the quantum result is verified against it.
