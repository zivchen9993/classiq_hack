# Project status

Updated as work proceeds, so a interrupted run can be resumed without
re-deriving context. See `background/implementation_details.md` for the
working conventions this file exists to satisfy.

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
| **Temporal: benchmark** | `temporal_benchmark.py` | **Running now** -- see "In flight". |

### In flight

- `temporal_benchmark.py` full run. Expected ~3-6 min on CPU (numpy only, no
  GPU used anywhere in this project). Sections 0-3 are sub-second; the time
  is almost entirely the two QAOA solves in sections 4 and 4b
  (48 qubits, 6^8 = 1,679,616-amplitude feasible subspace, p=4 with INTERP).

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
