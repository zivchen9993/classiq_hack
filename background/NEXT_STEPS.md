# Next Steps — Time-Marched DCQO for Antenna Tilt Optimization

**Project:** QUBIT Hackathon 2026 · AT&T Challenge 1 (Antenna tilt optimization)
**Approach:** Classical RF model + polynomial (HUBO) surrogate + **digitized counterdiabatic quantum optimization (DCQO)** with a bias field carried across timesteps.
**Explicitly not QAOA.**

**Hard deadline:** submission window 11:00–12:00, Wed 9 Sep 2026. Everything below is time-boxed against that.

---

## 0. Status tracker (update this section as you go)

Keep this table current. It is the resume point if a run dies or the session is restarted.

| Phase | Description | Status | Output dir | Notes |
|---|---|---|---|---|
| P0 | Repo scaffold, config, logging | ☐ not started | — | |
| P1 | RF/network model + KPIs | ☐ not started | — | |
| P2 | Classical baselines | ☐ not started | — | |
| P3 | Surrogate fit (order 1→4) | ☐ not started | — | |
| P4 | DCQO solver on Classiq | ☐ not started | — | |
| P5 | Static-snapshot validation | ☐ not started | — | |
| P6 | Mobility model | ☐ not started | — | |
| P7 | Time-marched BF-DCQO | ☐ not started | — | |
| P8 | Benchmarks + figures | ☐ not started | — | |
| P9 | Write-up + video | ☐ not started | — | |

Status values: `☐ not started` / `▶ running` / `✔ done` / `✖ blocked`.

---

## 1. Design decisions locked in

These follow from the algorithm analysis. Do not re-litigate them tonight.

1. **Quantum core = DCQO / BF-DCQO**, not QAOA. Non-variational, shallow, ingests 3-body terms natively, and has a published record of beating QAOA / quantum annealing / SA / Tabu on three-local Ising spin glasses.
2. **Surrogate order is measured, not assumed.** Fit orders 1–4, report held-out error, and feed the *smallest faithful* representation to the quantum solver.
3. **No quadratization** unless the surrogate turns out to be purely pairwise. Quadratizing cubic terms adds ancillas and penalty weights that change the instance.
4. **The switching cost is the bias field.** The operational penalty for moving an antenna, `C(θ_t, θ_{t-1})`, is a linear term in the cost function, i.e. a longitudinal bias `h^b`. Zero extra qubits for the temporal coupling. No `N·K·T` time-expanded encoding.
5. **Three time axes, kept separate in code and in naming:**
   - `tau` — imaginary time (not used in the main path; optional F-VQE branch).
   - `s` — annealing parameter *inside one circuit* (DCQO schedule).
   - `t` — physical network time (the outer mobility march).
6. **Benchmarks must include a locally-structured-graph-friendly classical solver** (DP/exact for the temporal layer, and a tensor-network or exhaustive baseline for the snapshot). Omitting it is the easiest way to lose the benchmarking 20%.

### Two traps to guard against in code

- **The diagonal-commuting trap.** `H(s) = (1-s)H_t + s·H_{t+Δt}` is diagonal for all `s`; its eigenvectors are computational basis states independent of `s`, so evolving under it cannot move amplitude between configurations. Any schedule must include a transverse-field term:
  ```
  H(s) = A(s)·H_X + B(s)·[(1-λ(s))·H_t + λ(s)·H_{t+Δt}]
  ```
  with `A(s)` non-zero in the interior. Add an assertion in the schedule class that rejects `A(s) ≡ 0`.
- **Silent quadratization.** If any code path converts a HUBO to a QUBO, it must log loudly and record the penalty weights used.

---

## 2. Repository layout

```
antenna_dcqo/
├── README.md
├── NEXT_STEPS.md                  # this file
├── requirements.txt
├── run_all.sh                     # sequential driver, `set -e`
├── configs/
│   ├── default.yaml               # master config, all toggles
│   ├── tiny.yaml                  # 6 sectors × 4 tilts = 12 qubits (dev)
│   └── medium.yaml                # 9 sectors × 4 tilts = 18 qubits (stretch)
├── src/
│   ├── common/
│   │   ├── config.py              # Config (loads yaml, validates)
│   │   ├── run_context.py         # RunContext -> output dir yyyy_MM_dd__hh_mm_ss
│   │   └── logging_utils.py       # ProgressLogger (stdout + file)
│   ├── rf/
│   │   ├── antenna_pattern.py     # AntennaPattern (azimuth/elevation, tilt)
│   │   ├── propagation.py         # PathLossModel, ShadowFading
│   │   ├── network.py             # Sector, Site, Network (hex grid + wrap-around)
│   │   └── kpi.py                 # KpiCalculator (SINR, spectral eff., utility)
│   ├── objective/
│   │   ├── tilt_config.py         # TiltConfiguration (encode/decode, Hamming ops)
│   │   ├── objective.py           # RfObjective (calls KpiCalculator)
│   │   └── surrogate.py           # PolynomialSurrogate (order 1..4), HuboModel
│   ├── classical/
│   │   ├── brute_force.py         # BruteForceSolver (parallel)
│   │   ├── simulated_annealing.py # SimulatedAnnealingSolver
│   │   ├── local_search.py        # GradientAscentSolver (Bell Labs style)
│   │   ├── tabu.py                # TabuSolver
│   │   └── temporal_dp.py         # TemporalDpSolver (shortest path over candidates)
│   ├── quantum/
│   │   ├── hamiltonian.py         # IsingHamiltonian (Pauli list, from HuboModel)
│   │   ├── schedule.py            # AnnealingSchedule (sin^2, asserts A(s)!=0)
│   │   ├── cd_terms.py            # CounterdiabaticTerms (1st-order nested comm.)
│   │   ├── dcqo.py                # DcqoSolver (Classiq suzuki_trotter)
│   │   └── bias_field.py          # BiasField (from previous solution / switching cost)
│   ├── dynamics/
│   │   ├── density.py             # UserDensityField (advection-diffusion)
│   │   ├── mobility.py            # MobilitySimulator (samples users per t)
│   │   └── time_march.py          # TimeMarchController (outer t loop + trigger)
│   └── analysis/
│       ├── metrics.py             # comparison metrics
│       └── plots.py               # all figures
├── scripts/
│   ├── run_p1_rf_sanity.py
│   ├── run_p2_baselines.py
│   ├── run_p3_surrogate.py
│   ├── run_p4_dcqo_smoke.py
│   ├── run_p5_snapshot.py
│   ├── run_p7_timemarch.py
│   └── run_p8_figures.py
├── tests/
│   ├── test_antenna_pattern.py
│   ├── test_propagation.py
│   ├── test_kpi.py
│   ├── test_tilt_config.py
│   ├── test_surrogate.py
│   ├── test_hamiltonian.py
│   ├── test_schedule.py
│   ├── test_cd_terms.py
│   ├── test_dcqo.py
│   ├── test_bias_field.py
│   ├── test_density.py
│   └── test_time_march.py
└── outputs/
    └── 2026_09_08__22_14_03/      # auto-generated, see RunContext
```

Every phase writes into `outputs/<timestamp>/` with subdirs `figures/`, `data/`, `logs/`, plus a `run_config.yaml` snapshot and a `STATUS.md` that the script updates after each stage.

---

## 3. Phase plan

Each phase: what to build, what to test, what to plot, expected runtime.

Runtime notes: a 12–18 qubit statevector simulation is not GPU-bound — it runs in well under a second per circuit on CPU. **The A100 matters for the classical RF sweep and brute-force enumeration**, which are embarrassingly parallel over (configuration × grid point × fading realization). Parallelize those; do not bother GPU-accelerating the simulator.

---

### P0 — Scaffold (30 min)

- `Config` loads YAML, validates, exposes attribute access, and every feature flag defaults to a safe value.
- `RunContext` creates `outputs/<yyyy_MM_dd__hh_mm_ss>/`, snapshots the config, opens the log.
- `ProgressLogger` prints `[P4 12/40] ...` style lines with elapsed + ETA, and mirrors to `logs/run.log`.

**Test:** `test_run_context` — directory name matches the format regex, config snapshot round-trips.

---

### P1 — RF model and KPIs (1.5 h)

Build the classical truth model. Parameters from the two supplied papers so the numbers are defensible.

- `AntennaPattern`: Gaussian main beam + sidelobe floor, azimuth and elevation cuts, total gain = max(Gaz + Gel, SLL0) + G0. Defaults: HPBW_el 6.5°, SLL_el −17 dB, HPBW_az 65°, SLL_az −25 dB, SLL0 −30 dB, G0 18 dBi (Ericsson Table I). Electrical tilt only, `±10°` around a nominal, per the AT&T brief.
- `PathLossModel`: `128.1 + 37.6·log10(d_km)` (Bell Labs) as default, `134 + 35·log10(d_km)` (Ericsson) as toggle.
- `ShadowFading`: lognormal, σ = 8 dB, spatially correlated, decorrelation length 50 m. **Toggle: on/off, and seed-controlled.**
- `Network`: hex grid, 3 sectors/site, inter-site distance 500 m, wrap-around toggle.
- `KpiCalculator`: per-user SINR, spectral efficiency `log2(1+SINR)`, then the utility
  ```
  U = (1/P) · Σ_m [ w_avg · s_m,avg + w_edge · s_m,edge ]
  ```
  with `w_avg = 1`, `w_edge = 10`, `s_edge` = 5% quantile (Bell Labs Eq. 1). Also report mean SINR, 5th-percentile SINR, mean throughput, and outage fraction so you can quote the brief's KPIs directly.

**Tests:** pattern peaks at boresight and is symmetric; tilt shifts the elevation peak by exactly the tilt angle; path loss monotone in distance; SINR improves when interferers are switched off; utility is invariant to sector relabeling.

**Plot (P1 figure set):** antenna gain vs elevation for 3 tilts; SINR heatmap over the grid for uniform tilt 0°, 5°, 10°; utility vs uniform tilt (should show a clear interior optimum — this is your sanity check that the coverage/interference trade-off is real).

**Runtime:** seconds per configuration on the grid; the uniform-tilt sweep parallelizes over tilt values.

---

### P2 — Classical baselines (1 h)

- `BruteForceSolver` — parallel over all `K^N` configurations. At `N=6, K=4` that is 4096 evaluations: exact ground truth. **This is what makes every later claim checkable.**
- `SimulatedAnnealingSolver` — geometric cooling, configurable reads/sweeps.
- `GradientAscentSolver` — the Bell Labs heuristic variant (cluster-centred, ±Δ probes, normalized step). This is the serious baseline; a judge who read the reference will ask about it.
- `TabuSolver` — optional if time is short.

**Tests:** brute force matches an independent exhaustive loop on `N=3`; SA and local search never return an infeasible tilt; both reach the brute-force optimum on `N=4` within tolerance.

**Plot:** utility vs objective evaluations for each solver, with the exact optimum as a horizontal line.

**Runtime:** brute force at `N=6, K=4` with 2000 grid points ≈ a few minutes on many cores; log progress every 100 configurations.

---

### P3 — Surrogate structure analysis (1.5 h) — **this is a result in itself**

- Sample `M` configurations (Latin hypercube or exhaustive at tiny size), evaluate `U_RF`.
- `PolynomialSurrogate.fit(order=k)` for `k = 1,2,3,4` in the ±1 spin basis over the binary encoding of tilt.
- Report per order: held-out MAE / R², number of non-negligible coefficients, and the coefficient-magnitude distribution by order.
- Decide the encoding: **binary** (`N·log2(K)` qubits) by default, **one-hot** (`N·K`) as a toggle for constraint-preserving experiments.

**Tests:** surrogate reproduces a synthetic known cubic exactly; fitted order-2 model on a synthetic pairwise objective has ~zero cubic coefficients; encode/decode round-trips for all configurations.

**Plots:** (a) held-out error vs polynomial order; (b) coefficient magnitude spectrum grouped by order (1-body / 2-body / 3-body); (c) coupling strength vs geographic sector distance — this is the locality evidence.

**Deliverable claim:** `ε_surrogate = E_θ|U_RF(θ) − Û(θ)|` measured, and later compared against `Δ_opt` = the utility gap between quantum and classical solutions. If `ε_surrogate ≫ |Δ_opt|`, say so on the slide. That number is more interesting than a fake advantage claim.

---

### P4 — DCQO solver on Classiq (2.5 h) — **the critical path**

- `IsingHamiltonian`: converts `HuboModel` coefficients into a Pauli term list (`Z_i`, `Z_iZ_j`, `Z_iZ_jZ_k`). No quadratization.
- `AnnealingSchedule`: `λ(t) = sin²[(π/2)·sin²(πt/2T)]`, with `A(s)` for the transverse field and `B(s)` for the cost. **Assert `A(s)` is not identically zero** (the diagonal-commuting trap).
- `CounterdiabaticTerms`: first-order nested-commutator approximation of the adiabatic gauge potential. With an X driver and Ising cost, the CD terms are Y-type: `Y_i`, `Y_iZ_j + Z_iY_j`, and the three-body analogues. Keep first order only — higher orders explode the depth.
- `DcqoSolver`: builds `H_total(s) = H_ad(s) + λ̇·A_λ(s)`, discretizes `s` into `n_steps`, and emits one `suzuki_trotter` call per step (Classiq exposes `suzuki_trotter(pauli_operator, evolution_coefficient, repetitions, order, qbv)`). Initial state = product state from `BiasField`. Measure, then classically evaluate the sampled configurations against the true objective and keep the best.
- `BiasField`: builds `h^b` from (a) a previous solution, (b) measured `⟨Z_j⟩` from the last DCQO iteration (the BF-DCQO update), or (c) zero (cold start). All three selectable.

**Tests:** Hamiltonian expectation on a basis state equals the classical cost of that configuration (this single test catches most sign and factor-of-2 bugs); schedule endpoints are `λ(0)=0`, `λ(T)=1` with vanishing derivative; CD terms vanish as `T→∞`; DCQO on a 4-qubit instance with a known optimum finds it with probability > 0.5; bias field pointing at the optimum increases success probability.

**Plots:** schedule curves `A(s)`, `B(s)`, `λ(s)`; circuit depth / CX count vs `n_steps` (pull from Classiq synthesis); sampled energy histogram vs the exact spectrum.

**Runtime:** synthesis is cloud-side and is the real cost — expect seconds to a minute per circuit. **Cache synthesized programs by hash of (Hamiltonian, schedule, n_steps)** or you will burn the night on repeated synthesis.

**Fallback if Classiq synthesis becomes a blocker:** keep a `numpy` reference DCQO that Trotterizes the same Hamiltonian by dense matrix exponentiation at 12 qubits. Same schedule, same CD terms, same results structure. Use it to develop P5–P7 in parallel and swap the Classiq backend in when it works. Do not let synthesis latency block the physics. The submission still needs the Classiq path to run — build both, present the Classiq one.

---

### P5 — Static snapshot validation (1 h)

Single timestep, one interference cluster. Compare on the *true* RF objective:

| Solver | What it shows |
|---|---|
| Brute force | exact optimum (ground truth) |
| Gradient ascent (Bell Labs) | the serious classical baseline |
| Simulated annealing | the standard baseline |
| DCQO cold start | quantum, no warm start |
| DCQO + bias field | the warm-start effect |

Report: best utility found, approximation ratio vs exact, objective evaluations used, circuit depth, shots.

**Plot:** grouped bar chart of approximation ratio per solver, plus a KPI table (mean SINR, 5%-ile SINR, mean throughput, outage) for the best configuration of each.

---

### P6 — Mobility model (1 h)

- `UserDensityField`: `∂ρ/∂t + ∇·(ρv) = D∇²ρ` on a coarse grid, explicit finite differences. Two scenarios, both toggleable: (a) commuter blob translating across the network, (b) hotspot dispersing (stadium emptying).
- `MobilitySimulator`: draws user positions from `ρ_t` at each timestep, with a fixed seed per timestep for reproducibility.

**Tests:** total mass conserved to numerical tolerance; pure advection translates the blob by `v·Δt`; pure diffusion monotonically increases variance; zero velocity + zero diffusion is a fixed point.

**Plot:** density field snapshots across `t`, with sector boundaries overlaid.

**Runtime:** negligible. Parallelize the per-timestep user sampling and RF evaluation across timesteps.

---

### P7 — Time-marched BF-DCQO (2 h) — **the novelty**

`TimeMarchController` loop, for `t = 1..T` (target `T = 8`):

1. Get `ρ_t`, sample users, build `U_RF`.
2. Refit / update the surrogate → `H_t` (toggle: refit vs incremental update).
3. Add the switching-cost bias: penalty `λ_switch · d(θ, θ_{t-1})` folded into the linear terms.
4. Build the bias field `h^b` from `θ*_{t-1}`.
5. Run DCQO with **fewer Trotter steps than cold start**, scaled by `‖δH‖` (this is the efficiency claim).
6. Evaluate samples on the true objective; record `θ*_t`.
7. **Re-solve trigger:** if the sample energy variance or bimodality exceeds a threshold — the near-degeneracy signature of a level crossing — fall back to a cold-start (full-schedule) DCQO run for that step. Log every trigger event.

**Tests:** with a static density the march is a fixed point (no tilt changes, no triggers); with `λ_switch = 0` the march reproduces per-step greedy snapshot solutions; with `λ_switch → ∞` tilts never move; the trigger fires on a hand-built degenerate instance.

**Comparison arms:**

| Arm | Description |
|---|---|
| Static | never change tilt (initial config held) |
| Greedy snapshot | jump to `θ*_t` every step, cold solve each time |
| Temporal DP | exact shortest path over per-step candidate sets (`TemporalDpSolver`) — the classical champion for the temporal layer |
| **Time-marched BF-DCQO** | ours |

**Plots — these are the money figures:**
1. **Trotter steps / circuit depth needed at step `t` vs `‖δH‖`**, warm vs cold. The headline.
2. Cumulative quantum work (total Trotter steps, total shots) over the trajectory, warm vs cold.
3. Cumulative utility vs cumulative number of tilt changes, for all four arms (the Pareto view).
4. Success probability per shot, warm vs cold, per timestep.
5. Timeline strip showing where the re-solve trigger fired, annotated against `‖δH‖`.

---

### P8 — Benchmarks and figure pass (1 h)

- Add the honest baselines: exhaustive/DP where feasible, SA with a matched evaluation budget, and — if there is any time — a tensor-network or treewidth-based solve of the snapshot instance. On a geometrically local graph this will likely be near-exact, and saying so *is* the "where is the boundary of quantum advantage" result.
- Assemble every figure into `outputs/<timestamp>/figures/` with consistent styling, and write `RESULTS.md` with the numbers inline.
- Compute and record: `ε_surrogate`, `Δ_opt`, and the ratio. State the conclusion either way.

---

### P9 — Write-up and video (2 h, non-negotiable)

- Technical write-up: formulation → surrogate structure result → why DCQO not QAOA → time-marching mechanism → benchmarks → limitations.
- Pre-recorded demo video embedded in the slides (nothing runs live).
- Rehearse the three-minute Q&A answers: (1) why not QAOA, (2) why parallel tempering / DMRG doesn't already win, (3) what breaks near a decision boundary.
- Set the presentation link to public and verify in an incognito window.

---

## 4. Timeline (from ~22:00 tonight)

| Slot | Phases | Cut if behind |
|---|---|---|
| 22:00–23:00 | P0, P1 | shadow fading off, single fading realization |
| 23:00–00:00 | P2, start P3 | drop Tabu |
| 00:00–01:30 | P3, P4 | CD terms first-order only, `n_steps` small |
| 01:30–02:30 | P5 | drop the KPI table, keep utility only |
| 02:30–03:30 | P6, P7 | `T = 4` instead of 8, one mobility scenario |
| 03:30–04:30 | P7 finish, P8 | figures 1 and 3 only |
| morning | P9 | — |

### Minimum viable submission (if everything slips)

`N = 6`, `K = 4`, 12 qubits, statevector, no shadow fading, one mobility scenario, `T = 4`:

1. Utility vs uniform tilt (the trade-off is real).
2. Held-out error vs surrogate order (the structure result).
3. DCQO reaches the brute-force optimum on the snapshot.
4. Depth-vs-`‖δH‖` warm vs cold (the novelty).
5. One honest limitations slide.

That is a complete, defensible submission. Everything beyond it is upside.

---

## 5. How to run

Single command, stops at the first error:

```bash
bash run_all.sh
```

`run_all.sh` starts with `set -euo pipefail` and calls, in order:

```bash
# run_all.sh
set -euo pipefail
CONFIG=${1:-configs/tiny.yaml}

pytest -q tests/                                            # line 5
python -m scripts.run_p1_rf_sanity   --config "$CONFIG"     # line 6
python -m scripts.run_p2_baselines   --config "$CONFIG"     # line 7
python -m scripts.run_p3_surrogate   --config "$CONFIG"     # line 8
python -m scripts.run_p4_dcqo_smoke  --config "$CONFIG"     # line 9
python -m scripts.run_p5_snapshot    --config "$CONFIG"     # line 10
python -m scripts.run_p7_timemarch   --config "$CONFIG"     # line 11
python -m scripts.run_p8_figures     --config "$CONFIG"     # line 12
```

To run a single phase, copy the matching line above. To use the larger instance, `bash run_all.sh configs/medium.yaml`.

**Parameter choices are automatic, never manual.** Each script reads the previous phase's artifacts from `outputs/<latest>/data/` (resolved by `RunContext.latest()`):

- P4/P5 take the polynomial order from P3's held-out-error curve: the lowest order within 5% of the best held-out error.
- P7 takes `λ_switch` from P5's measured utility spread: `λ_switch = 0.1 · (U_best − U_worst)` over the sampled configurations.
- P7 takes the cold-start `n_steps` from P4's depth sweep: the smallest `n_steps` reaching ≥ 0.9 success probability on the snapshot instance.
- P7's re-solve trigger threshold is set to the 95th percentile of the sample energy variance observed across P5 runs.

Resume behaviour: every script checks `outputs/<latest>/STATUS.md` and skips phases already marked `✔ done` unless `--force` is passed.

---

## 6. Config toggles (`configs/default.yaml`)

Everything below is switchable without touching code.

```yaml
network:      { n_sites, sectors_per_site, isd_m, wrap_around, seed }
antenna:      { hpbw_az, hpbw_el, sll_az, sll_el, sll_floor, g0_dbi, tilt_set }
propagation:  { model: bell_labs|ericsson, shadow_fading: on|off, sigma_db, decorr_m }
kpi:          { w_avg, w_edge, edge_quantile, grid_points, fading_realizations }
encoding:     { scheme: binary|one_hot }
surrogate:    { orders: [1,2,3,4], n_samples, holdout_frac, allow_quadratization: false }
classical:    { brute_force: on, sa: on, local_search: on, tabu: off, n_workers }
quantum:      { backend: classiq|numpy_reference, n_steps, trotter_order, trotter_reps,
                cd_order: 1, shots, cache_synthesis: true }
bias:         { source: none|previous_solution|measured_z, strength }
dynamics:     { scenario: commuter|dispersing, T, dt, velocity, diffusion }
time_march:   { lambda_switch: auto, adaptive_steps: on, trigger: on, trigger_threshold: auto }
output:       { figures: on, save_raw: on, verbose: on }
```

---

## 7. Optional branches (only if the main path finishes early)

| Branch | Value | Cost |
|---|---|---|
| Quantum-enhanced MCMC (short real-time evolution as a Metropolis proposal) | Best answer to "parallel tempering beats you" — it augments the classical champion instead of competing with it | ~2 h |
| F-VQE / filtering as an imaginary-time alternative at fixed `t` | Second quantum solver for the comparison plot | ~2 h |
| One-hot encoding with a constraint-preserving driver | Cleaner tilt semantics, more qubits | ~1 h |
| QRAO for the scalability slide | Best qubits-per-sector story | ~2 h |

Do not start any of these before P8 is done.
