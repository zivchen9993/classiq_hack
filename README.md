# Temporal antenna-tilt optimization: QAOA-XY vs DCQO vs BF-DCQO

**QUBIT Hackathon 2026 - AT&T antenna tilt challenge - Shiran branch**

This repository now treats **temporal antenna tilt optimization** as the main
problem.  The snapshot problem remains a correctness/encoding building block,
while the headline benchmark optimizes all sector tilts and timesteps jointly:

\[
J(x)=\sum_t C_t(x_t)+\lambda\sum_{t,s}\mathrm{switch}(x_{t,s},x_{t+1,s}),
\]

with one-hot variables `x[t,s,k]`.  Each `C_t` is the same validated antenna
coverage/interference/handover surrogate used by the snapshot QUBO, so the
interaction graph contains both **spatial RF edges** and **temporal switching
edges**.

## What is actually demonstrated

The canonical direct-temporal run is
`results/2026_09_09__07_05_41_utc/direct_temporal_results.json`:

- 3 sectors x 3 tilt levels x 3 timesteps = **27 one-hot qubits** and 19,683
  feasible trajectories.
- Exact MILP and brute force agree at **J = -4.5938700627**.
- Best static trajectory is worse by **0.1461604** and uses 0 switches.
- Snapshot chasing, myopic hysteresis, single/multi-start greedy, simulated
  annealing, QAOA-XY, DCQO and BF-DCQO all use the **same objective/scoring
  code**. On this tiny seed, every one of them samples/finds the exact
  trajectory; this makes the instance a correctness tier, not a quantum-win
  tier.
- The quantum-reference optimum probabilities on this run are **2.708% QAOA-XY,
  0.0205% no-CD annealing, 0.702% DCQO, 2.615% BF-DCQO**. QAOA required 160
  parameter-search circuit evaluations; DCQO required none.
- Strong classical methods solve the same instance in milliseconds: MILP
  ~12.8 ms, 32-start greedy ~13.2 ms, simulated annealing ~45.4 ms. **There is
  no present quantum advantage.**

Across five independently generated antenna/mobility seeds
(`results/multiseed_2026_09_09__07_07_57_utc/`), exact sampled-hit fractions
were **40% QAOA-XY, 40% no-CD, 80% DCQO, 60% BF-DCQO**.  On this small sample,
plain DCQO is the most robust; feedback is **not** consistently beneficial.

## Antenna KPI result: the operational guardrail matters

The unconstrained surrogate optimum improves mean SINR from **11.772 dB**
(best static) to **11.889 dB** and mean spectral efficiency from **4.1080** to
**4.1364**, but handover failure is slightly worse (**53.157% -> 53.221%**) and
outage is also slightly worse (**2.305% -> 2.362%**).  An exhaustive 19,683-
trajectory Pareto/guardrail check therefore rejects that point if the rule is
"do not worsen handover failure or outage relative to best static."  Under that
guardrail the optimum collapses back to the static trajectory, with an
objective penalty of **0.1461604** and no throughput gain.

That is an important negative result: on this toy RF model, optimizing the
surrogate more aggressively does **not** produce a clearly superior operational
antenna policy.

The fixed-service-area approximation was also tested by recomputing the serving
cell at every tilt configuration.  Fixed-vs-dynamic configuration rankings are
highly correlated (minimum Spearman rho **0.984** for mean SINR and **0.993**
for spectral efficiency), but the exact dynamic spectral-efficiency optimum is
**different at all 3 timesteps**.  So fixed service areas preserve the broad
landscape but can change the selected optimum; this approximation must remain a
stated limitation.

## Snapshot encoding check

`snapshot_validation_2026_09_09.json` is a fresh 9-qubit p=1 validation run:

| method | feasible shots | P(optimum) |
|---|---:|---:|
| QAOA-X + one-hot penalty | 33.8% | 1.85% |
| QAOA-XY + Dicke/W initial state | **100%** | **16.93%** |

The XY mixer itself is prior art.  The useful antenna-specific result is that a
Dicke/W state plus block-local XY mixer **structurally preserves one-hot
feasibility**, so no one-hot penalty is required.

## Classiq status - important caveat

A real synthesized/executed Classiq DCQO artifact is saved at
`results/classiq_tiny_validation_2026_09_09/three_step/`:

- **12 qubits**, 6 `(time, sector)` one-hot blocks;
- 3 Suzuki-Trotter schedule steps;
- **depth 726**, **1,377 gates**, including **897 CX** gates;
- Pauli counts: 12 initial, 42 final, 84 CD terms (12 weight-2 + 72 weight-3);
- 5,000 simulator shots, 100% feasible, exact best trajectory sampled;
- Classiq `P(optimum)=43.82%`, independent local Trotter reference
  `P(optimum)=37.80%`.

The absolute **6.02 percentage-point** P(optimum) difference is about **8.8
binomial standard errors** for 5,000 shots, so the saved run does **not** satisfy
a tight numerical-equivalence criterion.  Mean objective is much closer
(-2.9109 Classiq vs -2.8904 local), but this mismatch is still the main open
scientific blocker.  Do not present the Classiq port as fully validated until a
rerun resolves it.

`classiq_validation_rerun.py` implements the reproducible rerun (no-CD, DCQO,
BF-DCQO plus the independent full-`2^n` reference).  This container cannot run
it because the Classiq SDK/credentials/network are unavailable here.

## Scaling evidence

`results/scaling_2026_09_09__07_08_44_utc/scaling_results.json` varies the
horizon for S=3, T=3.  Measured direct encoding resources are:

| H | qubits | final-H Pauli terms | CD Pauli terms | MILP time |
|---:|---:|---:|---:|---:|
| 1 | 9 | 36 | 126 | 5.1 ms |
| 2 | 18 | 90 | 324 | 10.5 ms |
| 3 | 27 | 144 | 522 | 19.5 ms |
| 4 | 36 | 198 | 720 | 18.8 ms |

This supports **encoding/Hamiltonian scaling only** on the tested range.  There
is only one saved Classiq depth point (12 qubits, depth 726), so no Classiq
circuit-depth growth law is claimed or extrapolated.

## Reproduce the local evidence

```bash
pip install -r requirements.txt
pytest -q

python snapshot_validation_benchmark.py
python direct_temporal_benchmark.py --shots 600 --qaoa-depth 1 --qaoa-maxiter 40 --dcqo-steps 8 --bf-iters 3
python direct_temporal_multiseed.py --seeds 0,1,2,3,4 --shots 300 --qaoa-maxiter 24 --dcqo-steps 8
python direct_temporal_scaling.py --max-horizon 4
python direct_temporal_assignment_sensitivity.py --direct results/2026_09_09__07_05_41_utc/direct_temporal_results.json
python direct_temporal_pareto.py --direct results/2026_09_09__07_05_41_utc/direct_temporal_results.json
```

Generate all final figures strictly from saved artifacts:

```bash
python direct_temporal_visualize.py \
  --direct results/2026_09_09__07_05_41_utc/direct_temporal_results.json \
  --multiseed-summary results/multiseed_2026_09_09__07_07_57_utc/multiseed_summary.json \
  --multiseed-csv results/multiseed_2026_09_09__07_07_57_utc/multiseed_rows.csv \
  --scaling results/scaling_2026_09_09__07_08_44_utc/scaling_results.json
```

For an authenticated Classiq environment:

```bash
python -c "import classiq; classiq.authenticate()"
python classiq_validation_rerun.py --shots 10000 --steps 3 --time-scale 2
python show_classiq_circuit.py results/classiq_tiny_validation_2026_09_09/three_step/classiq_program.json
```

The current audit and acceptance-criteria status are in
[`FINAL_AUDIT_2026_09_09.md`](FINAL_AUDIT_2026_09_09.md).  Historical snapshot
claims were archived under `background/*_HISTORICAL_PRE_CANONICAL_2026_09_09.md`
and are not canonical evidence.
