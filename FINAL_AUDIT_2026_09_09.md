# Final audit against `CODEX_TEMPORAL_ANTENNA_CONTINUATION_BRIEF.md`

## Verdict

**Not fully complete under the brief's own acceptance criteria.** Most requested
implementation work is done, but the Classiq numerical-equivalence criterion is
still failing. The correct submission posture is to show the real Classiq
circuit/resource artifact and explicitly label the remaining validation gap.

## Checklist

| requirement | status | audit finding |
|---|---|---|
| audit/freeze old state and contradictions | DONE | `background/AUDIT_2026_09_09.md`; old canonical docs archived |
| machine-readable common result schema | DONE | `result_schema.py`, direct JSON records resource units separately |
| provenance for runs | PARTIAL | software/hardware/seeds/timestamps saved; uploaded zip contains no `.git`, so commit is `null` |
| baseline tests before changes | DONE | `results/baseline_2026_09_09/` |
| regression tests after changes | DONE | 49 passed; `results/final_audit_2026_09_09/pytest_after.txt` |
| retain snapshot problem | DONE | snapshot QUBO/tests + fresh 9-qubit XY feasibility benchmark |
| retain candidate-chain DP as negative control | DONE | `temporal.py`; docs state it is polynomial/easy |
| direct `x[t,s,k]` temporal QUBO | DONE | `direct_temporal.py` |
| spatial + temporal edges | DONE | exact QUBO construction and tests |
| exact/best-known solver | DONE | verified SciPy MILP + brute-force agreement on tiny tier |
| best static | DONE | canonical benchmark |
| snapshot chasing | DONE | canonical benchmark |
| hysteresis/rolling horizon | DONE | `myopic_hysteresis_trajectory` |
| single/multi-start local search | DONE | 1 and 32 restarts |
| simulated annealing/tabu | DONE | simulated annealing |
| QAOA-XY on same temporal objective | DONE | local feasible-subspace reference |
| no-CD annealing ablation | DONE | matched direct benchmark |
| DCQO first-order CD | DONE | local + Classiq gate-level implementation |
| BF-DCQO | DONE | local and Classiq implementation path |
| Classiq Dicke/W blocks | DONE | saved synthesized program + source |
| programmatic `i[Hd,Hf]` Pauli generation | DONE | includes measured weight-3 terms |
| Classiq Suzuki-Trotter | DONE | saved three-step program |
| Classiq execution | DONE | saved 5,000-shot simulator artifact |
| Classiq/local equivalence | **FAIL / BLOCKER** | P(opt) differs 43.82% vs 37.80%; ~8.8 sigma |
| exact RF KPI evaluation | DONE | SINR, edge SINR, spectral efficiency, outage, handover |
| switching cost reported with KPI | DONE | every direct method stores trajectory breakdown |
| operational guardrail/Pareto | DONE | all 19,683 tiny trajectories scored; guardrail optimum becomes static |
| moving serving-cell sensitivity | DONE | high rank correlation but different exact optimum 3/3 timesteps |
| multiple antenna/traffic seeds | DONE (small sample) | 5-seed distributions; enough to reject one-seed claims, not enough for strong statistics |
| scaling tiers | PARTIAL | H=1..4 encoding/Pauli + classical/local-reference measurements; no multi-point Classiq depth sweep |
| figures from saved data | DONE | `figures/final_temporal/FIGURE_MANIFEST.json` |
| README/WRITEUP/STATUS agree | DONE for canonical docs | historical documents moved under `background/` |
| measured vs projection separated | DONE | no missing Classiq depth points extrapolated |
| no unsupported quantum-advantage claim | DONE | explicit negative conclusion |

## Result analysis

### 1. Classical hardness is not demonstrated on the current direct benchmark

MILP solves the 27-qubit-equivalent direct instance in about 12.8 ms; 32-start
greedy and simulated annealing also hit the optimum. This is a correctness and
algorithm-comparison tier only. A larger hardness tier is still needed before
making any statement about relative scaling of quantum solution quality.

### 2. CD helps on the chosen seed, but robustness matters more

At 8 matched digitized steps, the no-CD local reference has P(opt)=0.0205%,
DCQO 0.702%, and BF-DCQO 2.615%. This makes the CD ablation look strong on that
seed. Across five seeds, however, DCQO has the best exact sampled-hit fraction
(80% vs BF 60% and QAOA 40%). BF's higher mean P(opt) is driven by a favourable
outlier. The safe claim is that **CD is useful here; BF is not yet robustly
better than DCQO**.

### 3. The antenna objective, not the optimizer, is the main business risk

The unconstrained optimum gives only +0.116 dB mean SINR and +0.0283 mean
spectral efficiency over best static, while edge SINR, outage and handover
failure are slightly worse. Under the simple "do not worsen outage or handover"
guardrail, the best feasible policy is exactly the static tilt trajectory. A
better RF objective/guardrail formulation is more important than squeezing a
higher P(opt) out of the quantum solver.

### 4. Fixed association is broadly stable but not exact

Recomputing strongest-server association keeps snapshot rankings very similar
(rho >=0.984/0.993), yet picks a different dynamic-throughput optimum in every
timestep. That is a meaningful modeling sensitivity and must be disclosed.

### 5. The Classiq artifact is real, but validation is not closed

The saved 12-qubit program is a genuine synthesized and executed Classiq DCQO
circuit: depth 726, 1,377 total gates, 897 CX, 5,000 shots. It reaches the exact
best trajectory and stays feasible. But its P(opt) distribution is not within
sampling tolerance of the local Trotter reference. The most likely next task is
to debug convention/order/indexing differences with a 1-step then 3-step exact
amplitude comparison, **before** any larger Classiq benchmark.

## Work added in this completion pass

- `myopic_hysteresis_trajectory` and benchmark row;
- fixed local DCQO benchmark resource bug;
- fresh direct canonical run;
- 5-seed benchmark + aggregate statistics;
- direct horizon scaling sweep;
- fresh snapshot X-vs-XY feasibility benchmark;
- dynamic serving-cell sensitivity evaluator + analysis;
- exhaustive direct temporal operational Pareto/guardrail analysis;
- figure generator using saved JSON/CSV only;
- Classiq logical/resource/validation figures;
- reproducible `classiq_validation_rerun.py`;
- `show_classiq_circuit.py` helper for the actual Classiq UI visualizer;
- canonical documentation rewrite;
- 49 passing tests.

## What I would do next, in order

1. Run `classiq_validation_rerun.py` in an authenticated Classiq environment.
2. If mismatch remains, reduce to one Trotter step and compare exact
   probabilities term-by-term/qubit-order-by-qubit-order; do not tune RF or BF
   until this is resolved.
3. After equivalence passes, collect at least 3 Classiq resource points varying
   H or S so circuit depth/CX scaling is measured rather than inferred.
4. Build a moderately harder direct antenna instance where strong classical
   heuristics do not trivially hit the MILP optimum, while retaining a MILP
   bound/time limit.
5. Re-tune or constrain the antenna objective so a temporal policy improves
   throughput **without** worsening handover/outage guardrails.
