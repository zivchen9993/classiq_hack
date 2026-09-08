# Quantum antenna-tilt optimisation driven by fluid-dynamic demand prediction

Working end-to-end pipeline. Runs in ~8 s on a laptop, pure NumPy/SciPy.

```
rho(t)  --lattice Boltzmann-->  rho_hat(t+dt)
        --3GPP RAN sim-->  4 KPIs  -->  F_safe(z)
        --Walsh-Hadamard-->  h_i, J_ij
        --QAOA-->  z*   (tilt +-2 deg per sector)
```

## Run it

```bash
python run_all.py     # full pipeline + results.json + landscape.npz
python plots.py       # 4 figures for the deck
python scaling.py     # M = 9, 12, 15 scaling study
```

## Files

| file | what it does |
|---|---|
| `netsim.py` | 3GPP TR 38.901 antenna + array factor, UMa pathloss, correlated Rayleigh fading, sticky A3 attachment, 4 KPIs |
| `objective.py` | KPI normalisation, weighted sum, guardrail penalties, `E = -F_safe` |
| `ising.py` | fast Walsh–Hadamard transform, Ising extraction, `kappa` truncation metric |
| `qaoa.py` | statevector QAOA for diagonal Ising cost (**swap point for Classiq**) |
| `lbm.py` | D2Q5 advection–diffusion for demand density (**swap point for QLBM**) |
| `run_all.py` | the experiment |
| `scaling.py` | M = 9/12/15 comparison vs multi-start greedy |

## Results as committed

```
kappa (deg<=2 Walsh energy)   0.956
2-local RMSE                  0.0394   (landscape sd 0.188)
rank correlation              0.977
2-local ground state          == true optimum  (gap 0.00000)

QAOA p=1  approx-ratio 0.478   finds optimum
QAOA p=2  approx-ratio 0.691   finds optimum
QAOA p=3  approx-ratio 0.756   finds optimum

HEADLINE, scored on true future demand:
  static tilts       0.696
  reactive           0.577
  proactive (LBM)    0.829     <- +0.252 over reactive
  oracle             0.915        closes 74.5% of the gap
```

## Swap point 1 — Classiq

`qaoa.py` simulates the circuit directly. The cost layer to build in Classiq is

```
for i in range(M):            RZ(2*gamma*h[i],  q[i])
for i < j:                    RZZ(2*gamma*J[i,j], q[i], q[j])   # CX–RZ–CX
mixer:                        RX(2*beta, q[i])  for all i
```

`h` and `J` come straight out of `ising.extract_ising()` and are saved in
`results.json`. Nothing else in the pipeline changes. Keep `qaoa.py` as the
reference to validate the Classiq circuit against.

## Swap point 2 — real demand data

`lbm.morning_commute()` is synthetic. Replace with the Telecom Italia Milan
grid: it is already a 100×100 lattice of 235 m cells at 10-minute resolution,
so an N×N sub-grid loads straight into `rho0`. Fit `u_field` from consecutive
frames (least-squares continuity fit or optical flow). No other change needed.

## Swap point 3 — QLBM

`lbm.DemandLBM` is written as the two canonical operators, `stream` and
`collide`, precisely because those are what a quantum LBM implements. Use this
class as the exact classical reference to validate any QLBM circuit —
compare `rho` field-by-field after each step.

## Honest limitations

- **QAOA does not beat greedy at M ≤ 15.** Multi-start coordinate ascent finds
  the same optimum. Do not claim otherwise on stage.
- κ ≈ 0.93–0.98 across all tested inter-site distances: the tilt landscape is
  genuinely dominated by first- and second-order effects. This *justifies* the
  2-local encoding but also explains why the instance is classically easy.
- The 5th/95th-percentile normalisation bounds, equal KPI weights, ±2° tilt
  step, and guardrail thresholds are design choices, not standardised values.
- RLF is a "SINR below threshold for N consecutive samples" proxy, not the
  full N310/T310/N311 procedure of TS 38.331 §5.3.10.
- Handover failure is modelled at the A3 trigger instant only; preparation and
  execution phases are not separated.

## Standards the model follows

- 3GPP TR 38.901 §7.3 — panel antenna pattern, electrical tilt via array weights
- 3GPP TR 38.901 §7.4 — UMa pathloss and shadow fading
- 3GPP TS 38.215 §5.1.5 — SS-SINR definition
- 3GPP TS 38.331 §5.5.4.4 — Event A3 handover trigger
- O'Donnell, *Analysis of Boolean Functions* Prop. 1.8 — Walsh expansion
- Farhi, Goldstone, Gutmann (2014) — QAOA

Verify version and clause numbers against the actual specs before publishing.
