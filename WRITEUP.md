# Antenna Tilt Optimization with Constrained QAOA

**QUBIT Hackathon 2026 - AT&T Challenge 1**

A quantum-classical hybrid for choosing per-sector antenna downtilt across a
cellular network, built on Classiq.

---

## 1. The problem

Every sector antenna has an electrically adjustable downtilt (+/-10 deg). The
tilt sets the trade-off the challenge brief describes:

- **too little downtilt** -> the beam overshoots into neighbouring cells -> inter-cell interference
- **too much downtilt** -> the footprint shrinks -> coverage holes and handover failures at the cell edge

The choices are **strongly coupled**: the best tilt for one sector depends on
what its neighbours do, because they share the same spectrum and the same cell
boundaries. With `T` tilt levels and `S` sectors there are `T^S` network-wide
configurations - 4^24 = 2.8e14 for just eight towers.

## 2. Our formulation

### 2.1 RF model (`rf_model.py`)

Classical, deliberately simple, fully vectorised:

- 3GPP-style vertical and horizontal antenna gain patterns
- log-distance path loss
- irregular geometry: jittered tower positions, varying heights (22-45 m), azimuths, transmit powers
- non-uniform **traffic demand** (Gaussian hotspots) weighting every grid point

The modelling decision that makes the quantum formulation possible: **each
sector's service area `S_i` is fixed** (assigned at a tilt-independent
reference tilt) and does not move when tilts change. That makes the coverage
term genuinely *unary* and the coupling terms genuinely *pairwise* - i.e. a
true QUBO with no higher-order terms.

### 2.2 The three cost terms

| term | type | meaning | pushes tilt |
|---|---|---|---|
| coverage | unary | demand-weighted mean SNR over the fixed service area | to best serve own demand |
| interference | pairwise | pairwise SINR degradation `P_i/(P_j+N)` in dB, both directions | **down** |
| handover | pairwise | fraction of cell-edge demand whose handover target drops below a 9 dB margin | **up** |

The last two **genuinely oppose each other**. That opposition is what makes the
landscape frustrated and multi-modal, and it is the honest reason a global
optimizer has anything to contribute.

### 2.3 QUBO / Ising (`qubo_builder.py`)

One-hot encoding: `x[i,t] = 1` iff sector `i` uses tilt level `t`, giving
`S x T` qubits.

```
C(x) = -w_cov * sum_i cov_i(t_i)  +  w_int * sum_ij ISR_ij(t_i,t_j)
       +w_ho  * sum_ij HO_ij(t_i,t_j)  +  P * sum_i (sum_t x[i,t] - 1)^2
```

Verified numerically (`python qubo_builder.py`):

- QUBO energy == physics objective on valid configurations (max error **1.0e-14**)
- Ising form == QUBO energy on *arbitrary* bitstrings, including infeasible
  ones (max error **1.3e-13**) - the penalty must survive the transformation,
  or QAOA would be optimizing a different Hamiltonian than we think

## 3. What we actually had to crack

### 3.1 The surrogate was initially invalid

Our first interference term was the natural one: mean linear
interference-to-signal ratio `P_j/P_i`. Measured against the exact joint-SINR
simulator over 400 random configurations, it gave a Spearman correlation of
**rho = -0.10** - essentially no relationship. Minimizing that QUBO would not
have improved the network at all.

**Cause:** the linear ratio is heavy-tailed; its mean is dominated by a handful
of deeply-faded points, while the KPI the network is judged on is a dB-domain
average. **Fix:** compute the pairwise term as a true SINR in dB,
`10*log10(P_i/(P_j+N))`. Correlation improved to **rho = -0.65**.

We now validate every term against the KPI it is meant to control:

| QUBO term | vs KPI | rho |
|---|---|---|
| coverage | outage % | **-0.85** |
| interference | mean SINR | **-0.65** |
| coverage + interference | mean SINR | **-0.64** |
| coverage + interference | spectral efficiency | **-0.59** |
| handover | handover failure % | **+0.89** |

### 3.2 The penalty term destroyed the quantum optimization

With one-hot encoding only `T^S` of `2^(S*T)` basis states are feasible -
**0.28%** on our flagship instance. The one-hot penalty must therefore be
large, and it then *dominates the spectrum*: on an 18-qubit instance the energy
range with the penalty is `-0.81 ... +57.6`, of which the physics we care about
occupies a sliver. Textbook QAOA spends its expressive power learning to
satisfy the encoding rather than optimizing the network.

Measured: at p=1, standard QAOA was **worse than random guessing**.

**Fix - a constrained (XY) ansatz.** We replaced the transverse-field mixer
with an XY mixer restricted to each sector's one-hot block, and the `|+>^n`
initial state with a **W state per block** (`prepare_dicke_state(1, block)`):

- `RXX(b)*RYY(b) = exp(-i*b*(XX+YY)/2)` acts only on the `{|01>,|10>}` subspace
  of a pair, so it **conserves Hamming weight**
- starting from exactly one excitation per block, the state can never leave the
  feasible subspace, **for any parameter values**
- so the penalty term is **dropped entirely**, and every measured shot is a
  valid network configuration by construction

The feasibility guarantee is structural: it does not depend on the Trotter
approximation in the mixer being accurate.

**Flagship 18-qubit instance, exact statevector simulation:**

| ansatz | p | P(optimum) | feasible shots |
|---|---|---|---|
| uniform random guess | - | 0.14% | - |
| QAOA-X (standard) | 1 | 0.02% | 14.1% |
| QAOA-X (standard) | 5 | 0.28% | 71.6% |
| **QAOA-XY (ours)** | **1** | **17.09%** | **100%** |
| **QAOA-XY (ours)** | **5** | **17.16%** | **100%** |

At p=1 the constrained ansatz is **855x better than standard QAOA** and
**122x better than random guessing**.

**Confirmed on Classiq** (same instance, p=2, synthesized and executed):

| ansatz | objective | gap | P(optimum) | feasible | circuit depth | gates |
|---|---|---|---|---|---|---|
| QAOA-X (standard) | -1.50106 | **+0.11418 (missed)** | 0.00% | 22.8% | 472 | 1008 |
| **QAOA-XY (ours)** | **-1.61524** | **0.00000 (found)** | 0.20% | **100%** | **454** | 1200 |

Standard QAOA never sampled the optimum. Ours found it - at **lower circuit
depth**, which is what matters on real hardware.

### 3.3 Reaching 36 qubits by simulating in the feasible subspace

The XY ansatz's invariant has a second payoff. Because the state provably never
leaves the one-hot subspace, it can be simulated exactly with `T^S` amplitudes
instead of `2^(S*T)`:

> 9 sectors x 4 tilts: **262,144** amplitudes instead of **68,719,476,736** -
> a 262,144x reduction.

Within one block the single-excitation subspace is spanned by the tilt levels,
so the whole XY chain collapses to a `T x T` unitary acting on that sector's
tilt index; the full mixer is that matrix applied along each axis of a rank-`S`
tensor (`qaoa_subspace.py`, verified against the QUBO objective to 2.2e-15).

This let us validate the algorithm on the **36-qubit instance where classical
greedy search genuinely fails** - out of reach for any full statevector
simulator. Combined with **INTERP** initialization (Zhou et al., PRX 10,
021067), which grows the schedule depth-by-depth instead of optimizing `2p`
parameters blind:

| depth p | P(optimum) | vs random |
|---|---|---|
| 1 | 0.170% | 444x |
| 3 | 0.842% | 2,207x |
| 5 | 1.215% | 3,184x |
| 8 | 2.000% | 5,244x |
| **12** | **5.290%** | **13,867x** |

## 4. On quantum advantage: what we can and cannot claim

**We did not achieve quantum advantage, and nothing below should be read as a
claim of it.**

The rigorous comparison is *resource-to-solution*: repetitions needed to hit the
optimum with 95% confidence, `R = ln(0.05)/ln(1 - p_success)`. On the 36-qubit
instance:

| method | P(hit)/repetition | repetitions | unit |
|---|---|---|---|
| random sampling | 0.00038% | 785,312 | objective evaluations |
| greedy local search | 42.5% | 5.4 restarts = **410** | objective evaluations |
| QAOA-XY, p=12 | 5.29% | **55** | circuit shots |

Sampling-phase-only, QAOA needs **7.4x fewer repetitions**. But that comparison
is incomplete, and the honest version reverses it: **training the parameters
cost ~900 additional circuit evaluations**, so for a single instance from cold,
classical greedy wins outright.

The only route to a favourable count is **amortization**: train the schedule
once, reuse it across many instances, paying only the sampling cost each time.
That matches the operational reality - a live network is re-optimized
continuously as traffic shifts. We measured whether that works rather than
asserting it, and it does: **zero-shot transfer to unseen instances retains 47%
of retrained performance (6,781x better than random), giving 1.9x fewer
repetitions per instance with training repaid after 9 instances** (§4.1, §4.2).

Even so, **a circuit shot is not equal in wall-clock cost to a classical
objective evaluation**. This is a resource-*count* argument about algorithmic
scaling, not a wall-clock win.

### 4.1 Parameter transfer: measured, not assumed (`param_transfer.py`)

We trained one p=8 schedule on a single 36-qubit instance and applied it
**verbatim, with no re-optimization**, to six unseen networks. The learned
schedule has the smooth annealing-like structure characteristic of good QAOA
parameters (|gamma| rising, beta falling), which is what makes transfer
plausible in the first place:

```text
gammas: [-1.271, -1.510, -1.857, -2.139, -2.417, -2.626, -2.820, -2.878]
betas : [ 0.649,  0.301,  0.177,  0.137,  0.123,  0.099,  0.092,  0.154]
```

| unseen instance | transferred | retrained (ceiling) | vs random | % of ceiling |
|---|---|---|---|---|
| 11 | 6.399% | 12.272% | 16,775x | 52% |
| 23 | 0.849% | 2.849% | 2,225x | 30% |
| 42 | 3.397% | 6.219% | 8,906x | 55% |
| 7 | 1.145% | 2.385% | 3,001x | 48% |
| 55 | 0.704% | 1.793% | 1,847x | 39% |
| 68 | 3.027% | 7.358% | 7,934x | 41% |
| **mean** | **2.587%** | **5.479%** | **6,781x** | **47%** |

**Zero-shot transfer retains 47% of the retrained performance and is 6,781x
better than random guessing.** Transfer is real, and it is the thing that makes
the amortization argument legitimate rather than hand-waving.

### 4.2 Amortized resource-to-solution

With transfer measured, the accounting closes:

| method | P(hit)/repetition | repetitions per instance | unit |
|---|---|---|---|
| greedy local search | 61.8% | 3.1 restarts = **217** | objective evaluations |
| QAOA-XY, transferred p=8 | 2.587% | **114** | circuit shots |
| one-off training | - | 903 | circuit evaluations |

**Per instance after training, QAOA needs 1.9x fewer repetitions, and the
training cost is repaid after 9 instances.**

That is the strongest claim the evidence supports. It is a **resource-count**
result under a same-family transfer assumption - not a wall-clock quantum
advantage, because a circuit shot on real hardware remains far more expensive
than a classical objective evaluation, and these simulations are noiseless.

## 5. Results

### 5.1 Business KPIs

Flagship instance, 6 sectors x 3 tilt levels. Baseline is the best single
network-wide tilt, which is how many networks are actually configured.

| KPI | uniform tilt | optimized | change |
|---|---|---|---|
| mean SINR | 7.23 dB | 8.63 dB | **+1.40 dB** |
| spectral efficiency | 2.777 | 3.184 | **+14.6% throughput** |
| outage | 6.2% | 6.0% | -0.2 pp |
| handover failures | 10.3% | 29.7% | +19.3 pp |

### 5.2 The trade-off is real - we report the frontier, not one point

The weights are an **operator policy dial**, so we show the whole frontier
rather than cherry-picking a flattering point:

| w_cov | w_int | w_ho | SINR | sp.eff | handover failures |
|---|---|---|---|---|---|
| - | uniform baseline | - | 7.23 dB | 2.777 | 10.3% |
| 0.40 | 0.35 | 0.25 | 8.63 dB | 3.184 | 29.7% |
| 0.35 | 0.30 | 0.35 | 7.41 dB | 2.828 | 18.5% |
| 0.30 | 0.20 | 0.50 | 7.23 dB | 2.777 | 10.3% |

Aggressive downtilt buys SINR and throughput and costs handover reliability.
The operator picks the point; the optimizer solves whichever point is chosen.

### 5.3 Where classical search actually breaks down

Measured, not asserted - greedy local search from 200 random starts:

| instance | qubits | configurations | greedy reaches global optimum |
|---|---|---|---|
| 6 sectors x 3 tilts | 18 | 729 | **100%** |
| 9 sectors x 4 tilts | 36 | 262,144 | **42%** |

**We are explicit: at 18 qubits classical greedy solves the problem and QAOA
offers no practical advantage.** The frustration a global optimizer can exploit
appears as the network grows.

## 6. Scaling

| towers | sectors | configurations | qubits | exhaustive search |
|---|---|---|---|---|
| 3 | 9 | 2.6e5 | 36 | 2.9 s |
| 5 | 15 | 1.1e9 | 60 | 3.3 hours |
| 6 | 18 | 6.9e10 | 72 | 8.8 days |
| 8 | 24 | 2.8e14 | 96 | 99 years |

Qubit count grows **linearly** (`S x T`) while the search space grows
exponentially. Dropping the penalty term also keeps the Hamiltonian's useful
spectral range from collapsing as the problem grows.

## 7. Limitations - what we did *not* prove

Stated plainly, because the honest version is more useful than the hyped one:

1. **No quantum advantage.** On every instance small enough to check, classical
   solvers match or beat QAOA in wall-clock time. Brute force solves the
   18-qubit instance in 5 ms; QAOA takes ~44 s.
2. **Simulator only, noiseless.** No real quantum hardware. At 36+ qubits and
   this circuit depth, current devices would be noise-dominated.
3. **QAOA parameter optimization is itself hard.** Without INTERP, deeper
   circuits performed *worse*; even with it, p=2 fell into a bad basin.
4. **The RF model is simplified**: no terrain, shadowing, ray tracing, user
   mobility or scheduler behaviour. Absolute KPI values are not predictions for
   a real AT&T network; the *relative* comparison between solvers is the result.
5. **The QUBO is a surrogate**, not the true objective. We measured how good a
   surrogate it is (§3.1) rather than assuming it.
6. **Fixed service areas** are an approximation - real cell boundaries move when
   tilts change.
7. **Parameter transfer was tested only within one instance family** (same
   generator, 9 sectors, 4 tilt levels).

## 8. Running it

```bash
pip install -r requirements.txt
python -c "import classiq; classiq.authenticate()"

python rf_model.py           # RF model + KPIs for uniform configurations
python qubo_builder.py       # QUBO construction + exactness self-checks
python classical_baseline.py # classical solvers
python qaoa_local.py         # X vs XY ansatz, exact statevector simulation
python qaoa_subspace.py      # 36-qubit run + resource-to-solution analysis
python param_transfer.py     # do trained parameters transfer?
python quantum_solve.py      # QAOA on Classiq (synthesis + execution)
python benchmark.py          # the full benchmark reproduced in this document
python visualize.py          # figures
```

## 9. Files

| file | role |
|---|---|
| `rf_model.py` | antenna patterns, path loss, demand, service areas, exact joint-SINR KPI simulator |
| `qubo_builder.py` | QUBO/Ising construction, encode/decode, exactness self-checks |
| `classical_baseline.py` | brute force, greedy local search, simulated annealing, uniform baselines |
| `qaoa_local.py` | exact local statevector QAOA (X and XY ansaetze) |
| `qaoa_subspace.py` | feasible-subspace simulation, INTERP, resource-to-solution |
| `param_transfer.py` | parameter-transfer experiment and amortized accounting |
| `quantum_solve.py` | QAOA on Classiq: Qmod ansatz construction, synthesis, execution |
| `benchmark.py` | surrogate validation, solver comparison, Pareto frontier, hardness, scaling |
| `hardness_probe.py` | instance-hardness diagnostics across random networks |
| `visualize.py` | figures for the pitch deck |
