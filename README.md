# Antenna Tilt Optimization with Constrained QAOA

**QUBIT Hackathon 2026 — AT&T Challenge 1** · built on [Classiq](https://classiq.io)

Choosing the electrical downtilt of every sector antenna in a cellular network,
formulated as a QUBO and solved with a **constrained (XY-mixer) QAOA**.

## The 30-second version

Tilt choices are strongly coupled — tilting one sector down cuts interference
into its neighbour but can strand that neighbour's cell-edge users. With `T`
tilt levels and `S` sectors there are `T^S` configurations.

The one-hot encoding needed for a clean QUBO puts only **0.28%** of the Hilbert
space inside the feasible set, so the constraint penalty dominates the
Hamiltonian and textbook QAOA wastes its whole circuit learning to satisfy the
encoding — at p=1 it performs **worse than random guessing**.

We replace the mixer with an **XY mixer on each sector's one-hot block**, seeded
by a **W state**. `RXX(β)·RYY(β)` conserves Hamming weight, so the state can
never leave the feasible subspace, the penalty term disappears, and every shot
is a valid network configuration by construction.

| ansatz (18 qubits, p=2, on Classiq) | P(optimum) | feasible shots | circuit depth |
|---|---|---|---|
| QAOA-X (standard + penalty) | 1.00% | 8.3% | 472 |
| **QAOA-XY (ours)** | **8.70%** | **100%** | **454** |

Better results at *lower* circuit depth.

## Scaling to 36 qubits

Because the XY ansatz provably never leaves the feasible subspace, we can
simulate it exactly with `T^S` amplitudes instead of `2^(S·T)` — **262,144
instead of 68.7 billion**. That let us validate on the 36-qubit instance where
classical greedy search genuinely fails, which no full statevector simulator
can reach. With INTERP initialization, P(optimum) climbs to **5.29% at p=12 —
13,867× better than random**.

## On quantum advantage

**We did not achieve it, and we don't claim it.** Per instance from cold,
classical greedy wins outright: training the QAOA parameters costs more than
just running the heuristic.

The one honest advantage-shaped result is **amortization**, which we measured
rather than assumed. A schedule trained once and applied **verbatim** to six
unseen 36-qubit networks retains **47%** of retrained performance and is
**6,781×** better than random:

| method | per-repetition P(hit) | repetitions/instance |
|---|---|---|
| greedy local search | 61.8% | 217 objective evaluations |
| QAOA-XY (transferred, p=8) | 2.587% | **114 circuit shots** |

**1.9× fewer repetitions per instance; training repaid after 9 instances.**

This is a resource-*count* result under a same-family transfer assumption — not
a wall-clock win. A circuit shot on real hardware costs far more than a
classical objective evaluation, and these simulations are noiseless. Full
caveats in [WRITEUP.md](WRITEUP.md) §7.

## Business results

- **+1.40 dB** mean SINR and **+14.6%** throughput over a uniform network-wide tilt
- QUBO validated against an exact joint-SINR simulator (rho = -0.85 to +0.89 per term)
- Coverage vs handover reliability reported as a **Pareto frontier**, not one cherry-picked point

## Run it

```bash
pip install -r requirements.txt
python -c "import classiq; classiq.authenticate()"

python benchmark.py      # the full benchmark
python quantum_solve.py  # QAOA on Classiq
python visualize.py      # figures for the deck
```

Full technical detail, including what broke and how we fixed it: **[WRITEUP.md](WRITEUP.md)**.
