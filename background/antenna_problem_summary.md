# AT&T QUBIT Hackathon 2026 — Antenna Tilt Optimization

## Central Challenge

Apply quantum computing to a computationally intensive telecommunications optimization problem at scale. The expected result is **not a production-ready system**, but a **credible algorithmic proof of concept** grounded in realistic problem parameters.

The solution must demonstrate either:
- a quantum algorithm implementation, or
- a rigorous quantum–classical hybrid approach.

## Antenna Tilt Optimization Problem

Every cellular tower has antennas whose tilt can be electronically fine-tuned by approximately **±10 degrees**.

The tilt angle affects the antenna radiation pattern and therefore:
- geographic coverage,
- signal quality,
- overlap with neighboring cells,
- inter-cell interference,
- handover performance.

The fundamental trade-off is:

- **Less down-tilt / higher tilt:** larger coverage footprint, but more overshoot and interference with neighboring cells.
- **More down-tilt / lower tilt:** smaller coverage footprint and potentially better local signal quality, but excessive down-tilt can shrink coverage and increase handover failures near cell edges.

The optimization problem is to choose the tilt of many interacting antenna sectors simultaneously so that network performance is maximized while interference and coverage problems are minimized.

## Optimization Formulation

For each antenna sector, define a **discrete set of feasible tilt values**.

The decision is a network-wide configuration vector:

`tilt_configuration = [tilt_1, tilt_2, ..., tilt_N]`

The objective can combine several competing quantities.

### Maximize

- Coverage quality
- SINR (Signal-to-Interference-plus-Noise Ratio)
- User throughput
- Energy efficiency

### Minimize

- Inter-cell interference
- Overshooting
- Outage
- Handover failures

A conceptual weighted objective is:

`maximize: coverage + SINR + throughput + energy_efficiency - interference - overshoot - outage`

The exact weighting/model is not specified by the supplied briefs and therefore must be chosen as part of the proof of concept.

## Why the Problem Is Difficult

The number of possible configurations grows rapidly with the number of sectors.

If each of `N` sectors has `K` possible tilt values, the search space contains:

`K^N`

possible configurations.

More importantly, sectors are **strongly coupled**. Changing one sector's tilt can affect neighboring sectors through:

- coverage overlap,
- interference,
- handover behavior.

Therefore, optimizing sectors independently is insufficient. The goal is to find a good **global configuration** across interacting sectors.

The challenge brief states that classical approaches generally rely on heuristics and local search and can miss better global configurations on complex networks.

## Why Quantum Computing Fits

The antenna problem is presented as a candidate for quantum optimization because it combines:

1. A large discrete/combinatorial search space.
2. Strong interactions between decision variables.
3. A need to search for high-quality global configurations.

A **hybrid quantum–classical architecture** is explicitly suggested:

### Classical component

Handle:
- RF simulation,
- propagation modeling,
- generation of the optimization instance,
- validation of candidate configurations,
- calculation of network KPIs.

### Quantum component

Handle the difficult combinatorial step:

> selecting a tilt configuration across many interacting antenna sectors.

The hackathon does not require demonstrating production-scale quantum advantage. The goal is a credible quantum proof of concept with a clear scalability/quantum-fit argument.

## Business Importance

AT&T estimates the potential value of solving antenna tilt optimization at approximately:

- **$500M/year in operational savings**
- **$600M/year in revenue uplift** from improved service quality

The brief also states that AT&T's CEO has publicly cited antenna optimization as an important frontier for quantum computing following AI.

## Metrics to Measure

The antenna challenge specifically identifies the following network KPIs:

### Coverage quality
Measure improvement in **SINR**.

### User throughput
Measure whether optimized tilt configurations improve achievable throughput.

### Drop and failure reduction
Measure reduction in network/service failures.

### Handover performance
Measure reduction in **handover failures**, particularly around cell boundaries.

A useful proof-of-concept comparison should evaluate the quantum/hybrid result against a classical baseline where possible.

## Hackathon Deliverables Relevant to This Challenge

The submission should include:

1. **Working implementation or simulation**
   - Quantum algorithm applied to antenna tilt optimization.
   - Must be built on the **Classiq platform**.

2. **Technical explanation**
   - Optimization formulation.
   - Quantum approach.
   - Quantum-fit / quantum-advantage argument.
   - Limitations encountered.

3. **Classical benchmark**
   - Compare against a classical baseline wherever possible.

4. **5-minute pitch**
   - Followed by 3 minutes of judges' questions.
   - The presentation must contain a **pre-recorded video demo**; nothing runs live on stage.

The team should be able to clearly explain what was technically difficult or novel about the implementation.

## Judging Criteria

| Criterion | Weight |
|---|---:|
| Business Impact & Relevance | 25% |
| Quantum Implementation & Code Quality | 20% |
| Quantum Fit, Scalability & Benchmarking | 20% |
| Novelty & Creative Approach | 20% |
| Presentation & Clarity | 15% |

For the antenna solution, this means the implementation should not only optimize a toy objective: it should preserve the actual telecom interpretation of **coverage versus interference**, demonstrate a meaningful quantum component, benchmark against a classical method, and clearly explain scalability and limitations.

## Pitch Guidance

The participant brief specifically recommends:

**Do**
- Explain what the team had to crack.
- Benchmark against a classical baseline.
- State limitations honestly.

**Do not**
- Present a long or clunky demo.
- Try to solve too many problems simultaneously.
- Present a solution the team cannot explain.

## Useful Background Areas

The challenge brief identifies:

- RF engineering
- Signal processing
- Wireless networking
- Optimization algorithms

## References Provided by the Antenna Challenge

The challenge brief lists:

- *Impact of Electrical and Mechanical Antenna Tilt on LTE Downlink System Performance*
- *Antenna Tilt Optimization: Mechanical vs Electrical, Coverage vs Interference*
- *Vertical Antenna Tilt Optimization for LTE Base Stations*

## Important Modeling Gap

The supplied hackathon materials describe the antenna optimization problem and its KPIs, but **do not provide a concrete RF dataset, propagation equation, antenna radiation model, objective weights, or exact discrete tilt set** in the text of the provided files.

Those details therefore need to be defined or simulated by the team when building the proof of concept, while keeping the model consistent with the stated coverage/interference trade-off.
