# AT&T Quantum Hackathon --- Dynamic Antenna Tilt Optimization

## User

How about this direction?

instead of solving the snapshot problem (a single timeslice), I would
assume to have a good solution for a single snapshot, and now I want to
solve the dynamical problem: keeping the angles as close as possible to
the optimum solution, for as long as possible, as users move in time.

Is that reasonable for the hackathon?

## Assistant

Yes --- **but only if you formulate it as a genuinely temporal
optimization problem, not just "rerun antenna optimization every few
seconds."** In that form, I think it is actually a stronger hackathon
direction than the pure snapshot problem.

The important warning is that **"continuous antenna re-optimization" by
itself is not novel**. The Bell Labs paper already proposes continuous
autonomous tilt optimization, uses a random-walk mobility model, and
says the algorithm can keep testing tilts and track the optimum when
system conditions change. It also demonstrates re-optimization after a
cell outage.

Your distinctive idea should therefore be:

> **Given a changing user distribution, optimize a trajectory of antenna
> configurations over time, trading network performance against the cost
> of changing antenna tilts.**

That is a different problem.

### A clean formulation

Suppose the "ideal snapshot optimizer" gives you

\[ `\theta`{=tex}*t\^\* =
`\arg`{=tex}`\max`{=tex}*`\theta `{=tex}U(`\theta`{=tex},x_t) \]

at each time (t), where (x_t) describes the current user distribution
and (`\theta`{=tex}) is the vector of antenna tilts.

Instead of simply setting

\[ `\theta`{=tex}\_t=`\theta`{=tex}\_t\^\* \]

every time, solve

\[ `\min`{=tex}\_{`\theta`{=tex}\_1,`\ldots`{=tex},`\theta`{=tex}*T}
`\sum`{=tex}*{t=1}\^{T} L(`\theta`{=tex}\_t,`\theta`{=tex}\_t\^\*) +
`\lambda `{=tex}C(`\theta`{=tex}*t,`\theta`{=tex}*{t-1}). \]

Here:

-   (L(`\theta`{=tex}\_t,`\theta`{=tex}\_t\^\*)): network-performance
    regret.
-   (C(`\theta`{=tex}*t,`\theta`{=tex}*{t-1})): cost of changing tilts.

For example,

\[ C(`\theta`{=tex}*t,`\theta`{=tex}*{t-1}) = `\sum`{=tex}*i
\|`\theta`{=tex}*{i,t}-`\theta`{=tex}\_{i,t-1}\| \]

or simply the number of antennas whose tilt changes.

This captures an operationally meaningful question:

**"Do I really need to move the antenna now, or is the current setting
still good enough for the next several minutes?"**

### Why this is useful for the hackathon

It gives a clear demo. Imagine users gradually moving from one region to
another. Compare:

1.  **Static:** never change tilt.
2.  **Greedy snapshot:** always jump immediately to
    (`\theta`{=tex}\_t\^\*).
3.  **Temporal optimizer:** anticipate movement and make fewer, smarter
    tilt changes.

Then compare cumulative SINR/throughput versus number or magnitude of
tilt adjustments.

The pitch becomes:

> Existing optimization asks, "What is the best network configuration
> now?"\
> We ask, "What sequence of configurations keeps the network
> near-optimal as demand evolves, without constantly reconfiguring it?"

### Where to be careful

The biggest problem is the **quantum component**.

If every antenna × angle × timestep is encoded directly, the problem
explodes. With (N) antennas, (K) possible angles and (T) timesteps, a
naïve one-hot encoding needs roughly

\[ NKT \]

binary variables.

A better approach is to use the assumed snapshot optimizer. For each
timestep, generate perhaps the best (M=3!-!5) candidate antenna
configurations:

\[ S_t={
`\theta`{=tex}\_t\^{(1)},`\theta`{=tex}\_t\^{(2)},`\ldots`{=tex},`\theta`{=tex}\_t\^{(M)}
}. \]

Then solve:

> Choose one candidate configuration at every timestep so that total
> performance loss + transition cost is minimized.

This gives a layered graph:

\[ S_1 `\rightarrow `{=tex}S_2 `\rightarrow `{=tex}S_3
`\rightarrow `{=tex}`\cdots `{=tex}`\rightarrow `{=tex}S_T. \]

### The classical-algorithm catch

If the temporal objective decomposes exactly into per-time performance
plus adjacent transition costs, it becomes a
shortest-path/dynamic-programming problem and is easy classically.

To make the optimization genuinely difficult, introduce global/temporal
constraints such as:

\[ `\sum`{=tex}\_t C(`\theta`{=tex}*t,`\theta`{=tex}*{t-1})
`\le `{=tex}B, \]

or:

-   no antenna may be adjusted more than (r) times per horizon;
-   only (q) antennas may be adjusted simultaneously;
-   configurations must be robust across possible future user
    trajectories;
-   optimize expected performance under uncertain mobility;
-   penalize handover instability caused by successive changes.

The uncertain-future-trajectories option is especially promising:

\[
`\min`{=tex}*`\Theta `{=tex}`\mathbb `{=tex}E*{`\omega`{=tex}}\[J(`\Theta`{=tex},`\omega`{=tex})\],
\]

where (`\omega`{=tex}) is a possible user-mobility trajectory.

That turns the problem into something closer to **stochastic
multi-period combinatorial optimization**.

### Assessment

A stronger framing is:

**Quantum optimization for predictive antenna-tilt control under dynamic
user mobility**

rather than simply:

**Continuous antenna tilt optimization.**

The latter has already been studied classically. The former emphasizes
temporal coupling, prediction, and reuse of previous solutions.

------------------------------------------------------------------------

## User

I thought about the following possibilities:

1.  using adiabatic quantum computation.
2.  treating the user density as a fluid, and solving it as a QCFD
    solver that optimizes the angles according to the density.
3.  what other algorithms can be relevant with the assumption that a
    delta small time step, would generate a small perturbation in the
    density?

## Assistant

Of these directions, **adiabatic continuation is the strongest fit to
the "small (`\Delta `{=tex}t) → small perturbation" assumption**. The
fluid idea is useful for modeling mobility, but I would not make QCFD
the quantum centerpiece.

### 1. Adiabatic quantum computation

Suppose the snapshot problem at time (t) is encoded by a Hamiltonian

\[ H_t \]

whose ground state represents the optimal antenna configuration.

After a short time step,

\[
`\rho`{=tex}\_{t+`\Delta `{=tex}t}=`\rho`{=tex}\_t+`\delta `{=tex}`\rho`{=tex},
\]

so the objective only changes slightly:

\[ H\_{t+`\Delta `{=tex}t}=H_t+`\delta `{=tex}H. \]

If (`\delta `{=tex}H) is small, the old optimum should often have
substantial overlap with the new optimum. Instead of solving the
optimization problem from scratch, continuously deform (H_t) into
(H\_{t+`\Delta `{=tex}t}):

\[ H(s)=(1-s)H_t+sH\_{t+`\Delta `{=tex}t},
`\qquad `{=tex}s`\in[0,1]`{=tex}. \]

Start in the previously obtained ground state and evolve adiabatically
toward the new one.

The central story is:

> We exploit temporal coherence in the cellular network. Because user
> density changes gradually, consecutive optimization Hamiltonians are
> close, so the previous solution becomes a quantum warm start for the
> next one.

A key caveat is that **small change in user density does not guarantee
small change in the optimal discrete tilt configuration**. Near a
decision boundary, a tiny density change can switch the optimum.

So measure something like

\[ \|`\rho`{=tex}\_{t+`\Delta `{=tex}t}-`\rho`{=tex}\_t\| \]

versus

\[ d(`\theta`{=tex}\^\*\_{t+`\Delta `{=tex}t},`\theta`{=tex}\^\*\_t). \]

If the second quantity is usually small in simulation, the assumption is
empirically defensible.

### 2. User density as a fluid

Rather than tracking thousands of individual users,

\[ {x_1(t),`\ldots`{=tex},x_M(t)}, \]

represent demand as a spatial density field

\[ `\rho`{=tex}(x,y,t). \]

It could evolve under a continuity/advection-diffusion equation such as

\[ `\frac{\partial \rho}{\partial t}`{=tex} +
`\nabla`{=tex}`\cdot`{=tex}(`\rho `{=tex}v) =
D`\nabla`{=tex}\^2`\rho`{=tex}. \]

This gives smooth temporal behavior and an intuitive demo, such as
commuters moving between regions or a stadium emptying.

However, I would stop before making **QCFD** the core algorithm. Users
are not literally a fluid whose Navier-Stokes equations need to be
solved. A continuity or diffusion model is enough.

More importantly, even if a quantum PDE solver predicts

\[ `\rho`{=tex}(x,y,t), \]

you still have not optimized the antenna angles.

A cleaner architecture is

\[ `\text{density evolution}`{=tex} `\rightarrow`{=tex} `\rho`{=tex}\_t
`\rightarrow`{=tex} `\text{RF objective}`{=tex} `\rightarrow`{=tex} H_t
`\rightarrow`{=tex} `\text{quantum optimization}`{=tex}. \]

Use the density evolution classically and put the quantum component in
the changing optimization problem.

### 3. Algorithms that exploit small perturbations

  -----------------------------------------------------------------------------------------------------
  Method                                                  Exploits        Hackathon fit Assessment
                     (H\_{t+`\Delta `{=tex}t}`\approx `{=tex}H_t)?                      
  ---------------- ----------------------------------------------- -------------------- ---------------
  **Adiabatic                                                ★★★★★                ★★★★☆ Best conceptual
  continuation**                                                                        fit

  **Warm-start VQE                                           ★★★★★                ★★★★★ Probably
  / variational                                                                         easiest to
  continuation**                                                                        demonstrate

  **Quantum                                                  ★★★★★                ★★★☆☆ Interesting and
  imaginary-time                                                                        distinctive
  evolution**                                                                           

  **Local                                                    ★★★★☆                ★★★★☆ Simple and
  Grover/minimum                                                                        defensible
  finding**                                                                             

  Quantum walk                                               ★★★★☆                ★★★☆☆ Interesting but
  search                                                                                harder to
                                                                                        justify

  Quantum PDE/QCFD                                           ★★☆☆☆                ★★☆☆☆ Wrong place to
                                                                                        spend effort

  Full-horizon                                               ★★☆☆☆                ★★★★☆ Conventional;
  QUBO/QAOA                                                                             preferably
                                                                                        avoid
  -----------------------------------------------------------------------------------------------------

#### A. Warm-start VQE / parameter continuation

Represent the solution with a parameterized state

\[ \|`\psi`{=tex}(`\alpha`{=tex})`\rangle`{=tex}. \]

At time (t), obtain parameters

\[ `\alpha`{=tex}\_t\^\*. \]

At the next timestep, initialize from

\[ `\alpha`{=tex}\_{t+`\Delta `{=tex}t}\^{(0)}=`\alpha`{=tex}\_t\^\* \]

rather than randomly.

Then minimize

\[ `\langle `{=tex}`\psi`{=tex}(`\alpha`{=tex})\|
H\_{t+`\Delta `{=tex}t} \|`\psi`{=tex}(`\alpha`{=tex})`\rangle`{=tex}.
\]

Because the Hamiltonian changed only slightly, ideally only a few
optimization iterations are required.

A useful benchmark is:

\[ `\text{cold-start iterations}`{=tex}
`\quad`{=tex}`\text{vs.}`{=tex}`\quad`{=tex}
`\text{warm-start iterations}`{=tex}. \]

Also measure circuit evaluations, convergence time, and solution
quality.

For a short hackathon, this may be easier than implementing genuine
adiabatic dynamics.

#### B. Quantum imaginary-time evolution

Imaginary-time evolution

\[ \|`\psi`{=tex}(`\tau`{=tex})`\rangle`{=tex} `\propto`{=tex}
e\^{-`\tau `{=tex}H}\|`\psi`{=tex}(0)`\rangle`{=tex} \]

suppresses excited states and drives a state toward the ground state.

For the dynamic setting:

\[ \|`\psi`{=tex}\_t\^\*`\rangle`{=tex} `\longrightarrow`{=tex}
e\^{-`\Delta`{=tex}`\tau `{=tex}H\_{t+`\Delta `{=tex}t}}
\|`\psi`{=tex}\_t\^\*`\rangle`{=tex}. \]

Because the starting state should already be near the new ground state,
relatively little evolution might be needed.

The implementation is more complex, so it is higher risk for the
hackathon.

#### C. Quantum search in the neighborhood of the previous optimum

If only a few antennas should change between (t) and
(t+`\Delta `{=tex}t), define a Hamming neighborhood:

\[ `\mathcal `{=tex}N_k(`\theta`{=tex}\_t) = { `\theta `{=tex}:
d_H(`\theta`{=tex},`\theta`{=tex}\_t)`\le `{=tex}k }. \]

Instead of searching the entire configuration space, search only
configurations where at most (k) antennas change.

For example, if each antenna has local actions

\[ {-1^`\circ`{=tex},0,+1^`\circ`{=tex}}, \]

and only (k) antennas are allowed to change, the local search space can
be much smaller than the full configuration space.

Then use **Grover search / Dürr-Høyer minimum finding** over that
neighborhood.

If there are (M) candidate perturbations, classical exhaustive search
requires roughly

\[ O(M) \]

objective evaluations, while quantum minimum finding requires
approximately

\[ O(`\sqrt `{=tex}M) \]

oracle calls under the standard oracle assumptions.

The main difficulty is implementing the RF objective as a practical
quantum oracle.

### A promising hybrid

A strong architecture is

\[ `\boxed{
\text{Density predictor}
+
\text{perturbative quantum optimizer}
}`{=tex} \]

At each timestep:

1.  Estimate current user density (`\rho`{=tex}\_t(x,y)).
2.  Predict a short horizon
    (`\rho`{=tex}*{t+1},`\ldots`{=tex},`\rho`{=tex}*{t+h}).
3.  Generate the corresponding antenna optimization Hamiltonian (H_t).
4.  Start from the previous quantum solution.
5.  Update it using adiabatic/variational continuation.
6.  Trigger a full reoptimization only if the perturbation becomes too
    large.

This gives two operating modes:

\[ \|`\delta `{=tex}H\| \< `\epsilon`{=tex}
`\quad`{=tex}`\Rightarrow`{=tex}`\quad`{=tex}
`\text{fast tracking}`{=tex} \]

and

\[ \|`\delta `{=tex}H\|`\geq`{=tex}`\epsilon`{=tex}
`\quad`{=tex}`\Rightarrow`{=tex}`\quad`{=tex}
`\text{global reoptimization}`{=tex}. \]

### Perturbation-theory interpretation

The assumption

\[ `\Delta `{=tex}t`\text{ small}`{=tex} `\Rightarrow`{=tex}
`\Delta`{=tex}`\rho`{=tex}`\text{ small}`{=tex} \]

also suggests perturbation theory directly.

If

\[ H\_{t+`\Delta `{=tex}t}=H_t+`\epsilon `{=tex}V, \]

then away from degeneracies the new ground state is approximately

\[ \|0'`\rangle`{=tex} `\approx`{=tex} \|0`\rangle`{=tex} +
`\epsilon`{=tex} `\sum`{=tex}\_{n\>0}
`\frac{\langle n|V|0\rangle}`{=tex} {E_0-E_n} \|n`\rangle`{=tex}. \]

This gives a theoretical explanation for why a warm start can work: the
state representing the optimum changes continuously while the spectral
gap remains open.

It also identifies the failure case. If two antenna configurations
become nearly equally good, the gap becomes small and a small
user-density perturbation can cause a large change in the optimum.

### Recommended ranking

1.  **Adiabatic/variational continuation of the optimum** --- strongest
    overall direction.
2.  **Local quantum minimum finding** --- strongest theoretical
    quantum-search argument.
3.  **QITE** --- novel but higher implementation risk.
4.  **Density-as-fluid** --- useful as the classical mobility model.
5.  **QCFD** --- not recommended unless a specific PDE formulation can
    be shown to directly produce the antenna controls.

The Bell Labs work provides precedent for continuously adapting antenna
tilts under changing network conditions. The novelty here should
therefore be **explicitly exploiting temporal correlation between
consecutive optimization problems**, rather than treating every timestep
independently.

------------------------------------------------------------------------

## Relevant project context

The hackathon challenge requires a quantum algorithm implementation or
rigorous quantum-classical hybrid, with a working
implementation/simulation on Classiq. The antenna challenge is a
discrete, strongly coupled optimization problem whose KPIs include SINR,
throughput, drop/failure reduction, and handover performance.

The supplied Bell Labs paper already demonstrates continuous autonomous
re-optimization of vertical antenna tilts using a heuristic
gradient-ascent method under changing network conditions, including cell
outage. This is why simply making the antenna optimizer dynamic is not
sufficient novelty.

The supplied Ericsson antenna paper models downlink SINR as a function
of user-dependent path gains and uses SINR-derived throughput metrics,
supporting the premise that changing user distributions can change the
antenna-tilt objective.

The Ericsson quantum-computing review also discusses antenna-tilt
optimization using a quantum neural network within an RL-related
approach, so a temporal continuation/perturbation approach provides
useful differentiation from both QAOA and QNN/RL approaches.
