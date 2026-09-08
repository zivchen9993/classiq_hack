# AT&T QUBIT Hackathon 2026 --- Antenna Tilt Quantum Advantage Discussion

## User

Is it possible to reach quantum advantage for the antenna problem? If
so, how?

## Assistant

### Verdict

**Yes, in principle---but there is no convincing end-to-end quantum
advantage for realistic antenna-tilt optimization today.**

The strongest defensible claim is currently one of these:

1.  **A provable reduction in the number of expensive objective
    evaluations**, using quantum minimum finding or amplitude
    estimation.
2.  **A possible approximation advantage from special problem
    structure**, using Decoded Quantum Interferometry (DQI).
3.  **A possible data-efficiency advantage during policy training**,
    using a quantum neural network.

None of those automatically implies that a present-day QPU will beat a
good classical antenna optimizer in wall-clock time.

The antenna problem is legitimately difficult: each sector has a
discrete tilt, and changing one tilt changes coverage, interference,
SINR, throughput, and handovers in nearby cells. The challenge brief
explicitly suggests keeping RF simulation classical and assigning the
coupled configuration-selection step to the quantum component.

However, the statement "there are (K\^N) configurations, therefore
quantum wins" is not sufficient. Classical methods do not enumerate all
(K\^N) states. A classical cluster-based gradient heuristic was already
demonstrated on 57 sectors and reported about a 10% increase in mean
spectral efficiency and a 100% increase at the cell edge, including
re-optimization after an outage. Classical antenna models can also
evaluate relative coverage and throughput quite efficiently, and have
been validated against detailed simulations and measured radiation
patterns.

As of September 2026, the closest direct quantum-optimization work I
found is a March 2026 IEICE technical report that formulates
base-station beam-tilt optimization as a QUBO for quantum annealing and
evaluates it on a regular hexagonal deployment with a known optimum. Its
public abstract does **not** report a quantum-versus-classical speed or
scaling advantage.

### The most defensible route: quantum search over local re-optimization clusters

Trying to optimize thousands of sectors simultaneously is not realistic.
Even Grover's quadratic speedup would leave an exponential problem:

\[ K\^N `\longrightarrow `{=tex}O!`\left`{=tex}(K\^{N/2}`\right`{=tex}).
\]

That is better asymptotically, but still hopeless for thousands of
sectors.

A more credible architecture is to borrow the cluster idea from the
classical antenna paper:

1.  Use current KPIs to identify a cluster of mutually interfering
    sectors.
2.  Freeze the rest of the network.
3.  Jointly optimize the tilts of only those (c) sectors.
4.  Validate the selected move in the full RF simulator.
5.  Move to another cluster and repeat.

For example, define a utility such as

\[ U(`\boldsymbol{\theta}`{=tex}) =
w\_{`\mathrm{avg}`{=tex}}`\bar`{=tex}{`\eta`{=tex}}
+w\_{`\mathrm{edge}`{=tex}}`\eta`{=tex}*{5%}
-`\lambda`{=tex}*{`\mathrm{out}`{=tex}}P(`\mathrm{SINR}`{=tex}\<`\tau`{=tex})
-`\lambda`{=tex}\_{`\mathrm{change}`{=tex}}
`\sum`{=tex}\_i\|`\theta`{=tex}\_i-`\theta`{=tex}\_i\^{(0)}\|, \]

where (`\bar`{=tex}{`\eta`{=tex}}) is mean spectral efficiency and
(`\eta`{=tex}\_{5%}) represents cell-edge performance.

#### Quantum minimum finding or Grover Adaptive Search

Suppose each of (c) active sectors has (K) allowed local actions. There
are

\[ M=K\^c \]

joint configurations. A threshold oracle marks configurations satisfying

\[ `\widehat `{=tex}U(`\boldsymbol{\theta}`{=tex}) \> T, \]

where (`\widehat `{=tex}U) is a compact RF surrogate. Grover Adaptive
Search repeatedly raises (T), while the Dürr--Høyer minimum-finding
algorithm gives an ideal query complexity of (O(`\sqrt `{=tex}M)),
compared with (O(M)) evaluations for unstructured exhaustive search.

As an illustrative scale:

-   12 sectors,
-   3 actions per sector: (-1^`\circ`{=tex},0,+1^`\circ`{=tex}),
-   (M=3\^{12}=531{,}441) configurations,
-   ideal quantum search scale: approximately (`\sqrt `{=tex}M=729)
    oracle uses.

That is a meaningful **query-complexity** advantage.

But it is not yet a wall-clock advantage, because each quantum oracle is
much more expensive than a classical arithmetic evaluation.

#### The oracle is the make-or-break issue

You cannot simply call a Python RF simulator "in quantum superposition."

For genuine advantage, the utility must be implemented as a reversible
circuit, for example through a low-degree local model:

\[ `\widehat `{=tex}U(`\mathbf `{=tex}z) = c_0 +`\sum`{=tex}*i h_i z_i
+`\sum`{=tex}*{(i,j)`\in `{=tex}E}J\_{ij}z_i z_j
+`\sum`{=tex}*{(i,j,k)`\in `{=tex}H}J*{ijk}z_i z_j z_k,
`\qquad `{=tex}z_i`\in`{=tex}{-1,+1}. \]

The coefficients can be estimated classically from RF perturbation
experiments or a digital twin. The quantum circuit then evaluates this
polynomial using fixed-point arithmetic.

A lookup table containing all (K\^c) utility values would be a bad
solution: building or loading that table already costs (O(K\^c)),
potentially erasing the search advantage. The compact analytical
surrogate is therefore essential.

The surrogate must also be validated on held-out configurations.
Otherwise the quantum algorithm may perfectly optimize an inaccurate
approximation of the network.

### A second theoretical advantage: quantum amplitude estimation

Antenna utility is often an expectation over user positions, shadow
fading, traffic states, and mobility:

\[ U(`\boldsymbol{\theta}`{=tex}) = `\mathbb `{=tex}E\_{`\omega`{=tex}}
`\big[f(\boldsymbol{\theta},\omega)\big]`{=tex}. \]

Ordinary Monte Carlo estimation generally needs
(O(1/`\epsilon`{=tex}\^2)) samples for additive error
(`\epsilon`{=tex}). Quantum amplitude estimation can reduce this to
approximately (O(1/`\epsilon`{=tex})) coherent oracle calls, giving a
quadratic improvement in precision dependence.

Conceptually, one could combine:

\[ `\text{classical exhaustive stochastic cost}`{=tex}
`\sim `{=tex}O!`\left`{=tex}(`\frac{K^c}{\epsilon^2}`{=tex}`\right`{=tex})
\]

with an idealized quantum cost of roughly

\[ O!`\left`{=tex}(`\frac{K^{c/2}}{\epsilon}`{=tex}`\right`{=tex}). \]

This is probably the cleanest theoretical quantum-advantage argument for
the antenna problem because it attacks both the combinatorial number of
tilt configurations and the stochastic cost of evaluating each
configuration.

The catch is severe: the random network scenario and RF utility must be
prepared and evaluated coherently. This is a fault-tolerant-computing
proposal, not a realistic NISQ circuit.

### Could DQI provide a stronger advantage?

**Possibly, but DQI is higher risk than Grover search for this
problem.**

For binary local tilt moves, let (x_i`\in`{=tex}{0,1}). Since

\[ z_i=(-1)\^{x_i}, `\qquad`{=tex} z_i
z_j=(-1)\^{x_i`\oplus `{=tex}x_j}, \]

a pairwise antenna surrogate can be written as weighted XOR terms:

\[ `\widehat `{=tex}U(`\mathbf `{=tex}x) = c_0+ `\sum`{=tex}*i
a_i(-1)\^{x_i} +
`\sum`{=tex}*{(i,j)}b\_{ij}(-1)\^{x_i`\oplus `{=tex}x_j}
+`\cdots`{=tex}. \]

This is a weighted Max-XORSAT-type objective. More generally, any local
Boolean utility can be expanded in a Walsh--Fourier basis, where every
Fourier term is a parity constraint. DQI is specifically designed to
exploit sparse Fourier spectra by reducing optimization to decoding.

A plausible DQI antenna pipeline would therefore be:

1.  Select an interference cluster.
2.  Generate RF utility samples for joint tilt moves.
3.  Compute the Walsh--Fourier expansion.
4.  Retain the most important low- and moderate-order parity terms.
5.  Quantize the coefficients into weighted or replicated XOR clauses.
6.  Run DQI with an appropriate decoder.
7.  Validate the sampled tilt configuration in the original RF model.

#### Why I would not claim DQI advantage yet

The 2025 DQI result is impressive, but its superpolynomial advantage
applies to a specially structured polynomial-intersection problem. For
sparse Max-XORSAT, DQI beat several general-purpose heuristics on a
constructed instance, but a tailored classical heuristic still produced
a better result.

There is another antenna-specific concern: a simple interference
surrogate is mostly **2-local**. For studied (k=2) and (k=3) Max-XORSAT
ensembles, DQI with classical decoders faces strong limits.

Therefore:

-   If the antenna utility really is captured by unary and pairwise
    terms, DQI may not be the strongest algorithm.
-   If genuine higher-order effects involving several neighboring
    sectors are important, DQI becomes more interesting---but the clause
    expansion and circuit size also grow.
-   Artificially adding higher-order terms just to make DQI look
    suitable would be scientifically weak.

I would use DQI only after demonstrating that the RF objective has a
sparse, useful Fourier representation and that the corresponding
decoding problem is favorable.

### What about quantum neural networks?

This is the most developed antenna-specific quantum direction, but it
does not currently demonstrate computational advantage.

Ericsson replaced part of an antenna-tilt reinforcement-learning
workflow with a QNN. It obtained similar prediction accuracy with
substantially fewer trainable parameters and found that moderate
entanglement could partially compensate for smaller training sets.
However, its 2025 review estimated one QNN inference at roughly 36 ms
while a classical multilayer perceptron performed the corresponding
inference in microseconds.

Reducing parameter count is not itself quantum advantage.

### Comparison of the realistic routes

  --------------------------------------------------------------------------
  Route             Possible advantage Main obstacle     Assessment
  ----------------- ------------------ ----------------- -------------------
  Grover/minimum    Quadratic          Deep reversible   Strongest
  finding           reduction in       utility oracle    defensible
                    candidate                            theoretical story
                    evaluations                          

  Amplitude         Quadratic          Coherent scenario Excellent future
  estimation        reduction in       preparation       extension
                    stochastic samples                   

  DQI               Potentially        Antenna mapping   Most novel, but
                    stronger           and reversible    high risk
                    approximation      decoder           
                    scaling for                          
                    special structure                    

  QNN-enhanced RL   Fewer parameters   Slow inference    Antenna-specific,
                    or training        and difficult     but no current
                    examples           training          speed advantage

  Quantum           Possible empirical Embedding,        Straightforward,
  annealing/QUBO    time-to-solution   chains, and       but least novel and
                    improvement        strong classical  no proven advantage
                                       solvers           
  --------------------------------------------------------------------------

### Recommendation for the project

For a credible non-QAOA solution, I would use:

> **Classical interference-cluster selection + compact RF surrogate +
> Grover Adaptive Search on Classiq + full-simulator validation.**

Present amplitude estimation as the fault-tolerant extension for
stochastic KPI evaluation.

DQI is worth attempting as a more original branch, but first run a
Fourier-sparsity test on the cluster utility. If a small number of
parity terms accurately predicts unseen configurations, proceed. If the
spectrum is dense or the model is almost entirely pairwise, Grover
search is the safer and more defensible choice.

The honest advantage claim would be:

> "We demonstrate a quantum architecture with a provable quadratic
> reduction in expensive cluster-configuration evaluations, while
> preserving antenna SINR and throughput metrics through a validated RF
> surrogate. Current hardware does not yet provide an end-to-end
> wall-clock crossover."

------------------------------------------------------------------------

## User

Claude answer:

Short answer: for antenna tilt as the brief poses it, no --- and not
with QAOA. But that is a *strong* pitch position, not a weak one, if you
say it precisely and back it up. Let me separate three different
questions that usually get conflated.

### 1. Is there provable asymptotic advantage for this problem class?

No. Nobody has a proven super-polynomial speedup for NP-hard
combinatorial optimization, and there is no reason to think tilt is the
exception. The provable results are quadratic-ish: Grover-based minimum
finding (Dürr--Høyer), and Montanaro's speedup in the number of nodes a
branch-and-bound solver explores. Quadratic speedups are widely argued
to be insufficient once you pay fault-tolerance overhead --- logical
gate rates in the kHz--MHz range against classical clocks in the GHz
range, plus massive classical parallelism. The usual conclusion is that
you need quartic or better before error-corrected quantum wins on wall
clock.

QAOA specifically has known theoretical ceilings at low depth
(Farhi--Gamarnik--Gutmann's "needs to see the whole graph", Bravyi et
al. on symmetry obstructions, Hastings' classical local algorithms
matching low-depth QAOA), and at high depth the variational parameter
optimization eats the budget.

### 2. What does the current empirical record say?

This is the part worth knowing before you stand in front of Classiq's
people. There's a March 2026 end-to-end benchmark from Kipu Quantum, IBM
Quantum and Zuse Institute Berlin that is close to the state of the art
on exactly this question. They ran a hybrid quantum pipeline on IBM
Heron hardware against CPU and GPU classical solvers with full
wall-clock accounting, and found it competitive with strong classical
solvers --- while stating plainly that the results do not establish
quantum advantage or an asymptotic separation. Their enhanced parallel
tempering implementation on 64 cores was the strongest baseline and beat
the hybrid quantum pipeline on time-to-solution across both instance
families.

The detail that matters most for you: they excluded QAOA from the
benchmark entirely, because the available evidence indicates low-depth
QAOA on this kind of hardware topology does not outperform simulated
annealing, while deeper variants carry too much variational overhead.

The one real scaling-advantage result you'll hear cited is Shaydulin et
al. on Low Autocorrelation Binary Sequences, where noiseless simulations
up to 40 qubits showed fixed-parameter QAOA scaling better than
branch-and-bound exact solvers, with QAOA plus quantum minimum finding
giving the best empirical scaling known for that problem. Note what LABS
is: dense, highly structured, no geometric locality, and a problem where
the classical champion is exact search. That profile is close to the
opposite of tilt.

### 3. Why tilt in particular is a poor advantage candidate

Four structural reasons, and you should own all four rather than hope no
judge notices:

**Geometric locality.** Interference decays with distance, so your
interaction graph is sparse and nearly planar --- exactly the regime
where classical methods thrive. Bounded-degree 2D structure is friendly
to tensor networks, treewidth-based exact methods, and belief
propagation. Katzgraber et al.'s warning about spin-glass benchmarks
applies directly: naively chosen instances are often just classically
easy.

**Tilt is ordinal and the objective is smooth in it.** Adjacent tilt
values give similar SINR. That means the landscape isn't rugged in the
spin-glass sense --- frustration is mild, and multi-start local search
will land within a fraction of a percent of optimal. Ruggedness is the
fuel quantum heuristics need.

**It decomposes.** Because interference is local, you can partition the
network into overlapping clusters, solve each, and iterate on
boundaries. This is what operators actually do. Decomposition destroys
the "3\^10000 configurations" scare number that the brief leans on ---
nobody ever solves the monolithic instance.

**The surrogate ceiling, and this is the sharpest one.** Your QUBO
optimizes the surrogate, not reality. Total error = surrogate fitting
error + RF model error, and propagation models carry shadow-fading
standard deviation around 6--8 dB. That combined error is almost
certainly larger than the gap between a good heuristic solution and the
true optimum. So a perfect oracle solver buys you nearly nothing over
simulated annealing. **Advantage in solving is not advantage in
outcome.**

That last point is a genuine, quantifiable result you can produce
tomorrow, and it's more interesting than anything you'd get from a fake
advantage claim: *your measured surrogate fidelity is an upper bound on
the value any better optimizer can deliver, quantum or classical.* An
AT&T judge should actually want to know that number.

### What would have to be true for advantage

Put this on a slide as a checklist, and score tilt honestly against each
row:

-   **Hardness of the right kind** --- dense, long-range, frustrated
    coupling with a rugged landscape. Tilt alone fails this. It gets
    closer if you densify the decision space: co-optimize tilt with
    transmit power, azimuth, beam selection and handover offsets, so
    couplings extend beyond immediate neighbours.
-   **Native higher-order structure** --- SINR genuinely isn't
    quadratic, so the faithful encoding is HUBO, not QUBO. Frustrated
    HUBO instances are where the recent competitive quantum results
    actually live. This is your strongest legitimate research direction,
    and it follows from your own surrogate work.
-   **Better than quadratic speedup**, or a claim reframed around
    latency and tail latency rather than asymptotics.
-   **Objective evaluation must not dominate the runtime.** If RF
    simulation is the bottleneck, no solver speedup matters at all.

### The claim I'd actually make tomorrow

> We do not claim quantum advantage for antenna tilt, and we can tell
> you why. We claim four things: a faithful encoding of a non-quadratic
> RF objective with measured fidelity; QAOA recovering the verified
> exact optimum on instances small enough to brute-force; a scaling
> analysis of qubit count and interaction density; and a stated set of
> conditions under which a quantum solver could matter --- plus evidence
> on whether tilt meets them. It mostly doesn't, and the reason is
> geometric locality.

Then prepare for the question you *will* get: "Enhanced parallel
tempering on 64 cores beats hybrid quantum solvers on published HUBO
benchmarks. Why would tilt be different?" Your answer is the
densification argument above --- and if you're honest, "it probably
isn't different, which is why we characterized the problem instead of
overclaiming."

Teams that claim advantage get taken apart in those three minutes. Teams
that show they know exactly where the boundary sits tend to win the
"quantum fit and benchmarking" 20% outright.

## Assistant

Claude's answer is directionally stronger than my previous one for the
hackathon strategy, but it contains several claims I would not repeat in
front of the judges.

I would update our position to:

> **There is currently no evidence of practical quantum advantage for
> antenna-tilt optimization, and the structure of tilt-only optimization
> gives us reasons to suspect that near-term advantage is difficult. But
> it is too strong to say that quantum advantage is impossible or that
> the problem is intrinsically classically easy.**

The distinction matters.

Claude is correct on the most important empirical point. The March 2026
Kipu/IBM/Zuse benchmark really does say that its hybrid quantum solver
does **not** establish quantum advantage or asymptotic separation.
Enhanced parallel tempering was the strongest time-to-solution baseline.
The paper also explicitly excluded QAOA because available evidence on
their higher-order heavy-hex setting suggested shallow QAOA does not
beat simulated annealing, while deeper QAOA incurs significant
variational overhead.

So I agree with Claude on one strategic change:

**I would not use QAOA for our antenna solution.**

That also fits our original goal of doing something different from the
likely majority of hackathon teams.

The Shaydulin/LABS point is also accurate. Their QAOA study used
noiseless simulations up to 40 qubits, and fixed-parameter QAOA scaled
empirically better than exact branch-and-bound for LABS; QAOA combined
with quantum minimum finding had the best empirical scaling they
studied. But LABS is a quartic, unusually structured problem. There is
no basis for transferring that result directly to antenna tilt.

Where I disagree with Claude is more important for how we formulate our
project.

First, the statement that antenna interference is "sparse and nearly
planar" is too strong. It is reasonable to say that antenna interactions
are **geographically localized**, and the Bell Labs paper explicitly
constructs optimization clusters based on mutually interfering
neighboring sectors. But an interference graph is not necessarily
planar: sectors can interfere across several geographic neighbors,
overshooting cells can create longer-range edges, and shadowing can
change the effective interaction structure. "Local" is defensible;
"nearly planar and therefore easy for treewidth/tensor-network
algorithms" has not been established by the supplied literature.

Second, this sentence from Claude is unsupported:

> "multi-start local search will land within a fraction of a percent of
> optimal."

We absolutely should not say that.

The Bell Labs study demonstrates that a distributed heuristic gradient
method works well on a 57-sector LTE-like simulation, achieving roughly
10% higher average spectral efficiency and 100% higher 5th-percentile
spectral efficiency. It does **not** establish that its answer is within
any particular optimality gap.

That paper is actually more useful to us if we frame it correctly:

> There exists a strong, physically motivated classical local optimizer,
> so any quantum proposal must beat something much more serious than
> brute force.

That makes our benchmark credible.

The biggest technical problem in Claude's answer is the "surrogate
ceiling" argument.

Claude says that because shadow fading has roughly 6--8 dB standard
deviation, this error is probably larger than the difference between
good optimizers, so solving the surrogate more accurately is pointless.

That mixes up two different things.

The Bell Labs simulation indeed uses spatially correlated shadow fading
with an 8 dB standard deviation. But **shadow fading variance is not the
same thing as surrogate-model error**.

If our objective is

\[
U(`\theta`{=tex})=`\mathbb `{=tex}E\_{`\omega`{=tex}}\[U(`\theta`{=tex},`\omega`{=tex})\],
\]

then (`\omega`{=tex}) can contain shadow fading. With enough samples, we
can estimate that expected utility accurately even though individual
channel realizations vary by many dB.

A much better quantity to measure is

\[ `\epsilon`{=tex}*{`\rm surrogate`{=tex}} =
`\mathbb `{=tex}E*{`\theta`{=tex}} `\left[
|U_{\rm RF}(\theta)-\hat U(\theta)|
\right]`{=tex}. \]

Then compare that against the optimization improvement:

\[ `\Delta`{=tex}*{`\rm opt`{=tex}} =
U*{`\rm RF`{=tex}}(`\theta`{=tex}*{`\rm quantum`{=tex}}) -
U*{`\rm RF`{=tex}}(`\theta`{=tex}\_{`\rm classical`{=tex}}). \]

If

\[
`\epsilon`{=tex}*{`\rm surrogate`{=tex}}`\gg `{=tex}\|`\Delta`{=tex}*{`\rm opt`{=tex}}\|,
\]

then Claude's conclusion becomes valid: better optimization of the
surrogate has little value.

That is an excellent experiment for us. But we have to **measure it**,
not infer it from the 8 dB shadow-fading parameter.

There is also evidence that simplified RF models can retain meaningful
relative accuracy. The Ericsson antenna paper compares a simple
analytical performance model with a detailed dynamic system simulator
and reports similar relative-throughput behavior as tilt changes. So the
claim that RF-model uncertainty necessarily overwhelms optimization
gains is particularly hard to defend.

I would also soften Claude's statement that SINR means the "faithful
encoding is HUBO, not QUBO."

SINR has the form

\[ `\mathrm{SINR}`{=tex}*u= `\frac{P_sg_{su}}`{=tex}
{`\sum`{=tex}*{j`\neq `{=tex}s}P_jg\_{ju}+N_0}, \]

and throughput is often approximately

\[ R_u=`\log`{=tex}\_2(1+`\mathrm{SINR}`{=tex}\_u). \]

That is neither a QUBO nor a HUBO naturally. It is a rational/nonlinear
function. A binary polynomial representation arises only after
discretization or approximation.

Higher-order terms may genuinely be useful. For example,

\[ `\hat `{=tex}U(z)= c+ `\sum`{=tex}*i h_i z_i+
`\sum`{=tex}*{ij}J\_{ij}z_iz_j+
`\sum`{=tex}*{ijk}K*{ijk}z_iz_jz_k+`\cdots`{=tex}. \]

But whether the cubic and higher-order coefficients are important is
something we should **discover from the RF data**, not assume because
HUBO is currently fashionable.

That actually suggests a much better project.

Instead of beginning with "antenna optimization is a HUBO," do this:

1.  Generate the true RF utility for configurations in a small
    interference cluster.
2.  Fit polynomial models of increasing order: \[
    U_1,`\quad `{=tex}U_2,`\quad `{=tex}U_3,`\quad `{=tex}U_4. \]
3.  Measure held-out error as polynomial order increases.
4.  Measure the number and magnitude of significant interaction terms.
5.  Determine empirically whether antenna optimization is mostly
    local/pairwise or whether meaningful higher-order frustration
    exists.
6.  Apply the quantum solver to the **smallest faithful
    representation**.

This gives us a concrete scientific question:

> **What structure in a real antenna-tilt objective would actually be
> required for quantum optimization to help?**

That is much stronger than merely writing down a QUBO.

The recent Kipu benchmark gives us an especially useful comparison
because its instances were deliberately engineered as frustrated,
higher-order problems: 156 variables, 1,128--1,323 one-, two- and
three-body terms, with heavy-tailed Cauchy coefficients and swap-layer
densification. The authors themselves warn that this hardware-aware
construction limits how far the results can be generalized to unrelated
optimization problems.

So we can ask:

\[
`\text{Does an antenna-derived Hamiltonian actually resemble that regime?}`{=tex}
\]

I suspect the answer for **tilt alone** will be "not very much."

That is useful, not a failure.

There is also one subtle error in Claude's opening. "Nobody has a proven
super-polynomial speedup for NP-hard combinatorial optimization" is too
categorical. What we can safely say is:

> **There is no known general super-polynomial quantum speedup for
> solving arbitrary NP-hard combinatorial optimization problems, and
> none has been established for antenna-tilt optimization.**

Likewise, "you need quartic or better before quantum wins" is a rule of
thumb about hardware overhead, not a theorem. I would not put it on a
slide.

For our hackathon, therefore, I would change Claude's proposed final
pitch fairly substantially. The official brief explicitly asks for a
credible quantum-advantage argument, benchmarking, and honest
limitations; it does not require us to demonstrate actual quantum
advantage. The judging guide even explicitly tells teams to benchmark
against a classical baseline and state limitations honestly.

I would make our central claim:

> **"Instead of assuming antenna tilt is quantum-hard because its search
> space is exponential, we characterize where its actual computational
> hardness comes from. We extract the interaction Hamiltonian directly
> from an RF model, measure locality, higher-order coupling, surrogate
> fidelity and landscape ruggedness, and then apply a non-QAOA quantum
> optimizer only to the part of the problem whose structure warrants
> it."**

And the punchline:

> **"Our result identifies the boundary for quantum advantage: tilt-only
> optimization is strongly local and already admits effective classical
> methods; quantum methods become more plausible only when the RF
> objective contains sufficiently strong higher-order, frustrated
> interactions that survive decomposition."**

For the quantum implementation, I would then choose between **DQI and a
non-variational Hamiltonian method**, not QAOA. DQI becomes especially
interesting if our Fourier/Walsh analysis reveals a sparse but
nontrivial set of higher-order parity interactions. If it does not, that
itself is evidence that DQI is not a natural fit.

This is also more intellectually defensible than artificially
"densifying" the problem by throwing power, azimuth, handover
parameters, etc. into it. We should only add those variables if we can
show they represent a real AT&T optimization problem. Artificially
making the Hamiltonian difficult so that quantum looks better would be
easy for a technical judge to attack.

So my assessment of Claude's answer is roughly: **the strategic
conclusion is good; the empirical QAOA/Kipu discussion is good; the
argument about locality is plausible but overstated; and the
surrogate-error argument as written is wrong.**

Most importantly, it changes what I would build. I would no longer
center our submission on "Grover gives a quadratic speedup over (K\^N)."
I would center it on **measuring the actual structure of the antenna
objective and demonstrating exactly which structural conditions
create---or destroy---the case for quantum advantage.** That is both
more novel and much harder for the judges to dismantle.

------------------------------------------------------------------------

## Sources mentioned in the discussion

### Uploaded project sources

-   *QUBIT Hackathon 2026 --- Challenge Briefs*
-   *QUBIT Hackathon 2026 --- Participant Brief*
-   *Vertical Antenna Tilt Optimization for LTE Base Stations* ---
    Eckhardt, Klein & Gruber
-   *Impact of Electrical and Mechanical Antenna Tilt on LTE Downlink
    System Performance* --- Athley & Johansson
-   *Exploring the Potential Advantages of Quantum Computing in
    Telecommunication Networks* --- Ericsson Technology Review, January
    2025

### External sources discussed

-   Dürr & Høyer --- quantum minimum finding
-   Classiq --- Grover Adaptive Search / amplitude estimation / DQI
    documentation
-   DQI / Max-XORSAT literature
-   March 2026 Kipu Quantum / IBM Quantum / Zuse Institute Berlin hybrid
    quantum optimization benchmark
-   Shaydulin et al. --- QAOA for Low Autocorrelation Binary Sequences
-   March 2026 IEICE report on base-station beam-tilt optimization using
    quantum annealing

> Note: This export preserves the substantive user/assistant discussion.
> Chat UI metadata, internal tool calls, and hidden reasoning are not
> included.
