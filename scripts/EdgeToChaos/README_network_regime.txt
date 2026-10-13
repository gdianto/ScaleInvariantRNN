ASYMPTOTIC NETWORK-REGIME ANALYSIS
==================================

Purpose
-------
These scripts test the asymptotic dynamical regime of the autonomous
heterogeneous recurrent neural network (hRNN). They support the statement in
Results, Section 2.1, that the network displays stable, non-chaotic asymptotic
dynamics for the parameters used in the paper: most connectivity realizations
converge to fixed points, while the remaining realizations are consistent with
stable periodic attractors.

The analysis uses the same N = 500 units, self-couplings, recurrent gain
g = 0.4, single-unit time constant tau_0 = 0.01 s, integration step
dt = 0.001 s, and polarized initial condition r_i(0) = 1 used in the paper.
The network is autonomous: external input and ongoing noise are zero.

Scripts and execution order
---------------------------
Run the scripts from the repository root in this order:

    julia scripts/analyze_network_regime.jl
    julia scripts/analyze_nonconverged_attractors.jl

Both scripts include src/project.jl and therefore activate the repository Julia
environment automatically. Existing output is never overwritten silently. Use
--force to replace an existing run or --tag=NAME to create a separate run.

Reproducibility note
--------------------
The numerical results reported below (including the 93 fixed-point and seven
periodic-candidate realizations) were obtained with Julia 1.8.5. Each network
uses a fixed seed, and BLAS is restricted to one thread, making repeated runs
deterministic within the original software environment.

The repository currently targets Julia 1.11.6. Julia does not guarantee that
its default random-number stream remains identical across Julia releases, even
when the same seed is used. A run under Julia 1.11.6 therefore reproduces the
same scientific protocol and samples the same connectivity distribution, but it
may generate different matrices and is not guaranteed to reproduce the exact
93/7 split.
Exact archival reproduction of the reported realization-level results requires
the original Julia 1.8.5 environment or preservation of the generated matrices
or output data. See the Julia documentation on random-number reproducibility:

    https://docs.julialang.org/en/v1.11/stdlib/Random/#Reproducibility

1. analyze_network_regime.jl
----------------------------
This script simulates 100 independent realizations of the off-diagonal
connectivity matrix, using seeds 20001--20100 by default. Each realization is
simulated for up to 10,000 s.

Fixed-point convergence is monitored using

    rho(t) = ||tanh(Jr(t)) - r(t)|| / sqrt(N).

A realization is classified as converged when rho remains below 1e-7 for 20
consecutive seconds. After convergence, the largest Lyapunov exponent is
estimated over a 300-s measurement window using tangent-vector renormalization,
following the algorithm of Benettin et al. A preceding 100-s interval is used
for tangent-vector alignment and is not included in the exponent estimate.

The script saves convergence times, Lyapunov exponents, bootstrap confidence
intervals, residual histories, Lyapunov blocks, and final network states in

    data/generated/network_regime_extended_100/

With the default paper parameters, 93 of 100 realizations converged to stable
fixed points. Their mean convergence time was 340 s (95% CI: 269--417 s), and
their mean largest Lyapunov exponent was -3.24 s^-1
(95% CI: -3.60 to -2.89 s^-1).

2. analyze_nonconverged_attractors.jl
-------------------------------------
This script analyses the realizations that did not satisfy the fixed-point
criterion within 10,000 s. It starts from the final states saved by
analyze_network_regime.jl, so it does not repeat the long simulations.

For each remaining realization, it computes the two largest Lyapunov exponents
over 300 s after a 100-s alignment interval. The calculation is repeated with
dt = 0.001 s and dt = 0.0005 s to check that the classification is not an
integration-step artifact. The script also estimates recurrence periods and
measures recurrence of the complete 500-dimensional activity vector.

A stable periodic candidate is identified by:

  - a largest Lyapunov exponent compatible with zero;
  - a negative second Lyapunov exponent;
  - a relative full-state recurrence error below 5%.

The outputs are written to

    data/generated/nonconverged_attractors/

For the default run, all seven remaining realizations met these criteria at both
integration steps. Their maximum relative recurrence error was 1.1%. These
results are consistent with stable periodic attractors rather than asymptotic
chaos.

Interpretation
--------------
Across the 100 tested connectivity realizations, no positive asymptotic largest
Lyapunov exponent was detected. The results support stable, bounded,
non-chaotic asymptotic activity under the tested parameters and polarized
initial condition: 93 realizations approached fixed points and seven displayed
dynamics consistent with stable periodic attractors.

This is numerical evidence for the sampled connectivity matrices and initial
condition, not a proof for every possible network realization or initial state.
The term "subcritical" should therefore be understood as below the observed
transition to asymptotic chaos.

Reference
---------
G. Benettin, L. Galgani, A. Giorgilli, and J.-M. Strelcyn,
"Lyapunov characteristic exponents for smooth dynamical systems and for
Hamiltonian systems: A method for computing all of them. Part 2: Numerical
application," Meccanica 15, 21--30 (1980).
https://doi.org/10.1007/BF02128237
