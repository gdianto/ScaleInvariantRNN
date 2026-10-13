ROBUSTNESS OF THE ASYMPTOTIC NETWORK REGIME
===========================================

Purpose
-------
This analysis tests whether the stable, non-chaotic dynamics and approximately
scale-invariant relaxation of the heterogeneous RNN persist over a finite region
of parameter space. It complements analyze_network_regime.jl and
analyze_nonconverged_attractors.jl; it does not modify or replace them.

Two independent parameter sweeps are performed:

  1. recurrent gain g, with the published self-couplings unchanged;
  2. quenched self-coupling perturbations

         s_i' = s_i + sigma_s z_i,     z_i ~ Normal(0,1),

     with g = 0.4.

The off-diagonal connectivity matrix is paired across all parameter conditions
for a given realization. The Gaussian perturbation vector is also paired across
all sigma_s values. Thus changes across a sweep are not confounded by drawing a
different connectivity matrix for every parameter value.

Files
-----
    analyze_network_robustness.jl   run the simulations and save numerical data
    plot_network_robustness.jl      plot saved results; does not run simulations

Model and diagnostics
---------------------
The script uses the repository network constructor and the same model and
integration conventions as the main analysis: N = 500, tau_0 = 0.01 s,
dt = 0.001 s, r_i(0) = 1, no external input, and no ongoing noise.

Every trajectory is first tested for fixed-point convergence using

    rho(t) = ||tanh(Jr(t)) - r(t)|| / sqrt(N).

A fixed point is recorded when rho stays below 1e-7 for 20 consecutive seconds.
Its largest Lyapunov exponent is then estimated with the Benettin method.

Trajectories that do not meet the fixed-point criterion are not automatically
called chaotic. Their first two Lyapunov exponents and full-state recurrence are
measured after the maximum convergence time. They are classified as:

  fixed_point         fixed-point criterion satisfied and lambda_1 negative;
  periodic_candidate lambda_1 compatible with zero, lambda_2 negative, and
                     relative recurrence error below 5%;
  chaotic_candidate  the approximate 95% block interval of lambda_1 is positive;
  unresolved         none of the above criteria is satisfied.

"Chaotic candidate" is deliberately cautious: a finite positive exponent over a
finite measurement window is evidence for chaos, not a mathematical proof of an
asymptotic chaotic attractor.

Scale-invariant relaxation
--------------------------
For each parameter condition, the population activity R(t) is averaged across
all connectivity realizations. The fit uses the same function and approximately
the same 0.07--0.9 s interval as Fig. 1. The output reports the fitted decay

    R(t) = amplitude * t^(-gamma)

and R2. The exponent should only be interpreted as evidence for power-law-like
relaxation when the fit quality is sufficiently high; the script reports the
numbers without imposing an arbitrary R2 threshold.

Running the analysis
--------------------
The default run uses 20 networks per condition:

    julia scripts/analyze_network_robustness.jl

Run a small gain-sweep pilot first:

    julia scripts/analyze_network_robustness.jl \
        --sweep=g --nets=3 --max-duration=1000 \
        --alignment=20 --measurement=50 --tag=pilot

Run only the self-coupling perturbation sweep:

    julia scripts/analyze_network_robustness.jl \
        --sweep=sigma --nets=20 \
        --sigma-values=0,0.05,0.1,0.2

Important options:

    --sweep=both|g|sigma
    --g-values=0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2
    --sigma-values=0,0.05,0.1,0.2
    --nets=20
    --max-duration=10000
    --alignment=100
    --measurement=300
    --seed=20001
    --perturbation-seed=70001
    --s=published|generated|FILE
    --tag=NAME
    --force

The pilot settings are intended only to check runtime and output. Use the full
convergence and Lyapunov windows for results intended for scientific reporting.

Outputs
-------
Numerical outputs are written to

    data/generated/network_robustness/

or to data/generated/TAG/network_robustness/ when --tag=TAG is used.

    realization_results.tsv   one row per network and parameter condition
    condition_summary.tsv     attractor fractions and fitted relaxation by condition
    population_relaxation.tsv population-averaged R(t) for every condition
    summary.txt                compact human-readable summary

Existing output is not overwritten unless --force is supplied.

After the analysis has finished, generate the summary figures with

    julia scripts/plot_network_robustness.jl

or, for a tagged run,

    julia scripts/plot_network_robustness.jl --tag=pilot

This creates separate gain- and self-coupling-sweep figures under
figures/generated/. Each figure shows individual largest Lyapunov exponents,
attractor-class fractions, the fitted power-law exponent, and R2.

Interpretation and limitations
------------------------------
The purpose is to determine whether non-chaotic dynamics and power-law-like
memory persist across a finite parameter region, and approximately where
positive asymptotic Lyapunov exponents begin to appear. The analysis samples a
finite connectivity ensemble and one initial condition; it does not prove the
absence of every possible critical transition or attractor.

The repository targets Julia 1.11.6. Random streams are reproducible within the
pinned environment but are not guaranteed to remain identical across Julia
versions. Report the Julia version and all command-line options with any final
numerical results.
