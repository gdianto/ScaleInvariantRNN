# Scale-invariant dynamics and context-dependent computation in heterogeneous neural networks

Code and data to reproduce the results and figures of

> F. Damiani, G. Di Antonio, J. Epifanio, G. V. Vinci, A. Galluzzi, M. Mattia,
> *Scale-invariant dynamics and context-dependent computation in heterogeneous neural networks without criticality.*

The model is a recurrent rate network, τ0 dr_i/dt = −r_i + tanh(s_i r_i + Σ_j W_ij r_j + inputs), in which every
unit has its own self-coupling s_i. A suitable distribution of s gives a broad distribution of unit timescales and a
scale-invariant (power-law) memory, without criticality. The repository covers:

| Figure | Content | Script |
|---|---|---|
| 1 | self-couplings, unit timescales τ(s), power-law decay of the population activity | `scripts/fig1.jl` |
| 2 | a decoupled network driven by Ornstein-Uhlenbeck (OU) noise reproduces the coupled network; escape-time theory | `scripts/fig2.jl` |
| 3 | short-term memory: heterogeneous network (hRNN) vs homogeneous network (hom) and echo-state network (esn) | `scripts/fig3.jl` |
| 4 | Wisconsin Card Sorting Test (WCST): strategy of the heterogeneous network | `scripts/fig4.jl` |
| 5 | WCST: heterogeneous self-couplings (s-network) vs heterogeneous time constants (τ0-network) | `scripts/fig5.jl` |

Each script writes the complete figure (SVG, PDF and PNG) at the published size and layout. By default the figures
are drawn from the data of the published runs (`data/published/`); with `--simulate` the scripts recompute the data
(`data/generated/`) and draw the figure from them.

## Installation

- [Julia](https://julialang.org) 1.11. The package versions in `Manifest.toml` were resolved with Julia 1.11.6;
  other Julia versions print a warning and may fail to install them. From the repository folder, install the
  packages once:

  ```bash
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```

  Packages: DelimitedFiles, Distributions, LsqFit, MultivariateStats, Plots and standard libraries.
  Every script activates the environment of the repository by itself, so it always uses exactly these versions,
  whatever the folder it is started from and whatever environment is active (also when run cell by cell in
  VS Code). A script started before the installation stops with a message asking to run `Pkg.instantiate()`.
- [Inkscape](https://inkscape.org) on the command line, to export PDF and PNG (without it only the SVG is written).
- The Arial font, for the PDF and PNG exports (e.g. `ttf-mscorefonts-installer` on Debian/Ubuntu).

## Quick start: the published figures

```bash
julia scripts/fig1.jl     # -> figures/published/fig1.svg, .pdf, .png
julia scripts/fig2.jl
julia scripts/fig3.jl
julia scripts/fig4.jl
julia scripts/fig5.jl
```

Each script takes a few seconds. Schematics and panels that show one particular network realization are embedded
as artwork (`assets/`); everything else is drawn from data.

## Data and figures

```
data/published/           data of the published figures: read by default, never written by the scripts
data/generated/[TAG/]     data computed by the scripts, same layout (not under version control)
figures/<source>/         rendered figures: figures/published/, figures/generated/[TAG/], ...
```

The layout and the file formats are described in [`data/README.md`](data/README.md). The same flags work in every
script:

| Flag | Effect |
|---|---|
| `--simulate` | compute new data, write them to `data/generated/` and draw the figure from them |
| `--data=published\|generated\|FOLDER` | data to draw from (default: `published`) |
| `--s=published\|generated\|FILE` | self-couplings used by the simulations (default: `published`) |
| `--tag=TAG` | use `data/generated/TAG/` instead of `data/generated/`, to keep several runs |
| `--force` | replace existing generated data (otherwise the script stops instead of overwriting them) |
| `--out=PREFIX` | output name of the figure (default: `figures/<source>/figN`) |

## Reproducing the results from scratch

All results can be recomputed from the code alone. Simulations are seeded, except the WCST runs (unseeded, as in
the study); new results are statistically equivalent to the published ones, not identical.

```bash
julia scripts/self_couplings.jl                        # self-couplings (~2 min)
julia scripts/fig1.jl --simulate --s=generated         # Fig. 1 (~2 min)
julia scripts/fig2.jl --simulate --s=generated         # Fig. 2 (~40 min)
julia -t 9 scripts/fig3.jl --simulate --s=generated    # Fig. 3 (~2 min on 9 threads)
julia -t 16 scripts/fig5.jl --simulate --s=generated   # WCST sweep and Fig. 5 (~20 CPU-hours)
julia scripts/fig4.jl --data=generated                 # Fig. 4 from the same sweep
```

The data go to `data/generated/` and the figures to `figures/generated/`. Without `--s=generated` the simulations
use the self-couplings of the paper. Adding the same `--tag=TAG` to every command keeps the whole run in
`data/generated/TAG/`. Runtimes are for a 16-core workstation.

## Figures

### Figure 1: heterogeneous self-couplings generate multiple timescales

```bash
julia scripts/fig1.jl                      # published data
julia scripts/fig1.jl --simulate           # new run: 500 networks x 500 units, seed 1234
julia scripts/fig1.jl --simulate --smoke   # quick test: 40 units, 2 networks (data/generated/smoke/)
```

| Panel | Content | Default source |
|---|---|---|
| a | schematic; distribution of s and the P(s) implied by the τ(s) fit | `fig1e_tau_vs_self_coupling.tsv` |
| b | s of each unit, activity of one network and network-averaged activity | artwork (generated data: `fig1b_*.tsv`) |
| c | single-trial activity of four units (s = 0.70, 0.80, 1.00, 1.23) | `fig1c_single_trial_activity.tsv` |
| d | the same units, averaged over 500 networks | `fig1d_mean_activity.tsv` |
| e | relaxation time τ of each unit vs s, fit τ(s) = a exp(b s^c) | `fig1e_tau_vs_self_coupling.tsv` |
| f | population activity ⟨R(t)⟩, power-law fit on 0.07–0.9 s | `fig1f_population_activity.tsv` |

Model: N = 500, g = 0.4 (Gaussian couplings W with spectral radius g), τ0 = 0.01 s, all units start at r = 1, no
input; explicit Euler with dt = 1 ms over 2 s; averages over 500 realizations of W with the same self-couplings.
The relaxation time of each unit is fitted as exp(−t/τ) for s < 1 and A exp(−t/τ) for s ≥ 1. The published run
gives a = 0.0165, b = 1.017, c = 2.99 and a decay ∝ t^−0.93; a new run gives a = 0.0154, b = 1.085, c = 2.89 and
t^−0.95.

### Figure 2: an effective colored noise replaces the recurrent input

```bash
julia scripts/fig2.jl                      # published data
julia scripts/fig2.jl --simulate           # new run: 500 coupled + 500 decoupled networks, seeds 1234 / 4321
julia scripts/fig2.jl --simulate --smoke   # quick test (data/generated/smoke/)
```

| Panel | Content | Default source |
|---|---|---|
| a | coupled network (case 1) and decoupled network driven by OU inputs (case 2) | schematic |
| b | ⟨R(t)⟩ of both networks, mean ± 3 SEM over 500 networks | `fig2b_*.tsv` |
| c | τ(s) of each unit in both networks | `fig2c_tau_vs_self_coupling.tsv` |
| d | coupled network, s ≥ 1: τ(s) and the theoretical escape time | `fig2d_tau_coupled_s_gt_1.txt`; theory computed |

Model: as Fig. 1 with dt = 0.1 ms. The decoupled network keeps only the self-couplings; unit i receives
Σ_k Jext_ik x_k(t), with Jext_ik ~ N(0, 1) and N independent stationary OU processes x_k (τ_OU = τ0,
σ_OU = 0.0187). `--simulate` also writes the statistics of the recurrent input of the coupled network.

The theoretical curve is the Kramers-type escape time of a bistable unit in the potential
U(x) = x²/2 − s log cosh x, with noise amplitude σ = √N σ_OU (`escape_time` in `src/timescales.jl`).

### Figure 3: delay-invariant short-term memory

```bash
julia scripts/fig3.jl                      # published values
julia -t 9 scripts/fig3.jl --simulate      # new simulation (--nets=50, --seed=2026)
```

| Panel | Content | Default source |
|---|---|---|
| a | stimulation scheme, activity of one hRNN realization | artwork |
| b | hRNN: correlation between evoked states vs signed position distance, for each T_sep | `fig3_values.csv` |
| c | correlation vs \|position distance\| for hRNN, hom and esn (mean ± SEM over 50 networks) | `fig3_values.csv` |

| Network | g | τ0 | self-coupling |
|---|---|---|---|
| hRNN | 0.4 | 0.01 s | the self-coupling vector |
| hom | 0.4 | 0.01 s | its mean, 1.026 |
| esn | 0.9999 | 0.04 s | 0 |

Protocol: N = 500; six orthogonal stimuli, each on 1/6 of the units, lasting 0.01 s and separated by
T_sep = 0.01, 0.05, 0.2 s; dt = 1 ms; 5 transient cycles, then 10 recorded cycles; 50 networks per condition.
The published values were computed with an earlier NumPy implementation of the same protocol (seed 2026), which
Julia cannot reproduce digit by digit; `--simulate` gives values within 2 SEM of them. The simulation is
reproducible for a given seed, whatever the number of threads.

### Figure 4: WCST strategy of the s-network (g_b = 0.8)

```bash
julia scripts/fig4.jl                      # published WCST results
julia -t 16 scripts/fig4.jl --simulate     # only the runs of Fig. 4: 50 s-network runs at g_b = 0.8 (~1 CPU-hour)
julia scripts/fig4.jl --kde                # panel b with kernel density estimates (figure fig4_kde)
```

| Panel | Content | Default source |
|---|---|---|
| a | reservoir and task | schematic |
| b | maximum readout weight vs self-coupling of the unit (response, context), Gaussian fits (`--kde`: kernel density estimates, Scott's bandwidth) | `wcst/` |
| c | sorted self-couplings before learning; after learning, mean ± std over 50 networks | `wcst/` |
| d, e | state space of one network (PCA, hysteresis) | artwork; see `scripts/explore_wcst.jl` |

### Figure 5: heterogeneous s vs heterogeneous τ0

```bash
julia scripts/fig5.jl                      # published WCST results
julia -t 16 scripts/fig5.jl --simulate     # the whole sweep (~20 CPU-hours)
```

| Panel | Content (all from `wcst/`) |
|---|---|
| a, b | self-couplings before and after learning, s-network and τ0-network, g_b = 0.8 |
| c | accuracy vs context feedback g_b, mean ± std over 50 networks |
| d | change of the synaptic matrix \|ΔJ\|/\|J0\| vs g_b, mean ± std over 50 networks |
| e | mean autocorrelation of the closed-loop network for each g_b (mean ± SEM); x axis: samples, one every 5 ms |

WCST model: N = 500, g = 0.4. The s-network has the self-couplings of Fig. 1 and τ0 = 0.01 s; the τ0-network has no
self-coupling and τ0 evenly spaced in [0.01, 0.05] s. Each run has 20 transient, 80 learning and 100 test trials, a
20 % probability of rule change per trial, inputs g_cards = 1.0 and g_error = 1.75, a ridge readout (β = 0.01) fed
back into the network, and a power-law test of the closed-loop network from 100 random initial states.

## Self-couplings

`scripts/self_couplings.jl` builds the self-coupling vector as described in the Methods: a pilot simulation with s
evenly spaced in [0.7, 1.7] measures τ(s), the fit τ(s) = a exp(b s^c) gives the distribution P(s) for which the unit
timescales follow P(τ) ∝ τ⁻², and 500 values are drawn from P(s) by inverse transform sampling. The pilot gives
a = 0.0156, b = 1.066, c = 2.85 (paper: 0.0165, 1.017, 2.99). With a new vector, Fig. 1 decays as t^−1.10 and the
hRNN of Fig. 3 at T_sep = 0.2 s gives ρ = 0.81, 0.57, 0.42 at distances 1, 2, 3 (published: 0.82, 0.60, 0.46).

The new vector is written to `data/generated/self_couplings.txt`; `--s=generated` uses it in the simulations
(`s_file` in the first cell of the explorations).

## Interactive exploration

Two scripts are meant to be run cell by cell (`##` cells, e.g. in VS Code). All parameters are in the first cell,
with the values of the paper as defaults; `save_figures = true` saves the plots to `figures/`.

- `scripts/explore_timescales.jl` (Figs. 1-2, ~30 s): rasters and traces; τ(s), its fit and the implied P(s); the
  power-law decay; amplitude, autocorrelation and time course of the recurrent input vs the OU input; coupled vs
  decoupled network with the escape-time theory; the fit of σ_OU as in the Methods (`run_grid_search = true`).
  It averages over 50 networks by default; with so few networks the timescale of a bistable unit is estimated as the
  first time its mean activity drops below 1/e (`tau_method`).
- `scripts/explore_wcst.jl` (Figs. 4-5, ~1 min): one network through the WCST task: inputs, readout fits, test
  behaviour and accuracy, change of the self-couplings and of J, readout weights, state space (PCA and hysteresis,
  as in Fig. 4d-e), decoding of the card features, power-law test.

```bash
julia scripts/explore_timescales.jl     # or open it in VS Code and run the cells
```

## Repository structure

```
scripts/
  fig1.jl ... fig5.jl     one script per figure
  self_couplings.jl       new self-coupling vector (Methods)
  explore_timescales.jl   interactive exploration of Figs. 1-2
  explore_wcst.jl         interactive exploration of the WCST (Figs. 4-5)
src/
  project.jl              paths, command-line flags, activation of the environment
  reservoir.jl            rate network: Network, simulate! (Euler), fit_readout, close_loop
  timescales.jl           Figs. 1-2: coupled and OU-driven networks, fits of τ(s) and of the power law,
                          P(s) and its sampling, escape-time theory
  memory_task.jl          Fig. 3: stimulation protocol and correlations
  wcst.jl                 Figs. 4-5: task, training, test, power-law test, sweep
  wcst_plots.jl           plots of explore_wcst.jl
  SVGFigure.jl            figure builder (layout in mm, Arial text, SVG/PDF/PNG)
data/
  README.md               layout and file formats
  published/              self_couplings.txt, fig1/, fig2/, fig3/, wcst/: data of the published figures
  generated/              data computed by the scripts (not under version control)
assets/                   artwork of the published figures, embedded by the scripts
figures/                  rendered figures (not under version control).
```
