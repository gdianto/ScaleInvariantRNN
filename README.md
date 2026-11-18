# Heterogeneous self-couplings enable scale-invariant memory and cognitive flexibility without criticality

Code and data to reproduce the figures of

> F. Damiani, G. Di Antonio, J. Epifanio, G. V. Vinci, A. Galluzzi, M. Mattia,
> *Heterogeneous self-couplings enable scale-invariant memory and cognitive flexibility without criticality*,
> bioRxiv (2026). https://doi.org/10.64898/2026.10.25.761657

The model is a recurrent rate network, τ0 dr_i/dt = −r_i + tanh(s_i r_i + Σ_j W_ij r_j + inputs), in which every
unit has its own self-coupling s_i. A suitable distribution of s gives a broad range of unit timescales and a
scale-invariant (power-law) memory without criticality, which the network uses for short-term memory and for a
context-dependent task (Wisconsin Card Sorting Test, WCST).

## Installation

- [Julia](https://julialang.org) 1.11 (with [juliaup](https://github.com/JuliaLang/juliaup): `juliaup add 1.11`, then
  `julia +1.11` in the commands below). Install the packages once, from the repository folder:

  ```bash
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```

- [Inkscape](https://inkscape.org) and the Arial font, to export PDF and PNG (without Inkscape only the SVG is written).

## Figures

Each script draws one figure of the paper from the data in `data/published/` and writes it to `figures/published/`
(SVG, PDF, PNG) in a few seconds:

```bash
julia scripts/fig1.jl
```

| Figure | Content | Script | `--simulate` |
|---|---|---|---|
| 1 | self-couplings, unit timescales, power-law decay | `scripts/fig1.jl` | ~2 min |
| 2 | decoupled network driven by Ornstein-Uhlenbeck noise; escape-time theory | `scripts/fig2.jl` | ~40 min |
| 3 | short-term memory: heterogeneous vs homogeneous and echo-state networks | `scripts/fig3.jl` | ~2 min |
| 4 | WCST: strategy of the heterogeneous network | `scripts/fig4.jl` | ~1 CPU-hour |
| 5 | WCST: heterogeneous self-couplings vs heterogeneous time constants | `scripts/fig5.jl` | ~20 CPU-hours |
| S1 | the autonomous network relaxes to a fixed point | `scripts/figS1.jl` | ~1 h on 16 threads |

## Reproducing the results

With `--simulate` a script recomputes its data (written to `data/generated/`) and draws the figure from them
(`figures/generated/`). Use several threads for the longer runs, e.g. `julia -t 16 scripts/fig5.jl --simulate`.

| Flag | Effect |
|---|---|
| `--simulate` | compute new data and draw the figure from them |
| `--data=published\|generated\|FOLDER` | data to draw from (default: `published`) |
| `--s=published\|generated\|FILE` | self-couplings of the simulations (default: those of the paper) |
| `--tag=TAG` | write to `data/generated/TAG/`, to keep several runs |
| `--force` | replace existing generated data |
| `--out=PREFIX` | output name of the figure |

`scripts/self_couplings.jl` builds a new self-coupling vector as in the Methods (use it with `--s=generated`).
The WCST sweep of `fig5.jl --simulate` includes the runs of Fig. 4: after it, run `fig4.jl --simulate --switch-only`.

The simulations are seeded, except the WCST runs of Figs. 4b-c and 5 (unseeded, as in the study), so new results are
statistically equivalent to the published ones, not identical. The data formats are
described in [`data/README.md`](data/README.md).

Two scripts explore the model interactively, cell by cell: `scripts/explore_timescales.jl` (Figs. 1-2) and
`scripts/explore_wcst.jl` (Figs. 4-5).

## Repository structure

```
scripts/     one script per figure, self_couplings.jl, two interactive explorations
src/         model, tasks and analyses (reservoir.jl, timescales.jl, memory_task.jl, wcst.jl, regime.jl, ...)
             and the figure builder (SVGFigure.jl)
data/        published/: data of the figures; generated/: new runs (not under version control)
assets/      artwork embedded in the figures
figures/     rendered figures; published/ is under version control
```

## Citation

Please cite the paper above (citation metadata in `CITATION.cff`, or "Cite this repository" on GitHub).

## License

MIT (`LICENSE`), for the code, the data and the figures.
