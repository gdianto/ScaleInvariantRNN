# CLAUDE.md

Code that reproduces the results and figures of Damiani et al. (heterogeneous self-couplings in recurrent
networks: multiple timescales, OU-driven decoupled network, short-term memory task, WCST). See README.md.

## Rules for this repository
- One script per paper figure in `scripts/` (`fig1.jl` ... `fig5.jl`). Each writes the *complete* figure,
  identical to the published one. Reusable code goes in `src/`.
- `scripts/explore_wcst.jl` and `scripts/explore_timescales.jl` are interactive (cell-by-cell) explorations.
  Keep their `##` cells, all parameters in the first cell (paper values as defaults) and the top-level
  variables the cells expose. `src/wcst_plots.jl` uses `using Plots`: never include it in the figure scripts.
- Style: Julia conventions (lowercase functions and variables, `!` for mutating functions, CamelCase types and
  modules, UPPERCASE constants; math symbols of the paper such as N, J, W, R, τ are fine). Keep code and
  comments simple and short; no over-engineering.
- Figure style: Arial; 7 pt labels and tick labels, sub/superscripts at 65%, 10 pt bold panel letters;
  page size, panel positions, colours and line widths as measured from the published SVG
  (all coordinates in mm from the top-left corner, see `src/SVGFigure.jl`).
- Data layout (see data/README.md): `data/published/` is the data of the published figures, never written by the
  code and never modified unless explicitly asked; `data/generated/[TAG/]` is written by the scripts, in the same
  layout. Scripts write data only through `new_data_dir`/`new_data_path` (src/project.jl), which stop if the target
  exists unless `--force`; figures go to `figures/<source>/` (`figure_path`). Figure scripts read through
  `data_dir(...)` (`--data=published|generated|FOLDER`) with one loader for published and generated data: keep the
  formats of the two identical.
- Do not change parameter values, random-number usage, algorithms or plot styles without approval.
- Everything must be reproducible from the repository alone (`--simulate`, `scripts/self_couplings.jl`, `--s`).
  The paper source and the published figures are not part of the repository: no script or doc may depend on them,
  and README.md must not mention files that are not in the repository.
- `data/published/fig1/` and `fig2/` hold the data of the published runs (one unseeded realization each). In the
  docs, describe them only as "one realization of the simulation, unknown seed" (the user's choice).
- No Git repository yet; the user will initialize it.

## Running
- Julia 1.11, environment at the root: `julia scripts/figN.jl` (from any directory). `src/project.jl` activates the
  repository environment (`Pkg.activate(PROJECT_ROOT)`): every script must include it before any `using`/`import`.
- PDF/PNG export needs Inkscape on the PATH (`SVGFigure.save`); otherwise only the SVG is written.
- Figure scripts `import Plots` (never `using Plots`): Plots exports names that clash with `SVGFigure`
  (`plot!`, `scatter!`, `xaxis!`, ...). They use Plots only for colour maps, `Plots._make_hist` (the bins of
  the original `stephist` panels) and colours.

## Code
- `src/reservoir.jl`: `Network` and `simulate!` (explicit Euler; τ dr/dt = -r + tanh(J r + Jext u) + noise),
  `fit_readout`, `close_loop`. It keeps the numerics and the random draws of the code used for the paper.
  Things that look removable but are not:
  - `simulate!` draws the noise matrix `randn(n, nt)` even when σ = 0;
  - `wcst_network` draws and discards `sprandn(N, N, 10/N); rand(N, n_inputs)` (the old constructor);
  - `run_wcst` fits the central-card readout ("pinv", noisy) before the power-law test;
  - `run_memory_task` keeps one network across realizations (each starts from the previous final state).
- `src/timescales.jl` (Figs. 1-2, `sample_self_couplings`, `escape_time` for the theory of Fig. 2d, which
  reproduces the published curve to 1e-10 with σ = √N σ_OU), `src/memory_task.jl` (Fig. 3), `src/wcst.jl`
  (Figs. 4-5): include after `project.jl` and `reservoir.jl`.
- `self_couplings_file()` (src/project.jl): `--s=published|generated|FILE` (default: data/published/self_couplings.txt).

## How the figures are built
- Quantitative panels are drawn from data with `SVGFigure` (axes boxes, tick positions and label positions
  measured from the published SVG).
- Artwork and single-realization panels (Fig. 3a and the timing schemes of 3b, Fig. 4a and 4d-e) are
  embedded from `assets/fig*_artwork.svg`, extracted from the published figures: the elements of
  the published SVG outside the redrawn panels, unchanged, with ids prefixed (`f3_`, `f4_`).
- Figs. 1-2 work the other way round. `assets/fig{1,2}_layout.svg` is the published SVG minus its data
  marks (axes, ticks, labels, legends and schematics are kept, ids prefixed `f1_`/`f2_`). The scripts draw only
  the data, through measured mappings `mm = c0 + c1*u` (the `AX` tables, `Axes(fig, box, xmap, ymap)`).
  Fig. 1b (one network) is embedded from `fig1_panel_b_example.svg`; generated data include the rasters
  (`fig1b_*.tsv`), drawn as PNG with a similar (not identical) threshold line.
- The default figure scripts are deterministic: check a change by comparing their SVG output before and after.
  Against the published figures (300 dpi, grey-level difference > 40) Figs. 1-2 differ in < 0.01% of the pixels,
  Figs. 3-4 in ~0.1%, Fig. 5 in ~1.5% (its panels c-d show mean ± std on purpose; the published bands were mean/√n).

## Simulations (`--simulate`)
- Figs. 1-2: `src/timescales.jl`, seeded (Fig. 1: 1234; Fig. 2: 1234 coupled, 4321 decoupled). `--smoke` runs
  40 units × 2 networks into `data/generated/smoke/`.
- Fig. 3: `src/memory_task.jl`, seeded (default 2026), BLAS forced to 1 thread: bit-reproducible. The
  published values (`data/published/fig3/fig3_values.csv`) come from a NumPy run and cannot be reproduced digit by digit.
- Figs. 4-5: `run_wcst` / `run_wcst_sweep` in `src/wcst.jl`, unseeded (as the original study).
- To verify a change of simulation code: put the same `Random.seed!(n)` after the includes in a scratch copy of
  the old and of the new code, run both with the same BLAS thread count (e.g. `OPENBLAS_NUM_THREADS=2`) and
  compare the result files byte for byte (`fig1.jl`/`fig2.jl --simulate --smoke` are seeded; `fig3.jl
  --simulate --nets=3` too; use `--tag` or `--force` for repeated runs). The default figure scripts are deterministic: compare their SVG output.
