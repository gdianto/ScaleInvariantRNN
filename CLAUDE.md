# CLAUDE.md

Code that reproduces the results and figures of Damiani et al. (heterogeneous self-couplings in recurrent
networks: multiple timescales, OU-driven decoupled network, short-term memory task, WCST). See README.md.

## Rules for this repository
- One script per paper figure in `scripts/` (`fig1.jl` ... `fig5.jl`, supplementary `figS1.jl`, `figS2.jl`). Each
  writes the *complete* figure, identical to the published one except for the revisions approved by the user (sub/
  superscripts enlarged to 5 pt in all figures; Fig. 4: the revised layout, see below; Fig. 5: revised panels).
  Reusable code goes in `src/`. The manuscript includes these renders (`manuscript/Figures/figN.pdf`, not in git).
- `scripts/explore_wcst.jl` and `scripts/explore_timescales.jl` are interactive (cell-by-cell) explorations.
  Keep their `##` cells, all parameters in the first cell (paper values as defaults) and the top-level
  variables the cells expose. `src/wcst_plots.jl` uses `using Plots`: never include it in the figure scripts.
- Style: Julia conventions (lowercase functions and variables, `!` for mutating functions, CamelCase types and
  modules, UPPERCASE constants; math symbols of the paper such as N, J, W, R, τ are fine). Keep code and
  comments simple and short; no over-engineering.
- Figure style: Arial; 7 pt labels and tick labels, 10 pt bold panel letters; no text below 5 pt, sub/superscripts
  included (`MIN_PT` in src/SVGFigure.jl: indices of 7 pt text at 72%; the artwork assets were enlarged the same way);
  page size, panel positions, colours and line widths as measured from the published SVG (Figs. 1-3, 5; Fig. 4 and
  Figs. S1-S2 follow the grids of their scripts). All coordinates in mm from the top-left corner (`src/SVGFigure.jl`).
- Data layout (see data/README.md): `data/published/` is the data of the published figures, never written by the
  code and never modified unless explicitly asked; `data/generated/[TAG/]` is written by the scripts, in the same
  layout. Scripts write data only through `new_data_dir`/`new_data_path` (src/project.jl), which stop if the target
  exists unless `--force`; figures go to `figures/<source>/` (`figure_path`). `figures/published/` is under version
  control: after changing a figure script, re-run it so that the committed figure matches the code. Figure scripts
  read through `data_dir(...)` (`--data=published|generated|FOLDER`) with one loader for published and generated
  data: keep the formats of the two identical.
- Do not change parameter values, random-number usage, algorithms or plot styles without approval.
- Everything must be reproducible from the repository alone (`--simulate`, `scripts/self_couplings.jl`, `--s`).
  The paper source and the journal's figure files are not part of the repository: no script or doc may depend on
  them, and README.md must not mention files that are not in the repository (`figures/published/` holds the figures
  rendered by the scripts from `data/published/`, not the journal's files).
- `data/published/fig1/` and `fig2/` hold the data of the published runs (one unseeded realization each). In the
  docs, describe them only as "one realization of the simulation, unknown seed" (the user's choice).

## Running
- Julia 1.11, environment at the root: `julia scripts/figN.jl` (from any directory; `julia +1.11 ...` with juliaup when
  the default Julia is another version). `src/project.jl` activates the repository environment
  (`Pkg.activate(PROJECT_ROOT)`): every script must include it before any `using`/`import`.
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
- `src/regime.jl` (Fig. S1; needs `timescales.jl`): fixed-point convergence and Lyapunov exponents of the autonomous
  network. `run_regime` was rewritten from the original analysis scripts and verified byte-identical to them;
  `run_lyapunov_g` and `write_example` were added for the figure. Keep the tangent seeds
  (regime `seed + 1_000_000` Xoshiro, lyapunov_g `seed + 2_000_000` MersenneTwister) and BLAS on one thread: the long
  irregular transients amplify rounding differences (with more BLAS threads the network of seed 20058 converged at
  390 s instead of 218 s).
- `src/switch.jl` (Fig. 4f, Fig. S2): result files, statistics and the bar panel (needs SVGFigure); the runs are
  `run_switch_sweep` in `src/wcst.jl`, seeded per network (2026 + k), paired across p, BLAS on 1 thread. Data:
  `data/published/switch/` (written by `fig4.jl --simulate` and `figS2.jl --simulate`). Fig. S2a (monkeys) reads
  `data/published/bernardi2020/` (values read off the figure of Bernardi et al., Cell 2020), always from published.
- `src/timescales.jl` (Figs. 1-2, `sample_self_couplings`, `escape_time` for the theory of Fig. 2d, which
  reproduces the published curve to 1e-10 with σ = √N σ_OU), `src/memory_task.jl` (Fig. 3), `src/wcst.jl`
  (Figs. 4-5): include after `project.jl` and `reservoir.jl`.
- `self_couplings_file()` (src/project.jl): `--s=published|generated|FILE` (default: data/published/self_couplings.txt).

## How the figures are built
- Quantitative panels are drawn from data with `SVGFigure` (Figs. 1-3, 5: axes boxes, tick positions and label
  positions measured from the published SVG).
- Artwork and single-realization panels (Fig. 3a and the timing schemes of 3b) are embedded from
  `assets/fig3_artwork.svg`, extracted from the published figure: the elements of the published SVG outside the
  redrawn panels, unchanged, with ids prefixed (`f3_`).
- Fig. 4 has a revised layout (approved by the user; reference: their Inkscape edit, reproduced to the pixel): four
  columns 41.25 mm wide with 3.5 mm gutters, a (two columns), b, c on top; d (two columns), e, f below. The reservoir of
  a and panels d, e are clean pieces of the published artwork, `assets/fig4_artwork_{network,d_left,d_right,e}.svg`
  (only the elements of the piece and the definitions they use, in the published page coordinates, ids prefixed
  `f4network_` ...), placed with `place!` (a translate). Never embed a whole artwork and show parts of it through
  clipped `<svg>`/`<use>` windows: the SVG then carries hidden copies of the old figure (visible in Inkscape).
  The task sketch of a, b, c, f and the legend of d, e are drawn by the script; f is `draw_switch!` (src/switch.jl),
  the same panel as Fig. S2.
- Figs. 1-2 work the other way round. `assets/fig{1,2}_layout.svg` is the published SVG minus its data
  marks (axes, ticks, labels, legends and schematics are kept, ids prefixed `f1_`/`f2_`). The scripts draw only
  the data, through measured mappings `mm = c0 + c1*u` (the `AX` tables, `Axes(fig, box, xmap, ymap)`).
  Fig. 1b (one network) is embedded from `fig1_panel_b_example.svg`; generated data include the rasters
  (`fig1b_*.tsv`), drawn as PNG with a similar (not identical) threshold line.
- Fig. 4b: the official version shows kernel density estimates (default of `fig4.jl`); `--gaussian` gives the
  Gaussian fits of the first published version, in the revised layout (`fig4_gaussian`).
- The default figure scripts are deterministic: check a change by comparing their SVG output before and after.
  Against the published figures (300 dpi, grey-level difference > 40) Figs. 1-2 differed in < 0.01% of the pixels and
  Fig. 3 in ~0.1% before the indices were enlarged to 5 pt (now they differ there too). Fig. 4 has the revised layout and Fig. 5 revised panels (c-d: mean ± std instead of mean/√n;
  d: plain tick labels; e: g_fb = 0.1, 0.3, 0.6, time in s, power-law fit of g_fb = 0.1 with
  `fit_acf_power_law` (src/wcst.jl, the fit of `power_law_test`: from t = 0.02 s on), dashed as in Fig. 1f): neither is compared with the published SVG any more. The paper calls the context feedback gain g_fb (figure labels too);
  the code and the data files keep g_b / gb.

## Simulations (`--simulate`)
- Figs. 1-2: `src/timescales.jl`, seeded (Fig. 1: 1234; Fig. 2: 1234 coupled, 4321 decoupled). `--smoke` runs
  40 units × 2 networks into `data/generated/smoke/`.
- Fig. 3: `src/memory_task.jl`, seeded (default 2026), BLAS forced to 1 thread: bit-reproducible. The published
  values (`data/published/fig3/fig3_values.csv`) come from a NumPy run and cannot be reproduced digit by digit.
- Figs. 4-5: `run_wcst` / `run_wcst_sweep` in `src/wcst.jl`, unseeded (as the original study).
- Fig. 4f and Fig. S2: `run_switch_sweep` in `src/wcst.jl`, seeded (network k: 2026 + k), bit-reproducible for any
  number of Julia threads (`figS2.jl --simulate --nets=2` for a quick check).
- Fig. S1: `src/regime.jl`, seeded per network (`--seed`, default 30001: network k has seed + k - 1; the example of
  panel a is the network with the median convergence time, `median_network`), bit-reproducible for any number of
  Julia threads (`figS1.jl --simulate --smoke` for a quick check). The published data were computed with Julia 1.11.9.
- To verify a change of simulation code: put the same `Random.seed!(n)` after the includes in a scratch copy of
  the old and of the new code, run both with the same BLAS thread count (e.g. `OPENBLAS_NUM_THREADS=2`) and
  compare the result files byte for byte (`fig1.jl`/`fig2.jl --simulate --smoke` are seeded; `fig3.jl
  --simulate --nets=3`, `figS1.jl --simulate --smoke` and `figS2.jl --simulate --nets=2` too; use `--tag` or `--force`
  for repeated runs). The default figure scripts are deterministic: compare their SVG output.
