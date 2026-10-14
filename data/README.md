# Data

```
data/
  published/        data of the published figures (never written by the scripts)
  generated/        data computed by the scripts, same layout (not under version control)
    TAG/            runs written with --tag=TAG (smoke tests: TAG = smoke)
```

Each figure script draws from `data/published/` by default, from `data/generated/` with `--data=generated`
(`--tag=TAG` for a tagged run), or from any folder with the same layout with `--data=FOLDER`. `--simulate` writes a
new dataset to `data/generated/` and draws the figure from it. Existing data are never replaced silently: the script
stops, unless `--force` is given (the old dataset is then deleted) or a new `--tag` is chosen.

The published data of Figs. 1-2 come from one realization of the simulation of the corresponding script (the seed of
these runs is not known); those of Fig. 3 from an earlier NumPy implementation of the same protocol (seed 2026); the
WCST results from unseeded runs of `run_wcst` (50 runs per network type and g_b).

## self_couplings.txt
The 500 self-couplings, comma-separated, sorted. `scripts/self_couplings.jl` writes a new vector to
`data/generated/self_couplings.txt`; simulations use it with `--s=generated`.

## fig1/ (`scripts/fig1.jl`)

| file | panel | content |
|---|---|---|
| `fig1b_single_trial_activity.tsv`, `fig1b_mean_activity.tsv` | b | generated data only: activity of every unit (rows) at every saved time (columns, t = 0 … 2 s every 2 ms), in one network and averaged over the networks |
| `fig1c_single_trial_activity.tsv` | c | `time_s`, `r_unit<k>`: activity of four units in one network |
| `fig1d_mean_activity.tsv` | d | `unit`, `time_s`, `mean_r`: activity of the same units averaged over the networks |
| `fig1e_tau_vs_self_coupling.tsv` | a, e | `self_coupling`, `tau_s`: relaxation time of each unit |
| `fig1f_population_activity.tsv` | f | `time_s`, `mean_R`: population activity ⟨R(t)⟩ |

Units of panels c-d: 7, 100, 280 and 400 (s = 0.70, 0.80, 1.00, 1.23). In the published data each trace of
`fig1d_mean_activity.tsv` stops where it leaves the axes of the panel, and panel b is embedded as artwork
(`assets/fig1_panel_b_example.svg`). Fits of the published data: τ(s) = a exp(b s^c) with a = 0.016462,
b = 1.01679, c = 2.99117; ⟨R(t)⟩ ∝ t^−0.9287 on 0.07–0.9 s.

## fig2/ (`scripts/fig2.jl`)

| file | panel | content |
|---|---|---|
| `fig2b_mean_activity_coupled.tsv`, `fig2b_mean_activity_decoupled.tsv` | b | `time_s`, `mean_R`: population activity ⟨R(t)⟩ (mean over the networks) |
| `fig2b_band_coupled.tsv`, `fig2b_band_decoupled.tsv` | b | `time_s`, `R`: outline of the ±3 SEM band (upper edge, then lower edge backwards) |
| `fig2b_mean_sem.tsv` | b | generated data only: `time_s` and mean and SEM of ⟨R(t)⟩ of both networks at every saved time |
| `fig2c_tau_vs_self_coupling.tsv` | c | `self_coupling`, `tau_coupled_s`, `tau_decoupled_s`: relaxation time of each unit in the two networks |
| `fig2d_tau_coupled_s_gt_1.txt` | d | s and τ (s) of the units with s ≥ 1 in the coupled network (no header) |
| `recurrent_input.txt` | – | generated data only: statistics of the recurrent input of the coupled network |

The theoretical curve of panel d is computed by `escape_time` in `src/timescales.jl` (σ = √N σ_OU). In the published
data the τ of panel d come from a different realization of the coupled network than those of panel c.

## fig3/ (`scripts/fig3.jl`)

`fig3_values.csv`: `network` (hRNN, hom, esn), `Tsep_s`, `d` (signed position distance, −3 … 3), `rho_mean`,
`rho_sem` (mean and SEM of the correlation over the networks).

## wcst/ (`scripts/fig4.jl`, `scripts/fig5.jl`)

One line per run; `<type>` is `s` (s-network) or `tau` (τ0-network), `<g_b>` = 0.1 … 0.8.

| file | content |
|---|---|
| `results_wcst_<type>.txt` | header, then g_b, accuracy, \|ΔJ\|/\|J0\| |
| `gds_post_wcst_<type>-gb_<g_b>.txt` | self-couplings after learning (500 values) |
| `readout_context_wcst_<type>-gb_<g_b>.txt`, `readout_response_wcst_<type>-gb_<g_b>.txt` | \|W_out\| of the two readouts (500 values) |
| `acfm_wcst_<type>-gb_<g_b>.txt` | mean autocorrelation of the closed-loop network (one value every 5 ms) |
| `self_couplings.txt` | the self-couplings of the s-network (before learning) |

`scripts/fig4.jl --simulate` computes only the runs of Fig. 4 (s-network, g_b = 0.8); `scripts/fig5.jl --simulate`
computes the whole sweep.

## fig4_switch/ (`scripts/fig4_switch.jl`, draft panel; generated data only)

One line per network (s-network, g_b = 0.8; during the test each error pulse halved with probability 0.5,
`run_wcst_switch`), one column per test trial (100).

| file | content |
|---|---|
| `hits_wcst_s-gb_0.8-phalf_0.5.txt` | 1 if the response is correct (no error signal) |
| `rules_wcst_s-gb_0.8.txt` | rule (±1) of each test trial |
| `self_couplings.txt` | the self-couplings used |
