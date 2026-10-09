# Figure 1: heterogeneous self-couplings generate multiple timescales and a scale-invariant decay.
#
#   a  network schematic; distribution of s and the P(s) derived from the τ(s) fit
#   b  self-couplings, activity of one network and network-averaged activity
#   c  single-trial activity of four units (s = 0.70, 0.80, 1.00, 1.23)
#   d  network-averaged activity of the same units
#   e  relaxation time τ vs s, fit τ(s) = a exp(b s^c)
#   f  population activity ⟨R(t)⟩ and power-law fit (0.07-0.9 s)
#
# Usage:
#   julia scripts/fig1.jl                     # published data (data/published/fig1)
#   julia scripts/fig1.jl --simulate          # new run, 500 networks x 500 units (seed 1234, ~2 min) -> data/generated/fig1
#   julia scripts/fig1.jl --simulate --smoke  # quick test, 40 units, 2 networks (data/generated/smoke/fig1)
#   julia scripts/fig1.jl --data=generated    # figure from the data of an earlier run (or --data=FOLDER)
# Other flags (--s, --tag, --force, --out): see README.md. The figure goes to figures/<data source>/fig1.
#
# Panel b of the published data is artwork (assets/fig1_panel_b_example.svg); generated data include the rasters.
# Axes, labels and schematic: assets/fig1_layout.svg.
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, Statistics, Printf
import Plots                                              # colour maps
include(projectdir("src", "SVGFigure.jl")); using .SVGFigure
include(projectdir("src", "reservoir.jl"))
include(projectdir("src", "timescales.jl"))

const TRACE_UNITS = [7, 100, 280, 400]
const TRACE_COLORS = ["#b9d7d9", "#95bcbf", "#73a3a6", "#406b6f"]
const TEAL, RED = "#4a7b81", "#bd1616"
const HIST_LIGHT, HIST_DARK = (185, 215, 217), (76, 129, 133)

# plot area [left, right, top, bottom] (mm) and data -> mm mappings mm = c0 + c1*u (u = value or log10 value),
# measured on the published figure
const AX = (
    a = ((35.27, 79.83, 2.59, 22.05),  (6.98085, 42.6178),     (21.47426, -9.04476), :identity, :identity),
    c = ((95.27, 131.31, 3.19, 27.31), (96.28685, 17.00116),   (15.25, -12.06),      :identity, :identity),
    d = ((142.38, 178.45, 3.47, 27.35), (142.38293, 120.22758), (3.47, -11.94),       :identity, :log10),
    e = ((96.74, 125.96, 40.60, 72.14), (77.17845, 28.97973),   (46.30114, -14.97955), :identity, :log10),
    f = ((138.39, 179.85, 39.88, 72.13), (174.75149, 13.03912), (39.88, -16.125),      :log10, :log10),
)
axes_for(fig, k) = (p = AX[k]; Axes(fig, p[1], p[2], p[3]; xscale=p[4], yscale=p[5]))
readtsv(f) = readdlm(f, '\t', Float64; header=true)[1]
hexrgb(t) = "#" * join(string.(round.(Int, t), base=16, pad=2))
inside(ax, x, y) = x > 0 && y > 0 && ax.left <= xmm(ax, x) <= ax.right && ax.top <= ymm(ax, y) <= ax.bottom

"Fig. 1 data in `dir` (layout of data/published/fig1); the rasters of panel b, if present."
function load_data(dir)
    rd(f) = readtsv(joinpath(dir, f))
    raster(f) = isfile(joinpath(dir, f)) ? readdlm(joinpath(dir, f), '\t', Float64) : nothing
    c, d = rd("fig1c_single_trial_activity.tsv"), rd("fig1d_mean_activity.tsv")
    e, f = rd("fig1e_tau_vs_self_coupling.tsv"), rd("fig1f_population_activity.tsv")
    return (; s=e[:, 1], tau=e[:, 2], traces_t=c[:, 1], traces=[c[:, k] for k in 2:size(c, 2)],
            mean_traces=[(d[d[:, 1] .== u, 2], d[d[:, 1] .== u, 3]) for u in unique(d[:, 1])],
            pop_t=f[:, 1], pop=f[:, 2],
            example=raster("fig1b_single_trial_activity.tsv"), mean_activity=raster("fig1b_mean_activity.tsv"))
end

"New run, written to data/generated[/TAG]/fig1 with the rasters of panel b (units × saved times)."
function simulate(smoke::Bool)
    dir = new_data_dir("fig1")
    c = smoke ? smoke_config(FIG1) : FIG1
    s = select_self_couplings(vec(readdlm(self_couplings_file(), ',', Float64)), c.n_units)
    println("Simulating Fig. 1: $(c.n_units) units, $(c.n_networks) networks, seed $(c.seed)")
    sim = simulate_fig1(c, s)
    t = saved_times(c)
    tau = fit_unit_timescales(sim.mean_activity, s, t, c.dt)
    n = length(s)
    units = n == 500 ? TRACE_UNITS : unique(clamp.(round.(Int, TRACE_UNITS .* n ./ 500), 1, n))
    table(f, header, cols) = writedlm(joinpath(dir, f), [permutedims(header); cols], '\t')
    writedlm(joinpath(dir, "fig1b_single_trial_activity.tsv"), sim.example, '\t')
    writedlm(joinpath(dir, "fig1b_mean_activity.tsv"), sim.mean_activity, '\t')
    table("fig1c_single_trial_activity.tsv", ["time_s"; ["r_unit$u" for u in units]], [t sim.example[units, :]'])
    table("fig1d_mean_activity.tsv", ["unit", "time_s", "mean_r"],
          reduce(vcat, [[fill(u, length(t)) t sim.mean_activity[u, :]] for u in units]))
    table("fig1e_tau_vs_self_coupling.tsv", ["self_coupling", "tau_s"], [s tau])
    table("fig1f_population_activity.tsv", ["time_s", "mean_R"], [t vec(mean(sim.mean_activity, dims=1))])
    println("Data in $dir")
end

function fig1(data, out)
    ok = isfinite.(data.tau) .& (data.tau .> 0)          # all units for the published data
    all(ok) || println("τ(s) fit: $(count(!, ok)) units with no positive finite τ excluded")
    fit = fit_tau(data.s[ok], data.tau[ok])
    p = fit.params
    @printf("τ(s) = a exp(b s^c): a = %.6g, b = %.6g, c = %.6g (R² log10 τ = %.4f)\n", p..., fit.r2_log)
    fig = Figure(180, 78.979912)
    embed!(fig, assetsdir("fig1_layout.svg"))

    # a: histogram of s (bins of 0.1, light to dark) and P(s) from the τ(s) fit
    ax = axes_for(fig, :a)
    edges = collect(0.7:0.1:1.7); edges[end] = 1.7
    counts = [count(v -> (edges[k] <= v < edges[k+1]) || (k == length(edges) - 1 && v == edges[end]), data.s) for k in 1:length(edges)-1]
    dens = counts ./ (length(data.s) * 0.1)
    for k in eachindex(dens)
        f = (k - 1) / (length(dens) - 1)
        x0, x1 = xmm(ax, edges[k]), xmm(ax, edges[k+1])
        rect!(fig, x0, ymm(ax, dens[k]), x1 - x0, ymm(ax, 0.0) - ymm(ax, dens[k]);
              fill=hexrgb((1 - f) .* HIST_LIGHT .+ f .* HIST_DARK), stroke="#ffffff", width=0.85)
    end
    sc = collect(range(0.7, 1.7; length=2001))
    plot!(ax, sc, self_coupling_density(sc, p); color=RED, width=0.99)

    # b: one network (artwork) or the new simulation
    data.example === nothing ? embed!(fig, assetsdir("fig1_panel_b_example.svg")) : draw_rasters!(fig, data)

    # c: single-trial activity
    ax = axes_for(fig, :c)
    for (r, col) in zip(data.traces, TRACE_COLORS); plot!(ax, data.traces_t, r; color=col, width=0.99); end

    # d: network-averaged activity
    ax = axes_for(fig, :d)
    for ((t, r), col) in zip(data.mean_traces, TRACE_COLORS)
        plot!(ax, t[r .> 0], r[r .> 0]; color=col, width=0.99)
    end

    # e: relaxation times and τ(s) fit (unclipped, as published)
    ax = axes_for(fig, :e)
    for (si, ti) in zip(data.s, data.tau)
        (data.example === nothing ? ti > 0 : inside(ax, si, ti)) && circle!(fig, xmm(ax, si), ymm(ax, ti), 0.3741; fill=TEAL)
    end
    plot!(ax, sc, tau_model(sc, p); color=RED, width=0.99, clip=data.example !== nothing)

    # f: population activity and power-law fit
    ax = axes_for(fig, :f)
    keep = (data.pop .> 0) .& (data.pop_t .> 0)
    plot!(ax, data.pop_t[keep], data.pop[keep]; color=TEAL, width=0.99)
    pp = fit_power_law(data.pop_t, data.pop)
    @printf("power-law fit of ⟨R(t)⟩ (0.07-0.9 s): α = %.5f, γ = %.4f\n", pp[1], -pp[2])
    t_in = (1 / pp[1])^(1 / pp[2])                       # the dashed line starts where it enters the axes
    tf = vcat(t_in, data.pop_t[keep][data.pop_t[keep] .> t_in])
    plot!(ax, tf, pp[1] .* tf .^ pp[2]; color=RED, width=0.99, dash=(7.937, 4.961), cap="round")

    return save(fig, out)
end

# rasters of panel b (RdBu, ±1, every n/100-th unit, unit 1 at the bottom) and the time at which each
# unit's mean activity drops to 0.4
function draw_rasters!(fig, data)
    cmap = Plots.cgrad(:RdBu)
    n = size(data.example, 1); units = collect(1:max(1, n ÷ 100):n)
    pixel(v) = (c = cmap[clamp((v + 1) / 2, 0, 1)]; round.(UInt8, 255 .* (Plots.Colors.red(c), Plots.Colors.green(c), Plots.Colors.blue(c))))
    boxes = ((22.00, 39.20, 28.41, 34.16), (55.09, 39.18, 28.40, 34.14))      # x, y, w, h
    for (M, (x, y, w, h)) in zip((data.example, data.mean_activity), boxes)
        image!(fig, x, y, w, h, png_bytes([pixel(M[u, j]) for u in reverse(units), j in 1:size(M, 2)]))
    end
    x, y, w, h = boxes[2]; T = data.traces_t[end]
    tcross = [something(findfirst(<=(0.4), data.mean_activity[u, :]), 1) for u in units]
    line!(fig, [x + w * data.traces_t[i] / T for i in tcross], [y + h * (1 - (k - 0.5) / length(units)) for k in eachindex(units)];
          color="#1a0e34", width=0.97)
end

hasflag("simulate") && simulate(hasflag("smoke"))
fig1(load_data(data_dir("fig1")), figure_path("fig1"))
