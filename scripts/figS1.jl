# Supplementary Figure S1: the autonomous network of the paper relaxes to a fixed point (no input, no noise, r = 1 at t = 0).
#
#   a  activity r_i(t) of one network (g = 0.4) up to its fixed point (neurons in order of increasing s); log time
#   b  fixed-point residual |tanh(Jr) - r|/√N of the networks that reach a fixed point (the network of a highlighted)
#   c  two largest Lyapunov exponents λ1 (top) and λ2 (bottom) of 100 networks for g = 0.38-0.42 (one point per network)
#
# Usage:
#   julia scripts/figS1.jl                              # published data (data/published/figS1)
#   julia -t 16 scripts/figS1.jl --simulate             # new run (~1 h on 16 threads) -> data/generated/figS1
#   julia -t 8 scripts/figS1.jl --simulate --smoke      # quick test: short runs, few networks (data/generated/smoke/figS1)
#   julia scripts/figS1.jl --data=generated             # figure from the data of an earlier run (or --data=FOLDER)
# Other flags (--s, --tag, --force, --out): see README.md. The figure goes to figures/<data source>/figS1.
#
# The simulation is seeded (network k: seed 20001 + k - 1): it is bit-reproducible whatever the number of threads.
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, LinearAlgebra, Random, Statistics, Printf
import Plots                                              # colour map
include(projectdir("src", "SVGFigure.jl")); using .SVGFigure
include(projectdir("src", "reservoir.jl"))
include(projectdir("src", "timescales.jl"))
include(projectdir("src", "regime.jl"))

const TEAL, DARK = "#4a7b81", "#1a0e34"

# ------------------------------------------------------------------ simulation (--simulate)
function simulate(smoke::Bool)
    dir = new_data_dir("figS1")
    s = vec(readdlm(self_couplings_file(), ',', Float64))
    c = smoke ? smoke_regime(REGIME) : REGIME
    run_regime(joinpath(dir, "regime"), s; c)
    write_example(joinpath(dir, "regime"), s; e=smoke ? smoke_example(EXAMPLE) : EXAMPLE, c)
    run_lyapunov_g(joinpath(dir, "lyapunov"), s; c=smoke ? smoke_lyapunov_g(LYAPUNOV_G) : LYAPUNOV_G)
    println("Data in $dir")
end

# ------------------------------------------------------------------ data
readtsv(f; T=Any) = readdlm(f, '\t', T; header=true)[1]

"Fig. S1 data in `dir` (layout of data/published/figS1)."
function load_data(dir)
    rd(f...; T=Any) = readtsv(joinpath(dir, f...); T)
    net, res = rd("regime", "network_results.tsv"), rd("regime", "residual_history.tsv"; T=Float64)
    states = rd("regime", "final_states.tsv")
    t = vec(rd("regime", "example_times.tsv"; T=Float64))
    converged = [x == true || x == "true" for x in net[:, 3]]
    confirmation = Float64.(net[:, 5])
    activity = readdlm(joinpath(dir, "regime", "example_activity.tsv"), '\t', Float64)
    # residual of each network that reaches a fixed point, up to the confirmation of the fixed point
    residuals = [(n = Int(net[k, 1]); m = (res[:, 1] .== n) .& (res[:, 3] .<= confirmation[k]); (res[m, 3], res[m, 4]))
                 for k in axes(net, 1) if converged[k]]
    # network of panel a: the one whose fixed point is the last state of the example
    fixed_point(k) = Float64.(states[states[:, 1] .== net[k, 1], 7])
    example = findfirst(k -> converged[k] && maximum(abs.(fixed_point(k) .- activity[:, end])) < 1e-4, axes(net, 1))
    ly = rd("lyapunov", "lyapunov_vs_g.tsv")
    return (; t, activity, residuals,
            example=example === nothing ? 0 : count(converged[1:example]),
            example_time=example === nothing ? NaN : Float64(net[example, 4]),
            g=Float64.(ly[:, 1]), λ1=Float64.(ly[:, 6]), λ2=Float64.(ly[:, 9]))
end

# ------------------------------------------------------------------ figure
"Deterministic horizontal jitter of the `k`-th point: `n` columns of half-width `w`."
jitter(k, w; n=9) = w * (2 * ((k - 1) % n) / (n - 1) - 1)

"Limits (lower: multiple of `step` below the data; upper: `top`) and ticks."
function limits(v; step=5, top=1.0)
    lo = step * floor(min(minimum(v), 0) / step)
    return (lo, top), collect(lo:step:0)
end

function figS1(data, out)
    fig = Figure(180, 45)
    letter!(x, y, s) = text!(fig, x, y, s; size=10, weight="bold")
    pow10(e) = "10^{$e}"
    ticklabel(v) = v == round(v) ? string(round(Int, v)) : string(v)

    # ---- a: activity of one network up to its fixed point (RdBu, ±1 as Fig. 1b); neurons in the order of s (increasing)
    letter!(1.5, 3.5, "a")
    N, t = size(data.activity, 1), data.t
    ax = Axes(fig; left=13.0, right=56.0, top=5.0, bottom=36.0, xscale=:log10, xlim=(t[1], t[end]), ylim=(0.5, N + 0.5))
    cmap = Plots.cgrad(:RdBu)
    pixel(v) = (c = cmap[clamp((v + 1) / 2, 0, 1)]; round.(UInt8, 255 .* (Plots.Colors.red(c), Plots.Colors.green(c), Plots.Colors.blue(c))))
    columns = [searchsortedlast(t, x) for x in exp10.(range(log10(t[1]), log10(t[end]), length=300))]   # uniform in log t
    image!(fig, ax.left, ax.top, ax.right - ax.left, ax.bottom - ax.top,
           png_bytes([pixel(data.activity[u, j]) for u in N:-1:1, j in columns]))
    if isfinite(data.example_time)                                           # time of the fixed point
        plot!(ax, fill(data.example_time, 2), [0.5, N + 0.5]; color="#000000", width=0.5, dash=(1.5, 1.0), cap="butt")
        text!(fig, xmm(ax, data.example_time) + 1.0, ax.top - 1.0, "fixed point"; anchor="end")
    end
    xaxis!(ax; ticks=[0.01, 1.0, 100.0], labels=pow10.([-2, 0, 2]))
    yticks = [k for k in (100, 300, 500) if k <= N]
    yaxis!(ax; ticks=yticks, labels=string.(yticks), baselineshift=0.75)
    text!(fig, 34.5, 42.0, "time, t [s]"; anchor="middle")
    text!(fig, 5.0, 20.5, "neuron index"; anchor="middle", rotate=-90)
    cb = (x=58.0, w=1.5, top=8.0, bottom=33.0)                                # colour bar
    image!(fig, cb.x, cb.top, cb.w, cb.bottom - cb.top, png_bytes([pixel(v) for v in range(1, -1, length=100), _ in 1:1]))
    for (v, y) in ((1, cb.top), (0, (cb.top + cb.bottom) / 2), (-1, cb.bottom))
        line!(fig, [cb.x + cb.w, cb.x + cb.w + 0.5], [y, y]; width=0.63, cap="butt")
        text!(fig, cb.x + cb.w + 1.0, y + 0.85, ticklabel(v))
    end
    text!(fig, cb.x + cb.w / 2, cb.top - 1.5, "r"; anchor="middle")

    # ---- b: fixed-point residual of the networks that reach a fixed point
    letter!(68.0, 3.5, "b")
    ax = Axes(fig; left=81.0, right=122.0, top=5.0, bottom=36.0, xscale=:log10, yscale=:log10, xlim=(1.0, 3e3), ylim=(1e-17, 10.0))
    for (k, (x, ρ)) in enumerate(data.residuals)
        k == data.example && continue
        keep = x .>= 1.0
        plot!(ax, x[keep], max.(ρ[keep], 1e-17); color=TEAL, width=0.3, opacity=0.4)
    end
    if data.example > 0
        x, ρ = data.residuals[data.example]
        plot!(ax, x[x .>= 1.0], max.(ρ[x .>= 1.0], 1e-17); color=DARK, width=0.7)
    end
    plot!(ax, [1.0, 3e3], [1e-7, 1e-7]; color="#000000", width=0.4, dash=(1.5, 1.0), cap="butt")
    text!(fig, ax.left + 1.0, ymm(ax, 1e-7) - 1.0, "tolerance")
    xaxis!(ax; ticks=10.0 .^ (0:3), labels=pow10.(0:3))
    yaxis!(ax; ticks=[1.0, 1e-8, 1e-16], labels=pow10.([0, -8, -16]), baselineshift=0.75)
    text!(fig, 101.5, 42.0, "time, t [s]"; anchor="middle")
    text!(fig, 70.5, 20.5, "‖tanh(Jr) − r‖ / √N"; anchor="middle", rotate=-90)

    # ---- c: two largest Lyapunov exponents vs g, shared x axis
    letter!(127.5, 3.5, "c")
    gs = sort(unique(data.g))
    xlim = (gs[1] - 0.006, gs[end] + 0.006)
    for (row, (λ, top, bottom, label)) in enumerate(((data.λ1, 5.0, 19.0, "λ_{1} [s^{-1}]"), (data.λ2, 22.0, 36.0, "λ_{2} [s^{-1}]")))
        ylim, yticks = limits(λ)
        ax = Axes(fig; left=142.0, right=177.0, top, bottom, xlim, ylim)
        plot!(ax, collect(xlim), [0.0, 0.0]; color="#000000", width=0.4, dash=(1.5, 1.0), cap="butt")
        row == 1 && text!(fig, ax.left + 1.0, ymm(ax, 0.0) - 1.7, "λ_{1} > 0: chaos")
        for g in gs
            v = λ[data.g .== g]
            x = [g + jitter(k, 0.0035; n=10) for k in eachindex(v)]
            scatter!(ax, x, v; r=0.3, fill=TEAL, fill_opacity=0.6)
            for k in findall(>(ylim[2]), v)                                  # above the axis: ▲ at the edge and the value
                X = xmm(ax, x[k])
                polygon!(fig, [X - 0.6, X + 0.6, X], [top - 0.1, top - 0.1, top - 1.1]; fill=TEAL)
                text!(fig, X - 0.9, top - 0.1, @sprintf("%.1f", v[k]); anchor="end")
            end
        end
        xaxis!(ax; ticks=gs, labels=row == 2 ? [@sprintf("%.2f", g) for g in gs] : fill("", length(gs)))
        yaxis!(ax; ticks=yticks, labels=ticklabel.(yticks), baselineshift=0.75)
        text!(fig, 133.0, (top + bottom) / 2, label; anchor="middle", rotate=-90)
    end
    text!(fig, 159.5, 42.0, "recurrent gain, g"; anchor="middle")

    return save(fig, out)
end

# ------------------------------------------------------------------ main
hasflag("simulate") && simulate(hasflag("smoke"))
figS1(load_data(data_dir("figS1")), figure_path("figS1"))
