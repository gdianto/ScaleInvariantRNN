# Figure 2: an OU-driven decoupled network reproduces the coupled one; escape-time theory.
#   julia scripts/fig2.jl              # from data/published/fig2
#   julia scripts/fig2.jl --simulate   # new run, seeds 1234 and 4321 (~40 min; --smoke: quick test)
# Other flags (--data, --s, --tag, --force, --out): see README.md.
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, Statistics, Printf
include(projectdir("src", "SVGFigure.jl")); using .SVGFigure
include(projectdir("src", "reservoir.jl"))
include(projectdir("src", "timescales.jl"))

const COUPLED, DECOUPLED, DARK, RED = "#4a7b81", "#f9a07b", "#256167", "#bb1616"

# axes boxes (mm) and mappings mm = c0 + c1*u (u = value or log10 value), measured on the published figure
const AX = (
    b = ((55.66, 84.83, 3.95, 28.59), (82.21, 6.43), (6.80, -9.47), :log10, :log10),
    c = ((11.81, 42.38, 40.77, 66.48), (-7.64691, 28.98218), (45.52, -12.07), :identity, :log10),
    d = ((55.56, 84.09, 41.49, 66.30), (17.91291, 38.45194), (47.25561, -13.05422), :identity, :log10),
)
axes_for(fig, k) = (p = AX[k]; Axes(fig, p[1], p[2], p[3]; xscale=p[4], yscale=p[5]))
readtsv(f) = readdlm(f, '\t', Float64; header=true)[1]
inside(ax, x, y) = x > 0 && y > 0 && ax.left <= xmm(ax, x) <= ax.right && ax.top <= ymm(ax, y) <= ax.bottom

"Fig. 2 data in `dir` (layout of data/published/fig2)."
function load_data(dir)
    rd(f) = readtsv(joinpath(dir, f))
    curve(f) = (m = rd(f); (m[:, 1], m[:, 2]))
    c = rd("fig2c_tau_vs_self_coupling.tsv")
    d = readdlm(joinpath(dir, "fig2d_tau_coupled_s_gt_1.txt"), Float64)
    return (; b=Dict(n => (curve("fig2b_mean_activity_$n.tsv"), curve("fig2b_band_$n.tsv")) for n in ("coupled", "decoupled")),
            s=c[:, 1], tau_coupled=c[:, 2], tau_decoupled=c[:, 3], d_s=d[:, 1], d_tau=d[:, 2])
end

"New run, written to data/generated[/TAG]/fig2."
function simulate(smoke::Bool)
    dir = new_data_dir("fig2")
    c = smoke ? smoke_config(FIG2) : FIG2
    s = select_self_couplings(vec(readdlm(self_couplings_file(), ',', Float64)), c.n_units)
    t = saved_times(c)
    println("Simulating Fig. 2: $(c.n_units) units, $(c.n_networks) networks per case")
    co = simulate_coupled(c, s)
    de = simulate_decoupled(c, s)
    tau_c = fit_unit_timescales(co.mean_activity, s, t, c.dt)
    tau_d = fit_unit_timescales(de.mean_activity, s, t, c.dt)
    table(f, header, cols) = writedlm(joinpath(dir, f), [permutedims(header); cols], '\t')
    i = 2:min(19_000, length(t))
    for (name, r) in (("coupled", co), ("decoupled", de))
        m, sem = r.population_mean[i], r.population_sem[i]
        lower = min.(3 .* sem, m .- 1e-12)
        table("fig2b_mean_activity_$name.tsv", ["time_s", "mean_R"], [t[i] m])
        table("fig2b_band_$name.tsv", ["time_s", "R"], [vcat(t[i], reverse(t[i])) vcat(m .+ 3 .* sem, reverse(m .- lower))])
    end
    table("fig2b_mean_sem.tsv", ["time_s", "coupled_mean", "coupled_sem", "decoupled_mean", "decoupled_sem"],
          [t co.population_mean co.population_sem de.population_mean de.population_sem])
    table("fig2c_tau_vs_self_coupling.tsv", ["self_coupling", "tau_coupled_s", "tau_decoupled_s"], [s tau_c tau_d])
    writedlm(joinpath(dir, "fig2d_tau_coupled_s_gt_1.txt"), [s tau_c][s .>= 1.0, :], '\t')
    open(joinpath(dir, "recurrent_input.txt"), "w") do io
        for (k, v) in pairs(co.currents); println(io, "$k = $v"); end
    end
    println("Data in $dir; recurrent input: ", co.currents)
end

function fig2(data, out)
    fig = Figure(89, 72.570633)
    embed!(fig, assetsdir("fig2_layout.svg"))

    # b: population activity with ±3 SEM bands
    ax = axes_for(fig, :b)
    for (name, col) in (("decoupled", DECOUPLED), ("coupled", COUPLED))
        (tm, m), (tb, rb) = data.b[name]
        keep = (tb .> 0) .& (rb .> 0)
        polygon!(fig, [xmm(ax, v) for v in tb[keep]], [ymm(ax, v) for v in rb[keep]]; fill=col, fill_opacity=0.4, clip=ax.clip)
        plot!(ax, tm[m .> 0], m[m .> 0]; color=col, width=0.99)
    end

    # c: τ(s), coupled and decoupled
    ax = axes_for(fig, :c)
    for (tau, col) in ((data.tau_coupled, COUPLED), (data.tau_decoupled, DECOUPLED)), (si, ti) in zip(data.s, tau)
        inside(ax, si, ti) && circle!(fig, xmm(ax, si), ymm(ax, ti), 0.3731; fill=col)
    end

    # d: coupled network for s >= 1 and theoretical escape time
    ax = axes_for(fig, :d)
    for (si, ti) in zip(data.d_s, data.d_tau)
        inside(ax, si, ti) && circle!(fig, xmm(ax, si), ymm(ax, ti), 0.3485; fill=DARK)
    end
    s_th = 1.0:0.01:1.7
    plot!(ax, s_th, escape_time.(s_th; σ=sqrt(FIG2.n_units) * FIG2.σ_ou, τ0=FIG2.τ); color=RED, width=0.99)

    return save(fig, out)
end

hasflag("simulate") && simulate(hasflag("smoke"))
fig2(load_data(data_dir("fig2")), figure_path("fig2"))
