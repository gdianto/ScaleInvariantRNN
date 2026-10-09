# Figure 5: heterogeneous self-couplings (s-network) vs heterogeneous time constants (τ0-network) in the WCST.
#
#   a, b  self-coupling distributions before/after learning (s-network, τ0-network), g_b = 0.8
#   c     accuracy vs context feedback g_b (mean ± std over 50 networks)
#   d     synaptic-matrix variation |ΔJ|/|J0| vs g_b (mean ± std over 50 networks)
#   e     mean autocorrelation of the closed-loop network for each g_b (mean ± SEM over 50 networks)
#
# Usage:
#   julia scripts/fig5.jl                    # published WCST results (data/published/wcst)
#   julia -t 16 scripts/fig5.jl --simulate   # whole sweep, 2 networks x 8 g_b x 50 runs (~20 CPU-hours)
#                                            # -> data/generated/wcst
#   julia scripts/fig5.jl --data=generated   # figure from the data of an earlier run (or --data=FOLDER)
# Other flags (--s, --tag, --force, --out): see README.md. The figure goes to figures/<data source>/fig5.
#
# Unlike the published panels (bands of mean/√n), the bands of c-d show the standard deviation.
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, Statistics, Printf
import Plots                                             # only for colour maps and histogram binning
include(projectdir("src", "SVGFigure.jl")); using .SVGFigure

const GBACKS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
const GB_PANELS = 0.8                                     # g_b of panels a-b

# colours of the published figure
hexcolor(c) = "#" * Plots.Colors.hex(Plots.Colors.RGB(c))
const BLUES = [hexcolor(c) for c in Plots.cgrad(:Blues, 10, categorical=true, rev=true)]  # one per g_b
const S_COLOR = BLUES[3]                                                                  # s-network
const T_COLOR = hexcolor(Plots.cgrad(:Reds, 10, categorical=true, rev=true)[3])            # τ0-network
const PRE_COLOR = hexcolor(Plots.cgrad(:batlow, 5, categorical=true)[2])
const POST_COLOR = hexcolor(Plots.cgrad(:batlow, 5, categorical=true)[4])

# ------------------------------------------------------------------ data
function load_results(dir)
    rd(f, delim=' ') = readdlm(joinpath(dir, f), delim)
    r_s = rd("results_wcst_s.txt")[2:end, :]; r_t = rd("results_wcst_tau.txt")[2:end, :]
    stat(r, col) = ([mean(r[r[:, 1] .== g, col]) for g in GBACKS], [std(r[r[:, 1] .== g, col]) for g in GBACKS])
    acfm = Dict(g => rd("acfm_wcst_s-gb_$g.txt") for g in GBACKS)
    return (; acc_s = stat(r_s, 2), acc_t = stat(r_t, 2), dJ_s = stat(r_s, 3), dJ_t = stat(r_t, 3), acfm,
            acfm_tau = rd("acfm_wcst_tau-gb_0.1.txt"),
            gds_pre = vec(readdlm(joinpath(dir, "self_couplings.txt"), ',')),
            gds_post = vec(rd("gds_post_wcst_s-gb_$GB_PANELS.txt")), gds_post_tau = vec(rd("gds_post_wcst_tau-gb_$GB_PANELS.txt")))
end

# normalized histogram with the automatic (Freedman–Diaconis) bins of Plots.stephist
function pdf_hist(v)
    h = Plots._make_hist((v,), :auto; normed=true)
    return collect(h.edges[1]), h.weights
end

# axes limits from two measured (value, mm) tick positions and the measured spine extent
lim_from(v1, m1, v2, m2, a, b; log=false) = begin
    t(v) = log ? log10(v) : v
    k = (m2 - m1) / (t(v2) - t(v1)); f(m) = t(v1) + (m - m1) / k
    lims = (f(a), f(b)); log ? (10 .^ lims) : lims
end

# ------------------------------------------------------------------ figure
function fig5(data, out)
    fig = Figure(89, 120)
    letter!(x, y, s; size=10) = text!(fig, x, y, s; size=size, weight="bold")
    legend_swatch!(x0, x1, y0, y1, c) = (rect!(fig, x0, y0, x1 - x0, y1 - y0; fill=c, fill_opacity=0.35);
                                         line!(fig, [x0, x1], [y0, y0]; color=c, width=0.48, cap="butt"))

    # ---- a: s-network, self-couplings pre/post learning
    letter!(5.48, 2.53, "a")
    ax = Axes(fig; left=15.522, right=40.331, top=3.927, bottom=29.817,
              xlim=lim_from(0.5, 17.858, 2.5, 39.63, 15.522, 40.331), ylim=lim_from(0.0, 29.085, 2.0, 5.37, 29.817, 3.927))
    e, h = pdf_hist(data.gds_pre);  steps!(ax, e, h; color=PRE_COLOR, fill=PRE_COLOR, width=0.63)
    e, h = pdf_hist(data.gds_post); steps!(ax, e, h; color=POST_COLOR, fill=POST_COLOR, width=0.63)
    xaxis!(ax; ticks=[0.5, 1.5, 2.5], labels=["0.5", "1.5", "2.5"], labeloffset=3.03, width=0.63)
    yaxis!(ax; ticks=[0.0, 1.0, 2.0], labels=["0", "1.0", "2.0"], labeloffset=1.11, baselineshift=0.77, width=0.63)
    text!(fig, 9.47, 19.04, "pdf"; rotate=-90)
    text!(fig, 20.47, 35.49, "self coupling, s")
    legend_swatch!(26.19, 28.20, 4.744, 6.41, PRE_COLOR);  text!(fig, 28.71, 6.25, "pre")
    legend_swatch!(34.18, 36.19, 4.77, 6.43, POST_COLOR);  text!(fig, 36.70, 6.32, "post")

    # ---- b: τ0-network, self-couplings pre (all 0) / post learning
    letter!(48.04, 2.53, "b")
    ax = Axes(fig; left=57.299, right=82.904, top=4.257, bottom=30.043,
              xlim=lim_from(-0.6, 58.994, 0.6, 78.383, 57.299, 82.904), ylim=(0.0, 30.0))
    plot!(ax, [0.0, 0.0], [0.0, 30.0]; color=PRE_COLOR, width=0.63, dash=(4*0.63, 2.5*0.63), cap="butt")
    e, h = pdf_hist(data.gds_post_tau); steps!(ax, e, h; color=POST_COLOR, fill=POST_COLOR, width=0.63)
    xaxis!(ax; ticks=[-0.6, 0.0, 0.6], labels=["-0.6", "0.0", "0.6"], labeloffset=2.80, width=0.63)
    yaxis!(ax; ticks=[0, 10, 20, 30], labels=["0", "10", "20", "30"], labeloffset=1.13, baselineshift=0.85, width=0.63)
    text!(fig, 51.93, 18.67, "pdf"; rotate=-90)
    text!(fig, 60.64, 35.49, "self coupling, s")

    # ---- c: accuracy vs g_b
    letter!(5.61, 43.91, "c")
    ax = Axes(fig; left=15.393, right=40.444, top=46.937, bottom=73.081,
              xlim=lim_from(0.2, 19.477, 0.8, 39.736, 15.393, 40.444), ylim=(0.0, 1.0))
    for ((m, s), c) in ((data.acc_s, S_COLOR), (data.acc_t, T_COLOR))
        band!(ax, GBACKS, m .- s, m .+ s; fill=c, fill_opacity=0.5)
        plot!(ax, GBACKS, m; color=c, width=0.64)
    end
    xaxis!(ax; ticks=[0.2, 0.4, 0.6, 0.8], labels=["0.2", "0.4", "0.6", "0.8"], labeloffset=2.78, width=0.64)
    yaxis!(ax; ticks=[0.0, 0.5, 1.0], labels=["0", "0.5", "1.0"], labeloffset=1.03, baselineshift=0.79, width=0.64)
    text!(fig, 9.12, 65.45, "accuracy"; rotate=-90)
    text!(fig, 28.41, 78.08, "g_{b}")
    rect!(fig, 34.43, 67.03, 5.18, 5.18; fill="#ffffff")
    legend_swatch!(30.87, 32.89, 69.464, 71.13, S_COLOR);  text!(fig, 33.39, 70.97, "s")
    legend_swatch!(36.74, 38.76, 69.49, 71.15, T_COLOR);   text!(fig, 39.26, 71.04, "τ_{0}")

    # ---- d: synaptic-matrix variation vs g_b (log scale)
    letter!(47.93, 43.91, "d")
    ax = Axes(fig; left=60.173, right=82.406, top=45.578, bottom=72.85, xscale=:identity, yscale=:log10,
              xlim=lim_from(0.2, 63.799, 0.8, 81.777, 60.173, 82.406),
              ylim=lim_from(10^0.3, 51.428, 10^-0.6, 70.652, 72.85, 45.578; log=true))
    for ((m, s), c) in ((data.dJ_s, S_COLOR), (data.dJ_t, T_COLOR))
        any(m .- s .<= 0) && @warn "mean - std <= 0 in panel d: the band is clipped on the log axis"
        band!(ax, GBACKS, max.(m .- s, 1e-12), m .+ s; fill=c, fill_opacity=0.5)
        plot!(ax, GBACKS, m; color=c, width=0.67)
    end
    xaxis!(ax; ticks=[0.2, 0.4, 0.6, 0.8], labels=["0.2", "0.4", "0.6", "0.8"], labeloffset=2.76, width=0.67)
    yaxis!(ax; ticks=10 .^ [0.3, 0.0, -0.3, -0.6], labels=["10^{0.3}", "10^{0}", "10^{-0.3}", "10^{-0.6}"],
           labelanchor="middle", labeloffset=60.173 - 57.21, baselineshift=0.0, width=0.67)
    text!(fig, 53.29, 73.20, "synaptic matrix variation"; rotate=-90)
    text!(fig, 72.04, 77.83, "g_{b}")

    # ---- e: autocorrelation (x = saved sample index, as in the published panel)
    letter!(9.28, 85.84, "e"; size=9.56)
    ax = Axes(fig; left=35.917, right=81.364, top=89.061, bottom=113.642, xscale=:log10, yscale=:log10,
              xlim=lim_from(1.0, 37.203, 100.0, 63.178, 35.917, 81.364; log=true),
              ylim=lim_from(1.0, 89.061, 0.01, 111.216, 113.642, 89.061; log=true))
    for (k, g) in enumerate(GBACKS)
        a = data.acfm[g]; m = vec(mean(a, dims=1)); se = vec(std(a, dims=1)) ./ sqrt(size(a, 1))
        c = BLUES[k]; x = 1:length(m)
        band!(ax, x, max.(m .- se, 1e-12), m .+ se; fill=c, fill_opacity=0.5)
        plot!(ax, x, m; color=c, width=0.64)
    end
    m_tau = vec(mean(data.acfm_tau, dims=1))
    plot!(ax, 1:length(m_tau), max.(m_tau, 1e-300); color=T_COLOR, width=0.64)
    xaxis!(ax; ticks=[1.0, 100.0], labels=["10^{0}", "10^{2}"], labeloffset=3.28, width=0.64)
    yaxis!(ax; ticks=[1.0, 0.1, 0.01], labels=["10^{0}", "10^{-1}", "10^{-2}"], labelanchor="middle",
           labeloffset=35.917 - 33.26, baselineshift=0.64, width=0.64)
    text!(fig, 29.65, 103.75, "C(t)"; rotate=-90)
    text!(fig, 55.37, 119.52, "time, t [s]")
    # legend swatches: 0.9 mm lines, 2.24 pt thick, with flattened round caps (0.1 mm)
    swatch!(y, c) = rect!(fig, 13.209, y - 0.395, 1.1, 0.79; fill=c, rx=0.1, ry=0.395)
    for (k, g) in enumerate(GBACKS)
        y = 89.187 + (k - 1) * 3.0384
        swatch!(y, BLUES[k])
        text!(fig, 15.16, 89.61 + (k - 1) * 3.0875, @sprintf("g_{b}=%g", g))
    end
    swatch!(113.494, T_COLOR)
    text!(fig, 15.16, 89.61 + 8 * 3.0875, "g_{b}=0.1 (τ_{0})")

    return save(fig, out)
end

# ------------------------------------------------------------------ main
if hasflag("simulate")
    include(projectdir("src", "reservoir.jl")); include(projectdir("src", "wcst.jl"))
    run_wcst_sweep(; variants=["s", "tau"], g_contexts=GBACKS, reps=50, outdir=new_data_dir("wcst"))
end
fig5(load_results(data_dir("wcst")), figure_path("fig5"))
