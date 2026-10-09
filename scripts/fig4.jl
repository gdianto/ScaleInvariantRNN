# Figure 4: strategy of the heterogeneous network in the WCST (s-network, g_b = 0.8).
#
#   a     schematic of the reservoir and of the task (artwork)
#   b     maximum readout weight of each network vs the self-coupling of the unit (response, context),
#         with Gaussian fits (50 networks); with --kde, kernel density estimates instead
#   c     sorted self-couplings before learning, and after learning (mean ± std over 50 networks)
#   d, e  state space of one network realization (PCA projections, hysteresis)
#
# Usage:
#   julia scripts/fig4.jl                    # published WCST results (data/published/wcst)
#   julia -t 16 scripts/fig4.jl --simulate   # only the runs of Fig. 4: 50 s-network runs at g_b = 0.8 (~1 CPU-hour)
#                                            # -> data/generated/wcst (scripts/fig5.jl --simulate makes the whole sweep)
#   julia scripts/fig4.jl --data=generated   # figure from the data of an earlier run (or --data=FOLDER)
#   julia scripts/fig4.jl --kde              # panel b with kernel density estimates (figure fig4_kde)
# Other flags (--s, --tag, --force, --out): see README.md. The figure goes to figures/<data source>/fig4.
#
# Panels a, d, e (one unseeded network) are artwork (assets/fig4_artwork.svg).
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, Statistics, LinearAlgebra, Printf
import Plots, Distributions                               # colour maps; Gaussian fit
include(projectdir("src", "SVGFigure.jl")); using .SVGFigure

const GB = 0.8
hexcolor(c) = "#" * Plots.Colors.hex(Plots.Colors.RGB(c))
const PRE_COLOR = hexcolor(Plots.cgrad(:batlow, 5, categorical=true)[2])
const POST_COLOR = hexcolor(Plots.cgrad(:batlow, 5, categorical=true)[4])
const CMAP_CONTEXT = Plots.cgrad([:purple4, :mediumorchid3, :plum3, :white])
const CMAP_RESPONSE = Plots.cgrad([:darkgreen, :seagreen, :darkseagreen, :white])

function load_results(dir)
    rd(f) = readdlm(joinpath(dir, f), ' ')
    return (; gds_pre = vec(readdlm(joinpath(dir, "self_couplings.txt"), ',')),
            gds_post = rd("gds_post_wcst_s-gb_$GB.txt"),
            read_context = rd("readout_context_wcst_s-gb_$GB.txt"), read_response = rd("readout_response_wcst_s-gb_$GB.txt"))
end

# Gaussian fit of the points (X, Y) on a 100 x 100 grid spanning mean ± 3 std
function gaussian_grid(X, Y)
    μ = [mean(X), mean(Y)]; C = cov(hcat(X, Y))
    xr = range(μ[1] - 3sqrt(C[1, 1]), μ[1] + 3sqrt(C[1, 1]), length=100)
    yr = range(μ[2] - 3sqrt(C[2, 2]), μ[2] + 3sqrt(C[2, 2]), length=100)
    Z = [Distributions.pdf(Distributions.MultivariateNormal(μ, C), [x, y]) for y in yr, x in xr]   # Z[i, j] at (x_j, y_i)
    return xr, yr, Z
end

# Gaussian kernel density estimate of the points (X, Y) (bandwidths: Scott's rule) on a 100 x 100 grid
function kde_grid(X, Y)
    n = length(X)
    hx, hy = std(X) * n^(-1 / 6), std(Y) * n^(-1 / 6)
    xr = range(minimum(X) - 3hx, maximum(X) + 3hx, length=100)
    yr = range(minimum(Y) - 3hy, maximum(Y) + 3hy, length=100)
    Z = [sum(@. exp(-((x - X) / hx)^2 / 2 - ((y - Y) / hy)^2 / 2)) / (2π * hx * hy * n) for y in yr, x in xr]
    return xr, yr, Z
end

function fig4(data, out; kde=false)
    # panel b: density, number of contour levels, contour opacity, marker radius, opacity and outline
    density, n_levels, line_opacity, r, marker_opacity, outline = kde ? (kde_grid, 6, 0.4, 0.5, 0.85, "#ffffff") :
                                                                        (gaussian_grid, 15, 1.0, 0.4205, 0.25, "none")
    fig = Figure(180, 88.437599)
    embed!(fig, assetsdir("fig4_artwork.svg"))       # panels a, d, e and the bottom legend

    # ---- b: maximum readout weight per network vs self-coupling of that unit
    text!(fig, 86.70, 2.63, "b"; size=10, weight="bold")
    ax = Axes(fig; left=95.852, right=122.734, top=5.884, bottom=33.58, xlim=(0.5, 2.0), ylim=(0.1, 1.1))
    pts(R) = (m = findmax(R, dims=2); (data.gds_post[m[2]][:], m[1][:] ./ maximum(m[1])))
    xc, yc = pts(data.read_context); xr, yr = pts(data.read_response)
    gc, gr = density(xc, yc), density(xr, yr)
    zlims = extrema(vcat(vec(gc[3]), vec(gr[3])))                          # shared colour range of the subplot
    levels = range(zlims..., length=n_levels + 2)[2:end-1]                  # Plots default: 15 levels
    for ((x, y), (gx, gy, Z), cmap, c) in (((xc, yc), gc, CMAP_CONTEXT, "#800080"), ((xr, yr), gr, CMAP_RESPONSE, "#2e8b57"))
        for l in levels
            k = round(Int, 255 * (l - zlims[1]) / (zlims[2] - zlims[1]))      # GR colour index of the level
            contour!(ax, gx, gy, Z, l; color=hexcolor(cmap[k / 255]), width=0.66, opacity=line_opacity)
        end
        scatter!(ax, x, y; r, fill=c, fill_opacity=marker_opacity, stroke=outline, width=0.25)
    end
    xaxis!(ax; ticks=[0.5, 1.0, 1.5, 2.0], labels=["0.5", "1.0", "1.5", "2.0"], labeloffset=3.13, width=0.66, ticklen=0.552)
    yaxis!(ax; ticks=[0.2, 0.6, 1.0], labels=fill("", 3), width=0.66, ticklen=0.552)
    for (s, yb) in (("1.0", 9.40), ("0.6", 20.38), ("0.2", 31.37)); text!(fig, 91.12, yb, s); end
    text!(fig, 101.06, 39.58, "self coupling, s")
    text!(fig, 89.00, 25.78, "max |W_{out}|"; rotate=-90)
    legend_response, legend_context = kde ? ("#2e8b57", "#800080") : ("#4b9968", "#b65acd")
    circle!(fig, 98.665, 3.7295, 0.7155; fill=legend_response);  text!(fig, 100.02, 4.40, "response")
    circle!(fig, 112.662, 3.7295, 0.7155; fill=legend_context); text!(fig, 114.02, 4.40, "context")

    # ---- c: sorted pre-learning self-couplings, post-learning mean ± std
    text!(fig, 128.71, 2.63, "c"; size=10, weight="bold")
    lim(v1, m1, v2, m2, a) = v1 + (a - m1) * (v2 - v1) / (m2 - m1)
    ax = Axes(fig; left=139.788, right=164.897, top=5.856, bottom=33.885,
              xlim=(lim(100, 145.198, 500, 164.187, 139.788), lim(100, 145.198, 500, 164.187, 164.897)),
              ylim=(lim(0.75, 28.559, 1.75, 6.672, 33.885), lim(0.75, 28.559, 1.75, 6.672, 5.856)))
    n = length(data.gds_pre); m = vec(mean(data.gds_post, dims=1)); s = vec(std(data.gds_post, dims=1))
    plot!(ax, 1:n, data.gds_pre; color=PRE_COLOR, width=0.69, opacity=0.5)
    band!(ax, 1:n, m .- s, m .+ s; fill=POST_COLOR, fill_opacity=0.5)
    plot!(ax, 1:n, m; color=POST_COLOR, width=0.69, opacity=0.2)
    xaxis!(ax; ticks=[100, 300, 500], labels=["100", "300", "500"], labeloffset=36.95 - 33.885, width=0.69, ticklen=0.578)
    yaxis!(ax; ticks=[0.75, 1.25, 1.75], labels=fill("", 3), width=0.69, ticklen=0.578)
    for (s_, yb) in (("1.75", 7.51), ("1.25", 18.37), ("0.75", 29.24)); text!(fig, 133.41, yb, s_); end
    text!(fig, 147.29, 39.87, "node index")
    text!(fig, 131.56, 25.96, "self coupling, s"; rotate=-90)
    circle!(fig, 147.3985, 3.6385, 0.7155; fill="#205e60", fill_opacity=0.501961); text!(fig, 148.67, 4.30, "pre")
    circle!(fig, 154.668, 3.6385, 0.7155; fill=POST_COLOR);                         text!(fig, 155.92, 4.30, "post")

    return save(fig, out)
end

# ------------------------------------------------------------------ main
if hasflag("simulate")
    include(projectdir("src", "reservoir.jl")); include(projectdir("src", "wcst.jl"))
    run_wcst_sweep(; variants=["s"], g_contexts=[GB], reps=50, outdir=new_data_dir("wcst"))
end
fig4(load_results(data_dir("wcst")), figure_path(hasflag("kde") ? "fig4_kde" : "fig4"); kde=hasflag("kde"))
