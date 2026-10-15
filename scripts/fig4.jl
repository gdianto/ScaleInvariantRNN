# Figure 4: strategy of the heterogeneous network in the WCST (s-network, g_b = 0.8).
#
#   a     the task (central card and the correct choice in each context) and the reservoir with its inputs and readouts
#   b     maximum readout weight of each network vs the self-coupling of the unit (response, context), with kernel
#         density estimates (50 networks); with --gaussian, Gaussian fits instead
#   c     sorted self-couplings before learning, and after learning (mean ± std over 50 networks)
#   d, e  state space of one network realization (PCA projections, hysteresis)
#   f     accuracy on the last trial before a rule switch and on the first four after it when, during the test, each
#         error signal is halved with probability 0.5 (mean ± SEM over 200 networks; other probabilities in Fig. S2)
#
# Usage:
#   julia scripts/fig4.jl                    # published data (data/published/wcst, data/published/switch)
#   julia -t 16 scripts/fig4.jl --simulate   # only the runs of Fig. 4: 50 s-network runs at g_b = 0.8 (~1 CPU-hour) and the
#                                            # switch runs of panel f and Fig. S2 (~5 min) -> data/generated/{wcst,switch}
#                                            # (scripts/fig5.jl --simulate makes the whole WCST sweep)
#   julia scripts/fig4.jl --data=generated   # figure from the data of an earlier run (or --data=FOLDER)
#   julia scripts/fig4.jl --gaussian         # panel b with Gaussian fits (figure fig4_gaussian)
# Other flags (--s, --tag, --force, --out): see README.md. The figure goes to figures/<data source>/fig4.
#
# Layout: four columns 41.25 mm wide with 3.5 mm gutters; a (two columns), b, c on top; d (two columns), e, f below.
# The reservoir of panel a and panels d, e (one unseeded network) are artwork: pieces of the published Fig. 4 in its
# page coordinates (assets/fig4_artwork_*.svg), placed here with a shift. The task sketch of panel a is drawn here.
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, Statistics, LinearAlgebra, Printf
import Plots, Distributions                               # colour maps; Gaussian fit
include(projectdir("src", "SVGFigure.jl")); using .SVGFigure
include(projectdir("src", "switch.jl"))

const GB = 0.8
hexcolor(c) = "#" * Plots.Colors.hex(Plots.Colors.RGB(c))
const PRE_COLOR = hexcolor(Plots.cgrad(:batlow, 5, categorical=true)[2])
const POST_COLOR = hexcolor(Plots.cgrad(:batlow, 5, categorical=true)[4])
const CMAP_CONTEXT = Plots.cgrad([:purple4, :mediumorchid3, :plum3, :white])
const CMAP_RESPONSE = Plots.cgrad([:darkgreen, :seagreen, :darkseagreen, :white])
const CARD_YELLOW, CARD_TEAL = "#fac938", "#61b5bd"      # card symbols and frames, as the card inputs of the schematic
const FRAME_YELLOW, FRAME_BLUE = "#ffbd39", "#5b91b2"
const RULE_COLOR, RULE_SHAPE = "#3399ff", "#ff4c4c"      # context: color, shape (as in d, e)
const ERROR_RED = "#d95b4f"

# grid (mm): columns, axes of b, c, f inside a column, rows
const COL = [2.5, 47.25, 92.0, 136.75]
const AXL, AXW, AXH = 10.4, 29.35, 27.7                   # left offset, width and height of the axes
const ROW2 = 43.57
# artwork pieces and their shift from the published figure (mm)
const PIECES = [("network", 5.3167, 0.5292), ("d_left", -39.883332, 0.0), ("d_right", -39.295832, 0.0), ("e", -36.3, 0.0)]
const TASK = (x=6.9307, y=4.1292, w=20.566, h=31.3, rx=1.028, cx=17.1457)   # task sketch: box and centre of its content
const INPUT_X, ERROR_Y, ERROR_NODE_X = 28.6167, 21.2622, 39.7597  # start of the input lines, error input: height, node centre

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

# ------------------------------------------------------------------ drawing helpers
letter!(fig, x, y, s) = text!(fig, x, y, s; size=10, weight="bold")
"Legend items (kind :dot or :ring, colour, label[, opacity]) at the x positions `xs`, baseline y."
function legend!(fig, xs, y, items)
    for (x, it) in zip(xs, items)
        kind, c, label = it[1], it[2], it[3]
        kind == :dot ? circle!(fig, x, y - 0.67, 0.7155; fill=c, fill_opacity=length(it) > 3 ? it[4] : 1.0) :
                       circle!(fig, x, y - 0.67, 0.6; fill="#ffffff", stroke=c, width=0.8)
        text!(fig, x + 1.36, y, label)
    end
end
"Axes of b, c or f in the column starting at X, top at y; x and y labels (y label at X + ylabelx)."
function cell_axes(fig, X, top, xlabel, ylabel; xlim, ylim, w=AXW, ylabelx=2.4)
    ax = Axes(fig; left=X + AXL, right=X + AXL + w, top, bottom=top + AXH, xlim, ylim)
    isempty(xlabel) || text!(fig, (ax.left + ax.right) / 2, ax.bottom + 6.0, xlabel; anchor="middle")
    text!(fig, X + ylabelx, (ax.top + ax.bottom) / 2, ylabel; anchor="middle", rotate=-90)
    return ax
end
const TICKS = (width=0.66, ticklen=0.552)

"Card centred at (x, y): white card, black face with a coloured frame, a circle or a triangle."
function card!(fig, x, y, shape, color, frame)
    rect!(fig, x - 2.3, y - 1.75, 4.6, 3.5; fill="#ffffff", stroke="#000000", width=0.5, rx=0.35)
    rect!(fig, x - 1.25, y - 1.25, 2.5, 2.5; fill="#000000", stroke=frame, width=1.1)
    shape == :circle ? circle!(fig, x, y, 0.7; fill=color) :
                       polygon!(fig, [x - 0.82, x + 0.82, x], [y + 0.68, y + 0.68, y - 0.78]; fill=color)
end

"Curved arrow from (x0, y0) leaving downwards to (x1, y1), entering from above, with a filled head."
function branch!(fig, x0, y0, x1, y1; color, width=0.75, head=1.0, half=0.5)
    yb = y1 - head; m = (y0 + yb) / 2
    raw!(fig, "<path d=\"M $x0 $y0 C $x0 $m $x1 $(m - 1.0) $x1 $yb\" fill=\"none\" stroke=\"$color\" " *
              "stroke-width=\"$(width * PT)\" stroke-linecap=\"butt\"/>")
    polygon!(fig, [x1, x1 - half, x1 + half], [y1, yb, yb]; fill=color)
end

"Polyline through `pts` with its corners rounded (radius r)."
function rounded_line!(fig, pts; r, color, width)
    d = IOBuffer(); print(d, "M ", pts[1][1], " ", pts[1][2])
    for k in 2:length(pts)-1
        (x0, y0), (x1, y1), (x2, y2) = pts[k-1], pts[k], pts[k+1]
        l1, l2 = hypot(x1 - x0, y1 - y0), hypot(x2 - x1, y2 - y1); rk = min(r, l1 / 2, l2 / 2)
        print(d, " L ", x1 - rk * (x1 - x0) / l1, " ", y1 - rk * (y1 - y0) / l1,
                 " Q ", x1, " ", y1, " ", x1 + rk * (x2 - x1) / l2, " ", y1 + rk * (y2 - y1) / l2)
    end
    print(d, " L ", pts[end][1], " ", pts[end][2])
    raw!(fig, "<path d=\"$(String(take!(d)))\" fill=\"none\" stroke=\"$color\" stroke-width=\"$(width * PT)\" " *
              "stroke-linecap=\"butt\" stroke-linejoin=\"round\"/>")
end

# ------------------------------------------------------------------ panels
"a: task sketch (central card, the correct choice in each context) and the reservoir; the inputs leave the task box."
function panel_a!(fig)
    xp, y = INPUT_X + 5.6, ERROR_Y                        # error input with a pulse, as the card inputs; under its node
    rounded_line!(fig, [(INPUT_X, y), (xp - 0.75, y), (xp - 0.5, y - 0.9), (xp + 0.5, y - 0.9), (xp + 0.75, y),
                        (ERROR_NODE_X, y)]; r=0.3, color=ERROR_RED, width=0.95)
    for (name, dx, dy) in PIECES
        place!(fig, assetsdir("fig4_artwork_$name.svg"), dx, dy)
    end
    letter!(fig, COL[1], 2.63, "a")
    (; x, y, w, h, rx, cx) = TASK
    xl, xr, yc0, yc = cx - 6.4, cx + 6.4, y + 6.9, y + h - 9.0          # central card, oriented cards
    rect!(fig, x, y, w, h; fill="#f0f0f0", rx)
    text!(fig, cx, y + 4.1, "central card"; anchor="middle")
    card!(fig, cx, yc0, :triangle, CARD_YELLOW, FRAME_YELLOW)
    branch!(fig, cx - 0.6, yc0 + 1.75, xl, yc - 1.95; color=RULE_COLOR)
    branch!(fig, cx + 0.6, yc0 + 1.75, xr, yc - 1.95; color=RULE_SHAPE)
    card!(fig, xl, yc, :circle, CARD_YELLOW, FRAME_YELLOW); card!(fig, xr, yc, :triangle, CARD_TEAL, FRAME_BLUE)
    text!(fig, xl, y + h - 4.5, "color"; anchor="middle", color=RULE_COLOR)
    text!(fig, xr, y + h - 4.5, "shape"; anchor="middle", color=RULE_SHAPE)
    text!(fig, cx, y + h - 1.4, "correct choice"; anchor="middle", color="#555555")
    text!(fig, INPUT_X + 1.0, ERROR_Y + 3.817, "error signal")
end

"b: maximum readout weight per network vs the self-coupling of that unit."
function panel_b!(fig, data, X; kde)
    # density, number of contour levels, contour opacity, marker radius, opacity and outline
    density, n_levels, line_opacity, r, marker_opacity, outline = kde ? (kde_grid, 6, 0.4, 0.5, 0.85, "#ffffff") :
                                                                        (gaussian_grid, 15, 1.0, 0.4205, 0.25, "none")
    letter!(fig, X, 2.63, "b")
    ax = cell_axes(fig, X, 5.9, "self coupling, s", "max |W_{out}|"; xlim=(0.5, 2.0), ylim=(0.1, 1.1))
    pts(R) = (m = findmax(R, dims=2); (data.gds_post[m[2]][:], m[1][:] ./ maximum(m[1])))
    xc, yc = pts(data.read_context); xr, yr = pts(data.read_response)
    gc, gr = density(xc, yc), density(xr, yr)
    zlims = extrema(vcat(vec(gc[3]), vec(gr[3])))                          # shared colour range of the subplot
    levels = range(zlims..., length=n_levels + 2)[2:end-1]
    for ((x, y), (gx, gy, Z), cmap, c) in (((xc, yc), gc, CMAP_CONTEXT, "#800080"), ((xr, yr), gr, CMAP_RESPONSE, "#2e8b57"))
        for l in levels
            k = round(Int, 255 * (l - zlims[1]) / (zlims[2] - zlims[1]))      # GR colour index of the level
            contour!(ax, gx, gy, Z, l; color=hexcolor(cmap[k / 255]), width=0.66, opacity=line_opacity)
        end
        scatter!(ax, x, y; r, fill=c, fill_opacity=marker_opacity, stroke=outline, width=0.25)
    end
    xaxis!(ax; ticks=[0.5, 1.0, 1.5, 2.0], labels=["0.5", "1.0", "1.5", "2.0"], labeloffset=3.13, TICKS...)
    yaxis!(ax; ticks=[0.2, 0.6, 1.0], labels=["0.2", "0.6", "1.0"], labeloffset=1.1, baselineshift=0.8, TICKS...)
    xm = (ax.left + ax.right) / 2
    legend!(fig, [xm - 12.6, xm + 1.4], 4.4, kde ? [(:dot, "#2e8b57", "response"), (:dot, "#800080", "context")] :
                                                   [(:dot, "#4b9968", "response"), (:dot, "#b65acd", "context")])
end

"c: sorted pre-learning self-couplings, post-learning mean ± std."
function panel_c!(fig, data, X)
    letter!(fig, X, 2.63, "c")
    lim(v1, m1, v2, m2, a) = v1 + (a - m1) * (v2 - v1) / (m2 - m1)             # axis limits of the published panel
    ax = cell_axes(fig, X, 5.9, "node index", "self coupling, s";
                   xlim=(lim(100, 145.198, 500, 164.187, 139.788), lim(100, 145.198, 500, 164.187, 164.897)),
                   ylim=(lim(0.75, 28.559, 1.75, 6.672, 33.885), lim(0.75, 28.559, 1.75, 6.672, 5.856)))
    n = length(data.gds_pre); m = vec(mean(data.gds_post, dims=1)); s = vec(std(data.gds_post, dims=1))
    plot!(ax, 1:n, data.gds_pre; color=PRE_COLOR, width=0.69, opacity=0.5)
    band!(ax, 1:n, m .- s, m .+ s; fill=POST_COLOR, fill_opacity=0.5)
    plot!(ax, 1:n, m; color=POST_COLOR, width=0.69, opacity=0.2)
    xaxis!(ax; ticks=[100, 300, 500], labels=["100", "300", "500"], labeloffset=3.13, TICKS...)
    yaxis!(ax; ticks=[0.75, 1.25, 1.75], labels=["0.75", "1.25", "1.75"], labeloffset=1.1, baselineshift=0.8, TICKS...)
    xm = (ax.left + ax.right) / 2
    legend!(fig, [xm - 7.2, xm + 0.6], 4.4, [(:dot, "#205e60", "pre", 0.5), (:dot, POST_COLOR, "post")])
end

"f: accuracy around a rule switch (bars as in Fig. S2)."
function panel_f!(fig, st, X)
    letter!(fig, X, ROW2 + 2.63, "f")
    ax = cell_axes(fig, X, ROW2 + 6.03, "", "accuracy"; xlim=(0.4, 5.6), ylim=(0.0, 1.0), w=29.4, ylabelx=3.99)
    draw_switch!(ax, st)
    text!(fig, (ax.left + ax.right) / 2, ax.bottom + 7.1, "trial"; anchor="middle")
end

function fig4(data, st, out; kde=true)
    fig = Figure(180, ROW2 + 46.0)
    panel_a!(fig)
    panel_b!(fig, data, COL[3]; kde)
    panel_c!(fig, data, COL[4])
    letter!(fig, COL[1], ROW2 + 2.63, "d"); letter!(fig, COL[3], ROW2 + 2.63, "e")
    panel_f!(fig, st, COL[4])
    xm = (COL[1] + COL[3] + 41.25) / 2                                       # legend of d, e, centred under them
    legend!(fig, xm - 59.5 .+ [0, 22.5, 47.5, 76.0, 96.5, 120.0], ROW2 + 43.7,
            [(:ring, "#661919", "central card"), (:dot, "#e54ce5", "response: left"), (:dot, "#33e54c", "response: right"),
             (:ring, "#000000", "error signal"), (:dot, "#ff4c4c", "context: shape"), (:dot, "#3399ff", "context: color")])
    return save(fig, out)
end

# ------------------------------------------------------------------ main
if hasflag("simulate")
    include(projectdir("src", "reservoir.jl")); include(projectdir("src", "wcst.jl"))
    run_wcst_sweep(; variants=["s"], g_contexts=[GB], reps=50, outdir=new_data_dir("wcst"))
    run_switch_sweep(new_data_dir("switch"); SWITCH.p_values, SWITCH.nets, SWITCH.seed, SWITCH.g_b)
end
fig4(load_results(data_dir("wcst")), switch_stats(load_switch(data_dir("switch"), 0.5), 0.5),
     figure_path(hasflag("gaussian") ? "fig4_gaussian" : "fig4"); kde=!hasflag("gaussian"))
