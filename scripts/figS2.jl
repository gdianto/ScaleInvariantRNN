# Supplementary Figure S2: accuracy of the s-network (g_b = 0.8) around a rule switch of the WCST when, during the test,
# each error signal is halved with probability p = 0, 0.25, 0.5, 0.75 (training with the full error signal).
#
#   a-d  accuracy (fraction of correct responses, the ones that give no error signal) on the last trial before a rule
#        switch and on the first four trials after it, mean ± SEM over 200 networks; only the switches between two
#        blocks of at least 4 trials, so that every bar averages the same switches
#
# Usage:
#   julia scripts/figS2.jl                              # published data (data/published/switch)
#   julia -t 16 scripts/figS2.jl --simulate             # 200 networks x 4 values of p (~5 min on 16 threads) -> data/generated/switch
#     --nets=200    networks
#     --seed=2026   network k uses the seed seed + k (the same networks for every p)
#   julia scripts/figS2.jl --data=generated             # figure from the data of an earlier run (or --data=FOLDER)
# Other flags (--s, --tag, --force, --out): see README.md. The figure goes to figures/<data source>/figS2.
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, LinearAlgebra, Random, Statistics, Printf
include(projectdir("src", "SVGFigure.jl")); using .SVGFigure
include(projectdir("src", "switch.jl"))

function figS2(stats, out)
    fig = Figure(180, 46)
    for (k, (p, st)) in enumerate(stats)
        x0 = 4.0 + 44.0 * (k - 1)                                              # left edge of the panel
        text!(fig, x0 - 2.5, 3.5, string('a' + k - 1); size=10, weight="bold")
        ax = Axes(fig; left=x0 + 10.0, right=x0 + 38.0, top=9.0, bottom=37.0, xlim=(0.4, 5.6), ylim=(0.0, 1.0))
        draw_switch!(ax, st; label=k == 1)
        text!(fig, (ax.left + ax.right) / 2, 3.5, @sprintf("p = %g", p); anchor="middle")
        text!(fig, (ax.left + ax.right) / 2, 43.5, "trial"; anchor="middle")
        text!(fig, x0 + 3.5, (ax.top + ax.bottom) / 2, "accuracy"; anchor="middle", rotate=-90)
    end
    return save(fig, out)
end

# ------------------------------------------------------------------ main
if hasflag("simulate")
    include(projectdir("src", "reservoir.jl")); include(projectdir("src", "wcst.jl"))
    run_switch_sweep(new_data_dir("switch"); SWITCH.p_values, nets=parse(Int, flagvalue("nets", string(SWITCH.nets))),
                     seed=parse(Int, flagvalue("seed", string(SWITCH.seed))), SWITCH.g_b)
end
dir = data_dir("switch")
figS2([(p, switch_stats(load_switch(dir, p), p)) for p in SWITCH.p_values], figure_path("figS2"))
