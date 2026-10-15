# Supplementary Figure S2: accuracy of the s-network (g_b = 0.8) around a rule switch of the WCST when, during the test,
# each error signal is halved with probability p = 0, 0.25, 0.5, 0.75 (training with the full error signal).
#
#   a  monkeys: performance on the last trial before a context switch and on the first four after it, read off the
#      figure of Bernardi et al., Cell 2020 (data/published/bernardi2020)
#   b  model, one plot per p: accuracy (fraction of correct responses, the ones that give no error signal) on the last
#      trial before a rule switch and on the first four trials after it, mean ± SEM over 200 networks; only the switches
#      between two blocks of at least 4 trials, so that every bar averages the same switches
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

const AXW = 27.3                                          # width of every plot (mm); a, then the four plots of b

"Mean and error of the monkeys' performance (Bernardi et al. 2020), as `switch_stats`."
function load_monkeys(path=published_dir("bernardi2020", "accuracy_around_switch.tsv"))
    d = readdlm(path, '\t'; header=true)[1]
    return (m=Float64.(d[:, 2]), se=Float64.(d[:, 3]))
end

function figS2(monkeys, stats, out)
    fig = Figure(180, 46)
    function plot_cell(left, title, st; ylabels=true, label=true, color=SWITCH_COLOR)
        ax = Axes(fig; left, right=left + AXW, top=9.0, bottom=37.0, xlim=(0.4, 5.6), ylim=(0.0, 1.0))
        draw_switch!(ax, st; label, ylabels, color)
        text!(fig, (ax.left + ax.right) / 2, 3.5, title; anchor="middle")
        text!(fig, (ax.left + ax.right) / 2, 43.5, "trial"; anchor="middle")
    end
    for (letter, x) in (("a", 12.0), ("b", 12.0 + AXW + 15.0))                 # letter and y label left of the axes
        text!(fig, x - 11.0, 3.5, letter; size=10, weight="bold")
        text!(fig, x - 7.5, 23.0, "accuracy"; anchor="middle", rotate=-90)
    end
    plot_cell(12.0, "monkeys", monkeys; color=MONKEY_COLOR)
    for (k, (p, st)) in enumerate(stats)
        plot_cell(12.0 + AXW + 15.0 + (k - 1) * (AXW + 5.0), @sprintf("p = %g", p), st; ylabels=k == 1, label=k == 1)
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
figS2(load_monkeys(), [(p, switch_stats(load_switch(dir, p), p)) for p in SWITCH.p_values], figure_path("figS2"))
