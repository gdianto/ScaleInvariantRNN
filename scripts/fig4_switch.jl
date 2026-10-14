# Draft panel for Fig. 4: accuracy of the s-network (g_b = 0.8) around a rule switch of the WCST, when during the test
# each error pulse is halved with probability p_half = 0.5 (run_wcst_switch; training with the full error pulse).
#
#   Accuracy (fraction of correct responses, the ones that give no error signal) on the last trial before a rule
#   switch and on the first four trials after it, mean ± SEM over the networks. Only the switches of the test period
#   between two blocks of at least 4 trials: every bar averages the same switches. Network k uses the seed seed + k.
#
# Usage:
#   julia -t 16 scripts/fig4_switch.jl --simulate             # 200 networks (~4 min on 16 threads) -> data/generated/fig4_switch
#     --nets=200    networks
#     --seed=2026   network k uses the seed seed + k
#   julia scripts/fig4_switch.jl --data=generated             # panel from the data of an earlier run (or --data=FOLDER)
# Other flags (--s, --tag, --force, --out): see README.md. The panel goes to figures/<data source>/fig4_switch.
#
# The published data have no per-trial results: the panel needs a run with --simulate.
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, LinearAlgebra, Random, Statistics, Printf
import Plots                                              # colour map
include(projectdir("src", "SVGFigure.jl")); using .SVGFigure

const GB = 0.8
const P_HALF = 0.5                                        # probability that an error pulse is halved at test
const MIN_BLOCK = 4                                       # trials of the blocks before and after a switch
const POSITIONS = ["last", "1^{st}", "2^{nd}", "3^{rd}", "4^{th}"]
hexcolor(c) = "#" * Plots.Colors.hex(Plots.Colors.RGB(c))
const BAR_COLOR = hexcolor(Plots.cgrad(:Reds)[0.7])
const CONTEXT_COLOR = "#800080"
hits_file(dir) = joinpath(dir, "hits_wcst_s-gb_$GB-phalf_$P_HALF.txt")
rules_file(dir) = joinpath(dir, "rules_wcst_s-gb_$GB.txt")

# ------------------------------------------------------------------ simulation (--simulate)
"Responses (1 correct, 0 wrong) of every test trial and the rules, one line per network."
function simulate(nets, seed)
    Threads.nthreads() > 1 && BLAS.set_num_threads(1)
    dir = new_data_dir("fig4_switch")
    println("Simulating $nets s-networks (g_b = $GB, p_half = $P_HALF) on $(Threads.nthreads()) thread(s)")
    res = Vector{Any}(undef, nets)
    Threads.@threads :dynamic for k in 1:nets
        Random.seed!(seed + k)
        res[k] = run_wcst_switch("s", GB; p_half=P_HALF)
    end
    writedlm(hits_file(dir), reduce(vcat, [Int.(r[1])' for r in res]), ' ')
    writedlm(rules_file(dir), reduce(vcat, [r[2]' for r in res]), ' ')
    cp(self_couplings_file(), joinpath(dir, "self_couplings.txt"))
    println("Data in $dir")
end

# ------------------------------------------------------------------ statistics
function load_switch(dir)
    isfile(rules_file(dir)) || error("no per-trial WCST results in $dir (the published data have none): run " *
                                     "julia -t 16 scripts/fig4_switch.jl --simulate, or choose a folder with --data")
    return (hits=readdlm(hits_file(dir), ' ', Int), rules=readdlm(rules_file(dir), ' ', Int))
end

"""
Accuracy on the last trial before a rule switch and on the first four after it, for each run (rows of `hits` and
`rules`, one column per test trial), averaged over the switches between two blocks of at least `MIN_BLOCK` trials.
Returns the runs × 5 accuracies (NaN for runs without such a switch) and the switches per run.
"""
function switch_accuracy(hits, rules)
    acc = fill(NaN, size(hits, 1), 5)
    n = zeros(Int, size(hits, 1))
    for i in axes(hits, 1)
        h, r = hits[i, :], rules[i, :]
        ks = [k for k in MIN_BLOCK+1:length(r)-MIN_BLOCK+1
              if r[k] != r[k-1] && all(==(r[k-1]), r[k-MIN_BLOCK:k-1]) && all(==(r[k]), r[k:k+MIN_BLOCK-1])]
        n[i] = length(ks)
        isempty(ks) || (acc[i, :] = mean(h[k-1:k+3] for k in ks))
    end
    return acc, n
end

"""
For each run: error rate within a rule block, P(wrong at t | correct at t-1, same rule), and switch after an error,
P(correct at t+1 | wrong at t, same rule at t+1).
"""
function error_rates(hits, rules)
    block_error, shift = fill(NaN, size(hits, 1)), fill(NaN, size(hits, 1))
    for i in axes(hits, 1)
        h, r = hits[i, :], rules[i, :]
        a = [h[t] for t in 2:length(h) if h[t-1] == 1 && r[t] == r[t-1]]
        b = [h[t+1] for t in 1:length(h)-1 if h[t] == 0 && r[t+1] == r[t]]
        isempty(a) || (block_error[i] = 1 - mean(a))
        isempty(b) || (shift[i] = mean(b))
    end
    return block_error, shift
end

meansem(v) = (v = filter(!isnan, v); (mean(v), std(v) / sqrt(length(v))))

function switch_stats(data)
    acc, n = switch_accuracy(data...)
    a = acc[n .> 0, :]
    m, se = vec(mean(a, dims=1)), vec(std(a, dims=1)) ./ sqrt(size(a, 1))
    be, sh = error_rates(data...)
    println("s-network, g_b = $GB, p_half = $P_HALF; accuracy around a rule switch (blocks of ≥ $MIN_BLOCK trials), mean ± SEM over the networks")
    @printf("networks %d, switches %d\n", size(a, 1), sum(n))
    println(join([@sprintf("%-5s %.3f ± %.3f", l, m[k], se[k]) for (k, l) in enumerate(["last", "1st", "2nd", "3rd", "4th"])], "\n"))
    @printf("all test trials %.3f; error within a block %.3f ± %.3f; switch after an error %.3f ± %.3f\n",
            mean(data.hits), meansem(be)..., meansem(sh)...)
    return (; m, se)
end

# ------------------------------------------------------------------ panel
function fig4_switch(stats, out)
    fig = Figure(40, 42)
    text!(fig, 1.0, 3.6, "b"; size=10, weight="bold")
    ax = Axes(fig; left=10.0, right=36.9, top=8.0, bottom=35.7, xlim=(0.4, 5.6), ylim=(0.0, 1.05))
    w = 0.36                                                                     # half-width of a bar
    for k in 1:5
        x0, x1, y = xmm(ax, k - w), xmm(ax, k + w), ymm(ax, stats.m[k])
        rect!(fig, x0, y, x1 - x0, ax.bottom - y; fill=BAR_COLOR, fill_opacity=0.4)
        line!(fig, [x0, x1], [y, y]; color=BAR_COLOR, width=0.7, cap="butt")
        X, m, se = xmm(ax, k), stats.m[k], stats.se[k]                           # error bar
        line!(fig, [X, X], [ymm(ax, m - se), ymm(ax, m + se)]; width=0.49, cap="butt")
        for v in (m - se, m + se)
            line!(fig, [X - 0.25, X + 0.25], [ymm(ax, v), ymm(ax, v)]; width=0.49, cap="butt")
        end
    end
    plot!(ax, [0.4, 5.6], [0.5, 0.5]; color="#000000", width=0.5, dash=(2.0, 1.5), cap="butt")    # chance
    X = xmm(ax, 1.5)
    line!(fig, [X, X], [ax.bottom, ax.top - 1.2]; color=CONTEXT_COLOR, width=1.0, cap="butt")
    text!(fig, X, ax.top - 2.0, "context switch"; anchor="middle", color=CONTEXT_COLOR)
    xaxis!(ax; ticks=1:5, labels=POSITIONS, ticklen=0.0, labeloffset=3.13, width=0.66)
    yaxis!(ax; ticks=[0.0, 0.5, 1.0], labels=["0", "0.5", "1.0"], labeloffset=1.03, baselineshift=0.79, width=0.66, ticklen=0.552)
    text!(fig, 23.45, 41.2, "trial"; anchor="middle")
    text!(fig, 4.0, 21.85, "accuracy"; anchor="middle", rotate=-90)
    return save(fig, out)
end

# ------------------------------------------------------------------ main
if hasflag("simulate")
    include(projectdir("src", "reservoir.jl")); include(projectdir("src", "wcst.jl"))
    simulate(parse(Int, flagvalue("nets", "200")), parse(Int, flagvalue("seed", "2026")))
end
fig4_switch(switch_stats(load_switch(data_dir("fig4_switch"))), figure_path("fig4_switch"))
