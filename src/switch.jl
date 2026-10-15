# Accuracy of the s-network (g_b = 0.8) around a rule switch of the WCST (Fig. 4f, Fig. S2): result files, statistics
# and the bar panel. During the test each error pulse is halved with probability p (`run_wcst_switch`, src/wcst.jl);
# network k has the seed SWITCH.seed + k for every p (paired runs: same network, task and training).
# The panel needs SVGFigure (include src/SVGFigure.jl first).

using DelimitedFiles, Statistics, Printf

const SWITCH = (g_b=0.8, p_values=[0.0, 0.25, 0.5, 0.75], nets=200, seed=2026, min_block=4)
const SWITCH_POSITIONS = ["last", "1^{st}", "2^{nd}", "3^{rd}", "4^{th}"]
const SWITCH_COLOR = "#2e8b57"          # bars: the response readout of Fig. 4b (accuracy of the card choice)
const CONTEXT_COLOR = "#800080"         # rule switch: the context readout of Fig. 4b
const MONKEY_COLOR = "#7f7f7f"          # bars of the monkeys (Fig. S2a)

switch_hits_file(dir, p; g_b=SWITCH.g_b) = joinpath(dir, "hits_wcst_s-gb_$(g_b)-phalf_$(p).txt")
switch_rules_file(dir; g_b=SWITCH.g_b) = joinpath(dir, "rules_wcst_s-gb_$(g_b).txt")

"Responses (1 correct, 0 wrong; networks × test trials) for halving probability `p`, and the rules (±1)."
function load_switch(dir, p)
    isfile(switch_hits_file(dir, p)) || error("no per-trial WCST results for p = $p in $dir: choose a folder with --data, " *
                                              "or run julia -t 16 scripts/figS2.jl --simulate")
    return (hits=readdlm(switch_hits_file(dir, p), ' ', Int), rules=readdlm(switch_rules_file(dir), ' ', Int))
end

"""
Accuracy on the last trial before a rule switch and on the first four after it, for each network (rows of `hits` and
`rules`), averaged over the switches between two blocks of at least `min_block` trials. Returns the networks × 5
accuracies (NaN for networks without such a switch) and the number of switches per network.
"""
function switch_accuracy(hits, rules; min_block=SWITCH.min_block)
    acc = fill(NaN, size(hits, 1), 5)
    n = zeros(Int, size(hits, 1))
    for i in axes(hits, 1)
        h, r = hits[i, :], rules[i, :]
        ks = [k for k in min_block+1:length(r)-min_block+1
              if r[k] != r[k-1] && all(==(r[k-1]), r[k-min_block:k-1]) && all(==(r[k]), r[k:k+min_block-1])]
        n[i] = length(ks)
        isempty(ks) || (acc[i, :] = mean(h[k-1:k+3] for k in ks))
    end
    return acc, n
end

"""
For each network: error rate within a rule block, P(wrong at t | correct at t-1, same rule), and switch after an error,
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

"Mean ± SEM over the networks of the accuracy around a switch; prints them with the error rates."
function switch_stats(data, p)
    acc, n = switch_accuracy(data.hits, data.rules)
    a = acc[n .> 0, :]
    m, se = vec(mean(a, dims=1)), vec(std(a, dims=1)) ./ sqrt(size(a, 1))
    be, sh = error_rates(data.hits, data.rules)
    @printf("p = %g: %d networks, %d switches; accuracy %s; all test trials %.3f; error within a block %.3f ± %.3f; switch after an error %.3f ± %.3f\n",
            p, size(a, 1), sum(n), join([@sprintf("%.3f", x) for x in m], " "), mean(data.hits), meansem(be)..., meansem(sh)...)
    return (; m, se)
end

"""
Bars of the accuracy (mean ± SEM) on the last trial before a rule switch and on the first four after it, in the axes
`ax` (xlim (0.4, 5.6), ylim (0, 1)), with the chance level and the switch; `label`: write "rule switch"; `ylabels`: tick
labels of the y axis. The same panel in Fig. 4f and Fig. S2.
"""
function draw_switch!(ax, stats; label=true, ylabels=true, color=SWITCH_COLOR)
    fig = ax.fig
    w = 0.34                                                                 # half-width of a bar
    for k in 1:5
        x0, x1, y = xmm(ax, k - w), xmm(ax, k + w), ymm(ax, stats.m[k])
        rect!(fig, x0, y, x1 - x0, ax.bottom - y; fill=color, fill_opacity=0.35)
        line!(fig, [x0, x1], [y, y]; color, width=0.69, cap="butt")
        X, m, se = xmm(ax, k), stats.m[k], stats.se[k]                       # error bar
        line!(fig, [X, X], [ymm(ax, m - se), ymm(ax, m + se)]; width=0.45, cap="butt")
        for v in (m - se, m + se)
            line!(fig, [X - 0.3, X + 0.3], [ymm(ax, v), ymm(ax, v)]; width=0.45, cap="butt")
        end
    end
    plot!(ax, [0.4, 5.6], [0.5, 0.5]; color="#000000", width=0.45, dash=(1.5, 1.0), cap="butt")    # chance
    X = xmm(ax, 1.5)
    line!(fig, [X, X], [ax.bottom, ax.top - 1.0]; color=CONTEXT_COLOR, width=0.8, cap="butt")
    label && text!(fig, X, ax.top - 1.8, "rule switch"; anchor="middle", color=CONTEXT_COLOR)
    xaxis!(ax; ticks=1:5, labels=SWITCH_POSITIONS, ticklen=0.0, labeloffset=3.13, width=0.66)
    yaxis!(ax; ticks=[0.0, 0.5, 1.0], labels=ylabels ? ["0", "0.5", "1.0"] : fill("", 3), labeloffset=1.03, baselineshift=0.79, width=0.66,
           ticklen=0.552)
end
