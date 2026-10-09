# Figure 3: delay-invariant short-term memory of the heterogeneous RNN.
#
#   a  stimulation scheme, population activity R(t) and raster of |r_j(t)| of one hRNN realization over one cycle
#   b  hRNN: Pearson correlation between evoked states vs signed position distance, one row per T_sep
#   c  the same vs |position distance| for hRNN, homogeneous network (hom) and echo-state network (esn)
#      (mean ± SEM over 50 networks)
#
# Usage:
#   julia scripts/fig3.jl                    # published values (data/published/fig3/fig3_values.csv)
#   julia -t 9 scripts/fig3.jl --simulate    # new simulation of the 9 conditions (~2 min) -> data/generated/fig3
#     --nets=50     networks per condition
#     --seed=2026   seed, the same for all conditions (same matrices W)
#   julia scripts/fig3.jl --data=generated   # figure from the data of an earlier run (or --data=FOLDER)
# Other flags (--s, --tag, --force, --out): see README.md. The figure goes to figures/<data source>/fig3.
#
# Panel a and the timing schemes of b are artwork (assets/fig3_artwork.svg). The published values come from an
# earlier NumPy implementation (seed 2026), which Julia cannot reproduce: --simulate gives statistically
# equivalent values.
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using LinearAlgebra, Random, Statistics, DelimitedFiles, Printf
include(projectdir("src", "SVGFigure.jl")); using .SVGFigure

const MTAU = 0.01
const MULTS = [1.0, 5.0, 20.0]                      # T_sep = 0.01, 0.05, 0.2 s
const TSEPS = [0.01, 0.05, 0.2]
const NETWORK_ORDER = ["hRNN", "hom", "esn"]
const COL = Dict("hRNN" => "#1b69af", "hom" => "#c2151b", "esn" => "#157333")

# ------------------------------------------------------------------ simulation (--simulate)
function simulate_all(nets::Int, seed::Int)
    BLAS.set_num_threads(1)                         # one BLAS thread per condition: results independent of -t
    jobs = [(net, m) for net in NETWORK_ORDER for m in MULTS]
    res = Dict{Tuple{String,Float64},Matrix{Float64}}()
    lk = ReentrantLock()
    Threads.@threads :dynamic for job in jobs
        net, m = job
        # same seed for every condition: the networks share the same random matrices W (paired comparison)
        c = run_memory_task(net, m; n_networks=nets, rng=Xoshiro(seed))
        lock(lk) do
            res[(net, m)] = c
            mu = vec(mean(c, dims=2))
            @printf("%-5s T_sep = %g s  rho(|d|=1,2,3) = [%.3f %.3f %.3f]\n", net, MTAU*m, mu[5], mu[6], mu[7])
        end
    end
    return res
end

# ------------------------------------------------------------------ table I/O
function write_csv(path, res)
    open(path, "w") do io
        println(io, "network,Tsep_s,d,rho_mean,rho_sem")
        for (net, m) in sort(collect(keys(res)))
            a = res[(net, m)]                       # distances x networks
            mu = vec(mean(a, dims=2)); se = vec(std(a, dims=2)) ./ sqrt(size(a, 2))
            for (i, d) in enumerate(-3:3)
                @printf(io, "%s,%s,%d,%.4f,%.4f\n", net, @sprintf("%g", MTAU*m), d, mu[i], se[i])
            end
        end
    end
end

function read_csv(path)
    tab = Dict{Tuple{String,Float64},Dict{Int,Tuple{Float64,Float64}}}()
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue
        f = split(line, ",")
        key = (String(f[1]), round(parse(Float64, f[2]), digits=4))
        get!(tab, key, Dict{Int,Tuple{Float64,Float64}}())[parse(Int, f[3])] = (parse(Float64, f[4]), parse(Float64, f[5]))
    end
    return tab
end

# ------------------------------------------------------------------ figure
# error bar: vertical segment and caps of half-width `cap` mm
function errorbar!(ax, x, y, e; color, width, cap)
    X = xmm(ax, x)
    line!(ax.fig, [X, X], [ymm(ax, y - e), ymm(ax, y + e)]; color=color, width=width, cap="butt")
    for v in (y - e, y + e)
        line!(ax.fig, [X - cap, X + cap], [ymm(ax, v), ymm(ax, v)]; color=color, width=width, cap="butt")
    end
end

function fig3(tab, out)
    fig = Figure(180, 98.957909)
    embed!(fig, assetsdir("fig3_artwork.svg"))      # panel a and the timing schemes of panel b

    # ---- b: hRNN, correlation vs signed position distance (one row per T_sep)
    # per row: frame [left, right, top, bottom], bar outline colour, bottom-spine width, label positions
    rows_b = [(100.388, 116.300, 5.240, 32.065, "#808080", 0.71, (98.04, 7.40), (97.71, 32.94), 26.27),
              (100.438, 116.351, 35.186, 62.011, "#808080", 0.71, (98.12, 37.35), (97.78, 62.88), 55.64),
              (100.413, 116.325, 65.133, 91.958, "#7f7f7f", 0.85, (98.08, 67.30), (97.74, 92.83), 85.59)]
    for (i, (l, r, t, b, edge, wbottom, lab1, lab0, ylab)) in enumerate(rows_b)
        ax = Axes(fig; left=l, right=r, top=t, bottom=b, xlim=(-3.5, 3.5), ylim=(0.0, 1.05))
        key = ("hRNN", TSEPS[i])
        for d in -3:3
            m = tab[key][d][1]
            x0, x1 = xmm(ax, d - 0.4), xmm(ax, d + 0.4)
            rect!(fig, x0, ymm(ax, m), x1 - x0, b - ymm(ax, m); fill="#cccccc", stroke=edge, width=0.57)
        end
        for d in -3:3
            errorbar!(ax, d, tab[key][d]...; color="#000000", width=0.49, cap=0.3485)
        end
        line!(fig, [l, r], [t, t]; width=0.71, cap="butt"); line!(fig, [l, r], [b, b]; width=wbottom, cap="butt")
        line!(fig, [l, l], [t, b]; width=0.71, cap="butt"); line!(fig, [r, r], [t, b]; width=0.71, cap="butt")
        yaxis!(ax; ticks=[0.0, 1.0], labels=["", ""], spine=false, ticklen=0.412, width=0.71)
        xaxis!(ax; ticks=[-3, 0, 3], labels=["", "", ""], spine=false, ticklen=0.412, width=0.71)
        text!(fig, lab1..., "1"); text!(fig, lab0..., "0")
        text!(fig, 97.94, ylab, "Corr. coef. ρ"; rotate=-90)
    end
    for (s, x) in (("-3", 100.47), ("0", 107.69), ("3", 114.51)); text!(fig, x, 94.93, s); end
    text!(fig, 99.14, 97.78, "position distance")

    # ---- c: correlation vs |position distance| for the three networks
    text!(fig, 126.29, 2.83, "c"; size=10, weight="bold")
    rows_c = [(133.783, 160.564, 5.948, 31.530, (131.28, 6.79, 7.0), (130.95, 32.33, 7.0), 25.48),
              (133.866, 160.566, 35.909, 61.567, (131.38, 36.67, 6.79), (131.29, 62.32, 6.79), 55.48),
              (133.826, 160.608, 65.330, 90.988, (131.37, 66.23, 6.79), (131.29, 91.87, 6.79), 84.90)]
    for (i, (l, r, t, b, lab1, lab0, ylab)) in enumerate(rows_c)
        ax = Axes(fig; left=l, right=r, top=t, bottom=b, xlim=(0.0, 3.25), ylim=(0.0, 1.0))
        for net in NETWORK_ORDER
            key = (net, TSEPS[i])
            plot!(ax, 0:3, [tab[key][d][1] for d in 0:3]; color=COL[net], width=0.68, clip=false)
            for d in 0:3
                errorbar!(ax, d, tab[key][d]...; color=COL[net], width=0.68, cap=0.465)
            end
        end
        xaxis!(ax; ticks=0:3, labels=fill("", 4), ticklen=0.585, width=0.71)
        yaxis!(ax; ticks=[0.0, 1.0], labels=["", ""], ticklen=0.55, width=0.71)
        text!(fig, lab1[1], lab1[2], "1"; size=lab1[3]); text!(fig, lab0[1], lab0[2], "0"; size=lab0[3])
        text!(fig, 131.11, ylab, "Corr. coef. ρ"; rotate=-90)
    end
    for (s, x, yb) in (("0", 133.15, 94.08), ("1", 141.47, 94.10), ("2", 149.65, 94.10), ("3", 157.87, 94.08))
        text!(fig, x, yb, s)
    end
    text!(fig, 137.38, 97.78, "|position distance|")
    for (net, y, tx, ty) in (("hRNN", 21.238, 136.02, 20.72), ("hom", 25.37, 136.66, 24.82), ("esn", 29.209, 137.26, 28.66))
        line!(fig, [138.474, 140.259], [y, y]; color=COL[net], width=1.42, cap="round")
        text!(fig, tx, ty, net)
    end

    return save(fig, out)
end

# ------------------------------------------------------------------ main
if hasflag("simulate")
    include(projectdir("src", "reservoir.jl")); include(projectdir("src", "memory_task.jl"))
    dir = new_data_dir("fig3")
    nets = parse(Int, flagvalue("nets", "50")); seed = parse(Int, flagvalue("seed", "2026"))
    println("Simulating $(length(NETWORK_ORDER)*length(MULTS)) conditions x $nets networks on $(Threads.nthreads()) thread(s), seed $seed")
    write_csv(joinpath(dir, "fig3_values.csv"), simulate_all(nets, seed))
end
fig3(read_csv(data_dir("fig3", "fig3_values.csv")), figure_path("fig3"))
