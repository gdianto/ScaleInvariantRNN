# Summary plots for analyze_network_robustness.jl. No simulation is run.
#
#   julia scripts/plot_network_robustness.jl
#   julia scripts/plot_network_robustness.jl --tag=pilot
#   julia scripts/plot_network_robustness.jl --data=/path/to/network_robustness
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using Statistics
include(projectdir("src", "SVGFigure.jl")); import .SVGFigure

const SOURCE_NAME = "network_robustness"
const COLORS = Dict(
    "fixed_point" => "#256167",
    "periodic_candidate" => "#e69f00",
    "chaotic_candidate" => "#bb1616",
    "unresolved" => "#888888",
)
const CLASS_ORDER = ["fixed_point", "periodic_candidate", "chaotic_candidate", "unresolved"]
const CLASS_LABEL = Dict(
    "fixed_point" => "fixed point",
    "periodic_candidate" => "periodic candidate",
    "chaotic_candidate" => "chaotic candidate",
    "unresolved" => "unresolved",
)


function input_directory()
    source = flagvalue("data", "generated")
    return source == "generated" ? generated_dir(SOURCE_NAME) : abspath(source)
end


function load_realizations(path)
    rows = NamedTuple[]
    for (row, line) in enumerate(eachline(path))
        row == 1 && continue
        f = split(line, '\t')
        push!(rows, (; sweep=String(f[1]), parameter=parse(Float64, f[2]),
                     realization=parse(Int, f[3]), classification=String(f[8]),
                     λ1=parse(Float64, f[11])))
    end
    return rows
end


function load_summaries(path)
    rows = NamedTuple[]
    for (row, line) in enumerate(eachline(path))
        row == 1 && continue
        f = split(line, '\t')
        push!(rows, (; sweep=String(f[1]), parameter=parse(Float64, f[2]),
                     fixed=parse(Float64, f[10]), periodic=parse(Float64, f[11]),
                     chaotic=parse(Float64, f[12]), unresolved=parse(Float64, f[13]),
                     gamma=parse(Float64, f[16]), r2=parse(Float64, f[17])))
    end
    return rows
end


function decorate_axes!(ax; xticks, xlabels, yticks, ylabels, xlabel, ylabel)
    SVGFigure.xaxis!(ax; ticks=xticks, labels=xlabels)
    SVGFigure.yaxis!(ax; ticks=yticks, labels=ylabels)
    SVGFigure.text!(ax.fig, (ax.left + ax.right) / 2, ax.bottom + 8, xlabel; anchor="middle")
    SVGFigure.text!(ax.fig, ax.left - 12, (ax.top + ax.bottom) / 2, ylabel;
                    anchor="middle", rotate=-90)
end


function finite_range(values; include_zero=false)
    values = filter(isfinite, values)
    isempty(values) && return (-1.0, 1.0)
    low, high = extrema(values)
    include_zero && (low, high = min(low, 0.0), max(high, 0.0))
    span = max(high - low, 0.1)
    return low - 0.08 * span, high + 0.08 * span
end


function draw_classification_bars!(ax, summaries, parameters, width)
    for (x, row) in zip(parameters, summaries)
        bottom = 0.0
        fractions = Dict("fixed_point" => row.fixed,
                         "periodic_candidate" => row.periodic,
                         "chaotic_candidate" => row.chaotic,
                         "unresolved" => row.unresolved)
        for class in CLASS_ORDER
            fraction = fractions[class]
            x0, x1 = SVGFigure.xmm(ax, x - width / 2), SVGFigure.xmm(ax, x + width / 2)
            y0, y1 = SVGFigure.ymm(ax, bottom), SVGFigure.ymm(ax, bottom + fraction)
            SVGFigure.rect!(ax.fig, x0, y1, x1-x0, y0-y1;
                            fill=COLORS[class], stroke="#ffffff", width=0.35, clip=ax.clip)
            bottom += fraction
        end
    end
end


function make_figure(sweep, realizations, summaries, output_prefix)
    summaries = sort(filter(x -> x.sweep == sweep, summaries); by=x -> x.parameter)
    isempty(summaries) && return String[]
    realizations = filter(x -> x.sweep == sweep, realizations)
    parameters = [x.parameter for x in summaries]
    spacing = length(parameters) > 1 ? minimum(diff(parameters)) : 1.0
    xlim = (minimum(parameters) - 0.55 * spacing, maximum(parameters) + 0.55 * spacing)
    xlabel = sweep == "g" ? "recurrent gain, g" : "self-coupling perturbation, sigma_s"
    title = sweep == "g" ? "Robustness to recurrent gain" : "Robustness to quenched self-coupling perturbations"
    xlabels = string.(parameters)

    fig = SVGFigure.Figure(180, 72)
    SVGFigure.text!(fig, 90, 6, title; size=9, anchor="middle", weight="bold")

    # Shared classification legend
    legend_x = [25, 65, 112, 151]
    for (x, class) in zip(legend_x, CLASS_ORDER)
        SVGFigure.circle!(fig, x, 11, 0.8; fill=COLORS[class])
        SVGFigure.text!(fig, x + 2, 11.8, CLASS_LABEL[class]; size=5.5)
    end

    # a: largest exponent of every realization
    λvalues = [x.λ1 for x in realizations if isfinite(x.λ1)]
    ylim = finite_range(λvalues; include_zero=true)
    ax = SVGFigure.Axes(fig; left=17, right=68, top=17, bottom=57, xlim, ylim)
    SVGFigure.plot!(ax, collect(xlim), [0.0, 0.0]; color="#bb1616", width=0.55, dash=(3, 2))
    for row in realizations
        isfinite(row.λ1) || continue
        nudge = spacing * 0.18 * ((row.realization - 0.5) / maximum(x.realization for x in realizations) - 0.5)
        SVGFigure.scatter!(ax, [row.parameter + nudge], [row.λ1]; r=0.55,
                           fill=COLORS[row.classification])
    end
    decorate_axes!(ax; xticks=parameters, xlabels, yticks=collect(range(ylim...; length=5)),
                   ylabels=string.(round.(collect(range(ylim...; length=5)); digits=2)),
                   xlabel, ylabel="largest exponent (1/s)")
    SVGFigure.text!(fig, 4, 16, "a"; size=10, weight="bold")

    # b: attractor fractions
    ax = SVGFigure.Axes(fig; left=77, right=123, top=17, bottom=57,
                        xlim, ylim=(0.0, 1.0))
    draw_classification_bars!(ax, summaries, parameters, 0.65 * spacing)
    decorate_axes!(ax; xticks=parameters, xlabels, yticks=[0.0, 0.5, 1.0],
                   ylabels=["0", "0.5", "1"], xlabel, ylabel="fraction")
    SVGFigure.text!(fig, 70, 16, "b"; size=10, weight="bold")

    # c: power-law exponent and fit quality
    gamma = [x.gamma for x in summaries]
    r2 = [x.r2 for x in summaries]
    combined = vcat(filter(isfinite, gamma), filter(isfinite, r2), [0.0, 1.0])
    ylow, yhigh = finite_range(combined)
    ax = SVGFigure.Axes(fig; left=134, right=176, top=17, bottom=57,
                        xlim, ylim=(ylow, yhigh))
    SVGFigure.plot!(ax, collect(xlim), [1.0, 1.0]; color="#777777", width=0.5, dash=(3, 2))
    valid_gamma = findall(isfinite, gamma)
    valid_r2 = findall(isfinite, r2)
    isempty(valid_gamma) || SVGFigure.plot!(ax, parameters[valid_gamma], gamma[valid_gamma];
                                             color="#256167", width=0.8)
    isempty(valid_r2) || SVGFigure.plot!(ax, parameters[valid_r2], r2[valid_r2];
                                         color="#e69f00", width=0.8)
    isempty(valid_gamma) || SVGFigure.scatter!(ax, parameters[valid_gamma], gamma[valid_gamma];
                                                r=0.65, fill="#256167")
    isempty(valid_r2) || SVGFigure.scatter!(ax, parameters[valid_r2], r2[valid_r2];
                                             r=0.65, fill="#e69f00")
    yticks = collect(range(ylow, yhigh; length=5))
    decorate_axes!(ax; xticks=parameters, xlabels, yticks,
                   ylabels=string.(round.(yticks; digits=2)), xlabel,
                   ylabel="power-law gamma / R2")
    SVGFigure.line!(fig, [144, 149], [14, 14]; color="#256167", width=0.8)
    SVGFigure.text!(fig, 150, 14.8, "gamma"; size=5.5)
    SVGFigure.line!(fig, [160, 165], [14, 14]; color="#e69f00", width=0.8)
    SVGFigure.text!(fig, 166, 14.8, "R2"; size=5.5)
    SVGFigure.text!(fig, 126, 16, "c"; size=10, weight="bold")

    return SVGFigure.save(fig, output_prefix)
end


function main()
    data_directory = input_directory()
    realizations_path = joinpath(data_directory, "realization_results.tsv")
    summaries_path = joinpath(data_directory, "condition_summary.tsv")
    isfile(realizations_path) || error("Results not found: $realizations_path")
    isfile(summaries_path) || error("Summary not found: $summaries_path")
    realizations = load_realizations(realizations_path)
    summaries = load_summaries(summaries_path)

    output_directory = figures_dir("generated")
    for sweep in ("g", "sigma_s")
        files = make_figure(sweep, realizations, summaries,
                            joinpath(output_directory, "network_robustness_$(sweep)"))
        isempty(files) || println("Saved: $(join(files, ", "))")
    end
end


main()
