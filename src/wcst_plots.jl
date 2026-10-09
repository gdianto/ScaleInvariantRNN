# Plots for scripts/explore_wcst.jl (kept out of src/wcst.jl so that the figure scripts do not load Plots).

using Plots, MultivariateStats

"Principal axes of the states X (units × time) and the entropy of the explained-variance fractions."
function pca(X)
    M = fit(PCA, X; pratio=1.0)
    λ = eigvals(M) / var(M)
    return projection(M), -sum(λ .* log.(λ))
end

function plot_spectrum(J)
    λ = eigvals(Matrix(J))
    t = range(0, 2π, length=500)
    plot(cos.(t), sin.(t), seriestype=:shape, lw=0.5, c=RGB(0, 0.5, 1), linecolor=:black, fillalpha=0.5, aspect_ratio=1, legend=false)
    return scatter!(λ, c=RGB(1, 0.4, 0.85), markersize=1, markerstrokewidth=0, title="max |λ| = $(round(maximum(abs.(λ)), digits=3))")
end

"Inputs over the steps `steps`, one offset trace per channel."
function plot_inputs(input, steps, n_cards, context_out)
    colors = palette(:roma, n_cards)
    p = plot(size=(600, 350))
    for k in 1:n_cards
        plot!(input[steps, k] .+ k * 2.1, label="card $k", color=colors[k])
    end
    plot!(input[steps, end-(1+context_out)] .+ ((n_cards + 1) * 2.4 + 2.2), label="error", color=RGB(0.8, 0.0, 0.2))
    context_out && plot!(input[steps, end-1] .+ ((n_cards + 1) * 2.4 + 4.6), label="rule", color=RGB(0.65, 0.6, 0.8), line=:dash)
    plot!(input[steps, end] .+ (n_cards + 1) * 2.4, label="optimal response", color=RGB(0.2, 0.2, 0.2), linestyle=:dash)
    plot!(yticks=[], xlabel="steps", ylabel="signals", legend=:outerbottom, legendcolumns=4, bottom_margin=3Plots.mm, left_margin=5Plots.mm)
    return p
end

"""
2D and 3D views of the trajectory `pcs` (rows: coordinates), marking the samples where `marked` > threshold
(white circles) and where `out` is above / below ±threshold (two colours).
"""
function plot_state_space(pcs, out, marked, labels, colors; size=(900, 600), threshold=0.0, plane=false, lims=[], camera=[45.0, 45.0, 1.0])
    up, down, on = findall(out .> threshold), findall(out .< -threshold), findall(marked .> threshold)
    p1 = plot(pcs[1, :], pcs[2, :], xlabel="PC₁", ylabel="PC₂", color=RGB(0.9, 0.9, 0.9), label="")
    p2 = plot(pcs[1, :], pcs[2, :], pcs[3, :], xlabel="PC₁", ylabel="PC₂", zlabel="PC₃", color=RGB(0.9, 0.9, 0.9), label="")
    for (p, d, ms) in ((p1, 1:2, 2), (p2, 1:3, 1.5))
        scatter!(p, eachrow(pcs[d, on])..., color=:white, markerstrokecolor=colors[3], ms=4, label=labels[3])
        scatter!(p, eachrow(pcs[d, up])..., color=colors[1], markerstrokecolor=colors[1], ms=ms, label=labels[1])
        scatter!(p, eachrow(pcs[d, down])..., color=colors[2], markerstrokecolor=colors[2], ms=ms, label=labels[2])
    end
    plane && surface!(p2, range(-20, 20, length=50), range(-20, 20, length=50), (x, y) -> 0, color=RGB(0.9, 0.8, 0.7), alpha=0.2, cbar=false)
    isempty(lims) || plot!(p2, xlims=lims[1], ylims=lims[2], zlims=lims[3])
    plot!(p2, camera=Tuple(camera))
    return plot(p1, p2, layout=@layout([a{0.35w} b{0.65w}]), legend=true, size=size, bottom_margin=5Plots.mm, left_margin=5Plots.mm)
end

plot_context_space(pcs, context, error; kw...) =
    plot_state_space(pcs, context, error, ("context 1", "context 2", "error"), (RGB(1.0, 0.3, 0.3), RGB(0.2, 0.6, 1.0), RGB(0.1, 0.1, 0.1)); kw...)
plot_response_space(pcs, response, central; kw...) =
    plot_state_space(pcs, response, central, ("response R", "response L", "central card"), (RGB(0.2, 0.9, 0.3), RGB(0.9, 0.3, 0.9), RGB(0.4, 0.1, 0.1)); kw...)

function plot_power_law(pl; n_starts=100, title="")
    (; t, acf_mean, r, params, i0) = pl
    p_acf = plot(t, acf_mean, color=:red, label="⟨C(t)⟩")
    p_traces = plot(r[1:10, :]', ylabel="r", xlabel="step", color=:black, lw=2)
    plot!(r[end-10:end, :]', color=:red, lw=2, legend=false)
    p_fit = plot(t, acf_mean, xlabel="t", ylabel="C(t)", label="C(t), $n_starts initial states", lw=2, size=(500, 500),
                 ylims=[minimum(abs.(acf_mean)), 1.0], xlims=[t[2], t[end]], xscale=:log10, yscale=:log10)
    plot!(t[2:end], power_law(t[2:end], params), lw=2, linestyle=:dash, legend=:bottom, title=title,
          label="fit α t^β, α=$(params[1]), β=$(params[2])\nfrom t = $(t[i0]) s")
    return (; p_acf, p_traces, p_fit)
end
