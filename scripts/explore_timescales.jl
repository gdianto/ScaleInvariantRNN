# Interactive exploration of Figs. 1-2, to run cell by cell (`##`). Parameters in the first cell (defaults: the
# paper, with 50 networks instead of 500); save_figures = true saves the plots to figures/explore_timescales/.
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using LinearAlgebra, Random, Statistics, DelimitedFiles, Printf, Plots
include(projectdir("src", "reservoir.jl"))
include(projectdir("src", "timescales.jl"))

## parameters
seed = 1234              # nothing = unseeded

# network
N = 500                  # units
g = 0.4                  # spectral radius of J before the self-couplings
τ0 = 0.01                # unit time constant (s)
s_file = self_couplings_file()   # published self-couplings; e.g. generated_dir("self_couplings.txt") for a new vector
custom_s = nothing       # N self-couplings (or one number) instead of the file, e.g. collect(range(0.7, 1.7; length=N))
n_nets = 50              # networks averaged (paper: 500)

# integration (Fig. 1: dt = 0.1 τ0, saved every 2 steps, 200 τ0; Fig. 2: dt = 0.01 τ0)
dt = 0.1 * τ0
duration = 200 * τ0      # s (the power-law fit uses 0.07-0.9 s)

# decoupled network: unit i receives Σ_j Jext_ij x_j(t), Jext_ij ~ N(0, 1), x_j independent OU processes
τ_ou = 0.01              # OU correlation time (s)
σ_ou = 0.0186734693877551   # OU amplitude (stationary std σ/√2), as in Fig. 2

# fit of σ_ou as in the Methods (slow): grid search, τ_ou fixed, loss = mean squared difference of the
# coupled and decoupled ⟨R(t)⟩ on 0.003-1.9 s
run_grid_search = false
σ_grid = range(0.012, 0.026; length=8)
n_nets_search = 50

# unit timescales: "paper" = Fig. 1 rule (exp(-t/τ) for s < 1, A exp(-t/τ) for s >= 1), reliable only with many
# networks (a bistable unit ends at ±r*, so few networks leave a random plateau); "crossing" = first time ⟨r⟩ <= 1/e
tau_method = "crossing"
trace_s = [0.70, 0.80, 1.00, 1.23]   # units of the traces (closest s), as in Fig. 1c-d
save_figures = false
default(left_margin=5Plots.mm, bottom_margin=6Plots.mm)
show!(p, name) = (display(p); save_figures && savefig(p, joinpath(figures_dir("explore_timescales"), "$name.png")); p)

## self-couplings and time grid
s = custom_s === nothing ? vec(readdlm(s_file, ',', Float64)) :
    custom_s isa Number ? fill(Float64(custom_s), N) : collect(Float64, custom_s)
length(s) == N || error("custom_s must contain N = $N values")
times = 0.0:dt:duration
t = collect(times)[1:2:end]                         # saved every 2 steps (as fit_unit_timescales assumes)
trace_units = [argmin(abs.(s .- v)) for v in trace_s]
@printf("N = %d, s in [%.3f, %.3f], %d units with s >= 1; dt = %.2g s, %d saved samples\n",
        N, minimum(s), maximum(s), count(>=(1), s), dt, length(t))
show!(histogram(s; bins=range(minimum(s), maximum(s) + 1e-9; length=11), normalize=:pdf, label="s",
                xlabel="self-coupling s", ylabel="P(s)", color=:teal), "s_distribution")

## coupled networks: relaxation from r = 1
# also the recurrent input of each unit without its self-coupling, Σ_{j≠i} J_ij r_j
function run_coupled(s, n_nets)
    mean_r = zeros(N, length(t)); pop = zeros(length(t), n_nets); input_std = zeros(N)
    example, example_input = similar(mean_r), similar(mean_r)
    for k in 1:n_nets
        r, cur = simulate!(coupled_network(s, g, τ0), (0.0, duration); dt, save_every=2, currents=true)
        mean_r .+= r; pop[:, k] .= vec(mean(r, dims=1)); input_std .+= vec(std(cur, dims=2))
        example .= r; example_input .= cur
    end
    return (; mean_r=mean_r ./ n_nets, pop, input_std=input_std ./ n_nets, example, example_input)
end
seed === nothing || Random.seed!(seed)
@time co = run_coupled(s, n_nets);

## rasters and traces (Fig. 1b-d)
order = sortperm(s)
p1 = heatmap(t, 1:N, co.example[order, :]; c=:RdBu, clims=(-1, 1), xlabel="t [s]", ylabel="unit (sorted by s)", title="one network")
p2 = heatmap(t, 1:N, co.mean_r[order, :]; c=:RdBu, clims=(-1, 1), xlabel="t [s]", title="mean over $n_nets networks")
show!(plot(p1, p2; layout=(1, 2), size=(1000, 400)), "rasters")
p1 = plot(xlabel="t [s]", ylabel="r(t)", title="one network")
p2 = plot(xlabel="t [s]", ylabel="⟨r(t)⟩", yscale=:log10, ylim=(1e-2, 1.1), title="mean over networks")
for u in trace_units
    plot!(p1, t, co.example[u, :]; label=@sprintf("s = %.2f", s[u]))
    keep = co.mean_r[u, :] .> 0
    plot!(p2, t[keep], co.mean_r[u, keep]; label=@sprintf("s = %.2f", s[u]))
end
show!(plot(p1, p2; layout=(1, 2), size=(1000, 380)), "traces")

## τ(s), its fit and the implied P(s) (Fig. 1a, e)
# P(s) makes P(τ) ∝ τ^-2 on [τ(min s), τ(max s)] for the fitted τ(s)
crossing_times(mean_r) = [(k = findfirst(<=(exp(-1)), view(mean_r, u, :)); k === nothing ? NaN : t[k]) for u in axes(mean_r, 1)]
unit_tau(mean_r) = tau_method == "paper" ? fit_unit_timescales(mean_r, s, t, dt) : crossing_times(mean_r)
τ = unit_tau(co.mean_r)
ok = isfinite.(τ) .& (τ .> 0)
τ_paper = readdlm(published_dir("fig1", "fig1e_tau_vs_self_coupling.tsv"), '\t', Float64; header=true)[1][:, 2]   # published run
@printf("%s τ: %d units not decayed; median |log10 τ/τ_paper| = %.3f decades\n", tau_method, count(!, ok),
        median(abs.(log10.(τ[ok] ./ τ_paper[ok]))))
fit = fit_tau(s[ok], τ[ok]); a, b, c = fit.params
@printf("τ(s) = a exp(b s^c): a = %.4g, b = %.4g, c = %.4g; R² (log10 τ) = %.4f, rmse = %.3g decades\n",
        a, b, c, fit.r2_log, fit.rmse_log10)
println("paper: a = 0.01646, b = 1.017, c = 2.991")
sc = range(minimum(s), maximum(s); length=500)
p1 = scatter(s, τ_paper; ms=2, msw=0, yscale=:log10, color=:lightgray, label="paper run", xlabel="s", ylabel="τ [s]")
scatter!(p1, s[ok], τ[ok]; ms=2, msw=0, label="units ($tau_method)", ylim=(5e-3, 10))
plot!(p1, sc, tau_model(sc, fit.params); lw=2, color=:red, label="a exp(b s^c)")
p2 = histogram(s; bins=range(minimum(s), maximum(s) + 1e-9; length=11), normalize=:pdf, label="s", xlabel="s", ylabel="P(s)", color=:teal)
plot!(p2, sc, self_coupling_density(sc, fit.params; smin=minimum(s), smax=maximum(s)); lw=2, color=:red, label="P(s) for P(τ) ∝ τ⁻²")
show!(plot(p1, p2; layout=(1, 2), size=(1000, 380)), "tau_vs_s")

## population activity: power law (Fig. 1f)
R = vec(mean(co.pop, dims=2)); R_sem = vec(std(co.pop, dims=2)) ./ sqrt(n_nets)
pp = fit_power_law(t, R)
win = (t .>= 0.07) .& (t .<= 0.9) .& (R .> 0)
r2 = r_squared(log10.(R[win]), log10.(power_law(t[win], pp)))
@printf("⟨R(t)⟩ ≈ α t^-γ on 0.07-0.9 s: α = %.4f, γ = %.4f, R² (log-log) = %.4f   (paper: γ = 0.93)\n", pp[1], -pp[2], r2)
keep = (t .> 0) .& (R .> 0)
band(m, sem) = (min.(3 .* sem, 0.9 .* m), 3 .* sem)      # ±3 SEM, positive on the log axis
p = plot(t[keep], R[keep]; ribbon=band(R[keep], R_sem[keep]), xscale=:log10, yscale=:log10,
         label="⟨R(t)⟩ ± 3 SEM", xlabel="t [s]", ylabel="⟨R(t)⟩", lw=2)
plot!(p, t[win], power_law(t[win], pp); ls=:dash, color=:red, lw=2, label=@sprintf("t^-%.3f", -pp[2]))
show!(p, "population_power_law")

## recurrent input vs OU input
# OU input Σ_j Jext_ij x_j: stationary std √N σ/√2, correlation time τ_ou. The recurrent input is not
# stationary: it changes as the network relaxes (right: std across units vs time, one network).
ou_std = sqrt(N) * σ_ou / sqrt(2)
@printf("recurrent input std (over time, mean over units and networks) = %.4f; OU input std = %.4f\n", mean(co.input_std), ou_std)
late = t .>= duration / 4
lags = 0:round(Int, 20τ0 / (t[2] - t[1]))
function autocorr(X, lags)
    X = X .- mean(X, dims=2)
    return [mean(sum(X[:, 1:end-l] .* X[:, 1+l:end], dims=2) ./ sum(X .^ 2, dims=2)) for l in lags]
end
ac = autocorr(co.example_input[:, late], lags)
p1 = scatter(s, co.input_std; ms=2, msw=0, label="recurrent input", xlabel="s", ylabel="std of the input")
hline!(p1, [ou_std]; color=:orange, lw=2, label="OU input")
p2 = plot(lags .* (t[2] - t[1]), ac; lw=2, label="recurrent input (one network, t > $(duration/4) s)", xlabel="lag [s]", ylabel="autocorrelation")
plot!(p2, lags .* (t[2] - t[1]), exp.(-lags .* (t[2] - t[1]) ./ τ_ou); lw=2, ls=:dash, color=:orange, label="OU, τ_ou = $τ_ou s")
p3 = plot(t, vec(std(co.example_input, dims=1)); lw=2, label="recurrent input", xlabel="t [s]", ylabel="std across units")
hline!(p3, [ou_std]; color=:orange, lw=2, label="OU input")
show!(plot(p1, p2, p3; layout=(1, 3), size=(1400, 380)), "recurrent_input")

## decoupled network driven by OU inputs (Fig. 2b-c)
function run_decoupled(s, n_nets, τ_ou, σ_ou; rng=Random.default_rng())
    mean_r = zeros(N, length(t)); pop = zeros(length(t), n_nets)
    for k in 1:n_nets
        x = ou_process(length(times), N, dt, τ_ou, σ_ou; rng)
        r = simulate!(decoupled_network(s, τ0, rng), (0.0, duration); dt, input=x, save_every=2)
        mean_r .+= r; pop[:, k] .= vec(mean(r, dims=1))
    end
    return (; mean_r=mean_r ./ n_nets, pop)
end
@time de = run_decoupled(s, n_nets, τ_ou, σ_ou; rng=seed === nothing ? Random.default_rng() : MersenneTwister(seed + 1));
τ_dec = unit_tau(de.mean_r)
R_dec = vec(mean(de.pop, dims=2)); R_dec_sem = vec(std(de.pop, dims=2)) ./ sqrt(n_nets)
okd = ok .& isfinite.(τ_dec)
@printf("median |log10 τ_dec/τ_coupled| = %.3f decades\n", median(abs.(log10.(τ_dec[okd] ./ τ[okd]))))
keep = (t .> 0) .& (R .> 0) .& (R_dec .> 0)
p1 = plot(xscale=:log10, yscale=:log10, xlabel="t [s]", ylabel="⟨R(t)⟩")
plot!(p1, t[keep], R_dec[keep]; ribbon=band(R_dec[keep], R_dec_sem[keep]), lw=2, color=:orange, label="decoupled + OU")
plot!(p1, t[keep], R[keep]; ribbon=band(R[keep], R_sem[keep]), lw=2, color=:teal, label="coupled")
p2 = scatter(s[ok], τ[ok]; ms=2, msw=0, color=:teal, yscale=:log10, label="coupled", xlabel="s", ylabel="τ [s]")
scatter!(p2, s[okd], τ_dec[okd]; ms=2, msw=0, color=:orange, label="decoupled + OU")
plot!(p2, 1:0.01:maximum(s), escape_time.(1:0.01:maximum(s); σ=sqrt(N) * σ_ou, τ0); lw=2, color=:red, label="escape time (theory, s > 1)")
show!(plot(p1, p2; layout=(1, 2), size=(1000, 380)), "coupled_vs_decoupled")

## optional: fit of σ_ou (Methods; paper: 0.0187)
if run_grid_search
    fit_window = (t .>= 0.003) .& (t .<= 1.9)
    loss = map(σ_grid) do σ_try
        R_try = vec(mean(run_decoupled(s, n_nets_search, τ_ou, σ_try; rng=MersenneTwister(something(seed, 1234))).pop, dims=2))
        L = mean(abs2, R[fit_window] .- R_try[fit_window])
        @printf("  σ_ou = %.5f: loss %.3e\n", σ_try, L)
        L
    end
    @printf("best σ_ou = %.5f (input std √N σ/√2 = %.3f)\n", σ_grid[argmin(loss)], sqrt(N) * σ_grid[argmin(loss)] / sqrt(2))
    show!(plot(σ_grid, loss; marker=:circle, xlabel="σ_ou", ylabel="loss", legend=false), "sigma_fit")
end
