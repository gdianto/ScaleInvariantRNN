# Figs. 1-2: networks with heterogeneous self-couplings relaxing from r = 1, unit timescales τ(s), the fit
# τ(s) = a exp(b s^c) and the P(s) it implies, and the decoupled network driven by OU inputs.
# Requires src/reservoir.jl.

using LinearAlgebra, Random, SparseArrays, Statistics, LsqFit

const FIG1 = (n_units=500, n_networks=500, g=0.4, τ=0.01, dt=0.1 * 0.01, save_every=2, duration=200 * 0.01, seed=1234)
const FIG2 = (n_units=500, n_networks=500, g=0.4, τ=0.01, dt=0.01 * 0.01, save_every=1, duration=200 * 0.01,
              τ_ou=0.01, σ_ou=0.0186734693877551, seed=1234, seed_decoupled=4321)
smoke_config(c) = merge(c, (n_units=40, n_networks=2))     # quick pipeline test

tau_model(s, p) = p[1] .* exp.(p[2] .* s .^ p[3])
log_tau_model(s, p) = log(p[1]) .+ p[2] .* s .^ p[3]

"Fit of τ(s) = a exp(b s^c) on log τ."
function fit_tau(s, τ)
    p = curve_fit(log_tau_model, s, log.(τ), [0.01, 1.0, 1.0]; lower=fill(1e-5, 3), upper=fill(10.0, 3), maxIter=10_000).param
    lt, lf = log10.(τ), log10.(tau_model(s, p))
    return (; params=p, r2_log=1 - sum(abs2, lt .- lf) / sum(abs2, lt .- mean(lt)), rmse_log10=sqrt(mean(abs2, lt .- lf)))
end

"P(s) on [smin, smax] for which P(τ) ∝ τ^-2, given the fit parameters p."
function self_coupling_density(s, p; smin=0.7, smax=1.7)
    a, b, c = p
    τmin, τmax = tau_model(smin, p), tau_model(smax, p)
    return (τmin * τmax / (τmax - τmin) * b * c / a) .* (s .^ (c - 1.0)) .* exp.(-b .* s .^ c)
end

"N self-couplings drawn from P(s) by inverse transform sampling (Methods), sorted."
function sample_self_couplings(p, n; smin=0.7, smax=1.7, rng=Random.default_rng())
    a, b, c = p
    τmin, τmax = tau_model(smin, p), tau_model(smax, p)
    C = τmin * τmax / (τmax - τmin)
    return sort([(-log(exp(-b * smin^c) - a / C * y) / b)^(1 / c) for y in rand(rng, n)])
end

"""
    escape_time(s; σ, τ0=0.01)

Theoretical relaxation time of a bistable unit (s > 1) driven by colored noise of amplitude σ with
τ_OU = τ0 (Fig. 2d): Kramers-type escape time in the potential U(x) = x²/2 - s log cosh(x), with diffusion
D = σ²/2 and the minimum of U approximated by x_m = 1.8 √(s - 1) (U(x_m) = 1.62 (s - 1) - s log cosh(x_m)).
"""
function escape_time(s; σ, τ0=0.01)
    xm = 1.8 * sqrt(s - 1)
    ΔU = abs(-1.62 + 1.62 * s - s * log(cosh(xm)))
    return τ0 * 2π * sqrt(abs(1 - s * sech(xm)^2 * abs(1 - s))) * exp(ΔU / (σ^2 / 2))
end

"All values, or an evenly spaced subset of them."
select_self_couplings(s, n) = n == length(s) ? collect(Float64, s) : collect(Float64, s[round.(Int, range(1, length(s); length=n))])

"Gaussian couplings with spectral radius g (zero diagonal), then self-couplings s on the diagonal; r = 1."
function coupled_network(s, g, τ)
    n = length(s)
    J = sprandn(n, n, 1.0)
    J[diagind(J)] .= 0.0
    J = Matrix(g * (J / maximum(abs.(eigvals(Matrix(J))))))
    J[diagind(J)] .= s
    return Network(ones(n), ones(n) .* τ, J, zeros(n, 0), zeros(n))
end

"Self-couplings only, inputs Σ_j Jext_ij x_j with Jext_ij ~ N(0, 1); r = 1."
decoupled_network(s, τ, rng) = (n = length(s); Network(ones(n), ones(n) .* τ, Diagonal(s), randn(rng, n, n), zeros(n)))

"Independent stationary OU processes dX = -X/τ dt + σ/√τ dW (stationary std σ/√2), Euler-Maruyama."
function ou_process(nt, n, dt, τ, σ; rng=Random.default_rng())
    x = zeros(nt, n)
    x[1, :] .= (σ / sqrt(2)) .* randn(rng, n)
    @inbounds for t in 1:nt-1
        x[t+1, :] .= x[t, :] .+ dt .* (-x[t, :] ./ τ) .+ σ / sqrt(τ) .* sqrt(dt) .* randn(rng, n)
    end
    return x
end

integration_times(c) = 0.0:c.dt:c.duration
saved_times(c) = collect(integration_times(c))[1:c.save_every:end]

"Fig. 1: unit activity averaged over networks, and the activity of the last network."
function simulate_fig1(c, s)
    Random.seed!(c.seed)
    t = integration_times(c)
    mean_activity = zeros(length(s), length(saved_times(c)))
    example = similar(mean_activity)
    for k in 1:c.n_networks
        progress("coupled", k, c.n_networks)
        net = coupled_network(s, c.g, c.τ)
        example .= simulate!(net, (first(t), last(t)); dt=c.dt, save_every=c.save_every)
        mean_activity .+= example
    end
    return (; mean_activity=mean_activity ./ c.n_networks, example)
end

"Fig. 2, coupled network: mean unit activity, population mean and SEM, statistics of the recurrent input."
function simulate_coupled(c, s)
    Random.seed!(c.seed)
    t = integration_times(c)
    nt = length(saved_times(c))
    mean_activity = zeros(length(s), nt); pop_sum = zeros(nt); pop_sum2 = zeros(nt)
    unit_std_mean = zeros(c.n_networks); unit_std_std = zeros(c.n_networks)
    csum = csum2 = 0.0; ccount = 0
    for k in 1:c.n_networks
        progress("coupled", k, c.n_networks)
        net = coupled_network(s, c.g, c.τ)
        r, cur = simulate!(net, (first(t), last(t)); dt=c.dt, save_every=c.save_every, currents=true)
        mean_activity .+= r
        pa = vec(mean(r, dims=1)); pop_sum .+= pa; pop_sum2 .+= pa .^ 2
        stds = vec(std(cur, dims=2)); unit_std_mean[k] = mean(stds); unit_std_std[k] = std(stds)
        csum += sum(cur); csum2 += sum(abs2, cur); ccount += length(cur)
    end
    m, sem = population_stats(pop_sum, pop_sum2, c.n_networks)
    currents = (mean_unit_std=mean(unit_std_mean), mean_std_across_units=mean(unit_std_std),
                total_std=sqrt(max((csum2 - csum^2 / ccount) / (ccount - 1), 0.0)))
    return (; mean_activity=mean_activity ./ c.n_networks, population_mean=m, population_sem=sem, currents)
end

"Fig. 2, decoupled network driven by OU inputs: mean unit activity, population mean and SEM."
function simulate_decoupled(c, s)
    rng = MersenneTwister(c.seed_decoupled)
    t = integration_times(c)
    nt = length(saved_times(c))
    mean_activity = zeros(length(s), nt); pop_sum = zeros(nt); pop_sum2 = zeros(nt)
    for k in 1:c.n_networks
        progress("decoupled", k, c.n_networks)
        x = ou_process(length(t), length(s), c.dt, c.τ_ou, c.σ_ou; rng)
        net = decoupled_network(s, c.τ, rng)
        r = simulate!(net, (first(t), last(t)); dt=c.dt, input=x, save_every=c.save_every)
        mean_activity .+= r
        pa = vec(mean(r, dims=1)); pop_sum .+= pa; pop_sum2 .+= pa .^ 2
    end
    m, sem = population_stats(pop_sum, pop_sum2, c.n_networks)
    return (; mean_activity=mean_activity ./ c.n_networks, population_mean=m, population_sem=sem)
end

function population_stats(pop_sum, pop_sum2, n)
    v = max.((pop_sum2 .- pop_sum .^ 2 ./ n) ./ (n - 1), 0.0)
    return pop_sum ./ n, sqrt.(v) ./ sqrt(n)
end

progress(name, k, n) = (k == 1 || k == n || k % max(1, n ÷ 10) == 0) && println("  $name network $k / $n")

exponential_decay(t, p) = p[1] .* exp.(-t ./ p[2])
unit_decay(t, p) = exp.(-t ./ p[1])

"""
    fit_unit_timescales(mean_activity, s, t, dt)

τ of each unit: exp(-t/τ) for s < 1, A exp(-t/τ) for s >= 1, on the positive samples of
t[1:i1], i1 = t[end]/(2dt) + 1. This is the whole run when saving every 2 steps (Fig. 1) and its first
half when saving every step (Fig. 2), as in the published analysis.
"""
function fit_unit_timescales(mean_activity, s, t, dt)
    i0, i1 = Int(first(t) / (2 * dt)) + 1, Int(last(t) / (2 * dt)) + 1
    x = t[i0:i1]
    τ = zeros(length(s))
    for u in eachindex(s)
        a = mean_activity[u, i0:i1]; pos = a .> 0.0
        τ[u] = s[u] < 1.0 ? curve_fit(unit_decay, x[pos], a[pos], [1.0]).param[1] :
                            curve_fit(exponential_decay, x[pos], a[pos], [1.0, 1.0]).param[2]
    end
    return τ
end

power_law(t, p) = p[1] .* t .^ p[2]

"Power-law fit of the population activity on 0.07-0.9 s (Fig. 1f); `t` starts at 0."
function fit_power_law(t, activity)
    step = t[2] - t[1]
    i0 = clamp(Int(round(0.07 / step)), 1, length(t))
    i1 = clamp(Int(round(0.9 / step)), i0, length(t))
    a, x = activity[i0:i1], t[i0:i1]; pos = a .> 0.0
    return curve_fit(power_law, x[pos], a[pos], [0.5, -0.5]).param
end

r_squared(y, yfit) = 1 - sum(abs2, y .- yfit) / sum(abs2, y .- mean(y))
