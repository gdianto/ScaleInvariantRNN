# New self-coupling vector (Methods): pilot run, fit of τ(s), N values drawn from the P(s) giving P(τ) ∝ τ^-2.
#   julia scripts/self_couplings.jl   # -> data/generated/self_couplings.txt (use it with --s=generated)
#     --seed=1  --pilot-networks=500
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, Printf, Random
include(projectdir("src", "reservoir.jl"))
include(projectdir("src", "timescales.jl"))

file = new_data_path("self_couplings.txt")
smin, smax = 0.7, 1.7
c = merge(FIG1, (n_networks=parse(Int, flagvalue("pilot-networks", "500")), seed=parse(Int, flagvalue("seed", "1"))))
pilot = collect(range(smin, smax; length=c.n_units))
println("Pilot: $(c.n_units) units with s evenly spaced in [$smin, $smax], $(c.n_networks) networks")
τ = fit_unit_timescales(simulate_fig1(c, pilot).mean_activity, pilot, saved_times(c), c.dt)
ok = isfinite.(τ) .& (τ .> 0)
fit = fit_tau(pilot[ok], τ[ok])
@printf("τ(s) = a exp(b s^c): a = %.5g, b = %.5g, c = %.5g (R² log10 τ = %.4f)\n", fit.params..., fit.r2_log)

s = sample_self_couplings(fit.params, c.n_units; smin, smax, rng=Xoshiro(c.seed))
writedlm(file, round.(s, sigdigits=5)', ',')
@printf("%d self-couplings in [%.4f, %.4f], mean %.4f -> %s\n", length(s), extrema(s)..., sum(s) / length(s), file)
