# Asymptotic regime of the autonomous network τ0 dr/dt = -r + tanh(J r) (Supplementary Fig. S1).
#
#   regime      networks of the paper (g = 0.4, r = 1 at t = 0, no input, no noise): fixed-point convergence, then the
#               largest Lyapunov exponent and the Jacobian at the fixed point
#   example     activity of one of them from r = 1 to its fixed point, at log-spaced times
#   lyapunov_g  the same networks for g around 0.4: the two largest Lyapunov exponents after the convergence, and the
#               full-state recurrence of the trajectories without a fixed point (periodic orbit or chaos)
#
# Lyapunov exponents: tangent-vector renormalization (Benettin et al., Meccanica 15, 21-30, 1980). Every network has its
# own seed and BLAS runs on one thread: the results do not depend on the number of Julia threads.
# Requires src/project.jl, src/reservoir.jl and src/timescales.jl.

using LinearAlgebra, Random, Statistics, Printf, DelimitedFiles

const REGIME = (N=FIG1.n_units, g=FIG1.g, τ=FIG1.τ, dt=FIG1.dt, nets=100, seed=20001, max_duration=10_000.0,
                tolerance=1e-7, hold=20.0, renormalization=0.05, alignment=100.0, measurement=300.0, block=5.0,
                residual_every=1.0, n_bootstrap=10_000, bootstrap_seed=91_337, tangent_offset=1_000_000)
const LYAPUNOV_G = (g_values=[0.38, 0.39, 0.4, 0.41, 0.42], nets=100, seed=20001, max_duration=10_000.0, alignment=100.0,
                    measurement=300.0, block=10.0, save_every=0.01, max_lag=10.0, min_period=0.05,
                    recurrence_tolerance=0.05, stride=20, tangent_offset=2_000_000)
const EXAMPLE = (realization=58, ncols=300, span=3.0)  # network with the median convergence time (218 s), up to 3 times it
smoke_regime(c) = merge(c, (nets=4, max_duration=200.0))                      # quick pipeline test
smoke_lyapunov_g(c) = merge(c, (g_values=[0.39, 0.41], nets=2, max_duration=60.0, alignment=10.0, measurement=20.0))
smoke_example(e) = merge(e, (realization=3, ncols=100))

# ------------------------------------------------------------------ dynamics
"Euler steps from `r` until ρ = |tanh(Jr) - r|/√N stays below `tolerance` for `hold` s or `max_duration` is reached.
Saves ρ every `residual_every` s and the state r at the (sorted) steps `record`."
function converge!(r, J; τ, dt, max_duration, tolerance, hold, residual_every=1.0, record=Int[])
    max_steps, hold_steps, save_steps = round(Int, max_duration / dt), round(Int, hold / dt), round(Int, residual_every / dt)
    α = dt / τ
    field, activated, drift = similar(r), similar(r), similar(r)
    times, residuals = Float64[], Float64[]
    below, rho = 0, Inf
    states, next = zeros(length(r), length(record)), 1
    for step in 0:max_steps
        next <= length(record) && record[next] == step && (states[:, next] .= r; next += 1)
        mul!(field, J, r)
        @. activated = tanh(field)
        @. drift = activated - r
        rho = norm(drift) / sqrt(length(r))
        step % save_steps == 0 && (push!(times, step * dt); push!(residuals, rho))
        below = rho < tolerance ? below + 1 : 0
        if below >= hold_steps + 1
            times[end] == step * dt || (push!(times, step * dt); push!(residuals, rho))
            return (; converged=true, time=(step - below + 1) * dt, confirmation_time=step * dt, residual=rho,
                    times, residuals, states=states[:, 1:next-1])
        end
        step == max_steps && break
        @. r = r * (1.0 - α) + α * activated
    end
    return (; converged=false, time=NaN, confirmation_time=max_duration, residual=rho, times, residuals,
            states=states[:, 1:next-1])
end

"Largest Lyapunov exponent (1/s) from `r`: `alignment` s of tangent alignment, then `measurement` s in blocks of
`block` s. Also the residual ρ every `residual_every` s and whether it stayed below `tolerance`. Updates `r`."
function lyapunov1!(r, J, rng; τ, dt, alignment, measurement, renormalization, block, residual_every=1.0, tolerance=1e-7)
    alignment_steps, measurement_steps = round(Int, alignment / dt), round(Int, measurement / dt)
    renormalization_steps, block_steps = round(Int, renormalization / dt), round(Int, block / dt)
    save_steps = round(Int, residual_every / dt)
    alignment_steps % renormalization_steps == 0 && measurement_steps % block_steps == 0 &&
        block_steps % renormalization_steps == 0 || error("windows must be multiples of the renormalization interval")
    tangent = randn(rng, length(r))
    tangent ./= norm(tangent)
    next_r, next_tangent, field, tangent_field, activated = similar(r), similar(r), similar(r), similar(r), similar(r)
    α = dt / τ
    total, block_sum = 0.0, 0.0
    block_end, block_log_growth = Float64[], Float64[]
    times, residuals = Float64[], Float64[]
    max_residual, below = 0.0, true
    for step in 1:alignment_steps+measurement_steps
        mul!(field, J, r)
        @. activated = tanh(field)
        mul!(tangent_field, J, tangent)
        @. next_r = activated - r
        rho = norm(next_r) / sqrt(length(r))
        max_residual = max(max_residual, rho)
        below &= rho < tolerance
        step > 1 && (step - 1) % save_steps == 0 && (push!(times, (step - 1) * dt); push!(residuals, rho))
        @. next_r = r * (1.0 - α) + α * activated
        @. next_tangent = tangent + α * ((1.0 - activated^2) * tangent_field - tangent)
        r, next_r = next_r, r
        tangent, next_tangent = next_tangent, tangent
        if step % renormalization_steps == 0
            growth = norm(tangent)
            isfinite(growth) && growth > 0.0 || error("invalid tangent-vector norm")
            tangent ./= growth
            if step > alignment_steps
                total += log(growth); block_sum += log(growth)
                if (step - alignment_steps) % block_steps == 0
                    push!(block_end, (step - alignment_steps) * dt); push!(block_log_growth, block_sum)
                    block_sum = 0.0
                end
            end
        end
    end
    return (; exponent=total / measurement, state=r, block_end, block_log_growth, block_exponent=block_log_growth ./ block,
            times, residuals, max_residual, below)
end

"Modified Gram-Schmidt of two tangent vectors; returns their growth factors."
function orthonormalize!(v)
    v1, v2 = view(v, :, 1), view(v, :, 2)
    g1 = norm(v1)
    isfinite(g1) && g1 > 0 || error("invalid first tangent-vector norm")
    v1 ./= g1
    v2 .-= dot(v1, v2) .* v1
    g2 = norm(v2)
    isfinite(g2) && g2 > 0 || error("invalid second tangent-vector norm")
    v2 ./= g2
    return g1, g2
end

"Autocorrelation of a signal up to `max_lag` samples."
function autocorrelation(x, max_lag)
    c = x .- mean(x)
    ac = zeros(max_lag + 1)
    for lag in 0:max_lag
        a, b = view(c, 1:length(c)-lag), view(c, 1+lag:length(c))
        scale = sqrt(sum(abs2, a) * sum(abs2, b))
        ac[lag+1] = scale > 0 ? dot(a, b) / scale : 0.0
    end
    return ac
end

"First autocorrelation peak ≥ 0.5 after `min_period`, otherwise the highest peak (in samples)."
function autocorrelation_period(ac, save_every, min_period)
    first_lag = max(2, round(Int, min_period / save_every))
    peaks = [lag for lag in first_lag:length(ac)-2 if ac[lag+1] > ac[lag] && ac[lag+1] >= ac[lag+2]]
    isempty(peaks) && return first_lag
    strong = findfirst(lag -> ac[lag+1] >= 0.5, peaks)
    return strong === nothing ? peaks[argmax(ac[peaks.+1])] : peaks[strong]
end

"RMS distance between saved states `lag` columns apart (every `stride`-th column)."
function recurrence_distance(states, lag, stride)
    n = size(states, 2) - lag
    n > 0 || return Inf
    columns = 1:stride:n
    d = 0.0
    @inbounds for k in columns, i in axes(states, 1)
        d += (states[i, k+lag] - states[i, k])^2
    end
    return sqrt(d / (size(states, 1) * length(columns)))
end

"RMS temporal fluctuation of the saved states."
function temporal_fluctuation(states)
    f = 0.0
    for row in eachrow(states)
        m = mean(row)
        f += sum(x -> (x - m)^2, row)
    end
    return sqrt(f / length(states))
end

"Best full-state recurrence near `lag` and its first six harmonics: period (s), distance, relative distance, fluctuation."
function best_recurrence(states, lag, save_every, stride)
    best_lag, best = lag, Inf
    for h in 1:6
        h * lag >= size(states, 2) && break
        for l in max(1, h*lag-3):min(size(states, 2)-1, h*lag+3)
            d = recurrence_distance(states, l, stride)
            d < best && ((best_lag, best) = (l, d))
        end
    end
    fluctuation = temporal_fluctuation(states)
    return best_lag * save_every, best, fluctuation > 0 ? best / fluctuation : Inf, fluctuation
end

"Mean ± 1.96 SEM of the block exponents."
block_interval(v) = (h = length(v) > 1 ? 1.96 * std(v) / sqrt(length(v)) : NaN; (mean(v) - h, mean(v) + h))

"Class of a trajectory without a fixed point from the 95% intervals of λ1, λ2 and the relative recurrence."
function classify(λ1, λ2, recurrence; tolerance)
    λ1[1] > 0 && return "chaotic_candidate"
    λ1[1] <= 0 && λ2[2] < 0 && recurrence < tolerance && return "periodic_candidate"
    return λ1[2] < 0 ? "nonchaotic_nonperiodic" : "uncertain"
end

"Two Lyapunov exponents from `r0` (blocks of `block` s) and the recurrence of the states saved every `save_every` s."
function lyapunov2(r0, J, rng; τ, dt, alignment, measurement, renormalization=0.05, block, save_every, max_lag, min_period,
                   recurrence_tolerance, stride)
    alignment_steps, measurement_steps = round(Int, alignment / dt), round(Int, measurement / dt)
    renormalization_steps, block_steps, save_steps = round(Int, renormalization / dt), round(Int, block / dt), round(Int, save_every / dt)
    alignment_steps % renormalization_steps == 0 && measurement_steps % block_steps == 0 &&
        block_steps % renormalization_steps == 0 && save_steps > 0 || error("windows must be multiples of the renormalization interval")
    N = length(r0)
    r = copy(r0)
    tangent = randn(rng, N, 2)
    orthonormalize!(tangent)
    field, activated, drift, next_r, derivative = similar(r), similar(r), similar(r), similar(r), similar(r)
    tangent_field, next_tangent = similar(tangent), similar(tangent)
    α = dt / τ
    nsave = fld(measurement_steps - 1, save_steps) + 1
    states, residual, population = zeros(N, nsave), zeros(nsave), zeros(nsave)
    col = 1
    total, block_sum = zeros(2), zeros(2)
    block_end, blocks1, blocks2 = Float64[], Float64[], Float64[]
    for step in 1:alignment_steps+measurement_steps
        mul!(field, J, r)
        @. activated = tanh(field)
        @. derivative = 1.0 - activated^2
        m = step - alignment_steps - 1
        if m >= 0 && m % save_steps == 0
            states[:, col] .= r
            @. drift = activated - r
            residual[col] = norm(drift) / sqrt(N)
            population[col] = mean(r)
            col += 1
        end
        mul!(tangent_field, J, tangent)
        @. next_tangent = tangent + α * (derivative * tangent_field - tangent)
        @. next_r = r * (1.0 - α) + α * activated
        r, next_r = next_r, r
        tangent, next_tangent = next_tangent, tangent
        if step % renormalization_steps == 0
            growth = orthonormalize!(tangent)
            if step > alignment_steps
                increments = log.([growth[1], growth[2]])
                total .+= increments; block_sum .+= increments
                if (step - alignment_steps) % block_steps == 0
                    push!(block_end, (step - alignment_steps) * dt)
                    push!(blocks1, block_sum[1] / block); push!(blocks2, block_sum[2] / block)
                    block_sum .= 0.0
                end
            end
        end
    end
    λ1, λ2 = total ./ measurement
    λ1_interval, λ2_interval = block_interval(blocks1), block_interval(blocks2)
    nlag = min(round(Int, max_lag / save_every), nsave - 2)
    ac = autocorrelation(residual, nlag)
    period, recurrence, relative_recurrence, fluctuation =
        best_recurrence(states, autocorrelation_period(ac, save_every, min_period), save_every, stride)
    classification = classify(λ1_interval, λ2_interval, relative_recurrence; tolerance=recurrence_tolerance)
    return (; dt, λ1, λ2, λ1_interval, λ2_interval, period, recurrence, relative_recurrence, fluctuation, classification,
            times=collect(0:nsave-1) .* save_every, residual, population, lags=collect(0:nlag) .* save_every, ac,
            block_end, blocks1, blocks2)
end

"Largest real part of the eigenvalues of the Jacobian at the fixed point `r` (1/s), and the exponent of the Euler map."
function fixed_point_exponents(J, r; τ, dt)
    A = J .* reshape(1.0 .- tanh.(J * r) .^ 2, :, 1)
    A[diagind(A)] .-= 1.0
    A ./= τ
    λ = eigvals(A)
    return maximum(real.(λ)), maximum(log.(abs.(1.0 .+ dt .* λ))) / dt
end

# ------------------------------------------------------------------ regime: the networks of the paper
"One network of the paper (seed `seed`): convergence, then the largest Lyapunov exponent and the Jacobian."
function regime_network(s, realization, seed; c=REGIME)
    Random.seed!(seed)
    net = coupled_network(s, c.g, c.τ)
    conv = converge!(net.r, net.J; c.τ, c.dt, c.max_duration, c.tolerance, c.hold, c.residual_every)
    if !conv.converged
        return (; realization, seed, converged=false, fixed_point_time=NaN, confirmation_time=conv.confirmation_time,
                final_residual=conv.residual, lyapunov=NaN, jacobian_exponent=NaN, euler_jacobian_exponent=NaN,
                local_relaxation_time=NaN, state_time=c.max_duration, final_state=copy(net.r), times=conv.times,
                residuals=conv.residuals, max_residual=NaN, below=false, block_end=Float64[], block_log_growth=Float64[],
                block_exponent=Float64[])
    end
    ly = lyapunov1!(net.r, net.J, Xoshiro(seed + c.tangent_offset); c.τ, c.dt, c.alignment, c.measurement,
                    c.renormalization, c.block, c.residual_every, c.tolerance)
    state = ly.state
    final_residual = norm(tanh.(net.J * state) .- state) / sqrt(length(state))
    jac, euler = fixed_point_exponents(net.J, state; c.τ, c.dt)
    state_time = conv.confirmation_time + c.alignment + c.measurement
    times = [conv.times; conv.confirmation_time .+ ly.times; state_time]
    residuals = [conv.residuals; ly.residuals; final_residual]
    return (; realization, seed, converged=true, fixed_point_time=conv.time, confirmation_time=conv.confirmation_time,
            final_residual, lyapunov=ly.exponent, jacobian_exponent=jac, euler_jacobian_exponent=euler,
            local_relaxation_time=jac < 0.0 ? -1.0 / jac : Inf, state_time, final_state=copy(state), times, residuals,
            max_residual=max(ly.max_residual, final_residual), below=ly.below && final_residual < c.tolerance,
            ly.block_end, ly.block_log_growth, ly.block_exponent)
end

"`c.nets` networks (seeds c.seed, c.seed + 1, ...) in parallel threads; results in `dir`."
function run_regime(dir, s; c=REGIME)
    BLAS.set_num_threads(1)
    println("Regime: $(c.nets) networks, N = $(c.N), g = $(c.g), up to $(c.max_duration) s, $(Threads.nthreads()) thread(s)")
    res = Vector{Any}(undef, c.nets)
    Threads.@threads :dynamic for k in 1:c.nets
        res[k] = regime_network(s, k, c.seed + k - 1; c)
        x = res[k]
        x.converged ? @printf("  network %3d: fixed point at %.0f s, λ1 = %.3g 1/s\n", k, x.fixed_point_time, x.lyapunov) :
                      @printf("  network %3d: no fixed point by %.0f s\n", k, c.max_duration)
    end
    write_regime(dir, res, s; c)
end

function write_regime(dir, res, s; c)
    mkpath(dir)
    open(joinpath(dir, "network_results.tsv"), "w") do io
        println(io, join(["realization", "seed", "converged", "fixed_point_time_s", "confirmation_time_s", "final_residual",
                          "largest_lyapunov_per_s", "jacobian_max_real_eigenvalue_per_s", "euler_jacobian_exponent_per_s",
                          "local_relaxation_time_s", "maximum_post_convergence_residual",
                          "remained_below_tolerance_post_convergence"], '\t'))
        for x in res
            @printf(io, "%d\t%d\t%s\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%s\n", x.realization, x.seed,
                    x.converged, x.fixed_point_time, x.confirmation_time, x.final_residual, x.lyapunov, x.jacobian_exponent,
                    x.euler_jacobian_exponent, x.local_relaxation_time, x.max_residual, x.below)
        end
    end
    open(joinpath(dir, "residual_history.tsv"), "w") do io
        println(io, "realization\tseed\ttime_s\tfixed_point_residual")
        for x in res, (t, rho) in zip(x.times, x.residuals)
            @printf(io, "%d\t%d\t%.12g\t%.12g\n", x.realization, x.seed, t, rho)
        end
    end
    open(joinpath(dir, "final_states.tsv"), "w") do io
        println(io, "realization\tseed\tconverged\tstate_time_s\tunit\tself_coupling\tactivity")
        for x in res, i in eachindex(x.final_state)
            @printf(io, "%d\t%d\t%s\t%.12g\t%d\t%.12g\t%.12g\n", x.realization, x.seed, x.converged, x.state_time, i, s[i],
                    x.final_state[i])
        end
    end
    open(joinpath(dir, "lyapunov_blocks.tsv"), "w") do io
        println(io, "realization\tseed\tmeasurement_start_s\tmeasurement_end_s\tsimulation_start_s\tsimulation_end_s\tlog_growth\tlyapunov_per_s")
        for x in res, b in eachindex(x.block_end)
            e = x.confirmation_time + c.alignment + x.block_end[b]
            @printf(io, "%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n", x.realization, x.seed, x.block_end[b] - c.block,
                    x.block_end[b], e - c.block, e, x.block_log_growth[b], x.block_exponent[b])
        end
    end
    write_regime_summary(joinpath(dir, "summary.txt"), res; c)
end

"Bootstrap 95% interval of the mean."
function bootstrap_ci(v, rng, n)
    means = [mean(v[rand(rng, 1:length(v), length(v))]) for _ in 1:n]
    return quantile(means, 0.025), quantile(means, 0.975)
end

function write_regime_summary(path, res; c)
    conv = filter(x -> x.converged, res)
    T, λ = [x.fixed_point_time for x in conv], [x.lyapunov for x in conv]
    open(path, "w") do io
        for (k, v) in (("model", "tau0 dr/dt = -r + tanh(Jr)"), ("n_units", c.N), ("n_networks", c.nets),
                       ("n_converged", length(conv)), ("n_not_converged", c.nets - length(conv)),
                       ("fraction_converged", length(conv) / c.nets), ("first_seed", c.seed), ("recurrent_gain", c.g),
                       ("tau0_s", c.τ), ("dt_s", c.dt), ("maximum_duration_s", c.max_duration),
                       ("residual_tolerance", c.tolerance), ("convergence_hold_s", c.hold),
                       ("post_convergence_duration_s", c.alignment + c.measurement), ("lyapunov_alignment_s", c.alignment),
                       ("lyapunov_measurement_s", c.measurement), ("renormalization_interval_s", c.renormalization),
                       ("lyapunov_block_duration_s", c.block), ("residual_save_interval_s", c.residual_every),
                       ("initial_activity", "1 for every unit"), ("external_input", "zero"), ("ongoing_noise", "zero"),
                       ("self_couplings", relpath(self_couplings_file(), PROJECT_ROOT)))
            println(io, "$k = $v")
        end
        isempty(conv) && return
        rng = Xoshiro(c.bootstrap_seed)
        Tci, λci = bootstrap_ci(T, rng, c.n_bootstrap), bootstrap_ci(λ, rng, c.n_bootstrap)
        nb = count(x -> x.below, conv)
        for (k, v) in (("mean_fixed_point_time_s", mean(T)), ("mean_fixed_point_time_95ci_s", "$(Tci[1]), $(Tci[2])"),
                       ("median_fixed_point_time_s", median(T)),
                       ("fixed_point_time_iqr_s", "$(quantile(T, 0.25)), $(quantile(T, 0.75))"),
                       ("mean_largest_lyapunov_per_s", mean(λ)), ("mean_largest_lyapunov_95ci_per_s", "$(λci[1]), $(λci[2])"),
                       ("median_largest_lyapunov_per_s", median(λ)),
                       ("largest_lyapunov_iqr_per_s", "$(quantile(λ, 0.25)), $(quantile(λ, 0.75))"),
                       ("fraction_positive_lyapunov", count(>(0.0), λ) / length(λ)),
                       ("n_remained_below_tolerance_post_convergence", nb),
                       ("fraction_remained_below_tolerance_post_convergence", nb / length(conv)),
                       ("mean_abs_benettin_minus_euler_jacobian_per_s", mean(abs(x.lyapunov - x.euler_jacobian_exponent) for x in conv)))
            println(io, "$k = $v")
        end
    end
end

# ------------------------------------------------------------------ lyapunov_g: two exponents vs g
"The networks of the paper (seeds c.seed, ...) for each g: convergence from r = 1, then the two largest Lyapunov
exponents from the final state (fixed point or not)."
function run_lyapunov_g(dir, s; c=LYAPUNOV_G, r=REGIME)
    BLAS.set_num_threads(1)
    jobs = [(g, k) for g in c.g_values for k in 1:c.nets]
    println("Lyapunov exponents vs g: $(length(c.g_values)) values x $(c.nets) networks on $(Threads.nthreads()) thread(s)")
    res = Vector{Any}(undef, length(jobs))
    Threads.@threads :dynamic for j in eachindex(jobs)
        g, k = jobs[j]
        seed = c.seed + k - 1
        Random.seed!(seed)
        net = coupled_network(s, g, r.τ)
        conv = converge!(net.r, net.J; r.τ, r.dt, c.max_duration, r.tolerance, r.hold)
        ly = lyapunov2(net.r, net.J, MersenneTwister(seed + c.tangent_offset); r.τ, r.dt, c.alignment, c.measurement,
                       c.block, c.save_every, c.max_lag, c.min_period, c.recurrence_tolerance, c.stride)
        res[j] = (; g, k, seed, conv, ly, class=conv.converged ? "fixed_point" : ly.classification)
        @printf("  g = %g, network %d: %s, λ = (%.3g, %.3g) 1/s\n", g, k, res[j].class, ly.λ1, ly.λ2)
    end
    mkpath(dir)
    open(joinpath(dir, "lyapunov_vs_g.tsv"), "w") do io
        println(io, join(["g", "realization", "seed", "converged", "fixed_point_time_s", "lambda1_per_s", "lambda1_ci_low",
                          "lambda1_ci_high", "lambda2_per_s", "lambda2_ci_low", "lambda2_ci_high", "classification"], '\t'))
        for x in res
            @printf(io, "%.12g\t%d\t%d\t%s\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%s\n", x.g, x.k, x.seed,
                    x.conv.converged, x.conv.time, x.ly.λ1, x.ly.λ1_interval..., x.ly.λ2, x.ly.λ2_interval..., x.class)
        end
    end
    open(joinpath(dir, "summary.txt"), "w") do io
        for (k, v) in (("n_units", r.N), ("n_networks_per_g", c.nets), ("first_seed", c.seed), ("tau0_s", r.τ), ("dt_s", r.dt),
                       ("maximum_duration_s", c.max_duration), ("lyapunov_alignment_s", c.alignment),
                       ("lyapunov_measurement_s", c.measurement), ("lyapunov_block_duration_s", c.block),
                       ("self_couplings", relpath(self_couplings_file(), PROJECT_ROOT)), ("g_values", join(c.g_values, ", ")))
            println(io, "$k = $v")
        end
        println(io)
        for g in c.g_values
            sub = [x for x in res if x.g == g]
            λ1, λ2 = [x.ly.λ1 for x in sub], [x.ly.λ2 for x in sub]
            @printf(io, "g = %g: fixed points %d/%d, chaotic candidates %d; lambda1 = %.4g ± %.4g, lambda2 = %.4g ± %.4g 1/s (mean ± std)\n",
                    g, count(x -> x.conv.converged, sub), length(sub), count(x -> x.class == "chaotic_candidate", sub),
                    mean(λ1), std(λ1), mean(λ2), std(λ2))
        end
    end
end

# ------------------------------------------------------------------ example: activity up to the fixed point
"Activity (units × times) of network `e.realization` of the paper from r = 1 to `e.span` times the time of its fixed
point, at `e.ncols` log-spaced times from dt on."
function activity_example(s; e=EXAMPLE, c=REGIME)
    BLAS.set_num_threads(1)                               # the trajectory of run_regime
    function run(hold, record)
        Random.seed!(c.seed + e.realization - 1)
        net = coupled_network(s, c.g, c.τ)
        return converge!(net.r, net.J; c.τ, c.dt, c.max_duration, c.tolerance, hold, record)
    end
    conv = run(c.hold, Int[])
    conv.converged || error("network $(e.realization) has no fixed point")
    hold = (e.span - 1) * conv.time                       # stays below the tolerance until span * time
    steps = unique(round.(Int, exp10.(range(0, log10(conv.time / c.dt + hold / c.dt), length=e.ncols))))
    A = run(hold, steps).states
    return steps[1:size(A, 2)] .* c.dt, A
end

function write_example(dir, s; e=EXAMPLE, c=REGIME)
    t, A = activity_example(s; e, c)
    mkpath(dir)
    writedlm(joinpath(dir, "example_times.tsv"), ["time_s"; t])
    writedlm(joinpath(dir, "example_activity.tsv"), round.(A, sigdigits=6), '\t')
end
