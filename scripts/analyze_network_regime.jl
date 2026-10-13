# Asymptotic regime of the autonomous heterogeneous network used in the paper.
#
# For each network realization:
#   1. start every unit at r = 1 and simulate until the fixed-point residual
#      rho = ||tanh(Jr) - r|| / sqrt(N) stays below tolerance for 20 s;
#   2. after convergence, continue for 400 s: use 100 s for tangent alignment,
#      then estimate the largest Lyapunov exponent over 300 s.
#
# The exponent is computed with tangent-vector renormalization following
# Benettin et al., Meccanica 15, 21--30 (1980), doi:10.1007/BF02128237.
# Numerical results and diagnostics are written to data/generated; no figure is produced.
#
#   julia scripts/analyze_network_regime.jl
#     --nets=100            number of independent networks
#     --seed=20001          seed of the first network
#     --max-duration=10000  maximum convergence time in seconds
#     --s=published|generated|FILE, --tag=TAG, --force: see README.md
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, LinearAlgebra, Printf, Random, Statistics
include(projectdir("src", "reservoir.jl"))
include(projectdir("src", "timescales.jl"))

# ------------------------------------------------------------------ settings
const N_NETWORKS = parse(Int, flagvalue("nets", "100"))
const FIRST_SEED = parse(Int, flagvalue("seed", "20001"))
const MAX_DURATION = parse(Float64, flagvalue("max-duration", "10000"))
const OUTPUT_NAME = "network_regime_extended_100"

const N = FIG1.n_units
const G = FIG1.g
const TAU0 = FIG1.τ
const DT = FIG1.dt

const RESIDUAL_TOLERANCE = 1e-7
const CONVERGENCE_HOLD = 20.0       # rho must stay below tolerance for this long
const LYAPUNOV_ALIGNMENT = 100.0    # extra settling and tangent-vector alignment
const LYAPUNOV_DURATION = 300.0
const RENORMALIZATION_INTERVAL = 0.05
const LYAPUNOV_BLOCK_DURATION = 5.0
const RESIDUAL_SAVE_INTERVAL = 1.0
const N_BOOTSTRAP = 10_000
const BOOTSTRAP_SEED = 91_337


"Fixed-point residual rho = ||tanh(Jr) - r|| / sqrt(N)."
function fixed_point_residual!(field, drift, J, r)
    mul!(field, J, r)
    @. drift = tanh(field) - r
    return norm(drift) / sqrt(length(r))
end


"Simulate until rho remains below tolerance for the full hold interval."
function converge_to_fixed_point!(r, J)
    max_steps = round(Int, MAX_DURATION / DT)
    hold_steps = round(Int, CONVERGENCE_HOLD / DT)
    progress_steps = round(Int, 500.0 / DT)
    save_steps = round(Int, RESIDUAL_SAVE_INTERVAL / DT)
    α = DT / TAU0

    field = similar(r)
    drift = similar(r)
    below_tolerance = 0
    rho = Inf
    saved_time = Float64[]
    saved_residual = Float64[]

    for step in 0:max_steps
        rho = fixed_point_residual!(field, drift, J, r)
        below_tolerance = rho < RESIDUAL_TOLERANCE ? below_tolerance + 1 : 0

        if step % save_steps == 0
            push!(saved_time, step * DT)
            push!(saved_residual, rho)
        end

        if below_tolerance >= hold_steps + 1
            first_step_below = step - below_tolerance + 1
            if isempty(saved_time) || saved_time[end] != step * DT
                push!(saved_time, step * DT)
                push!(saved_residual, rho)
            end
            return (; converged=true, time=first_step_below * DT,
                    confirmation_time=step * DT, residual=rho,
                    residual_time=saved_time, residual_history=saved_residual)
        end

        step == max_steps && break
        @. r = r * (1.0 - α) + α * tanh(field)

        if step > 0 && step % progress_steps == 0
            @printf("    t = %.0f s, rho = %.3e\n", step * DT, rho)
        end
    end

    return (; converged=false, time=NaN, confirmation_time=MAX_DURATION, residual=rho,
            residual_time=saved_time, residual_history=saved_residual)
end


"Largest Lyapunov exponent after convergence, in inverse seconds."
function benettin_exponent!(r, J, seed)
    alignment_steps = round(Int, LYAPUNOV_ALIGNMENT / DT)
    measurement_steps = round(Int, LYAPUNOV_DURATION / DT)
    renormalization_steps = round(Int, RENORMALIZATION_INTERVAL / DT)
    block_steps = round(Int, LYAPUNOV_BLOCK_DURATION / DT)
    save_steps = round(Int, RESIDUAL_SAVE_INTERVAL / DT)
    total_steps = alignment_steps + measurement_steps

    alignment_steps % renormalization_steps == 0 || error(
        "LYAPUNOV_ALIGNMENT must be a multiple of RENORMALIZATION_INTERVAL."
    )
    measurement_steps % renormalization_steps == 0 || error(
        "LYAPUNOV_DURATION must be a multiple of RENORMALIZATION_INTERVAL."
    )
    measurement_steps % block_steps == 0 || error(
        "LYAPUNOV_DURATION must be a multiple of LYAPUNOV_BLOCK_DURATION."
    )
    block_steps % renormalization_steps == 0 || error(
        "LYAPUNOV_BLOCK_DURATION must be a multiple of RENORMALIZATION_INTERVAL."
    )

    rng = Xoshiro(seed + 1_000_000)
    tangent = randn(rng, length(r))
    tangent ./= norm(tangent)

    next_r = similar(r)
    next_tangent = similar(tangent)
    field = similar(r)
    tangent_field = similar(tangent)
    activated = similar(r)
    α = DT / TAU0
    sum_log_growth = 0.0
    block_log_growth = 0.0
    block_end_time = Float64[]
    block_log_growth_values = Float64[]
    block_exponents = Float64[]
    residual_time = Float64[]
    residual_history = Float64[]
    maximum_residual = 0.0
    remained_below_tolerance = true

    for step in 1:total_steps
        mul!(field, J, r)
        @. activated = tanh(field)
        mul!(tangent_field, J, tangent)
        @. next_r = activated - r
        rho = norm(next_r) / sqrt(length(r))
        maximum_residual = max(maximum_residual, rho)
        remained_below_tolerance &= rho < RESIDUAL_TOLERANCE

        elapsed_steps = step - 1
        if elapsed_steps > 0 && elapsed_steps % save_steps == 0
            push!(residual_time, elapsed_steps * DT)
            push!(residual_history, rho)
        end

        @. next_r = r * (1.0 - α) + α * activated
        @. next_tangent = tangent + α * ((1.0 - activated^2) * tangent_field - tangent)
        r, next_r = next_r, r
        tangent, next_tangent = next_tangent, tangent

        if step % renormalization_steps == 0
            growth = norm(tangent)
            isfinite(growth) && growth > 0.0 || error("Invalid tangent-vector norm.")
            tangent ./= growth
            if step > alignment_steps
                increment = log(growth)
                sum_log_growth += increment
                block_log_growth += increment
                if (step - alignment_steps) % block_steps == 0
                    push!(block_end_time, (step - alignment_steps) * DT)
                    push!(block_log_growth_values, block_log_growth)
                    push!(block_exponents, block_log_growth / LYAPUNOV_BLOCK_DURATION)
                    block_log_growth = 0.0
                end
            end
        end
    end

    return (; exponent=sum_log_growth / LYAPUNOV_DURATION, state=r,
            block_end_time, block_log_growth=block_log_growth_values,
            block_exponent=block_exponents, residual_time, residual_history,
            maximum_residual, remained_below_tolerance)
end


"Continuous-time and Euler-map exponents of the final fixed-point Jacobian."
function fixed_point_exponents(J, r)
    field = J * r
    derivative = @. 1.0 - tanh(field)^2
    A = J .* reshape(derivative, :, 1)
    A[diagind(A)] .-= 1.0
    A ./= TAU0
    eigenvalues = eigvals(A)
    continuous = maximum(real.(eigenvalues))
    euler = maximum(log.(abs.(1.0 .+ DT .* eigenvalues))) / DT
    return continuous, euler
end


function analyze_network(s, realization, seed)
    Random.seed!(seed)
    net = coupled_network(s, G, TAU0)
    convergence = converge_to_fixed_point!(net.r, net.J)

    if !convergence.converged
        return (; realization, seed, converged=false,
                fixed_point_time=NaN, confirmation_time=convergence.confirmation_time,
                final_residual=convergence.residual, lyapunov=NaN,
                jacobian_exponent=NaN, euler_jacobian_exponent=NaN,
                local_relaxation_time=NaN, state_time=MAX_DURATION,
                final_state=copy(net.r), residual_time=convergence.residual_time,
                residual_history=convergence.residual_history,
                maximum_post_convergence_residual=NaN,
                remained_below_tolerance=false,
                lyapunov_block_end=Float64[], lyapunov_block_log_growth=Float64[],
                lyapunov_block_exponent=Float64[])
    end

    lyapunov_result = benettin_exponent!(net.r, net.J, seed)
    lyapunov = lyapunov_result.exponent
    final_state = lyapunov_result.state
    final_residual = fixed_point_residual!(similar(final_state), similar(final_state), net.J, final_state)
    maximum_post_convergence_residual = max(lyapunov_result.maximum_residual, final_residual)
    remained_below_tolerance = lyapunov_result.remained_below_tolerance &&
                               final_residual < RESIDUAL_TOLERANCE
    jacobian_exponent, euler_jacobian_exponent = fixed_point_exponents(net.J, final_state)
    local_relaxation_time = jacobian_exponent < 0.0 ? -1.0 / jacobian_exponent : Inf
    state_time = convergence.confirmation_time + LYAPUNOV_ALIGNMENT + LYAPUNOV_DURATION
    append!(convergence.residual_time, convergence.confirmation_time .+ lyapunov_result.residual_time)
    append!(convergence.residual_history, lyapunov_result.residual_history)
    push!(convergence.residual_time, state_time)
    push!(convergence.residual_history, final_residual)

    return (; realization, seed, converged=true,
            fixed_point_time=convergence.time,
            confirmation_time=convergence.confirmation_time,
            final_residual, lyapunov, jacobian_exponent,
            euler_jacobian_exponent, local_relaxation_time, state_time,
            final_state=copy(final_state), residual_time=convergence.residual_time,
            residual_history=convergence.residual_history,
            maximum_post_convergence_residual, remained_below_tolerance,
            lyapunov_block_end=lyapunov_result.block_end_time,
            lyapunov_block_log_growth=lyapunov_result.block_log_growth,
            lyapunov_block_exponent=lyapunov_result.block_exponent)
end


"Bootstrap confidence interval for the mean across network realizations."
function bootstrap_mean_ci(values, rng)
    n = length(values)
    means = zeros(N_BOOTSTRAP)
    for b in eachindex(means)
        means[b] = mean(values[rand(rng, 1:n, n)])
    end
    return quantile(means, 0.025), quantile(means, 0.975)
end


function write_results(path, results)
    open(path, "w") do io
        println(io, join([
            "realization", "seed", "converged", "fixed_point_time_s",
            "confirmation_time_s", "final_residual", "largest_lyapunov_per_s",
            "jacobian_max_real_eigenvalue_per_s", "euler_jacobian_exponent_per_s",
            "local_relaxation_time_s", "maximum_post_convergence_residual",
            "remained_below_tolerance_post_convergence",
        ], '\t'))
        for x in results
            @printf(io, "%d\t%d\t%s\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%s\n",
                    x.realization, x.seed, x.converged, x.fixed_point_time,
                    x.confirmation_time, x.final_residual, x.lyapunov,
                    x.jacobian_exponent, x.euler_jacobian_exponent,
                    x.local_relaxation_time, x.maximum_post_convergence_residual,
                    x.remained_below_tolerance)
        end
    end
end


function initialize_diagnostic_files(residual_path, states_path, blocks_path)
    open(residual_path, "w") do io
        println(io, "realization\tseed\ttime_s\tfixed_point_residual")
    end
    open(states_path, "w") do io
        println(io, "realization\tseed\tconverged\tstate_time_s\tunit\tself_coupling\tactivity")
    end
    open(blocks_path, "w") do io
        println(io, "realization\tseed\tmeasurement_start_s\tmeasurement_end_s\tsimulation_start_s\tsimulation_end_s\tlog_growth\tlyapunov_per_s")
    end
end


"Append the lightweight diagnostics of one completed network."
function append_diagnostics(residual_path, states_path, blocks_path, result, s)
    open(residual_path, "a") do io
        for (time, rho) in zip(result.residual_time, result.residual_history)
            @printf(io, "%d\t%d\t%.12g\t%.12g\n",
                    result.realization, result.seed, time, rho)
        end
    end

    open(states_path, "a") do io
        for unit in eachindex(result.final_state)
            @printf(io, "%d\t%d\t%s\t%.12g\t%d\t%.12g\t%.12g\n",
                    result.realization, result.seed, result.converged,
                    result.state_time, unit, s[unit], result.final_state[unit])
        end
    end

    open(blocks_path, "a") do io
        for block in eachindex(result.lyapunov_block_end)
            block_end = result.lyapunov_block_end[block]
            simulation_end = result.confirmation_time + LYAPUNOV_ALIGNMENT + block_end
            @printf(io, "%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",
                    result.realization, result.seed,
                    block_end - LYAPUNOV_BLOCK_DURATION, block_end,
                    simulation_end - LYAPUNOV_BLOCK_DURATION, simulation_end,
                    result.lyapunov_block_log_growth[block],
                    result.lyapunov_block_exponent[block])
        end
    end
end


function write_summary(path, results, self_coupling_path)
    converged = filter(x -> x.converged, results)
    times = [x.fixed_point_time for x in converged]
    exponents = [x.lyapunov for x in converged]
    rng = Xoshiro(BOOTSTRAP_SEED)

    open(path, "w") do io
        println(io, "model = tau0 dr/dt = -r + tanh(Jr)")
        println(io, "n_units = $N")
        println(io, "n_networks = $N_NETWORKS")
        println(io, "n_converged = $(length(converged))")
        println(io, "n_not_converged = $(N_NETWORKS - length(converged))")
        println(io, "fraction_converged = $(length(converged) / N_NETWORKS)")
        println(io, "first_seed = $FIRST_SEED")
        println(io, "recurrent_gain = $G")
        println(io, "tau0_s = $TAU0")
        println(io, "dt_s = $DT")
        println(io, "maximum_duration_s = $MAX_DURATION")
        println(io, "residual_tolerance = $RESIDUAL_TOLERANCE")
        println(io, "convergence_hold_s = $CONVERGENCE_HOLD")
        println(io, "post_convergence_duration_s = $(LYAPUNOV_ALIGNMENT + LYAPUNOV_DURATION)")
        println(io, "lyapunov_alignment_s = $LYAPUNOV_ALIGNMENT")
        println(io, "lyapunov_measurement_s = $LYAPUNOV_DURATION")
        println(io, "renormalization_interval_s = $RENORMALIZATION_INTERVAL")
        println(io, "lyapunov_block_duration_s = $LYAPUNOV_BLOCK_DURATION")
        println(io, "residual_save_interval_s = $RESIDUAL_SAVE_INTERVAL")
        println(io, "initial_activity = 1 for every unit")
        println(io, "external_input = zero")
        println(io, "ongoing_noise = zero")
        println(io, "self_couplings = $(relpath(self_coupling_path, PROJECT_ROOT))")

        isempty(converged) && return

        time_ci = bootstrap_mean_ci(times, rng)
        exponent_ci = bootstrap_mean_ci(exponents, rng)
        println(io, "mean_fixed_point_time_s = $(mean(times))")
        println(io, "mean_fixed_point_time_95ci_s = $(time_ci[1]), $(time_ci[2])")
        println(io, "median_fixed_point_time_s = $(median(times))")
        println(io, "fixed_point_time_iqr_s = $(quantile(times, 0.25)), $(quantile(times, 0.75))")
        println(io, "mean_largest_lyapunov_per_s = $(mean(exponents))")
        println(io, "mean_largest_lyapunov_95ci_per_s = $(exponent_ci[1]), $(exponent_ci[2])")
        println(io, "median_largest_lyapunov_per_s = $(median(exponents))")
        println(io, "largest_lyapunov_iqr_per_s = $(quantile(exponents, 0.25)), $(quantile(exponents, 0.75))")
        println(io, "fraction_positive_lyapunov = $(count(>(0.0), exponents) / length(exponents))")
        println(io, "n_remained_below_tolerance_post_convergence = ",
                count(x -> x.remained_below_tolerance, converged))
        println(io, "fraction_remained_below_tolerance_post_convergence = ",
                count(x -> x.remained_below_tolerance, converged) / length(converged))
        println(io, "mean_abs_benettin_minus_euler_jacobian_per_s = ",
                mean(abs(x.lyapunov - x.euler_jacobian_exponent) for x in converged))
    end
end


function main()
    N_NETWORKS > 0 || error("--nets must be positive.")
    MAX_DURATION > CONVERGENCE_HOLD || error("--max-duration must exceed the convergence hold interval.")
    BLAS.set_num_threads(1)

    self_coupling_path = self_couplings_file()
    s = vec(readdlm(self_coupling_path, ',', Float64))
    length(s) == N || error("Expected $N self-couplings, found $(length(s)).")
    all(isfinite, s) || error("Self-couplings contain non-finite values.")

    output_dir = new_data_dir(OUTPUT_NAME)
    results_path = joinpath(output_dir, "network_results.tsv")
    summary_path = joinpath(output_dir, "summary.txt")
    residual_path = joinpath(output_dir, "residual_history.tsv")
    states_path = joinpath(output_dir, "final_states.tsv")
    blocks_path = joinpath(output_dir, "lyapunov_blocks.tsv")
    results = NamedTuple[]
    initialize_diagnostic_files(residual_path, states_path, blocks_path)

    println("Network-regime analysis: $N_NETWORKS networks, N = $N")
    println("Fixed point: rho < $RESIDUAL_TOLERANCE for $CONVERGENCE_HOLD s")
    println("After convergence: $LYAPUNOV_ALIGNMENT s tangent-vector alignment + " *
            "$LYAPUNOV_DURATION s Lyapunov measurement")
    println("Output: $output_dir")

    for realization in 1:N_NETWORKS
        seed = FIRST_SEED + realization - 1
        println("\nNetwork $realization / $N_NETWORKS (seed $seed)")
        result = analyze_network(s, realization, seed)
        push!(results, result)
        write_results(results_path, results)       # preserve completed networks if the run is interrupted
        append_diagnostics(residual_path, states_path, blocks_path, result, s)

        if result.converged
            @printf("  T_fixed = %.3f s, lambda_max = %.6g 1/s, rho_final = %.3e, remained below tolerance = %s\n",
                    result.fixed_point_time, result.lyapunov, result.final_residual,
                    result.remained_below_tolerance)
        else
            @printf("  no fixed point by %.1f s, rho = %.3e\n", MAX_DURATION, result.final_residual)
        end
    end

    write_summary(summary_path, results, self_coupling_path)
    println("\nCompleted.\n  Results: $results_path\n  Summary: $summary_path")
    println("  Residuals: $residual_path\n  Final states: $states_path\n  Lyapunov blocks: $blocks_path")
end


main()
