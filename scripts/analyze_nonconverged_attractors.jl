# Persistent dynamics of networks that did not reach a fixed point by 10,000 s.
# Run analyze_network_regime.jl first. This script starts from its saved final
# states, so the long simulations are not repeated.
#
# For each network and integration step:
#   - estimate the two largest Lyapunov exponents with two tangent vectors;
#   - save block exponents to assess finite-time convergence;
#   - estimate a period from the residual autocorrelation;
#   - test full-state recurrence at that period and its harmonics.
#
# Stable periodic candidates have lambda1 compatible with zero, lambda2 < 0,
# and relative full-state recurrence error below 5%. The integration-step check
# distinguishes persistent dynamics from an Euler-discretization artifact.
# Lyapunov exponents follow Benettin et al., Meccanica 15, 21--30 (1980),
# doi:10.1007/BF02128237.
#
# Run directly from VS Code, or from a terminal:
#   julia scripts/analyze_nonconverged_attractors.jl
# Optional: --alignment=100 --measurement=300 --dts=0.001,0.0005
#           --s=published|generated|FILE, --tag=TAG, --force: see README.md
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using DelimitedFiles, LinearAlgebra, Printf, Random, SparseArrays, Statistics

const SOURCE_NAME = "network_regime_extended_100"
const OUTPUT_NAME = "nonconverged_attractors"

const N = 500
const G = 0.4
const TAU0 = 0.01
const ALIGNMENT = parse(Float64, flagvalue("alignment", "100"))
const MEASUREMENT = parse(Float64, flagvalue("measurement", "300"))
const DT_VALUES = parse.(Float64, split(flagvalue("dts", "0.001,0.0005"), ','))

const RENORMALIZATION_INTERVAL = 0.05
const LYAPUNOV_BLOCK_DURATION = 10.0
const DIAGNOSTIC_SAVE_INTERVAL = 0.01
const MAX_AUTOCORRELATION_LAG = 10.0
const MIN_PERIOD = 0.05
const RECURRENCE_TOLERANCE = 0.05
const RECURRENCE_EVALUATION_STRIDE = 20
const TANGENT_SEED_OFFSET = 2_000_000


"Realization numbers and seeds of networks that did not converge."
function load_nonconverged(path)
    networks = NamedTuple[]
    for (row, line) in enumerate(eachline(path))
        row == 1 && continue
        fields = split(line, '\t')
        parse(Bool, fields[3]) && continue
        push!(networks, (realization=parse(Int, fields[1]), seed=parse(Int, fields[2])))
    end
    return networks
end


"Final state saved at 10,000 s for each requested realization."
function load_final_states(path, networks)
    states = Dict(x.realization => zeros(N) for x in networks)
    counts = Dict(x.realization => 0 for x in networks)
    for (row, line) in enumerate(eachline(path))
        row == 1 && continue
        fields = split(line, '\t')
        realization = parse(Int, fields[1])
        haskey(states, realization) || continue
        unit = parse(Int, fields[5])
        states[realization][unit] = parse(Float64, fields[7])
        counts[realization] += 1
    end
    all(counts[x.realization] == N for x in networks) || error("Some final states are incomplete.")
    return states
end


"Reconstruct exactly the coupling matrix used by analyze_network_regime.jl."
function reconstruct_coupling(s)
    J = sprandn(N, N, 1.0)
    J[diagind(J)] .= 0.0
    J = Matrix(G * (J / maximum(abs.(eigvals(Matrix(J))))))
    J[diagind(J)] .= s
    return J
end


"Modified Gram-Schmidt for two tangent vectors; returns their growth factors."
function orthonormalize!(vectors)
    v1 = view(vectors, :, 1)
    v2 = view(vectors, :, 2)
    growth1 = norm(v1)
    isfinite(growth1) && growth1 > 0 || error("Invalid first tangent-vector norm.")
    v1 ./= growth1
    v2 .-= dot(v1, v2) .* v1
    growth2 = norm(v2)
    isfinite(growth2) && growth2 > 0 || error("Invalid second tangent-vector norm.")
    v2 ./= growth2
    return growth1, growth2
end


"Autocorrelation of a scalar signal up to max_lag samples."
function autocorrelation(signal, max_lag)
    centered = signal .- mean(signal)
    ac = zeros(max_lag + 1)
    for lag in 0:max_lag
        a = view(centered, 1:length(centered)-lag)
        b = view(centered, 1+lag:length(centered))
        scale = sqrt(sum(abs2, a) * sum(abs2, b))
        ac[lag + 1] = scale > 0 ? dot(a, b) / scale : 0.0
    end
    return ac
end


"First strong autocorrelation peak; otherwise the largest available peak."
function autocorrelation_period(ac, save_interval)
    first_lag = max(2, round(Int, MIN_PERIOD / save_interval))
    peaks = Int[]
    for lag in first_lag:length(ac)-2
        ac[lag + 1] > ac[lag] && ac[lag + 1] >= ac[lag + 2] && push!(peaks, lag)
    end
    isempty(peaks) && return first_lag
    strong = findfirst(lag -> ac[lag + 1] >= 0.5, peaks)
    return strong === nothing ? peaks[argmax(ac[peaks .+ 1])] : peaks[strong]
end


"RMS distance between full network states separated by lag columns."
function recurrence_distance(states, lag)
    n_pairs = size(states, 2) - lag
    n_pairs > 0 || return Inf
    squared_distance = 0.0
    evaluated_columns = 1:RECURRENCE_EVALUATION_STRIDE:n_pairs
    @inbounds for column in evaluated_columns, unit in 1:size(states, 1)
        difference = states[unit, column + lag] - states[unit, column]
        squared_distance += difference * difference
    end
    return sqrt(squared_distance / (size(states, 1) * length(evaluated_columns)))
end


"RMS temporal fluctuation of the full state."
function temporal_fluctuation(states)
    squared_fluctuation = 0.0
    for unit in axes(states, 1)
        row = view(states, unit, :)
        unit_mean = mean(row)
        squared_fluctuation += sum(x -> (x - unit_mean)^2, row)
    end
    return sqrt(squared_fluctuation / length(states))
end


"Choose the best full-state recurrence near the autocorrelation period or its harmonics."
function best_recurrence(states, basic_lag, save_interval)
    best_lag = basic_lag
    best_distance = Inf
    for harmonic in 1:6
        central_lag = harmonic * basic_lag
        central_lag >= size(states, 2) && break
        for lag in max(1, central_lag-3):min(size(states, 2)-1, central_lag+3)
            distance = recurrence_distance(states, lag)
            if distance < best_distance
                best_lag = lag
                best_distance = distance
            end
        end
    end
    fluctuation = temporal_fluctuation(states)
    relative_distance = fluctuation > 0 ? best_distance / fluctuation : Inf
    return best_lag * save_interval, best_distance, relative_distance, fluctuation
end


"Approximate 95% interval for the mean across Lyapunov blocks."
function block_interval(values)
    center = mean(values)
    half_width = length(values) > 1 ? 1.96 * std(values) / sqrt(length(values)) : NaN
    return center - half_width, center + half_width
end


function classify_attractor(λ1_interval, λ2_interval, relative_recurrence)
    λ1_low, λ1_high = λ1_interval
    _, λ2_high = λ2_interval
    λ1_low > 0 && return "chaotic_candidate"
    relative_recurrence < RECURRENCE_TOLERANCE && λ1_low <= 0 && λ2_high < 0 &&
        return "periodic_candidate"
    λ1_high < 0 && return "nonchaotic_nonperiodic"
    return "uncertain"
end


"Two-exponent Benettin calculation and recurrence diagnostics."
function analyze_trajectory(initial_state, J, seed, dt)
    alignment_steps = round(Int, ALIGNMENT / dt)
    measurement_steps = round(Int, MEASUREMENT / dt)
    renormalization_steps = round(Int, RENORMALIZATION_INTERVAL / dt)
    block_steps = round(Int, LYAPUNOV_BLOCK_DURATION / dt)
    save_steps = round(Int, DIAGNOSTIC_SAVE_INTERVAL / dt)

    alignment_steps % renormalization_steps == 0 || error("Alignment is not divisible by the renormalization interval.")
    measurement_steps % block_steps == 0 || error("Measurement is not divisible by the block duration.")
    block_steps % renormalization_steps == 0 || error("Block duration is not divisible by the renormalization interval.")
    save_steps > 0 || error("Diagnostic save interval must be at least one integration step.")

    r = copy(initial_state)
    rng = MersenneTwister(seed + TANGENT_SEED_OFFSET)
    tangent = randn(rng, N, 2)
    orthonormalize!(tangent)

    field = similar(r)
    activated = similar(r)
    drift = similar(r)
    next_r = similar(r)
    tangent_field = similar(tangent)
    next_tangent = similar(tangent)
    derivative = similar(r)
    α = dt / TAU0

    nsave = fld(measurement_steps - 1, save_steps) + 1
    saved_states = zeros(N, nsave)
    saved_residual = zeros(nsave)
    saved_population_mean = zeros(nsave)
    saved_column = 1

    total_log_growth = zeros(2)
    block_log_growth = zeros(2)
    block_end_times = Float64[]
    block_exponents_1 = Float64[]
    block_exponents_2 = Float64[]
    total_steps = alignment_steps + measurement_steps

    for step in 1:total_steps
        mul!(field, J, r)
        @. activated = tanh(field)
        @. derivative = 1.0 - activated^2

        measurement_step = step - alignment_steps - 1
        if measurement_step >= 0 && measurement_step % save_steps == 0
            saved_states[:, saved_column] .= r
            @. drift = activated - r
            saved_residual[saved_column] = norm(drift) / sqrt(N)
            saved_population_mean[saved_column] = mean(r)
            saved_column += 1
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
                total_log_growth .+= increments
                block_log_growth .+= increments
                measured_steps = step - alignment_steps
                if measured_steps % block_steps == 0
                    push!(block_end_times, measured_steps * dt)
                    push!(block_exponents_1, block_log_growth[1] / LYAPUNOV_BLOCK_DURATION)
                    push!(block_exponents_2, block_log_growth[2] / LYAPUNOV_BLOCK_DURATION)
                    block_log_growth .= 0.0
                end
            end
        end
    end

    λ1, λ2 = total_log_growth ./ MEASUREMENT
    λ1_interval = block_interval(block_exponents_1)
    λ2_interval = block_interval(block_exponents_2)
    max_lag = min(round(Int, MAX_AUTOCORRELATION_LAG / DIAGNOSTIC_SAVE_INTERVAL), nsave - 2)
    ac = autocorrelation(saved_residual, max_lag)
    basic_lag = autocorrelation_period(ac, DIAGNOSTIC_SAVE_INTERVAL)
    period, recurrence, relative_recurrence, fluctuation =
        best_recurrence(saved_states, basic_lag, DIAGNOSTIC_SAVE_INTERVAL)
    classification = classify_attractor(λ1_interval, λ2_interval, relative_recurrence)
    saved_times = collect(0:nsave-1) .* DIAGNOSTIC_SAVE_INTERVAL
    autocorrelation_lags = collect(0:max_lag) .* DIAGNOSTIC_SAVE_INTERVAL

    return (; dt, λ1, λ2, λ1_interval, λ2_interval, period, recurrence,
            relative_recurrence, fluctuation, classification,
            residual_mean=mean(saved_residual), residual_min=minimum(saved_residual),
            residual_max=maximum(saved_residual), saved_times, saved_residual,
            saved_population_mean, autocorrelation_lags, autocorrelation=ac,
            block_end_times, block_exponents_1, block_exponents_2)
end


function initialize_output_files(output_dir)
    results_path = joinpath(output_dir, "attractor_results.tsv")
    blocks_path = joinpath(output_dir, "lyapunov_blocks.tsv")
    timeseries_path = joinpath(output_dir, "timeseries.tsv")
    autocorrelation_path = joinpath(output_dir, "residual_autocorrelation.tsv")

    open(results_path, "w") do io
        println(io, join([
            "realization", "seed", "dt_s", "lambda1_per_s", "lambda1_ci_low", "lambda1_ci_high",
            "lambda2_per_s", "lambda2_ci_low", "lambda2_ci_high", "period_s",
            "recurrence_rms", "relative_recurrence", "temporal_fluctuation_rms",
            "residual_mean", "residual_min", "residual_max", "classification",
        ], '\t'))
    end
    open(blocks_path, "w") do io
        println(io, "realization\tseed\tdt_s\tblock_end_s\tlambda1_per_s\tlambda2_per_s")
    end
    open(timeseries_path, "w") do io
        println(io, "realization\tseed\tdt_s\ttime_since_alignment_s\tpopulation_mean\tfixed_point_residual")
    end
    open(autocorrelation_path, "w") do io
        println(io, "realization\tseed\tdt_s\tlag_s\tresidual_autocorrelation")
    end
    return results_path, blocks_path, timeseries_path, autocorrelation_path
end


function append_result(paths, network, result)
    results_path, blocks_path, timeseries_path, autocorrelation_path = paths
    open(results_path, "a") do io
        @printf(io, "%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%s\n",
                network.realization, network.seed, result.dt,
                result.λ1, result.λ1_interval[1], result.λ1_interval[2],
                result.λ2, result.λ2_interval[1], result.λ2_interval[2],
                result.period, result.recurrence, result.relative_recurrence,
                result.fluctuation, result.residual_mean, result.residual_min,
                result.residual_max, result.classification)
    end
    open(blocks_path, "a") do io
        for k in eachindex(result.block_end_times)
            @printf(io, "%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\n",
                    network.realization, network.seed, result.dt,
                    result.block_end_times[k], result.block_exponents_1[k],
                    result.block_exponents_2[k])
        end
    end
    open(timeseries_path, "a") do io
        for k in eachindex(result.saved_times)
            @printf(io, "%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\n",
                    network.realization, network.seed, result.dt,
                    result.saved_times[k], result.saved_population_mean[k],
                    result.saved_residual[k])
        end
    end
    open(autocorrelation_path, "a") do io
        for k in eachindex(result.autocorrelation_lags)
            @printf(io, "%d\t%d\t%.12g\t%.12g\t%.12g\n",
                    network.realization, network.seed, result.dt,
                    result.autocorrelation_lags[k], result.autocorrelation[k])
        end
    end
end


function write_summary(path, networks, completed_results)
    open(path, "w") do io
        println(io, "Analysis of persistent dynamics after 10,000 s")
        println(io, "n_networks = $(length(networks))")
        println(io, "n_units = $N")
        println(io, "recurrent_gain = $G")
        println(io, "tau0_s = $TAU0")
        println(io, "integration_steps_s = $(join(DT_VALUES, ", "))")
        println(io, "alignment_s = $ALIGNMENT")
        println(io, "measurement_s = $MEASUREMENT")
        println(io, "renormalization_interval_s = $RENORMALIZATION_INTERVAL")
        println(io, "lyapunov_block_duration_s = $LYAPUNOV_BLOCK_DURATION")
        println(io, "diagnostic_save_interval_s = $DIAGNOSTIC_SAVE_INTERVAL")
        println(io, "period_search_max_lag_s = $MAX_AUTOCORRELATION_LAG")
        println(io, "periodic_candidate_relative_recurrence_tolerance = $RECURRENCE_TOLERANCE")
        println(io, "recurrence_evaluation_stride = $RECURRENCE_EVALUATION_STRIDE saved samples")
        println(io)
        for (network, result) in completed_results
            @printf(io, "network %d, seed %d, dt %.4g: lambda=(%.6g, %.6g) 1/s, period=%.4g s, relative recurrence=%.4g, %s\n",
                    network.realization, network.seed, result.dt, result.λ1,
                    result.λ2, result.period, result.relative_recurrence,
                    result.classification)
        end
    end
end


function main()
    ALIGNMENT > 0 || error("--alignment must be positive.")
    MEASUREMENT > 0 || error("--measurement must be positive.")
    all(>(0), DT_VALUES) || error("All integration steps must be positive.")
    BLAS.set_num_threads(1)

    source_dir = generated_dir(SOURCE_NAME)
    networks = load_nonconverged(joinpath(source_dir, "network_results.tsv"))
    states = load_final_states(joinpath(source_dir, "final_states.tsv"), networks)
    s = vec(readdlm(self_couplings_file(), ',', Float64))
    length(s) == N || error("Expected $N self-couplings, found $(length(s)).")

    output_dir = new_data_dir(OUTPUT_NAME)
    output_paths = initialize_output_files(output_dir)
    completed_results = Tuple{NamedTuple,NamedTuple}[]

    println("Persistent-attractor analysis: $(length(networks)) networks")
    println("Time steps: $(join(DT_VALUES, ", ")) s")
    println("Per run: $ALIGNMENT s alignment + $MEASUREMENT s measurement")
    println("Output: $output_dir")

    for (network_index, network) in enumerate(networks)
        println("\nNetwork $network_index / $(length(networks)): realization $(network.realization), seed $(network.seed)")
        Random.seed!(network.seed)
        J = reconstruct_coupling(s)

        for dt in DT_VALUES
            println("  dt = $dt s")
            result = analyze_trajectory(states[network.realization], J, network.seed, dt)
            append_result(output_paths, network, result)
            push!(completed_results, (network, result))
            write_summary(joinpath(output_dir, "summary.txt"), networks, completed_results)
            @printf("    lambda1 = %.6g, lambda2 = %.6g 1/s, period = %.4g s, relative recurrence = %.4g, %s\n",
                    result.λ1, result.λ2, result.period,
                    result.relative_recurrence, result.classification)
            GC.gc()
        end
    end

    println("\nCompleted. Results saved in $output_dir")
end


main()
