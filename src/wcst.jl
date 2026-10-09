# Wisconsin Card Sorting Test solved by a reservoir (Figs. 4-5).
#
# Cards have two properties (colour, shape). Each trial shows a central card and two oriented cards; the correct
# response is the oriented card that shares the property selected by the current rule with the central card.
# The rule changes at each trial with probability change_prob, and a wrong response triggers an error signal.
# Requires src/project.jl and src/reservoir.jl. Keep the order of the random draws: seeded results depend on it.

using LinearAlgebra, Random, Statistics, Printf, DelimitedFiles, LsqFit, SparseArrays

other(i) = 3 - i                                          # 1 <-> 2
correct_response(central, card1, rule) = central[rule] == card1[rule] ? -1 : 1
card_code(card, n_shapes) = (card[1] - 1) * n_shapes + card[2]
different_value(v, m) = v == 1 ? rand(2:m) : v == m ? rand(1:m-1) : rand([1:v-1; v+1:m])

"Smoothed square pulse of length `len` with cosine ramps of `ramp` steps."
function smoothed_square(len, ramp)
    r = ones(len)
    v = 0.5 * (cos.(range(-π, 0.0, length=ramp)) .+ 1.0)
    r[1:ramp] = v
    r[end-ramp+1:end] = -v .+ 1.0
    return r
end

# ------------------------------------------------------------------ task
"Cards, rules and optimal responses of every trial. `n_values` = number of colours and of shapes."
function generate_trials(n_values, n_trials, change_prob, error_magnitude)
    central = [[rand(1:n_values[1]), rand(1:n_values[2])] for _ in 1:n_trials]
    rule_changes = Int[]
    rules_signed = ones(Int, n_trials)
    for k in 2:n_trials
        if rand() < change_prob
            rules_signed[k:end] = -rules_signed[k:end]
            push!(rule_changes, k)
        end
    end
    rules = Int.(0.5 * rules_signed .+ 1.5)

    oriented = [[[rand(1:n_values[1]), rand(1:n_values[2])], [rand(1:n_values[1]), rand(1:n_values[2])]] for _ in 1:n_trials]
    correct_cards = rand(1:2, n_trials)
    correct_responses = Int.(2.0 * (correct_cards .- 1.5))
    optimal_responses = copy(correct_responses)            # responses with the previous rule after a change
    for k in eachindex(correct_cards)
        c, rule = correct_cards[k], rules[k]
        # the correct card matches the central card on the rule property only, the other card on the other one
        oriented[k][c][rule] = central[k][rule]
        oriented[k][other(c)][rule] = different_value(central[k][rule], n_values[rule])
        oriented[k][other(c)][other(rule)] = central[k][other(rule)]
        oriented[k][c][other(rule)] = different_value(central[k][other(rule)], n_values[other(rule)])
        k in rule_changes && (optimal_responses[k] = correct_response(central[k], oriented[k][1], rules[k-1]))
    end
    optimal_error = error_magnitude * abs.(0.5 * (correct_responses - optimal_responses))

    central_index = [card_code(central[k], n_values[2]) for k in 1:n_trials]
    oriented_index = [card_code(oriented[k][i], n_values[2]) for i in 1:2, k in 1:n_trials]
    return (; central, rule_changes, rules_signed, rules, oriented, correct_cards, correct_responses,
            optimal_responses, optimal_error, central_index, oriented_index)
end

"""
    trial_timing(τu, dt, n_trials, error_magnitude; ...)

Signals of one trial. Durations are in units of τu (defaults: the paper): central card, oriented cards,
error signal, and the waits central -> oriented, response delay (within the oriented cards), oriented ->
error, and between trials.
"""
function trial_timing(τu, dt, n_trials, error_magnitude; central=9.0, oriented=9.0, error_signal=6.0,
                      central_to_oriented=1.0, response_delay=3.0, oriented_to_error=1.0, inter_trials=10.0)
    steps(x) = round(Int, x * τu / dt)
    ramp = round(Int, 1.0τu / dt)
    trial_duration = central * τu + central_to_oriented * τu + oriented * τu + oriented_to_error * τu +
                     error_signal * τu + inter_trials * τu
    trial_steps = round(Int, trial_duration / dt)
    signal = (central=zeros(trial_steps), oriented=zeros(2, trial_steps), error=zeros(trial_steps), response=zeros(trial_steps))

    t1_central = steps(central)
    signal.central[1:t1_central] .= smoothed_square(t1_central, ramp)
    t0_oriented = 1 + t1_central + steps(central_to_oriented)
    t1_oriented = t0_oriented + steps(oriented) - 1
    signal.oriented[1, t0_oriented:t1_oriented] .= -smoothed_square(t1_oriented - t0_oriented + 1, ramp)   # right
    signal.oriented[2, t0_oriented:t1_oriented] .= smoothed_square(t1_oriented - t0_oriented + 1, ramp)    # left
    t0_response, t1_response = t0_oriented + steps(response_delay), t1_oriented
    signal.response[t0_response:t1_response] .= smoothed_square(t1_response - t0_response + 1, ramp)
    t0_error = t1_oriented + steps(oriented_to_error)
    t1_error = t0_error + steps(error_signal)
    signal.error[t0_error:t1_error] .= error_magnitude * smoothed_square(t1_error - t0_error + 1, ramp)

    return (; ramp, trial_duration, trial_steps, total_steps=round(Int, n_trials * trial_duration / dt), signal,
            t0_central=1, t0_response, t1_response, t0_error)
end

"Time spans (s) and step ranges of the transient, learning and test periods."
function task_spans(dt, trial_duration, trial_steps, n_transient, n_learning, n_test)
    transient = (dt, n_transient * trial_duration)
    learning = (transient[end] + dt, transient[end] + n_learning * trial_duration)
    test = (learning[end] + dt, learning[end] + n_test * trial_duration)
    a, b = trial_steps * n_transient, trial_steps * (n_transient + n_learning)
    return (; transient, learning, test, steps_transient=1:a, steps_learning=a+1:b, steps_test=b+1:b+trial_steps*n_test)
end

"""
Input of the whole task, columns: cards (n_cards), error, [context], optimal response. The error signal is set
for the transient and learning trials only. Also returns the central-card features (colour, shape) as ±1.
"""
function build_input(trials, timing, n_trials, n_learned, n_cards, context_out, error_sign)
    (; total_steps, trial_steps, signal, ramp, t0_central, t1_response, t0_error) = timing
    input = zeros(total_steps, n_cards + 2 + context_out)
    features = zeros(total_steps, 2)
    feature_signal = zeros(trial_steps)
    feature_signal[t0_central:t1_response] .= smoothed_square(t1_response - t0_central + 1, ramp)
    ramp_context = 5ramp
    first_step = [1 + (k - 1) * trial_steps for k in 1:n_trials]
    for k in 1:n_trials
        start = first_step[k]
        steps = start:start+trial_steps-1
        input[steps, trials.central_index[k]] .+= signal.central
        features[steps, 1] .+= feature_signal * sign(trials.central[k][1] - 1.5)
        features[steps, 2] .+= feature_signal * sign(trials.central[k][2] - 1.5)
        input[steps, trials.oriented_index[1, k]] .+= signal.oriented[1, :]
        input[steps, trials.oriented_index[2, k]] .+= signal.oriented[2, :]
        if k <= n_learned
            input[steps, end-(1+context_out)] .= error_sign ? trials.rules_signed[k] * trials.optimal_error[k] * signal.error :
                                                              trials.optimal_error[k] * signal.error
        end
        if context_out
            if trials.optimal_error[k] == 1.0                  # the context switches with the error signal
                input[start:start+t0_error, end-1] .= trials.rules_signed[k-1]
                input[start+t0_error+1:start+t0_error+ramp_context, end-1] .=
                    (smoothed_square(ramp_context, ramp_context) * 2.0 .- 1.0) * trials.rules_signed[k-1]
                input[start+t0_error+ramp_context:steps[end], end-1] .= trials.rules_signed[k]
            else
                input[steps, end-1] .= trials.rules_signed[k]
            end
        end
        input[steps, end] .= trials.optimal_responses[k] * signal.response
    end
    return (; input, features, first_step)
end

# ------------------------------------------------------------------ network
"""
Dense Gaussian J with spectral radius g and the given self-couplings (vector, or scalar for the τ0-network);
input weights for the cards, error, [context] and response channels.
"""
function wcst_network(; N, τ, g, self_couplings, n_cards, context_out, g_cards, g_error, g_context, g_response)
    n_inputs = n_cards + 2 + context_out
    sprandn(N, N, 10 / N); rand(N, n_inputs)    # unused draws, kept for the random stream of the published runs
    J = randn(N, N)
    J = g * J / maximum(abs.(eigvals(J)))
    J[diagind(J)] .= self_couplings
    Jext = zeros(N, n_inputs)
    Jext[:, 1:n_cards] = g_cards * randn(N, n_cards)
    Jext[:, end-(1+context_out)] = g_error * randn(N)
    context_out && (Jext[:, end-1] = g_context * randn(N))
    Jext[:, end] = g_response * randn(N)
    return Network(zeros(N), ones(N) .* τ, J, Jext, zeros(N))
end

"Transient and learning periods; readout (response, [context]) and closed-loop network."
function train_readout!(net, input, spans, dt; β, method, context_out, response_feedback_only)
    net.r .= 0.0
    simulate!(net, spans.transient; dt, input=input[spans.steps_transient, :])
    input_learning = input[spans.steps_learning, :]
    r_learning = simulate!(net, spans.learning; dt, input=input_learning, save_every=1)
    target_learning = copy(transpose(input[spans.steps_learning, end-context_out:end]))
    target_test = copy(transpose(input[spans.steps_test, end-context_out:end]))
    W_out = fit_readout(r_learning, target_learning, β; method)
    closed = close_loop(net, response_feedback_only ? W_out[1+context_out, :]' : W_out)
    return (; input_learning, r_learning, target_learning, target_test, W_out, closed, out_learning=W_out * r_learning)
end

"""
Test period: the closed-loop network responds autonomously; the error signal is given only after a wrong
response (written into `input`).
"""
function run_test!(closed, net, input, W_out, trials, timing, stim, spans, dt; N, n_learned, n_test, context_out, error_sign)
    (; trial_steps, signal, t0_central, t0_response, t1_response) = timing
    closed.r = net.r
    r_test = zeros(N, length(spans.test[1]:dt:spans.test[end]))
    response = zeros(size(r_test, 2))
    err = size(input, 2) - (1 + context_out)                       # error channel
    input[stim.first_step[n_learned+1]:end, err] .= 0.0
    response_steps = t1_response - t0_response
    span_pre, span_post = (dt, t1_response * dt), (dt, (trial_steps - t1_response) * dt)
    hit_optimal, hit_correct, optimal_correct = zeros(n_test), zeros(n_test), zeros(n_test)
    for k in 1:n_test
        trial = n_learned + k
        steps = stim.first_step[trial]:stim.first_step[trial]+trial_steps-1
        trial_input = input[steps, :]
        pre = ((k - 1) * trial_steps + 1):(t1_response + (k - 1) * trial_steps)
        post = (pre[end] + 1):trial_steps*k
        r_test[:, pre] = simulate!(closed, span_pre; dt, input=trial_input[t0_central:t1_response, :], save_every=1)
        window = pre[end-response_steps:end]
        response[window] = W_out[end, :]' * r_test[:, window]
        answer = sign(sum(response[window]))
        hit_optimal[k] = Int(trials.optimal_responses[trial] == answer)
        hit_correct[k] = Int(trials.correct_responses[trial] == answer)
        optimal_correct[k] = Int(trials.correct_responses[trial] == trials.optimal_responses[trial])
        if hit_correct[k] == 0.0
            e = error_sign ? trials.rules_signed[trial] * signal.error : signal.error
            trial_input[:, err] .= e
            input[steps, err] .= e
        end
        r_test[:, post] = simulate!(closed, span_post; dt, input=trial_input[(t1_response+1):end, :], save_every=1)
    end
    return (; r_test, response, input_test=input[spans.steps_test, :], accuracy=mean(hit_correct),
            accuracy_optimal=mean(hit_optimal), optimal_accuracy=mean(optimal_correct))
end

power_law(t, p) = p[1] .* (t .^ p[2])

"""
Mean autocorrelation of the closed-loop network started from `n_starts` random states (saved every 5 steps
over 1000 τu), and its power-law fit p[1] t^p[2] from sample `i0` on.
"""
function power_law_test(closed, N, τu, dt; n_starts=100, i0=5)
    tspan = (0.0, 1000 * τu)
    t = (tspan[1]:dt:tspan[end])[1:5:end]
    net = Network(closed.r, closed.τ, closed.J, zeros(N, 0), closed.σ)
    acf, acf_mean = zeros(length(t)), zeros(length(t))
    r = zeros(N, length(t))
    for _ in 1:n_starts
        net.r = randn(N)
        net.r ./= maximum(abs.(net.r))
        r = simulate!(net, tspan; dt, save_every=5)
        for k in eachindex(t)
            acf[k] = (r[:, 1]' * r[:, k]) / sqrt((r[:, 1]' * r[:, 1]) * (r[:, k]' * r[:, k]))
        end
        acf_mean += acf
    end
    acf_mean ./= n_starts
    params = round.(curve_fit(power_law, t[i0:end], acf_mean[i0:end], [1.0, -1.0]).param, digits=2)
    return (; t, acf_mean, r, params, i0)
end

# ------------------------------------------------------------------ runs and result files
append_row(path, values) = open(io -> println(io, join(map(string, values), " ")), path, "a")

function append_results(path, g_context, accuracy, dJ)
    open(path, "a") do io
        filesize(path) == 0 && println(io, "gbackContext accuracy_from_optimal dJ_J0")
        println(io, "$g_context $accuracy $dJ")
    end
end

"""
    run_wcst(variant, g_context; outdir, verbose=false)

One WCST run with the parameters of the paper, appended to the result files in `outdir`
(results_wcst, gds_post_wcst, readout_context_wcst, readout_response_wcst, acfm_wcst).
variant "s": self-couplings of `self_couplings_file()`, τ0 = 0.01 s; "tau": no self-coupling, τ0 in [0.01, 0.05] s.
"""
function run_wcst(variant, g_context; outdir, verbose=false)
    variant in ("s", "tau") || error("variant must be \"s\" or \"tau\"")
    n_values = [2, 2]; n_cards = prod(n_values)
    n_transient, n_learning, n_test = 20, 80, 100
    n_trials = n_transient + n_learning + n_test
    error_magnitude = 1.0; error_sign = false; context_out = true
    trials = generate_trials(n_values, n_trials, 0.2, error_magnitude)

    N, g = 500, 0.4
    if variant == "s"
        τ = 0.01; τu = mean(τ)
        self_couplings = readdlm(self_couplings_file(), ',')[:]
    else
        τ = 0.001 * Vector(LinRange(10, 50, N)); τu = minimum(τ)
        self_couplings = 0.0
    end
    dt = τu / 10.0

    timing = trial_timing(τu, dt, n_trials, error_magnitude)
    spans = task_spans(dt, timing.trial_duration, timing.trial_steps, n_transient, n_learning, n_test)
    stim = build_input(trials, timing, n_trials, n_transient + n_learning, n_cards, context_out, error_sign)
    net = wcst_network(; N, τ, g, self_couplings, n_cards, context_out, g_cards=1.0, g_error=1.75, g_context, g_response=0.0)
    learn = train_readout!(net, stim.input, spans, dt; β=0.01, method="ridge", context_out, response_feedback_only=false)
    (; W_out, closed) = learn
    test = run_test!(closed, net, stim.input, W_out, trials, timing, stim, spans, dt; N, n_learned=n_transient + n_learning,
                     n_test, context_out, error_sign)
    verbose && println("accuracy $(test.accuracy), accordance with optimal $(test.accuracy_optimal)")

    tag = "$(variant)-gb_$g_context"
    dJ = norm(net.J - closed.J) / norm(net.J)
    append_row(joinpath(outdir, "gds_post_wcst_$tag.txt"), closed.J[diagind(closed.J)])
    append_results(joinpath(outdir, "results_wcst_$(variant).txt"), g_context, test.accuracy_optimal, dJ)
    append_row(joinpath(outdir, "readout_context_wcst_$tag.txt"), abs.(W_out[1, :]))
    append_row(joinpath(outdir, "readout_response_wcst_$tag.txt"), abs.(W_out[2, :]))

    # decoding of the central-card features (unused here, but its random draws precede the power-law test)
    fit_readout(learn.r_learning, copy(stim.features[spans.steps_learning, :]'), 0.01)

    pl = power_law_test(closed, N, τu, dt)
    verbose && println("power-law fit: ", pl.params)
    append_row(joinpath(outdir, "acfm_wcst_$tag.txt"), pl.acf_mean)
    return (; accuracy=test.accuracy_optimal, dJ)
end

"""
    run_wcst_sweep(; variants, g_contexts, reps=50, outdir)

`reps` unseeded runs for each network type and g_b, (type, g_b) pairs in parallel threads, merged into
`outdir` (an empty folder) with the layout of data/published/wcst, together with the self-couplings used.
"""
function run_wcst_sweep(; variants=["s", "tau"], g_contexts=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], reps=50, outdir)
    Threads.nthreads() > 1 && BLAS.set_num_threads(1)
    staging(v, gb) = mkpath(joinpath(outdir, "staging", "$(v)-gb_$(gb)"))
    jobs = [(v, gb) for v in variants for gb in g_contexts]
    println("WCST sweep: $(length(jobs)) (network, g_b) pairs x $reps runs on $(Threads.nthreads()) thread(s) -> $outdir")
    lk = ReentrantLock()
    Threads.@threads :dynamic for (v, gb) in jobs
        for k in 1:reps
            res = run_wcst(v, gb; outdir=staging(v, gb))
            lock(lk) do
                @printf("  %-3s g_b=%.1f run %2d/%d: accuracy %.2f, |dJ|/|J0| %.3f\n", v, gb, k, reps, res.accuracy, res.dJ)
            end
        end
    end
    for v in variants
        open(joinpath(outdir, "results_wcst_$(v).txt"), "w") do io
            println(io, "gbackContext accuracy_from_optimal dJ_J0")
            for gb in sort(g_contexts)
                foreach(l -> println(io, l), readlines(joinpath(staging(v, gb), "results_wcst_$(v).txt"))[2:end])
            end
        end
        for gb in g_contexts, prefix in ("acfm", "gds_post", "readout_context", "readout_response")
            name = "$(prefix)_wcst_$(v)-gb_$(gb).txt"
            cp(joinpath(staging(v, gb), name), joinpath(outdir, name))
        end
    end
    rm(joinpath(outdir, "staging"); recursive=true)
    cp(self_couplings_file(), joinpath(outdir, "self_couplings.txt"))
    println("WCST results in $outdir")
    return outdir
end
