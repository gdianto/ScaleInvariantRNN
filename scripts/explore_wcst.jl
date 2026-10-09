# Interactive exploration of the WCST task (Figs. 4-5): one network from the task inputs to the readout,
# the test behaviour, the change of the couplings, the readout weights, the state space (PCA, Fig. 4d-e) and the
# power-law test of the closed-loop network.
#
# Run cell by cell (`##`): change the parameters in the first cell and re-run the following cells.
# Defaults: the parameters of the paper. With save_figures = true the plots go to figures/explore_wcst/.
#   julia scripts/explore_wcst.jl     # all cells (GKSwstype=100 for no windows)
include(joinpath(@__DIR__, "..", "src", "project.jl"))
using LinearAlgebra, Random, Statistics, DelimitedFiles, Printf, Plots
include(projectdir("src", "reservoir.jl"))
include(projectdir("src", "wcst.jl"))
include(projectdir("src", "wcst_plots.jl"))

## ===================== parameters =====================
variant = "s"            # "s": heterogeneous self-couplings, τ0 = 0.01 s | "tau": no self-coupling, heterogeneous τ0
seed = nothing           # e.g. 1234 for a reproducible run; nothing = unseeded (as in the paper)

# network
N = 500                  # units
g = 0.4                  # spectral radius of J before the self-couplings
s_file = self_couplings_file()   # "s": published self-couplings; e.g. generated_dir("self_couplings.txt")
custom_s = nothing       # "s": N self-couplings (or one number) instead of the file
τ_range_ms = (10, 50)    # "tau": τ0 evenly spaced in this range (ms)

# inputs, feedback and readout
g_context = 0.8          # context feedback g_b (paper: 0.1-0.8; Fig. 4: 0.8)
g_cards = 1.0            # card inputs
g_error = 1.75           # error signal
g_response = 0.0         # response feedback
response_feedback_only = false   # close the loop on the response channel only
β = 0.01                 # ridge regularization
method = "ridge"         # "ridge" or "pinv"

# task
n_transient, n_learning, n_test = 20, 80, 100   # trials
change_prob = 0.2        # probability of a rule change at each trial
error_magnitude = 1.0
error_sign = false       # true: the error signal carries the sign of the rule
timing_params = (central=9.0, oriented=9.0, error_signal=6.0,    # in units of τ0
                 central_to_oriented=1.0, response_delay=3.0, oriented_to_error=1.0, inter_trials=10.0)

# analysis
n_starts = 100           # initial states of the power-law test
camera = [70.0, 50.0, 1.0]
save_figures = false

context_out = true       # the readout also learns the context (needed by the plots below)
show!(p, name) = (display(p); save_figures && savefig(p, joinpath(figures_dir("explore_wcst"), "$(variant)_$name.png")); p)

## ===================== task =====================
seed === nothing || Random.seed!(seed)
n_values = [2, 2]; n_cards = prod(n_values)
n_trials = n_transient + n_learning + n_test
n_learned = n_transient + n_learning
trials = generate_trials(n_values, n_trials, change_prob, error_magnitude);
println("rule changes at trials: ", trials.rule_changes)

if variant == "s"
    τ = 0.01; τu = τ
    self_couplings = custom_s === nothing ? readdlm(s_file, ',')[:] : custom_s
else
    τ = 0.001 * Vector(LinRange(τ_range_ms..., N)); τu = minimum(τ)
    self_couplings = 0.0
end
dt = τu / 10

timing = trial_timing(τu, dt, n_trials, error_magnitude; timing_params...);
spans = task_spans(dt, timing.trial_duration, timing.trial_steps, n_transient, n_learning, n_test);
stim = build_input(trials, timing, n_trials, n_learned, n_cards, context_out, error_sign);
input = stim.input;      # columns: cards, error, context, optimal response

## trial prototype
sig = timing.signal
plot(sig.central, label="central", lw=2); plot!(sig.oriented[1, :], label="right card", lw=2)
plot!(sig.oriented[2, :], label="left card", lw=2); plot!(sig.response, label="response", lw=2); plot!(sig.error, label="error", lw=2)
show!(plot!(size=(500, 350), title="Trial prototype", xlabel="steps", ylabel="signal"), "trial_prototype")

## inputs during learning (whole period and first trial)
plot_inputs(input, spans.steps_learning, n_cards, context_out);
show!(plot!(size=(750, 400), title="Learning period inputs"), "inputs_learning")
first_learning = spans.steps_learning[1]
plot_inputs(input, first_learning:first_learning+timing.trial_steps, n_cards, context_out);
show!(plot!(size=(300, 400), title="Zoom-in inputs", legend=false), "inputs_zoom")

## ===================== network =====================
net = wcst_network(; N, τ, g, self_couplings, n_cards, context_out, g_cards, g_error, g_context, g_response);
p1 = histogram(net.J[diagind(net.J)], legend=false, title="self-couplings s", xlabel="s", ylabel="count", color=RGB(0.8, 0.88, 0.94), bins=30);
show!(plot(p1, plot_spectrum(net.J), layout=(1, 2), size=(600, 300)), "network")

## ===================== learning =====================
# transient, learning, readout (context, response) and closed loop J + Jext W_out
learn = train_readout!(net, input, spans, dt; β, method, context_out, response_feedback_only);
(; r_learning, input_learning, W_out, closed, out_learning, target_learning, target_test) = learn;

plot(target_learning[end, :], label="optimal response")
show!(plot!(out_learning[end, :], label="fitted response", size=(700, 300), xlabel="steps", title="Learning: response", bottom_margin=5Plots.mm), "fit_response")
plot(target_learning[1, :], label="context")
show!(plot!(out_learning[1, :], label="fitted context", size=(700, 300), xlabel="steps", title="Learning: context", bottom_margin=5Plots.mm), "fit_context")

## ===================== test (autonomous network) =====================
test = run_test!(closed, net, input, W_out, trials, timing, stim, spans, dt; N, n_learned, n_test, context_out, error_sign);
(; r_test, response, input_test) = test;
@printf("accuracy: %.1f%%   accordance with optimal: %.1f%%   optimal accuracy: %.1f%%\n",
        100test.accuracy, 100test.accuracy_optimal, 100test.optimal_accuracy)
context_test = (W_out * r_test)[1, :]

plot(target_test[1, :], label="context")
show!(plot!(context_test, label="decoded context", size=(700, 300), xlabel="steps", title="Test: context", bottom_margin=5Plots.mm), "test_context")
plot(target_test[end, :], label="optimal response", color=RGBA(0.5, 0.5, 0.5, 0.5), lw=1.5)
show!(plot!(response, label="network response", size=(700, 300), xlabel="steps", title="Test: response", bottom_margin=5Plots.mm, color=RGBA(0.2, 0.5, 0.7, 1)), "test_response")

## network behaviour during the test (whole test and first 5000 steps)
for (name, steps) in (("test_task", spans.steps_test), ("test_task_zoom", spans.steps_test[1]:spans.steps_test[1]+5000))
    n = length(steps)
    plot_inputs(input, steps, n_cards, context_out)
    plot!(context_test[1:n] .+ ((n_cards + 1) * 2.4 + 4.6), label="network context", color=RGB(0.5, 0.8, 0.4))
    plot!(response[1:n] .+ (n_cards + 1) * 2.4, color=RGBA(0.2, 0.5, 0.7, 1), label="network response")
    annotate!(n / 2, -6.0, text(@sprintf("accuracy %.0f%%, accordance with optimal %.0f%%", 100test.accuracy, 100test.accuracy_optimal), :black, :center, 10))
    show!(plot!(size=(1000, 650), title="Network behaviour during the test", left_margin=8Plots.mm, bottom_margin=15Plots.mm), name)
end

## ===================== what learning changed =====================
s_pre, s_post = net.J[diagind(net.J)], closed.J[diagind(closed.J)]
p1 = scatter(s_post, ms=3, alpha=0.2, xlabel="unit", ylabel="self-coupling", label="post")
scatter!(s_pre, ms=3, alpha=0.2, label="pre")
p2 = histogram(s_pre, legend=false, title="s (pre)", xlabel="s", color=RGB(0.8, 0.88, 0.94), bins=50)
p3 = histogram(s_post, legend=false, title="s (post)", xlabel="s", color=RGB(0.8, 0.88, 0.94), bins=50)
show!(plot(p1, p2, p3, layout=(1, 3), size=(900, 300), bottom_margin=5Plots.mm, left_margin=5Plots.mm), "selfcouplings_pre_post")

ΔJ = net.J - closed.J
@printf("|ΔJ|/|J0| = %.4f\n", norm(ΔJ) / norm(net.J))
show!(heatmap(ΔJ, title=@sprintf("|ΔJ|/|J₀| = %.3f", norm(ΔJ) / norm(net.J)), size=(500, 400), c=:balance,
              clims=(-maximum(abs.(ΔJ)), maximum(abs.(ΔJ))), xlabel="index", ylabel="index", colorbar_title="ΔJ"), "deltaJ")

## readout weights vs the self-coupling (s-network) or the time constant (τ0-network) of each unit
xv, xlab = variant == "s" ? (s_pre, "self coupling s") : (τ, "τ₀ [s]")
p1 = plot(xv, abs.(W_out[1, :]), seriestype=:sticks, xlabel=xlab, ylabel="|W context|", color=:deepskyblue4, lw=2.5, alpha=0.7, legend=false)
p2 = plot(xv, abs.(W_out[2, :]), seriestype=:sticks, xlabel=xlab, ylabel="|W response|", color=:deepskyblue4, lw=2.5, alpha=0.7, legend=false)
show!(plot(p1, p2, layout=(1, 2), size=(800, 400), left_margin=3Plots.mm, bottom_margin=2Plots.mm), "readout_weights")

## ===================== state space (PCA) =====================
err = n_cards + 1                                     # error channel
Ψ, H = pca(r_learning); println("embedding dimension (learning): ", round(exp(H - 1), digits=2))
plot_context_space(Ψ' * r_learning, target_learning[1, :], input_learning[:, err])
show!(plot!(title="PCA of the learning activity"), "pca_learning")

## PCA of the test activity: context attractors (Fig. 4d right)
Ψ, H = pca(r_test); println("embedding dimension (test): ", round(exp(H - 1), digits=2))
pcs = Ψ' * r_test
plot_context_space(pcs, context_test, input_test[:, err]; camera)
show!(plot!(title="PCA of the test activity: context"), "pca_test_context")

## response zones and central-card patterns (Fig. 4d left)
central_on = vec(sum(input_test[:, 1:n_cards], dims=2))
plot_response_space(pcs[1:3, :], response, central_on; threshold=0.5, camera)
show!(plot!(legend=false), "pca_test_response")

## hysteresis in the (context readout, PC1, response) space (Fig. 4e)
plot_context_space(permutedims(hcat(context_test, pcs[1, :], response)), context_test, input_test[:, err];
                   plane=true, lims=[[-1.5, 1.5], [-15, 15], [-1, 1]], camera)
show!(plot!(xlabel="W_ctx·r", ylabel="PC₁", zlabel="W_resp·r", legend=false), "hysteresis")

## linear decoding of the central-card features (colour, shape)
W_central = fit_readout(r_learning, copy(stim.features[spans.steps_learning, :]'), β)
features_test = stim.features[spans.steps_test, :]'
println("correlation between true and decoded central-card features: ", cor(features_test', (W_central * r_test)'))
plot_response_space(permutedims(hcat(response, features_test[1, :], features_test[2, :])), response, central_on; threshold=0.25, camera)
show!(plot!(xlabel="W_resp·r", ylabel="W_C₁·r", zlabel="W_C₂·r"), "central_card_decoding")

## ===================== power-law test of the closed-loop network =====================
pl = power_law_test(closed, N, τu, dt; n_starts);
println("power-law fit α tᵝ: ", pl.params)
pls = plot_power_law(pl; n_starts, title="g=$g, g_b=$g_context, $variant-network")
show!(pls.p_acf, "power_law_acf"); show!(pls.p_traces, "power_law_traces"); show!(pls.p_fit, "power_law_fit")
