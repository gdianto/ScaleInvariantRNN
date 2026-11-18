# Short-term memory task of Fig. 3: correlation between the states evoked by six stimuli vs their position distance.
# Requires src/project.jl and src/reservoir.jl.

using LinearAlgebra, Random, Statistics, DelimitedFiles

# network types of Fig. 3c (Methods)
const MEMORY_NETWORKS = Dict(
    "hRNN" => (g=0.4,    τ=0.01),     # heterogeneous self-couplings (self_couplings_file())
    "hom"  => (g=0.4,    τ=0.01),     # their mean, 1.026
    "esn"  => (g=0.9999, τ=0.04),     # echo-state network, no self-coupling
)

function memory_self_couplings(network; P=MEMORY_PROTOCOL)
    network == "esn" && return 0.0
    s = vec(readdlm(self_couplings_file(), ','))
    return network == "hRNN" ? s : round(sum(s) / P.N, digits=3)
end

const MEMORY_PROTOCOL = (
    N = 500,
    τu = 0.01,                 # time unit of T_sep (not the network time constant)
    dt = 0.001,
    n_patterns = 6,
    n_transient = 5,           # transient cycles
    n_cycles = 10,             # saved cycles
    n_networks = 50,
    stimulus = 1.0e16,         # stimulus intensity
    pulse = 0.01,              # stimulus duration (s)
    save_every = 2,
)

"Input of one cycle (pulse k on channel k, then a pause) and the step ranges of the pauses."
function pulse_sequence(n, pulse, pause, dt)
    n_pulse, n_pause = round(Int, pulse / dt), round(Int, pause / dt)
    input = zeros((n_pulse + n_pause) * n, n)
    pauses = UnitRange{Int}[]
    for k in 1:n
        t0 = (k - 1) * (n_pulse + n_pause) + 1
        t1 = t0 + n_pulse
        input[t0:t1, k] .= 1.0
        push!(pauses, t1+1:t1+n_pause)
    end
    return pauses, input
end

"Sequence protocol for T_sep = multiplier * τu: input, saved samples and the saved samples of each pause."
function memory_protocol(multiplier; P=MEMORY_PROTOCOL)
    (; n_patterns, pulse, dt, save_every, τu) = P
    pauses, input = pulse_sequence(n_patterns, pulse, τu * multiplier, dt)
    span = (dt, size(input, 1) * dt)
    m = multiplier * save_every
    # max(1, ...): for multiplier = 20 the first index would round to 0
    saved_pauses = [max(1, round(Int, p[1] / m)):round(Int, p[end] / m) for p in pauses]
    t = span[1]:multiplier*(save_every*dt):span[2]
    return (; input, span, t, saved_pauses, saved_per_cycle=round(Int, size(input, 1) / m), save_every=Int(m))
end

"New random J (spectral radius g) with the self-couplings of `network`, and the stimulus weights."
function draw_couplings!(net, network; P=MEMORY_PROTOCOL, rng=Random.default_rng())
    p = MEMORY_NETWORKS[network]
    (; N, n_patterns, stimulus) = P
    J = randn(rng, N, N)
    net.J = p.g * J / maximum(abs.(eigvals(J)))
    net.J[diagind(net.J)] .= memory_self_couplings(network; P)
    net.Jext = zeros(N, n_patterns)
    for k in 1:n_patterns
        net.Jext[k:n_patterns:end, k] .= stimulus
    end
    return net
end

"Mean state in each pause (patterns × units × cycles)."
function pause_states(r, n_cycles, per_cycle, pauses)
    states = zeros(length(pauses), size(r, 1), n_cycles)
    for c in 1:n_cycles, k in eachindex(pauses)
        states[k, :, c] = mean(r[:, (c-1)*per_cycle .+ pauses[k]], dims=2)
    end
    return states
end

"Mean correlation between pause states at signed position distance -d ... d (cycles 2 to n-1 as reference)."
function correlation_by_distance(states)
    n = size(states, 1)
    nd = round(Int, 0.5n - 0.499)
    cors = [Float64[] for _ in 1:2nd+1]
    cors[nd+1] = [1.0, 1.0]
    for c in 2:size(states, 3)-1, k in 1:n, d in 1:nd, sgn in (1, -1)
        k2, c2 = k + sgn * d, c
        k2 <= 0 && (k2 += n; c2 -= 1)
        k2 > n && (k2 -= n; c2 += 1)
        push!(cors[nd+1+sgn*d], cor(states[k, :, c], states[k2, :, c2]))
    end
    return mean.(cors)
end

"""
    run_memory_task(network, multiplier; n_networks, rng)

Correlation vs signed distance (rows -3:3) for `n_networks` realizations of `network` ("hRNN", "hom", "esn")
at T_sep = multiplier * 0.01 s. J is drawn from `rng`; each realization starts from the final state of the
previous one.
"""
function run_memory_task(network, multiplier; P=MEMORY_PROTOCOL, n_networks=P.n_networks, rng=Random.default_rng())
    pr = memory_protocol(multiplier; P)
    r = zeros(P.N, P.n_cycles * length(pr.t))
    cor_matrix = zeros(P.n_patterns + 1, n_networks)
    net = Network(zeros(P.N), ones(P.N) .* MEMORY_NETWORKS[network].τ, zeros(P.N, P.N), zeros(P.N, P.n_patterns), zeros(P.N))
    for k in 1:n_networks
        draw_couplings!(net, network; P, rng)
        for _ in 1:P.n_transient
            simulate!(net, pr.span; dt=P.dt, input=pr.input)
        end
        for c in 1:P.n_cycles
            r[:, 1+(c-1)*length(pr.t):c*length(pr.t)] = simulate!(net, pr.span; dt=P.dt, input=pr.input, save_every=pr.save_every)
        end
        cor_matrix[:, k] = correlation_by_distance(pause_states(r, P.n_cycles, pr.saved_per_cycle, pr.saved_pauses))
    end
    return cor_matrix
end
