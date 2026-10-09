# Rate network τ dr/dt = -r + tanh(J r + Jext u(t)) + noise, integrated with explicit Euler steps.

using LinearAlgebra, Random, SparseArrays, Statistics

mutable struct Network
    r::Vector{Float64}             # rates
    τ::Vector{Float64}             # time constants (s)
    J::AbstractMatrix{Float64}     # recurrent couplings, self-couplings on the diagonal
    Jext::AbstractMatrix{Float64}  # input weights
    σ::Vector{Float64}             # noise amplitude
end

"""
    simulate!(net, tspan; dt, input=nothing, save_every=0, currents=false)

Integrate from `tspan[1]` to `tspan[2]`, updating `net.r` in place. `input[k, :]` is the input at step k.
With `save_every = m > 0`, returns the rates every m steps (first column: initial state); with
`currents = true`, also the input of each unit without its self-coupling, `(rates, currents)`.
"""
function simulate!(net::Network, tspan; dt, input=nothing, save_every=0, currents=false)
    times = tspan[1]:dt:tspan[end]
    n, nt = length(net.r), length(times)
    α = dt ./ net.τ
    noise = (net.σ * sqrt(dt)) .* randn(n, nt)
    r = net.r
    h, hext = zeros(n), zeros(n)
    nsave = save_every > 0 ? length(1:save_every:nt) : 0
    rates = zeros(n, nsave)
    nsave > 0 && (rates[:, 1] = r)
    if currents
        cur = zeros(n, nsave)
        Joff = copy(net.J)
        Joff[diagind(Joff)] .= 0
    end
    col = 2
    @inbounds for k in 2:nt
        mul!(h, net.J, r)
        input === nothing || mul!(hext, net.Jext, view(input, k, :))
        @views r .= r .* (1 .- α) .+ α .* tanh.(h .+ hext) .+ noise[:, k]
        if save_every > 0 && (k - 1) % save_every == 0
            rates[:, col] .= r
            currents && (mul!(view(cur, :, col), Joff, r); cur[:, col] .+= hext)
            col += 1
        end
    end
    save_every > 0 || return nothing
    return currents ? (rates, cur) : rates
end

"""
    fit_readout(X, Y, β; method="pinv")

Readout W with W X ≈ Y (states X: units × time). "pinv": least squares on X plus Gaussian noise of
relative amplitude β; "ridge": ridge regression with parameter β.
"""
function fit_readout(X, Y, β; method="pinv")
    β == 0 && return Y / X
    method == "pinv" && return Y / (X .+ randn(size(X)) * β * std(X))
    method == "ridge" && return Matrix((inv(X * X' + β * I(size(X, 1))) * X * Y')')
    error("method must be \"pinv\" or \"ridge\"")
end

"Closed-loop network: the readout W feeds back through the last size(W, 1) input channels."
function close_loop(net::Network, W)
    k = size(W, 1)
    closed = deepcopy(net)
    closed.J = Matrix(net.J .+ net.Jext[:, end-k+1:end] * W)
    closed.Jext = copy(net.Jext)
    closed.Jext[:, end-k+1:end] .= 0.0
    return closed
end
