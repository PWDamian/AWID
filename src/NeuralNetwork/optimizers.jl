module Optimizers

using ..AutoDiff: Variable

abstract type Optimizer end

mutable struct Adam <: Optimizer
    lr::Float32 # Współczynnik uczenia (learning rate)
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    t::Int # Licznik kroków (iteracji)

    # m_value i v_value mają taki sam typ i kształt jak parametr.output
    m::Dict{Variable,Any} # bezwładność, pierwszy moment
    v::Dict{Variable,Any} # drugi moment (kwadrat gradientu)

    function Adam(; lr::Float32=0.001f0, beta1::Float32=0.9f0, beta2::Float32=0.999f0, epsilon::Float32=eps(Float32))
        new(lr, beta1, beta2, epsilon, 0, Dict{Variable,Any}(), Dict{Variable,Any}())
    end
end

function update!(opt::Adam, params::Vector{<:Variable})
    opt.t += 1

    for p in params
        g = p.gradient

        # Inicjalizacja m i v dla parametru, jeśli to pierwszy krok dla niego
        if !haskey(opt.m, p)
            opt.m[p] = zero(p.output)
            opt.v[p] = zero(p.output)
        end

        # Aktualizacja pierwszego momentu (m)
        # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        opt.m[p] = opt.beta1 .* opt.m[p] .+ (1 - opt.beta1) .* g

        # Aktualizacja drugiego momentu (v)
        # v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t)^2
        opt.v[p] = opt.beta2 .* opt.v[p] .+ (1 - opt.beta2) .* (g .^ 2)

        # Korekcja biasu dla momentów (bias correction)
        # m_hat_t = m_t / (1 - beta1^t)
        m_hat = opt.m[p] ./ (1 - opt.beta1^opt.t)

        # v_hat_t = v_t / (1 - beta2^t)
        v_hat = opt.v[p] ./ (1 - opt.beta2^opt.t)

        # Aktualizacja parametru
        # p_t = p_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
        p.output .-= opt.lr .* m_hat ./ (sqrt.(v_hat) .+ opt.epsilon)
    end
    return nothing
end

export Optimizer, Adam, update!

end