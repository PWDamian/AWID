module Layers

using ..AutoDiff: Variable, GraphNode
using Random: randn

struct Dense
    W::Variable
    b::Variable
    activation_fn::Function
end

function init_xavier_glorot(in_dim, out_dim)
    scale = sqrt(6f0 / (in_dim + out_dim))
    return rand(Float32, out_dim, in_dim) .* 2f0 .* scale .- scale
end

function init_zeros(dims...)
    return zeros(Float32, dims...)
end

# Konstruktor dla Dense
function Dense(input_size::Int, output_size::Int, activation_fn::Function; init_W::Function=init_xavier_glorot, init_b::Function=init_zeros)
    W_values = init_W(input_size, output_size)
    b_values = init_b(output_size)
    W = Variable(W_values)
    b = Variable(b_values)

    return Dense(W, b, activation_fn)
end

# Przejście w przód dla warstwy Dense
function (layer::Dense)(input::GraphNode)::GraphNode # Czyni obiekty Dense "wywoływalnymi": dense_layer(x)
    # input.output to macierz cech (n_features np. 17703 - liczba cech z TF-IDF, n_samples_in_batch np. 64)
    affine_transformation = (layer.W * input) .+ layer.b
    return layer.activation_fn(affine_transformation)
end

struct Chain
    layers::Tuple{Vararg{Dense}}
end

# Konstruktor dla Chain, przyjmuje zmienną liczbę warstw
Chain(layers::Vararg{Dense}) = Chain((layers...,))

# Przejście w przód dla Chain
function (chain::Chain)(input::GraphNode)::GraphNode # Czyni obiekty Chain "wywoływalnymi": model(x)
    output = input
    for layer in chain.layers
        output = layer(output) # przejście w przód przez każdą warstwę
    end
    return output
end

# Zbieranie parametrów dla optymalizatora
function parameters(layer::Dense)
    return [layer.W, layer.b]
end

function parameters(chain::Chain)
    all_params = Variable[] # pusta lista
    for layer in chain.layers
        append!(all_params, parameters(layer))
    end
    return all_params
end

export Dense, Chain, parameters, init_xavier_glorot, init_zeros

end