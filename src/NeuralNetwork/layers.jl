module Layers

using ..AutoDiff
using Random: randn

struct Dense
    W::Variable
    b::Variable
    activation_fn::Function
end

struct Conv
    W::Variable  # weights/kernel
    b::Variable  # bias
    kernel_size::Constant
    in_channels::Constant
    out_channels::Constant
    activation_fn::Function
end

struct MaxPool
    kernel_size::Constant
    stride::Constant
end

struct Flatten end

struct Permute
    dims::Tuple{Vararg{Int}}
end

struct Embedding
    W::Variable
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

# Konstruktor dla Conv
function Conv(kernel_size::Tuple{Vararg{Int}}, in_channels::Int, out_channels::Int, activation_fn::Function; init_W::Function=init_xavier_glorot, init_b::Function=init_zeros)
    # Initialize weights with Xavier/Glorot initialization
    W_shape = (kernel_size..., in_channels, out_channels)
    W_values = init_W(prod(kernel_size) * in_channels, out_channels)
    W_values = reshape(W_values, W_shape...)
    W = Variable(W_values)
    
    # Initialize bias
    b_values = init_b(out_channels)
    b = Variable(b_values)
    
    return Conv(W, b, Constant(kernel_size), Constant(in_channels), Constant(out_channels), activation_fn)
end

# Konstruktor dla MaxPool
function MaxPool(kernel_size::Tuple{Vararg{Int}}; stride::Tuple{Vararg{Int}}=kernel_size)
    return MaxPool(Constant(kernel_size), Constant(stride))
end

# Konstruktor dla Embedding
function Embedding(vocab_size::Int, embedding_dim::Int; init_W::Function=init_xavier_glorot)
    W_values = init_W(vocab_size, embedding_dim)
    W = Variable(W_values)
    return Embedding(W)
end

# Przejście w przód dla warstwy Dense
function (layer::Dense)(input::GraphNode)::GraphNode # Czyni obiekty Dense "wywoływalnymi": dense_layer(x)
    println("layer: Dense")
    # input.output to macierz cech (n_features np. 17703 - liczba cech z TF-IDF, n_samples_in_batch np. 64)
    affine_transformation = (layer.W * input) .+ layer.b
    return layer.activation_fn(affine_transformation)
end

# Przejście w przód dla warstwy Conv
function (layer::Conv)(input::GraphNode)::GraphNode
    println("layer: Conv")
    
    return layer.activation_fn(conv_op(layer.kernel_size, layer.in_channels, layer.out_channels, layer.W, layer.b, input))
end

# Przejście w przód dla warstwy MaxPool
function (layer::MaxPool)(input::GraphNode)::GraphNode
    println("layer: MaxPool")
    
    return maxpool_op(layer.kernel_size, layer.stride, input)
end

# Przejście w przód dla warstwy Flatten
function (layer::Flatten)(input::GraphNode)::GraphNode
    println("layer: Flatten")
    return flatten_op(input)
end

# Przejście w przód dla warstwy Embedding
function (layer::Embedding)(input::GraphNode)::GraphNode
    return embedding_op(layer.W, input)
end


struct Chain
    layers::Tuple{Vararg{Union{Dense,Conv,MaxPool,Flatten,Permute,Embedding}}}
end

# Konstruktor dla Chain, przyjmuje zmienną liczbę warstw
Chain(layers::Vararg{Union{Dense,Conv,MaxPool,Flatten,Permute,Embedding}}) = Chain((layers...,))

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

function parameters(layer::Conv)
    return [layer.W, layer.b]
end

function parameters(layer::MaxPool)
    return []  # MaxPool has no parameters
end

function parameters(layer::Flatten)
    return []  # Flatten has no parameters
end

function parameters(layer::Permute)
    return []  # Permute has no parameters
end

function parameters(layer::Embedding)
    return [layer.W]
end

function parameters(chain::Chain)
    all_params = Variable[] # pusta lista
    for layer in chain.layers
        append!(all_params, parameters(layer))
    end
    return all_params
end

export Dense, Conv, MaxPool, Flatten, Permute, Embedding, Chain, parameters, init_xavier_glorot, init_zeros

end