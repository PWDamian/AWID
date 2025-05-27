module Layers

using ..AutoDiff: Variable, GraphNode
using Random: randn

struct Dense
    W::Variable
    b::Variable
    activation_fn::Function
end

struct Conv
    W::Variable  # weights/kernel
    b::Variable  # bias
    kernel_size::Tuple{Vararg{Int}}
    in_channels::Int
    out_channels::Int
    activation_fn::Function
end

struct MaxPool
    kernel_size::Tuple{Vararg{Int}}
    stride::Tuple{Vararg{Int}}
end

struct Flatten end

struct Permute
    dims::Tuple{Vararg{Int}}
end

struct Embedding
    W::Variable  # embedding matrix
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
    
    return Conv(W, b, kernel_size, in_channels, out_channels, activation_fn)
end

# Konstruktor dla MaxPool
function MaxPool(kernel_size::Tuple{Vararg{Int}}; stride::Tuple{Vararg{Int}}=kernel_size)
    return MaxPool(kernel_size, stride)
end

# Konstruktor dla Embedding
function Embedding(vocab_size::Int, embedding_dim::Int; init_W::Function=init_xavier_glorot)
    W_values = init_W(vocab_size, embedding_dim)
    W = Variable(W_values)
    return Embedding(W)
end

# Przejście w przód dla warstwy Dense
function (layer::Dense)(input::GraphNode)::GraphNode # Czyni obiekty Dense "wywoływalnymi": dense_layer(x)
    # input.output to macierz cech (n_features np. 17703 - liczba cech z TF-IDF, n_samples_in_batch np. 64)
    affine_transformation = (layer.W * input) .+ layer.b
    return layer.activation_fn(affine_transformation)
end

# Przejście w przód dla warstwy Conv
function (layer::Conv)(input::GraphNode)::GraphNode
    # Perform convolution
    # For simplicity, we'll use a basic convolution implementation
    # This is a simplified version and might need to be optimized
    input_size = size(input.output)
    output_size = (input_size[1:end-2]..., layer.out_channels, input_size[end])
    output = zeros(Float32, output_size)
    
    # Simple convolution implementation
    for i in 1:input_size[end]  # batch dimension
        for j in 1:layer.out_channels
            for k in 1:layer.in_channels
                # Apply convolution
                # This is a simplified version - in practice, you'd want to use a more efficient implementation
                output[:,:,:,j,i] .+= conv2d(input.output[:,:,:,k,i], layer.W.output[:,:,:,k,j])
            end
            output[:,:,:,j,i] .+= layer.b.output[j]
        end
    end
    
    # Apply activation function
    return layer.activation_fn(Variable(output))
end

# Przejście w przód dla warstwy MaxPool
function (layer::MaxPool)(input::GraphNode)::GraphNode
    # Get input size from the output of the operator
    input_size = size(input.output)
    
    # Calculate output size based on input size, kernel size, and stride
    output_size = Tuple(
        ceil(Int, (input_size[i] - layer.kernel_size[i] + 1) / layer.stride[i])
        for i in 1:length(layer.kernel_size)
    )
    # Add channel and batch dimensions
    output_size = (output_size..., input_size[end-1], input_size[end])
    
    output = zeros(Float32, output_size)
    
    # Perform max pooling
    for i in 1:input_size[end]  # batch dimension
        for j in 1:input_size[end-1]  # channel dimension
            output[:,:,:,j,i] = maxpool2d(input.output[:,:,:,j,i], layer.kernel_size, layer.stride)
        end
    end
    
    return Variable(output)
end

# Przejście w przód dla warstwy Flatten
function (layer::Flatten)(input::GraphNode)::GraphNode
    # Reshape the input to a 2D matrix where each column is a flattened sample
    input_size = size(input.output)
    flattened_size = prod(input_size[1:end-1])  # All dimensions except the last (batch) dimension
    output = reshape(input.output, flattened_size, input_size[end])
    return Variable(output)
end

# Przejście w przód dla warstwy Permute
function (layer::Permute)(input::GraphNode)::GraphNode
    # Permute the dimensions of the input tensor
    println("Permute: ", layer.dims)
    println("Input: ", input.output)
    output = permutedims(input.output, layer.dims)
    return Variable(output)
end

# Przejście w przód dla warstwy Embedding
function (layer::Embedding)(input::GraphNode)::GraphNode
    # For a single index
    if ndims(input.output) == 1
        return Variable(layer.W.output[:, input.output])
    end
    
    # For a batch of indices (sequence_length x batch_size)
    if ndims(input.output) == 2
        sequence_length, batch_size = size(input.output)
        embedding_dim = size(layer.W.output, 1)
        output = zeros(Float32, embedding_dim, sequence_length, batch_size)
        
        for i in 1:sequence_length
            for j in 1:batch_size
                idx = input.output[i, j]
                if idx > 0  # Skip padding tokens
                    output[:, i, j] = layer.W.output[:, idx]
                end
            end
        end
        
        return Variable(output)
    end
    
    # For higher dimensional inputs
    if ndims(input.output) > 2
        input_size = size(input.output)
        embedding_dim = size(layer.W.output, 1)
        output_size = (embedding_dim, input_size...)
        output = zeros(Float32, output_size)
        
        # Reshape input to 2D for easier processing
        flat_input = reshape(input.output, :, input_size[end])
        flat_output = zeros(Float32, embedding_dim, size(flat_input, 2))
        
        for i in 1:size(flat_input, 2)
            idx = flat_input[i]
            if idx > 0  # Skip padding tokens
                flat_output[:, i] = layer.W.output[:, idx]
            end
        end
        
        # Reshape back to original dimensions
        output = reshape(flat_output, output_size)
        return Variable(output)
    end
end

# Helper function for 2D convolution
function conv2d(input, kernel)
    # Simple 2D convolution implementation
    # This is a basic version - in practice, you'd want to use a more efficient implementation
    input_size = size(input)
    kernel_size = size(kernel)
    output_size = (input_size[1] - kernel_size[1] + 1, input_size[2] - kernel_size[2] + 1)
    output = zeros(Float32, output_size)
    
    for i in 1:output_size[1]
        for j in 1:output_size[2]
            output[i,j] = sum(input[i:i+kernel_size[1]-1, j:j+kernel_size[2]-1] .* kernel)
        end
    end
    
    return output
end

# Helper function for 2D max pooling
function maxpool2d(input, kernel_size, stride)
    input_size = size(input)
    output_size = Tuple(
        ceil(Int, (input_size[i] - kernel_size[i] + 1) / stride[i])
        for i in 1:length(kernel_size)
    )
    output = zeros(Float32, output_size)
    
    for i in 1:output_size[1]
        for j in 1:output_size[2]
            # Calculate window boundaries
            start_i = (i-1) * stride[1] + 1
            end_i = min(start_i + kernel_size[1] - 1, input_size[1])
            start_j = (j-1) * stride[2] + 1
            end_j = min(start_j + kernel_size[2] - 1, input_size[2])
            
            # Get the window and find maximum
            window = input[start_i:end_i, start_j:end_j]
            output[i,j] = maximum(window)
        end
    end
    
    return output
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