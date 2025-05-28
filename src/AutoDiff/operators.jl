import Base: ^
^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x^(n - 1), g * log(abs(x)) * x^n)

import Base: sin
sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

import Base: *
import LinearAlgebra: mul!
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x # matrix multiplication
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

Base.Broadcast.broadcasted(::typeof(+), x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x::Number, y::Number, g::Number) = tuple(g, g)
backward(::BroadcastedOperator{typeof(+)}, x::AbstractArray, y::Number, g::AbstractArray) = tuple(g, sum(g))
backward(::BroadcastedOperator{typeof(+)}, x::Number, y::AbstractArray, g::AbstractArray) = tuple(sum(g), g)
function backward(::BroadcastedOperator{typeof(+)}, x_matrix::AbstractMatrix, y_vector::AbstractVector, g_matrix::AbstractMatrix)
    grad_x = g_matrix
    grad_y = vec(sum(g_matrix, dims=2))
    return tuple(grad_x, grad_y)
end
function backward(::BroadcastedOperator{typeof(+)}, x_vector::AbstractVector, y_matrix::AbstractMatrix, g_matrix::AbstractMatrix)
    grad_x = vec(sum(g_matrix, dims=2))
    grad_y = g_matrix
    return tuple(grad_x, grad_y)
end
function backward(::BroadcastedOperator{typeof(+)}, x_array::AbstractArray, y_array::AbstractArray, g_array::AbstractArray)
    return tuple(g_array, g_array)
end

Base.Broadcast.broadcasted(::typeof(-), x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

Base.Broadcast.broadcasted(::typeof(*), x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y # element-wise multiplication
backward(::BroadcastedOperator{typeof(*)}, x, y, g) = tuple(g .* y, g .* x)

Base.Broadcast.broadcasted(::typeof(/), x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(::BroadcastedOperator{typeof(/)}, x, y, g) = (g ./ y, -g .* x ./ (y .^ 2))

import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = tuple(fill(g, size(x))) # tworzy macierz o tym samym rozmiarze co x, wypełnioną wartością g

import Base: max
Base.Broadcast.broadcasted(::typeof(max), x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = tuple(g .* (x .>= y), g .* (x .< y)) # >= a potem <, żeby gradient popłynął tylko do jednej zmiennej

softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = return exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) =
    let
        y = node.output
        tuple(y .* (g .- sum(y .* g)))
    end

import Base: log
Base.Broadcast.broadcasted(::typeof(log), x::GraphNode) = BroadcastedOperator(log, x) # potrzebny do binarycrossentropy
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) = tuple(g ./ (x))

function σ_internal(x)
    return 1.0f0 ./ (1.0f0 .+ exp.(-x))
end
σ(x::GraphNode) = BroadcastedOperator(σ_internal, x) # sigmoid
forward(::BroadcastedOperator{typeof(σ_internal)}, x) = return σ_internal(x)
backward(node::BroadcastedOperator{typeof(σ_internal)}, x, g) =
    let
        y = node.output
        tuple(g .* y .* (1f0 .- y))
    end

# convs

embedding_op(W::GraphNode, x::GraphNode) = BroadcastedOperator(W, x)
forward(::BroadcastedOperator, W, x) = begin
    println("layer: Embedding forward")
    println("input: ", size(x.output))
    # For a single index
    if ndims(x.output) == 1
        return Variable(W.output[:, x.output])
    end

    # For a batch of indices (sequence_length x batch_size)
    if ndims(x.output) == 2
        sequence_length, batch_size = size(x.output)
        embedding_dim = size(W.output, 1)
        output = zeros(Float32, embedding_dim, sequence_length, batch_size)

        for i in 1:sequence_length
            for j in 1:batch_size
                idx = x.output[i, j]
                if idx > 0  # Skip padding tokens
                    output[:, i, j] = W.output[:, idx]
                end
            end
        end

        return Variable(output)
    end

    # For higher dimensional inputs
    if ndims(x.output) > 2
        input_size = size(x.output)
        embedding_dim = size(W.output, 1)
        output_size = (embedding_dim, input_size...)
        output = zeros(Float32, output_size)

        # Reshape input to 2D for easier processing
        flat_input = reshape(x.output, :, input_size[end])
        flat_output = zeros(Float32, embedding_dim, size(flat_input, 2))

        for i in 1:size(flat_input, 2)
            idx = flat_input[i]
            if idx > 0  # Skip padding tokens
                flat_output[:, i] = W.output[:, idx]
            end
        end

        # Reshape back to original dimensions
        output = reshape(flat_output, output_size)
        return Variable(output)
    end
end
backward(::BroadcastedOperator, W, x, g) = tuple(g, sum(g, dims=2))

conv_op(kernel_size::Constant, in_channels::Constant, out_channels::Constant, W::GraphNode, b::GraphNode, x::GraphNode) =
    BroadcastedOperator(conv_op, kernel_size, in_channels, out_channels, W, b, x)
forward(::BroadcastedOperator{typeof(conv_op)}, kernel_size::Constant, in_channels::Constant, out_channels::Constant, W, b, x) = begin
    println("layer: Conv forward")


    # Perform convolution
    # For simplicity, we'll use a basic convolution implementation
    # This is a simplified version and might need to be optimized
    input_size = size(x.output)
    output_size = (input_size[1:end-2]..., out_channels, input_size[end])
    output = zeros(Float32, output_size)

    # Simple convolution implementation
    for i in 1:input_size[end]  # batch dimension
        for j in 1:out_channels
            for k in 1:in_channels
                # Apply convolution
                # This is a simplified version - in practice, you'd want to use a more efficient implementation
                output[:, :, :, j, i] .+= conv2d(x.output[:, :, :, k, i], W.output[:, :, :, k, j])
            end
            output[:, :, :, j, i] .+= b.output[j]
        end
    end
    # Apply activation function
    return Variable(output)
end
backward(::BroadcastedOperator{typeof(conv_op)}, W, x, g) = tuple(g, sum(g, dims=2))


maxpool_op(kernel_size::Constant, stride::Constant, x::GraphNode) = BroadcastedOperator(maxpool_op, kernel_size, stride, x)
forward(::BroadcastedOperator{typeof(maxpool_op)}, kernel_size::Constant, stride::Constant, x) = begin
    println("layer: MaxPool forward")
    println("input: ", size(x.output))
    println("kernel_size: ", kernel_size)
    println("stride: ", stride)

    # Get input size from the output of the operator
    input_size = size(x.output)
    
    # Calculate output size based on input size, kernel size, and stride
    output_size = Tuple(
        ceil(Int, (input_size[i] - kernel_size[i] + 1) / stride[i])
        for i in 1:length(kernel_size)
    )
    # Add channel and batch dimensions
    output_size = (output_size..., input_size[end-1], input_size[end])
    
    output = zeros(Float32, output_size)
    
    # Perform max pooling
    for i in 1:input_size[end]  # batch dimension
        for j in 1:input_size[end-1]  # channel dimension
            output[:,:,:,j,i] = maxpool2d(x.output[:,:,:,j,i], kernel_size, stride)
        end
    end
    return Variable(output)
end
backward(::BroadcastedOperator{typeof(maxpool_op)}, kernel_size::Constant, stride::Constant, x, g) = tuple(g, sum(g, dims=2))

flatten_op(x::GraphNode) = BroadcastedOperator(flatten_op, x)
forward(::BroadcastedOperator{typeof(flatten_op)}, x) = begin
    println("layer: Flatten forward")
    println("input: ", size(x.output))
    # Reshape the input to a 2D matrix where each column is a flattened sample
    input_size = size(x.output)
    flattened_size = prod(input_size[1:end-1])  # All dimensions except the last (batch) dimension
    output = reshape(x.output, flattened_size, input_size[end])
    return Variable(output)
end
backward(::BroadcastedOperator{typeof(flatten_op)}, x, g) = tuple(g, sum(g, dims=2))



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