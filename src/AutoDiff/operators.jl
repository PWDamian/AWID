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
backward(::BroadcastedOperator{typeof(+)}, x::AbstractMatrix, y::AbstractVector, g::AbstractMatrix) =
    let
        grad_x = g
        grad_y = vec(sum(g, dims=2))
        tuple(grad_x, grad_y)
    end
backward(::BroadcastedOperator{typeof(+)}, x::AbstractVector, y::AbstractMatrix, g::AbstractMatrix) =
    let
        grad_x = vec(sum(g, dims=2))
        grad_y = g
        tuple(grad_x, grad_y)
    end
backward(::BroadcastedOperator{typeof(+)}, x::AbstractArray, y::AbstractArray, g::AbstractArray) =
    let
        tuple(g, g)
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

function embedding_lookup_internal(embedding_weights::Matrix{Float32}, input_indices::AbstractMatrix{<:Int})
    embedding_dim = size(embedding_weights, 1)
    seq_len, batch_size = size(input_indices)
    selected_embeddings = Array{Float32,3}(undef, embedding_dim, seq_len, batch_size) # Dla każdego słowa w każdej sekwencji i batchu, przechowujemy jego embedding jako kolumnę
    for sample_idx in 1:batch_size
        for token_idx in 1:seq_len
            selected_embeddings[:, token_idx, sample_idx] = @view embedding_weights[:, input_indices[token_idx, sample_idx]] # @view automatycznie zamienia wycinek tablicy (np. A[:, i]) w danym bloku na view(A, :, i)
        end
    end
    return selected_embeddings
end
embedding_lookup(embedding_weights_node::GraphNode, indices_node::GraphNode) = BroadcastedOperator(embedding_lookup_internal, embedding_weights_node, indices_node)
forward(::BroadcastedOperator{typeof(embedding_lookup_internal)}, embedding_weights::Matrix{Float32}, input_indices::AbstractMatrix{<:Int}) = embedding_lookup_internal(embedding_weights, input_indices)
backward(::BroadcastedOperator{typeof(embedding_lookup_internal)}, embedding_weights::Matrix{Float32}, input_indices::AbstractMatrix{<:Int}, g::AbstractArray) =
    let
        grad_weights = zero(embedding_weights)
        _embedding_dim, seq_len, batch_size = size(g)
        for sample_idx in 1:batch_size
            for token_idx in 1:seq_len
                @views grad_weights[:, input_indices[token_idx, sample_idx]] .+= g[:, token_idx, sample_idx]
            end
        end
        return tuple(grad_weights, nothing) # nothing to miejsce, gdzie powinien być gradient względem input_indices, ale indeksy są dyskretne i nie różniczkowalne
    end

# zachowuje się jak dla domyślnych wartości we Flux, czyli stride = 1, pad = 0
function conv1d_internal(input_data::Array{Float32,3}, weights::Array{Float32,3}, bias::Vector{Float32})::Array{Float32,3}
    in_channels, sequence_length, batch_size = size(input_data)
    kernel_width, _, out_channels = size(weights)
    output_length = sequence_length - kernel_width + 1

    output = zeros(Float32, out_channels, output_length, batch_size)

    for b in 1:batch_size
        for out_c in 1:out_channels
            filter = view(weights, :, :, out_c)
            for pos in 1:output_length # które przesunięcie filtra
                window = view(input_data, :, pos:pos+kernel_width-1, b)
                sumval = 0.0f0
                for in_c in 1:in_channels
                    for k in 1:kernel_width
                        sumval += window[in_c, k] * filter[k, in_c] # liczenie konwolucji (iloczyn skalarny)
                    end
                end
                output[out_c, pos, b] = sumval + bias[out_c]
            end
        end
    end

    return output
end
conv1d(input_node::GraphNode, weight_node::GraphNode, bias_node::GraphNode)::GraphNode = BroadcastedOperator(conv1d_internal, input_node, weight_node, bias_node)
forward(::BroadcastedOperator{typeof(conv1d_internal)}, input_data::Array{Float32,3}, weights::Array{Float32,3}, bias::Vector{Float32})::Array{Float32,3} = conv1d_internal(input_data, weights, bias)
backward(::BroadcastedOperator{typeof(conv1d_internal)}, input::Array{Float32,3}, weights::Array{Float32,3}, bias::Vector{Float32}, grad_output::Array{Float32,3})::Tuple{Array{Float32,3},Array{Float32,3},Vector{Float32}} =
    let
        in_channels, input_len, batch_size = size(input)
        kernel_width, _, out_channels = size(weights)
        output_len = input_len - kernel_width + 1

        grad_input = zero(input)
        grad_weights = zero(weights)
        grad_bias = zero(bias)

        for out_c in 1:out_channels
            grad_bias[out_c] = sum(view(grad_output, out_c, :, :)) # Bias jest tylko dodawany do każdego wyniku filtru, więc gradient to suma po batchu i czasie
        end

        for b in 1:batch_size
            for out_c in 1:out_channels
                filter = view(weights, :, :, out_c)
                for out_idx in 1:output_len
                    input_start = out_idx
                    input_end = input_start + kernel_width - 1

                    input_slice = view(input, :, input_start:input_end, b)
                    grad = grad_output[out_c, out_idx, b]

                    for in_c in 1:in_channels
                        for k in 1:kernel_width
                            grad_weights[k, in_c, out_c] += input_slice[in_c, k] * grad
                            grad_input[in_c, input_start+k-1, b] += filter[k, in_c] * grad
                        end
                    end
                end
            end
        end

        return tuple(grad_input, grad_weights, grad_bias)
    end

# stride = pool_size
function maxpool1d_internal(input_data::Array{Float32,3}, pool_size::Int)::Array{Float32,3}
    channels, sequence_length, batch_size = size(input_data)
    output_sequence_length = fld(sequence_length, pool_size) # wydajniejsza wersja floor(sequence_length / pool_size)

    output_data = Array{Float32,3}(undef, channels, output_sequence_length, batch_size)

    for b in 1:batch_size
        for c in 1:channels
            for out_seq_idx in 1:output_sequence_length
                start_idx = (out_seq_idx - 1) * pool_size + 1
                end_idx = start_idx + pool_size - 1
                input_window = view(input_data, c, start_idx:end_idx, b)
                output_data[c, out_seq_idx, b] = maximum(input_window)
            end
        end
    end

    return output_data
end
maxpool1d(input_node::GraphNode, pool_size_node::GraphNode) = BroadcastedOperator(maxpool1d_internal, input_node, pool_size_node)
forward(::BroadcastedOperator{typeof(maxpool1d_internal)}, input_data::AbstractArray{T,3}, pool_size::Int) where {T} = maxpool1d_internal(input_data, pool_size)
backward(::BroadcastedOperator{typeof(maxpool1d_internal)}, input_data::Array{Float32,3}, pool_size::Int, g_output::Array{Float32,3})::Tuple{Array{Float32,3},Nothing} =
    let
        channels, _sequence_length, batch_size = size(input_data)
        _, output_sequence_length, _ = size(g_output)

        grad_input = zeros(Float32, size(input_data))

        for b in 1:batch_size
            for c in 1:channels
                for out_idx in 1:output_sequence_length
                    start_idx = (out_idx - 1) * pool_size + 1
                    end_idx = start_idx + pool_size - 1

                    window = @view input_data[c, start_idx:end_idx, b]
                    _, local_max_idx = findmax(window)
                    max_pos = start_idx + local_max_idx - 1

                    grad_input[c, max_pos, b] += g_output[c, out_idx, b] # Tylko pozycja z maksimum dostaje gradient
                end
            end
        end

        return tuple(grad_input, nothing) # Gradient nie przepływa przez pool_size, więc zwracamy nothing jako jego "gradient"
    end

function flatten_internal(input::Array{Float32,N}) where {N}
    dims = size(input)
    batch_size = dims[end]
    num_features = prod(dims[1:end-1])
    return reshape(input, num_features, batch_size)
end
flatten(input_node::GraphNode) = BroadcastedOperator(flatten_internal, input_node)
forward(::BroadcastedOperator{typeof(flatten_internal)}, input::AbstractArray) = flatten_internal(input)
backward(::BroadcastedOperator{typeof(flatten_internal)}, input::Array{Float32,N}, g::Matrix{Float32}) where {N} = tuple(reshape(g, size(input)))