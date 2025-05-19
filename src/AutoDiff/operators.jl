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