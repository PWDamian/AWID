# funkcje pomocnicze
update!(node::Constant{T}, gradient::S) where {T,S<:Union{AbstractArray{Float32},Float32,Nothing}} = nothing
update!(node::GraphNode, gradient::S) where {S<:Union{AbstractArray{Float32},Float32,Nothing}} =
    if isnothing(node.gradient)
        node.gradient = gradient # pierwszy gradient dla tego węzła
    else
        node.gradient = node.gradient .+ gradient # (.+) - broadcast/elementwise
    end

function backward!(order::Vector{GraphNode}; seed::Float32=1.0f0)::Nothing
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

# Constant i Variable to zawsze liście (pozostałe węzły to zawsze Operator), na liściach się zatrzymujemy, dlatego nothing
function backward!(node::Constant{T})::Nothing where {T} end
function backward!(node::Variable{T})::Nothing where {T} end
function backward!(node::Operator)::Nothing
    gradients = backward(node, extract_input_values(node)..., node.gradient)
    for (input, gradient) in zip(node.inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end