reset!(node::Constant{T}) where {T} = nothing
reset!(node::Variable{T}) where {T} = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant{T}) where {T} = nothing
compute!(node::Variable{T}) where {T} = nothing
compute!(node::Operator) =
    node.output = forward(node, extract_input_values(node)...) # forward z operators.jl

function forward!(order::Vector{GraphNode})
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end