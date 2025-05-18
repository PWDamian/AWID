module Activations

using ..AutoDiff: GraphNode, Constant, σ

function relu(x::GraphNode)::GraphNode
    return max.(x, Constant(0.0f0))
end

sigmoid(x::GraphNode)::GraphNode = σ(x)

export relu, sigmoid

end