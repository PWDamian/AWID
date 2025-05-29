module Activations

using ..AutoDiff: GraphNode, Constant, σ

function relu(x::GraphNode)::GraphNode
    # println("relu: ", size(x.output))
    v = max.(x, Constant(0.0f0))
    println("v: ", v)
    return v
end

sigmoid(x::GraphNode)::GraphNode = σ(x)

export relu, sigmoid

end