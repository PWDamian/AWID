using ..AutoDiff: flatten

struct Flatten <: AbstractLayer end # nie ma parametrów do nauczenia ani konfiguracji

function (layer::Flatten)(input_node::GraphNode)::GraphNode
    return flatten(input_node)
end

function parameters(layer::Flatten)
    return Variable[]
end

function layer_output_shape(layer::Flatten, input_shape::Tuple{Vararg{Int}})
    flat_dim = prod(input_shape[1:end-1])
    batch_size = input_shape[end]
    return (flat_dim, batch_size)
end
