using ..AutoDiff: maxpool1d

# stride = pool_size (tak jak domyślnie we Fluxie), czyli okna nie nakładają się
struct MaxPool1D <: AbstractLayer
    pool_size::Int
end

function (layer::MaxPool1D)(input_node::GraphNode)::GraphNode
    pool_size_node = Constant(layer.pool_size)
    return maxpool1d(input_node, pool_size_node)
end

function parameters(layer::MaxPool1D)
    return Variable[]
end

function layer_output_shape(layer::MaxPool1D, input_shape)
    channels, width, batch_size = input_shape
    out_width = fld(width, layer.pool_size)
    return (channels, out_width, batch_size)
end
