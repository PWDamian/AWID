using ..AutoDiff: conv1d

struct Conv1D <: AbstractLayer
    weight::Variable{Array{Float32,3}} # (kernel_width, in_channels, out_channels) - out_channels filtrów o wymiarach (kernel_width, in_channels)
    bias::Variable{Vector{Float32}} # (out_channels,) - przesunięcie dodawane do każdego wyjściowego kanału
    activation_fn::Function
end

function Conv1D(
    kernel_width::Int,
    channels_pair::Pair{Int,Int},
    activation_fn::Function;
    init_W::Function=init_xavier_glorot_conv1d,
    init_b::Function=init_zeros,
)
    in_channels = channels_pair.first
    out_channels = channels_pair.second

    W_values = init_W(kernel_width, in_channels, out_channels)
    b_values = init_b(out_channels)
    W = Variable(W_values)
    b = Variable(b_values)

    return Conv1D(W, b, activation_fn)
end

function (layer::Conv1D)(input_node::GraphNode)::GraphNode
    conv_output = conv1d(input_node, layer.weight, layer.bias)
    return layer.activation_fn(conv_output)
end

function parameters(layer::Conv1D)
    return [layer.weight, layer.bias]
end

function layer_output_shape(layer::Conv1D, input_shape::Tuple{Int,Int,Int})
    _in_channels, width, batch_size = input_shape
    kernel_width = size(layer.weight.output, 1)
    out_channels = size(layer.weight.output, 3)
    out_width = width - kernel_width + 1
    return (out_channels, out_width, batch_size)
end
