using ..AutoDiff: embedding_lookup

struct Embedding <: AbstractLayer
    weight::Variable
end

function Embedding(vocab_size::Int, embedding_dim::Int; init_W::Function=init_xavier_glorot)
    W_values = init_W(vocab_size, embedding_dim)
    W = Variable(W_values)
    return Embedding(W)
end

function (layer::Embedding)(input_indices_node::GraphNode)::GraphNode
    return embedding_lookup(layer.weight, input_indices_node)
end

function parameters(layer::Embedding)
    return [layer.weight]
end

function layer_output_shape(layer::Embedding, input_shape::Tuple{Int,Int})
    embedding_dim = size(layer.weight.output, 1)
    seq_len, batch_size = input_shape
    return (embedding_dim, seq_len, batch_size)
end
