struct Dense <: AbstractLayer
    W::Variable
    b::Variable
    activation_fn::Function
end

# Konstruktor dla Dense
function Dense(
    input_size::Int,
    output_size::Int,
    activation_fn::Function;
    init_W::Function=init_xavier_glorot,
    init_b::Function=init_zeros
)
    W_values = init_W(input_size, output_size)
    b_values = init_b(output_size)
    W = Variable(W_values)
    b = Variable(b_values)

    return Dense(W, b, activation_fn)
end

# Przejście w przód dla warstwy Dense
function (layer::Dense)(input::GraphNode)::GraphNode # Czyni obiekty Dense "wywoływalnymi": dense_layer(x)
    # input.output to macierz cech (n_features np. 17703 - liczba cech z TF-IDF, n_samples_in_batch np. 64)
    affine_transformation = (layer.W * input) .+ layer.b
    return layer.activation_fn(affine_transformation)
end

# Zbieranie parametrów dla optymalizatora
function parameters(layer::Dense)
    return [layer.W, layer.b]
end

function layer_output_shape(layer::Dense, input_shape)
    out_features = size(layer.W.output, 1)
    batch_size = input_shape[end]
    return (out_features, batch_size)
end
