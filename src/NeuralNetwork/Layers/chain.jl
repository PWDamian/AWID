struct Chain
    layers::Tuple{Vararg{AbstractLayer}}
end

# Konstruktor dla Chain, przyjmuje zmienną liczbę warstw
Chain(layers::Vararg{AbstractLayer}) = Chain((layers...,))

# Przejście w przód dla Chain
function (chain::Chain)(input::GraphNode)::GraphNode # Czyni obiekty Chain "wywoływalnymi": model(x)
    output = input
    for layer in chain.layers
        output = layer(output) # przejście w przód przez każdą warstwę
    end
    return output
end

function parameters(chain::Chain)
    all_params = Variable[] # pusta lista
    for layer in chain.layers
        append!(all_params, parameters(layer))
    end
    return all_params
end

import Base: summary
function summary(chain::Chain, input_shape::Tuple{Vararg{Int}})
    println("Model Summary:")
    println("Input shape: ", input_shape)
    current_shape = input_shape
    for (i, layer) in enumerate(chain.layers)
        out_shape = layer_output_shape(layer, current_shape)
        println("Layer $i: $(typeof(layer)) -> Output shape: $out_shape")
        current_shape = out_shape
    end
end