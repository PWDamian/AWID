module Losses

using ..AutoDiff: GraphNode, Constant, Variable
using ..Layers

# ŷ - przewidywane etykiety
# y - rzeczywiste etykiety
# epsilon - mała wartość, aby uniknąć log(0) w skrajnych przypadkach
function _binary_crossentropy_graph(ŷ::GraphNode, y::GraphNode, batch_size_val::Int; epsilon::Float32=eps(Float32))
    one_const = Constant(1.0f0)
    eps_const = Constant(epsilon)

    term1 = y .* log.(ŷ .+ eps_const)
    term2 = (one_const .- y) .* log.(one_const .- ŷ .+ eps_const)

    total_loss_for_batch_node = sum(Constant(-1.0f0) .* (term1 .+ term2))

    mean_loss_node = total_loss_for_batch_node ./ Constant(Float32(batch_size_val))

    return mean_loss_node
end

function binary_crossentropy(model::Layers.Chain, x_data::AbstractArray, y_data::AbstractArray; epsilon::Float32=eps(Float32))
    x_node = Variable(x_data)
    y_node = Constant(y_data) # stałe cele

    y_pred_node = model(x_node)

    if ndims(y_data) == 1 # obsługa, gdy y_data to wektor
        batch_size = length(y_data)
    else # obsługa, gdy y_data to macierz 1xN
        batch_size = size(y_data, 2)
    end

    return _binary_crossentropy_graph(y_pred_node, y_node, batch_size; epsilon=epsilon)
end

export binary_crossentropy

end