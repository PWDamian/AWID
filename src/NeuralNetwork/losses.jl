module Losses

using ..AutoDiff: GraphNode, Constant

# ŷ - przewidywane etykiety
# y - rzeczywiste etykiety
# epsilon - mała wartość, aby uniknąć log(0) w skrajnych przypadkach
function binary_crossentropy(ŷ::GraphNode, y::GraphNode; batch_size::Int, epsilon::Float32)
    one_const = Constant(1.0f0)
    eps_const = Constant(epsilon)

    term1 = y .* log.(ŷ .+ eps_const)
    term2 = (one_const .- y) .* log.(one_const .- ŷ .+ eps_const)

    total_loss_for_batch_node = sum(Constant(-1.0f0) .* (term1 .+ term2))

    mean_loss_node = total_loss_for_batch_node ./ Constant(Float32(batch_size))

    return mean_loss_node
end

export binary_crossentropy

end