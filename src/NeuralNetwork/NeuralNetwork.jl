module NeuralNetwork

using ..AutoDiff

include("layers.jl")
include("activations.jl")
include("losses.jl")
include("optimizers.jl")

using .Layers: Dense, Chain, parameters
export Dense, Chain, parameters

using .Activations: relu, sigmoid
export relu, sigmoid

using .Losses: binary_crossentropy
export binary_crossentropy

using .Optimizers: Adam, update!
export Adam, update!

end