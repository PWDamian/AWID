module NeuralNetwork

using ..AutoDiff

include("layers.jl")
include("activations.jl")
include("losses.jl")
include("optimizers.jl")
include("metrics.jl")
include("training.jl")
include("dataloader.jl")

using .Layers: Dense, Chain, parameters, init_xavier_glorot, init_zeros
export Dense, Chain, parameters, init_xavier_glorot, init_zeros

using .Activations: relu, sigmoid
export relu, sigmoid

using .Losses: binary_crossentropy
export binary_crossentropy

using .Optimizers: Adam, update!
export Adam, update!

using .Metrics: accuracy
export accuracy

using .Training: setup_training_functions
export setup_training_functions

using .DataLoader: get_epoch_batches
export get_epoch_batches

end