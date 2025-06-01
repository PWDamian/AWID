module NeuralNetwork

using ..AutoDiff

include("Layers/Layers.jl")
include("activations.jl")
include("losses.jl")
include("optimizers.jl")
include("metrics.jl")
include("training.jl")
include("dataloader.jl")

using .Layers: AbstractLayer, Dense, Embedding, Conv1D, MaxPool1D, Flatten, Chain, parameters, summary, init_xavier_glorot, init_xavier_glorot_conv1d, init_zeros
export AbstractLayer, Dense, Embedding, Conv1D, MaxPool1D, Flatten, Chain, parameters, summary, init_xavier_glorot, init_xavier_glorot_conv1d, init_zeros

using .Activations: relu, sigmoid
export relu, sigmoid

using .Losses: binary_crossentropy
export binary_crossentropy

using .Optimizers: Optimizer, Adam, update!
export Optimizer, Adam, update!

using .Metrics: accuracy
export accuracy

using .Training: setup_training_functions
export setup_training_functions

using .DataLoader: get_epoch_batches
export get_epoch_batches

end