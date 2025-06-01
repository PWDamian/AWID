module Layers

using ..AutoDiff: Variable, GraphNode, Constant

abstract type AbstractLayer end

include("init.jl")
include("dense.jl")
include("embedding.jl")
include("conv1d.jl")
include("maxpool1d.jl")
include("flatten.jl")
include("chain.jl")

export AbstractLayer, Dense, Embedding, Conv1D, MaxPool1D, Flatten, Chain, parameters, summary, init_xavier_glorot, init_xavier_glorot_conv1d, init_zeros

end