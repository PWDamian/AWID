using Test
using MLP.AutoDiff
using MLP.NeuralNetwork

@testset "Tests" begin
    include("test_autodiff.jl")
    include("test_neuralnetwork.jl")
end