using Test
using AWID.AutoDiff
using AWID.NeuralNetwork

@testset "Tests" begin
    include("test_autodiff.jl")
    include("test_neuralnetwork.jl")
end