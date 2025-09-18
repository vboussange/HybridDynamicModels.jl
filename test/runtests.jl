using HybridDynamicModels
using Test
using Random
using StableRNGs
using DifferentiationInterface
using Zygote


@testset "HybridDynamicModels.jl" begin
    include("qa.jl")
    
    include("data.jl")
    include("constraints.jl")
    
    # Layer tests
    @testset "Layer Utilities" begin
        include("layers/utilities/parameter.jl")
        include("layers/utilities/bayesian_layer.jl")
    end
    
    @testset "Model Layers" begin
        include("layers/models/odemodel.jl")
        include("layers/models/analyticmodel.jl")
        include("layers/models/armodel.jl")
    end
    
    # Training API tests
    @testset "Training API" begin
        include("train/generics.jl")
        include("train/lux_trainer.jl")
        include("train/mcmc_trainer.jl")
        include("train/vi_trainer.jl")
    end
end
