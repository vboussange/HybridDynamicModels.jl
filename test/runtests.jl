using HybridDynamicModels
using Test
using Random
using StableRNGs
using DifferentiationInterface
using Zygote
using LuxCore
using ComponentArrays
using Lux
using SciMLSensitivity
using OrdinaryDiffEq

@testset "HybridDynamicModels.jl" begin
    include("qa.jl")

    include("data.jl")
    include("constraints.jl")

    
    @testset "Layers" begin
        include("layers/parameter.jl")
        include("layers/initialconditions.jl")
        include("layers/models.jl")
    end
    
    # Training API tests
    @testset "HybridDynamicModelsLuxExt" begin
        include("ext/HybridDynamicModelsLuxExt.jl")
    end

    @testset "HybridDynamicModelsTuringExt" begin
        include("ext/HybridDynamicModelsTuringExt.jl")
    end
end
