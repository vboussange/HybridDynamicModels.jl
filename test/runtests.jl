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
    
    @testset "Layer Utilities" begin
        include("layers/utilities/parameter.jl")
        include("layers/utilities/initialconditions.jl")
    end
    
    @testset "Model Layers" begin
        include("layers/models/odemodel.jl")
        include("layers/models/analyticmodel.jl")
        include("layers/models/armodel.jl")
    end
    
    # Training API tests
    @testset "HybridDynamicModelsLuxExt" begin
        include("ext/HybridDynamicModelsLuxExt.jl")
    end

    @testset "HybridDynamicModelsTuringExt" begin
        include("ext/HybridDynamicModelsTuringExt.jl")
    end
end
