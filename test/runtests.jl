# using HybridModelling
using Test

@testset "HybridModelling.jl" begin
    # Write your tests here.
    include("data.jl")
    include("constraints.jl")
    include("layers/parameter.jl")
    include("layers/odemodel.jl")
    
    # Training API tests
    @testset "Training API" begin
        include("train/generics.jl")
        include("train/lux_trainer.jl")
        include("train/mcmc_trainer.jl")
        include("train/vi_trainer.jl")
    end
end
