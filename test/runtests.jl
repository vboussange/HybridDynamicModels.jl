# using HybridModelling
using Test

@testset "HybridModelling.jl" begin
    # Write your tests here.
    include("data.jl")
    include("constraints.jl")
    include("layers/parameter.jl")
    include("layers/odemodel.jl")
end
