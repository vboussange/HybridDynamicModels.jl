using Test
using HybridDynamicModels
import LuxCore
using Distributions
using Random

@testset "Constraint" begin
    rng = StableRNG(42)

    @testset "NoConstraint" begin
        constraint = NoConstraint()
        ps, st = LuxCore.setup(rng, constraint)
        y, _ = HybridDynamicModels.inverse(constraint, [0.5], st)
        x, _ = constraint(y, st)
        @test all(x .== [0.5])
    end

    @testset "BoxConstraint" begin
        constraint = BoxConstraint([0.0, 0.5, 1.0], [1.0, 1.5, 2.0])
        _, st = LuxCore.setup(rng, constraint)
        x = [0.5, 1.0, 1.5]
        y, _ = HybridDynamicModels.inverse(constraint, x, st)
        @test !all(isapprox.(x, y))
        x_recovered, _ = constraint(y, st)
        @test all(isapprox.(x, x_recovered))
    end

    @testset "NamedTupleConstraint" begin
        constraint = HybridDynamicModels.NamedTupleConstraint((
            a = HybridDynamicModels.BoxConstraint([0.0], [1.0]),
            b = HybridDynamicModels.NoConstraint(),
            c = HybridDynamicModels.BoxConstraint([0.0], [1.0])
        ))

        _, st = LuxCore.setup(rng, constraint)
        x = (a = [0.0, 0.5, 1.0], b = [-10], c = [0.1])

        y, st = HybridDynamicModels.inverse(constraint, x, st)
        x_recovered, st = constraint(y, st)
        @test all([isapprox(x[k], x_recovered[k]; atol = 1e-8) for k in keys(x)])
    end
end
