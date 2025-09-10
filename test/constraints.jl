using Test
using HybridModelling
import LuxCore
using Distributions, Bijectors
using Random


@testset "Constraint" begin

    constraint = NoConstraint()
    ps, st = LuxCore.setup(Random.default_rng(), constraint)
    y, _ = HybridModelling.inverse(constraint, [0.5], st)
    x, _ = constraint(y, st)
    @test all(x .== [0.5])

    constraint = BoxConstraint(0., 1.)
    _, st = LuxCore.setup(Random.default_rng(), constraint)
    y, _ = HybridModelling.inverse(constraint, [0.5], st)
    x, _ = constraint(y, st)
    @test all(isapprox.(x, [0.5]))

    constraint = BoxConstraint([0., 0.5, 1.], [1., 1.5, 2.])
    _, st = LuxCore.setup(Random.default_rng(), constraint)
    x = [0.5, 1., 1.5]
    y, _ = HybridModelling.inverse(constraint, x, st)
    @test !all(isapprox.(x, y))
    x_recovered, _ = constraint(y, st)
    @test all(isapprox.(x, x_recovered))

    # Scalar bounds
    constraint = HybridModelling.NamedTupleConstraint((
                                        a = HybridModelling.BoxConstraint(0.0, 1.0),
                                        b = HybridModelling.NoConstraint(),
                                        c = HybridModelling.BoxConstraint(0.0, 1.0),
                                    ))

    _, st = LuxCore.setup(Random.default_rng(), constraint)
    x = (a = [0.0, 0.5, 1.0], b = [-10], c = [0.1])

    y, st = HybridModelling.inverse(constraint, x, st)
    x_recovered, st = constraint(y, st)
    @test all([isapprox(x[k], x_recovered[k]; atol=1e-8) for k in keys(x)])

end
