using Test
using HybridModelling
using Distributions, Bijectors


@testset "Constraint" begin
    # Scalar bounds
    x = (a = [0.0, 0.5, 1.0], b = [-10], c = 0.1)
    transform = Bijectors.NamedTransform((
                                        a = bijector(Uniform(0.0, 1.0)),
                                        b = identity,
                                        c = bijector(Uniform(0.0, 1.0)),
                                    ))
    bc = Constraint(transform)

    
    y = HybridModelling._to_optim_space(bc, x)
    x_recovered = bc(y)
    @test all([isapprox(x[k], x_recovered[k]; atol=1e-8) for k in keys(x)])
end
