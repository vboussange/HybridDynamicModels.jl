using Bijectors, Distributions
using HybridDynamicModelling: NamedTupleConstraint, NoConstraint, BoxConstraint
using Lux
using Random
using Test
using ComponentArrays
using DifferentiationInterface
using Zygote

using Test
using HybridDynamicModelling
using Random
using Lux
using Bijectors
using Distributions

######## ParameterLayer tests ########
@testset "ParameterLayer" begin
    # no constraint
    param = ParameterLayer(;constraint = NoConstraint(), 
                        init_value = (;u0 = ones(10)))

    ps, st = Lux.setup(Random.default_rng(), param)
    kwargs, _ = param(ps, st)
    @test all(kwargs.u0 .≈ ones(10))

    # test with a constraint
    constraint = NamedTupleConstraint((;
        a = BoxConstraint(0., 3.0))
    )
    param = ParameterLayer(; constraint, init_value = (;a = ones(3), b = randn(3)))

    ps, st = Lux.setup(Random.default_rng(), param)
    @test !any(ps.a .≈ ones(3)) # should be transformed
    kwargs, _ = param(ps, st)
    @test all(kwargs.a .≈ ones(3))

    # LEGACY BEHAVIOR
    # # overwriting the parameter; is there a case where this is desired? This is an alternative to freezing the parameter
    # st = merge(st, (a = 3 * ones(3),)) # states are not transformed
    # kwargs, _ = Lux.apply(param, ps, st)
    # @test all(kwargs.a .≈ 3.)

    # gradient
    fun = ps -> sum(param(ps, st)[1].a)
    grad = value_and_gradient(fun, AutoZygote(), ps)[2]
    @test !isnothing(grad.a)
    @test isnothing(grad.b)
end
