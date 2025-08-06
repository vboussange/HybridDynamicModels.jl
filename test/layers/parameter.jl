
######## ParameterLayer tests ########
@testset "ParameterLayer" begin
    # no constraint
    param = ParameterLayer(;constraint = NoConstraint(), 
                        init_value = (;u0 = ones(10)),
                        init_state_value = (;tspan = (0.0, 1.0), saveat = range(0.0, stop=1.0, length=100)))

    ps, st = Lux.setup(Random.default_rng(), param)
    kwargs, _ = Lux.apply(param, ps, st)
    @test all(kwargs.u0 .≈ ones(10))

    # test with a constraint
    transform = Bijectors.NamedTransform((
        a = bijector(Uniform(0., 3.0)),
        b = identity)
    )
    constraint = Constraint(transform)
    param = ParameterLayer(; constraint, init_value = (;a = ones(3), b = randn(3)))

    ps, st = Lux.setup(Random.default_rng(), param)
    @test !any(ps.a .≈ ones(3)) # should be transformed
    kwargs, _ = Lux.apply(param, ps, st)
    @test all(kwargs.a .≈ ones(3))

    # LEGACY BEHAVIOR
    # # overwriting the parameter; is there a case where this is desired? This is an alternative to freezing the parameter
    # st = merge(st, (a = 3 * ones(3),)) # states are not transformed
    # kwargs, _ = Lux.apply(param, ps, st)
    # @test all(kwargs.a .≈ 3.)

    # gradient
    transform = Bijectors.NamedTransform((
        a = bijector(Uniform(0., 3.0)),
        b = identity)
    )
    constraint = Constraint(transform)
    param = ParameterLayer(;constraint, init_value = (;a = ones(3), b = randn(3)))

    ps, st = Lux.setup(Random.default_rng(), param)
    fun = ps -> sum(param(ps, st)[1].a)
    grad = value_and_gradient(fun, AutoZygote(), ps)[2]
    @test !isnothing(grad.a)
    @test isnothing(grad.b)
end
