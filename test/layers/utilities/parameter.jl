@testset "ParameterLayer" begin
    rng = StableRNG(42)

    # no constraint
    param = ParameterLayer(; constraint = NoConstraint(),
        init_value = (; u0 = ones(10)))

    ps, st = Lux.setup(Random.default_rng(), param)
    kwargs, _ = param(ps, st)
    @test all(kwargs.u0 .≈ ones(10))

    # test with a constraint
    constraint = NamedTupleConstraint((;
        a = BoxConstraint(0.0, 3.0))
    )
    param = ParameterLayer(; constraint, init_value = (; a = ones(3), b = randn(3)))

    ps, st = Lux.setup(Random.default_rng(), param)
    @test !any(ps.a .≈ ones(3)) # should be transformed
    kwargs, _ = param(ps, st)
    @test all(kwargs.a .≈ ones(3))

    # gradient
    fun = ps -> sum(param(ps, st)[1].a)
    grad = value_and_gradient(fun, AutoZygote(), ps)[2]
    @test !isnothing(grad.a)
    @test isnothing(grad.b)
end
