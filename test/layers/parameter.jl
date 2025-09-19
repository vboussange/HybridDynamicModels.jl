@testset "ParameterLayer" begin
    rng = StableRNG(42)
    init_value = (; a = ones(3), b = randn(3))

    # no constraint
    param = ParameterLayer(; constraint = NoConstraint(),
        init_value)

    ps, st = LuxCore.setup(rng, param)
    kwargs, _ = param(ps, st)
    @test all(kwargs.a == ones(3))

    # test with a constraint
    constraint = NamedTupleConstraint((;
        a = BoxConstraint([0.0], [3.0]))
    )
    param = ParameterLayer(; constraint, init_value)

    ps, st = LuxCore.setup(rng, param)
    @test !any(ps.a .== ones(3)) # should be transformed
    kwargs, _ = param(ps, st)
    @test all(kwargs.a .== ones(3))

end
