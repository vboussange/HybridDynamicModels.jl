@testset "ICLayer" begin
    rng = StableRNG(42)

     # initial conditions with no constraints
    lics = ICLayer(ParameterLayer(;init_value = (;u0 = rand(3),)))
    ps, st = LuxCore.setup(rng, lics)
    ps = ComponentArray(ps)
    u0, _, = LuxCore.apply(lics, (), ps, st)
    @test hasproperty(u0, :u0)

    # initial conditions with multiple ParameterLayers
    # should return intitial conditions associated to indices
    initial_ics = [ParameterLayer(init_value = (;u0 = rand(10))) for _ in 1:5]
    lics = ICLayer(initial_ics)
    ps, st = LuxCore.setup(rng, lics)
    @test hasproperty(ps, :u0_1)
    u0, _ = lics((u0 = 1,),  ps, st) 
    @test hasproperty(u0, :u0)

    # batch mode
    u0s, _ = lics([(;u0 = 1,),(;u0 = 2,)],  ps, st)
    @test isa(u0s, AbstractVector)
    @test length(u0s) == 2
    @test hasproperty(u0s[1], :u0)

    # Testing Chain
    chain_ics = Lux.Chain(lics)
    ps, st = LuxCore.setup(rng, chain_ics)
    u0, _ = chain_ics((u0 = 1,), ps, st)
    @test hasproperty(u0, :u0)

    # Testing frozen ics
    lics_frozen = Lux.Experimental.FrozenLayer(ICLayer(initial_ics))
    ps, st = LuxCore.setup(rng, lics_frozen)
    u0, _ = lics_frozen((u0 = 1,), ps, st)
    @test hasproperty(u0, :u0)
    @test isempty(ps)

    # testing initial conditions where a Lux layer predicts ics based on predictors
    initial_ics = Dense(1, 10)
    lics = ICLayer(initial_ics)
    ps, st = LuxCore.setup(rng, lics)
    u0, _ = LuxCore.apply(lics, (u0 = [1.],),  ps, st)
    @test hasproperty(u0, :u0)

    # testing gradients
    fun = ps -> sum(lics((;u0 = [1.],),ps, st)[1].u0)
    grad = value_and_gradient(fun, AutoZygote(), ps)[2]
    @test all(!isnothing(grad[k] for k in keys(grad)))
end