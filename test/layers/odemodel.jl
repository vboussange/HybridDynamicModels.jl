using Bijectors, Distributions
using HybridModelling
using Lux
using Random
using Test
using DifferentiationInterface
import ForwardDiff, Zygote, Enzyme
using SciMLSensitivity
using OrdinaryDiffEq
using ComponentArrays

######## Initial Conditions tests ########
@testset "Initial Conditions" begin
    lics = InitialConditions(ParameterLayer(;init_value = (;u0 = rand(3),))) # initial conditions with no constraints
    ps, st = Lux.setup(Random.default_rng(), lics)
    ps = ComponentArray(ps)
    u0, _, = Lux.apply(lics, (), ps, st) # expected to work, returns all initial conditions
    @test hasproperty(u0, :u0)

    initial_ics = [ParameterLayer(init_value = (;u0 = rand(10))) for _ in 1:5] # a should be in [0, 1], b has no constraints
    lics = InitialConditions(initial_ics)
    ps, st = Lux.setup(Random.default_rng(), lics)
    @test hasproperty(ps, :u0_1)
    u0, _ = lics((u0 = 1,),  ps, st) # expected to work, returns intitial conditions associated to indices
    @test hasproperty(u0, :u0)
    # batch mode
    u0s, _ = lics([(;u0 = 1,),(;u0 = 2,)],  ps, st) # expected to work, returns intitial conditions associated to indices
    @test isa(u0s, AbstractVector)
    @test length(u0s) == 2
    @test hasproperty(u0s[1], :u0)

    # Testing Chain
    many_ics = Chain(lics)
    ps, st = Lux.setup(Random.default_rng(), many_ics)
    u0, _ = many_ics((u0 = 1,), ps, st)
    @test hasproperty(u0, :u0)

    # testing Neural net
    initial_ics = Dense(1, 10)
    lics = InitialConditions(initial_ics)
    ps, st = Lux.setup(Random.default_rng(), lics)
    u0, _ = Lux.apply(lics, (u0 = [1.],),  ps, st) # expected to work, returns intitial conditions associated to indices
    @test hasproperty(u0, :u0)

    # testing gradients
    fun = ps -> sum(lics((;u0 = [1.],),ps, st)[1].u0)
    grad = value_and_gradient(fun, AutoZygote(), ps)[2]
    @test all(!isnothing(grad[k] for k in keys(grad)))
end

######### ODEModel tests ########
@testset "ODEModel" begin
    components = (; layer1 = Lux.Dense(10, 10, relu))
    dudt(components, u, ps, t) = components.layer1(u, ps.layer1)
    ode_model = ODEModel(components, 
                        dudt,
                        tspan = (0.0f0, 1.0f0), 
                        saveat = range(0.0f0, stop=1.0f0, length=100), 
                        alg = Tsit5(),
                        abstol = 1f-6,
                        reltol = 1f-6,
                        sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)))
    ps, st = Lux.setup(Random.default_rng(), ode_model)
    ps = ComponentArray(ps)
    @testset "No chain" begin
        u0 = (;u0 = ones(Float32, 10))
        @testset "Forward pass" begin
            ys = ode_model(u0, ps, st)[1]
            @test size(ys) == (10, 100)
        end

        @testset "Backward pass" begin
            fun = ps -> sum(ode_model(u0, ps, st)[1])
            fun(ps)
            grad = value_and_gradient(fun, AutoZygote(), ps)[2] # fails
            @test all(!isnothing(grad[k] for k in keys(grad)))
        end
    end

    @testset "Neural net as initial conditions" begin
        # Neural net as initial conditions
        initial_ics = Dense(1, 10)
        lics = InitialConditions(initial_ics)
        rng = MersenneTwister(1234)
        model_with_ics = Chain(lics, ode_model)
        ps, st = Lux.setup(rng, model_with_ics)
        ps = ComponentArray(ps)

        # forward pass
        ys = model_with_ics((u0 = [1.],), ps, st)[1]
        @test size(ys) == (10, 100)

        # batched forward pass
        ys = model_with_ics([(u0 = [1.],), (u0 = [2.],)], ps, st)[1]
        @test size(ys) == (10, 100, 2)

        fun = ps -> sum(model_with_ics((u0 = [1.],), ps, st)[1])
        fun(ps)
        grad = value_and_gradient(fun, AutoZygote(), ps)[2]
        @test all(!isnothing(grad[k] for k in keys(grad)))
    end

    @testset "Single initial condition" begin
        # multiple initial conditions, tokenized
        initial_ics = ParameterLayer(constraint = NoConstraint(), 
                                        init_value = (;u0 = rand(Float32, 10)))
        lics = InitialConditions(initial_ics)

        model_with_ics = Chain(lics, ode_model)
        ps, st = Lux.setup(Random.default_rng(), model_with_ics)
        ps = ComponentArray(ps)
        # forward pass
        x, _ = model_with_ics((), ps, st)
        @test size(x) == (10, 100)

        # backward pass
        fun = ps -> sum(model_with_ics((), ps, st)[1])
        fun(ps)
        grad = value_and_gradient(fun, AutoZygote(), ps)[2]
        @test all(!isnothing(grad[k] for k in keys(grad)))
    end

    @testset "Multiple initial conditions" begin
        # multiple initial conditions, tokenized
        initial_ics = [ParameterLayer(constraint = NoConstraint(), 
                    init_value = (;u0 = rand(Float32, 10))) for i in 1:5]
        lics = InitialConditions(initial_ics)

        model_with_ics = Chain(lics, ode_model)
        ps, st = Lux.setup(Random.default_rng(), model_with_ics)
        ps = ComponentArray(ps)

        # forward pass
        x, _ = model_with_ics((u0 = 3,), ps, st)
        @test size(x) == (10, 100)

        # batched forward pass
        x, _ = model_with_ics([(u0 = 3,), (u0 = 4,)], ps, st)
        @test size(x) == (10, 100, 2)

        # backward pass
        fun = ps -> sum(model_with_ics((u0 = 3,), ps, st)[1])
        fun(ps)
        grad = value_and_gradient(fun, AutoZygote(), ps)[2]
        @test all(!isnothing(grad[k] for k in keys(grad)))
    end

end
