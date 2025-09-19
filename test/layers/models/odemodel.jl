using HybridDynamicModels
using DifferentiationInterface
import Zygote
using SciMLSensitivity
using OrdinaryDiffEq
using ComponentArrays
using Lux
using StableRNGs
using Test

@testset "ODEModel" begin
    rng = StableRNG(42)

    layers = (; layer1 = Lux.Dense(10, 10, relu))
    dudt(layers, u, ps, t) = layers.layer1(u, ps.layer1)
    ode_model = ODEModel(layers,
        dudt,
        tspan = (0.0f0, 1.0f0),
        saveat = range(0.0f0, stop = 1.0f0, length = 100),
        alg = Tsit5(),
        abstol = 1.0f-6,
        reltol = 1.0f-6,
        sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)))
    ps, st = Lux.setup(rng, ode_model)
    ps = ComponentArray(ps)

    @testset "ICs in states" begin
        u0 = (; u0 = ones(Float32, 10))
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

    @testset "Neural net as ICLayer" begin
        initial_ics = Dense(1, 10)
        lics = ICLayer(initial_ics)
        model_with_ics = Chain(lics, ode_model)
        ps, st = Lux.setup(rng, model_with_ics)
        ps = ComponentArray(ps)

        # forward pass
        ys = model_with_ics((u0 = [1.0],), ps, st)[1]
        @test size(ys) == (10, 100)

        # batched forward pass
        ys = model_with_ics([(u0 = [1.0],), (u0 = [2.0],)], ps, st)[1]
        @test size(ys) == (10, 100, 2)

        fun = ps -> sum(model_with_ics((u0 = [1.0],), ps, st)[1])
        fun(ps)
        grad = value_and_gradient(fun, AutoZygote(), ps)[2]
        @test all(!isnothing(grad[k] for k in keys(grad)))
    end

    @testset "Single initial condition" begin
        # multiple initial conditions, tokenized
        initial_ics = ParameterLayer(constraint = NoConstraint(),
            init_value = (; u0 = rand(Float32, 10)))
        lics = ICLayer(initial_ics)

        model_with_ics = Chain(lics, ode_model)
        ps, st = LuxCore.setup(rng, model_with_ics)
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
                           init_value = (; u0 = rand(Float32, 10))) for i in 1:5]
        lics = ICLayer(initial_ics)

        model_with_ics = Chain(lics, ode_model)
        ps, st = Lux.setup(rng, model_with_ics)
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
