using HybridDynamicModels
using DifferentiationInterface
import Zygote
using SciMLSensitivity
using OrdinaryDiffEqTsit5
using ComponentArrays
using Lux
using StableRNGs
using Test

rng = StableRNG(42)
layers = (; layer1 = Lux.Dense(10, 10, relu))
tspan = (0.0f0, 1.0f0)
saveat = range(0.0f0, stop = 1.0f0, length = 11)

dudt(layers, u, ps, t) = layers.layer1(u, ps.layer1)
odemodel = ODEModel(layers,
    dudt;
    tspan,
    saveat,
    alg = Tsit5(),
    abstol = 1.0f-6,
    reltol = 1.0f-6,
    sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)))

arstep(layers, u, ps, t) = layers.layer1(u, ps.layer1)
armodel = ARModel(layers, arstep; tspan, saveat, dt = 0.1)

fun(layers, u0, t0, ps, t) = u0 .* exp.(layers.layer1(u0, ps.layer1) .* (t .- t0))
analyticmodel = AnalyticModel(layers, fun; tspan, saveat)

@testset "Dynamic model layers" begin
    @testset "$(nameof(typeof(model)))" for model in (odemodel, armodel, analyticmodel)
        ps, st = Lux.setup(rng, model)
        ps = ComponentArray(ps)

        @testset "ICs in states" begin
            u0 = (; u0 = ones(Float32, 10))
            @testset "Forward pass" begin
                ys = model(u0, ps, st)[1]
                @test size(ys) == (10, 11)
            end

            @testset "Backward pass" begin
                f = ps -> sum(model(u0, ps, st)[1])
                f(ps)
                grad = value_and_gradient(f, AutoZygote(), ps)[2] # fails
                @test all(!isnothing(grad[k] for k in keys(grad)))
            end
        end

        @testset "Neural net as ICLayer" begin
            initial_ics = Dense(1, 10)
            lics = ICLayer(initial_ics)
            model_with_ics = Chain(lics, model)
            ps, st = Lux.setup(rng, model_with_ics)
            ps = ComponentArray(ps)

            # forward pass
            ys = model_with_ics((u0 = [1.0],), ps, st)[1]
            @test size(ys) == (10, 11)

            # batched forward pass
            ys = model_with_ics([(u0 = [1.0],), (u0 = [2.0],)], ps, st)[1]
            @test size(ys) == (10, 11, 2)

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

            model_with_ics = Chain(lics, model)
            ps, st = LuxCore.setup(rng, model_with_ics)
            ps = ComponentArray(ps)
            # forward pass
            x, _ = model_with_ics((), ps, st)
            @test size(x) == (10, 11)

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

            model_with_ics = Chain(lics, model)
            ps, st = Lux.setup(rng, model_with_ics)
            ps = ComponentArray(ps)

            # forward pass
            x, _ = model_with_ics((u0 = 3,), ps, st)
            @test size(x) == (10, 11)

            # batched forward pass
            x, _ = model_with_ics([(u0 = 3,), (u0 = 4,)], ps, st)
            @test size(x) == (10, 11, 2)

            # backward pass
            fun = ps -> sum(model_with_ics((u0 = 3,), ps, st)[1])
            fun(ps)
            grad = value_and_gradient(fun, AutoZygote(), ps)[2]
            @test all(!isnothing(grad[k] for k in keys(grad)))
        end
    end
end
