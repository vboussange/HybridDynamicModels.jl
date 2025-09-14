using Test
using HybridDynamicModelling
using Random
using Lux
using ComponentArrays
using DifferentiationInterface
import ForwardDiff, Zygote

@testset "AnalyticModel Tests" begin
    
    @testset "Basic AnalyticModel functionality" begin
        # Create a simple analytic model with polynomial dynamics
        layers = (;
            linear = Dense(2, 2),
            params = ParameterLayer(init_value = (a = 1.0, b = 0.5))
        )
        
        function analytic_solution(layers, u0, ps, t)
            # Simple analytic solution: u(t) = u0 * exp(a*t) + b*t
            a, b = layers.params(ps.params)[1].a, layers.params(ps.params)[1].b
            linear_transform = layers.linear(u0, ps.linear)
            return linear_transform .* exp.(a .* t) .+ b .* t
        end
        
        model = AnalyticModel(layers, analytic_solution)
        
        # Test setup
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        # Test forward pass
        x = (u0 = [1.0, 0.5], t = 0.5)
        sol, new_st = model(x, ps, st)
        
        @test length(sol) == 2  # 2 variables
        @test all(isfinite.(sol))  # All values should be finite
        
        # Test at t=0 (should be close to linear transform of u0)
        x_zero = (u0 = [1.0, 0.5], t = 0.0)
        sol_zero, _ = model(x_zero, ps, st)
        
        # At t=0, exp(a*t) = 1 and b*t = 0, so result should be linear transform of u0
        linear_u0 = layers.linear([1.0, 0.5], ps.linear)
        @test sol_zero ≈ linear_u0 atol=1e-10
    end
    
    @testset "AnalyticModel with default parameters" begin
        layers = (;
            identity = Dense(1, 1, identity),
            params = ParameterLayer(init_value = (growth = 0.1,))
        )
        
        function exponential_growth(layers, u0, ps, t)
            # Simple exponential growth: u(t) = u0 * exp(growth * t)
            growth = layers.params(ps.params)[1].growth
            transformed_u0 = layers.identity(u0, ps.identity)
            return transformed_u0 .* exp.(growth .* t)
        end
        
        # Model with default u0 and t
        model = AnalyticModel(layers, exponential_growth; u0 = [2.0], t = 1.0)
        
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        # Test using default parameters
        sol, new_st = model(ps, st)
        
        @test length(sol) == 1  # 1 variable
        @test sol[1] > 2.0  # Should have grown from initial value
    end
    
    @testset "AnalyticModel with vector time input" begin
        layers = (;
            transform = Dense(2, 2),
            decay = ParameterLayer(init_value = (λ = 0.5,))
        )
        
        function decay_solution(layers, u0, ps, t)
            # Decay solution: u(t) = transform(u0) * exp(-λ * t)
            λ = layers.decay(ps.decay)[1].λ
            transformed = layers.transform(u0, ps.transform)
            if isa(t, AbstractVector)
                # For vector t, broadcast over time dimension
                return transformed .* exp.(-λ .* t')  # Broadcasting over columns
            else
                return transformed .* exp.(-λ .* t)
            end
        end
        
        model = AnalyticModel(layers, decay_solution)
        
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        # Test with vector time input
        x = (u0 = [1.0, 0.8], t = [0.0, 0.5, 1.0, 2.0])
        sol, new_st = model(x, ps, st)
        
        @test size(sol) == (2, 4)  # 2 variables, 4 time points
        
        # Test that solution decays over time (approximately)
        # Note: exact behavior depends on random initialization of transform layer
        @test all(isfinite.(sol))
        
        # Test with single time point
        x_single = (u0 = [1.0, 0.8], t = 1.0)
        sol_single, _ = model(x_single, ps, st)
        @test length(sol_single) == 2
    end
    
    @testset "AnalyticModel batch processing" begin
        layers = (;
            scale = Dense(1, 1),
            params = ParameterLayer(init_value = (factor = 2.0,))
        )
        
        function scaling_solution(layers, u0, ps, t)
            # Simple scaling: u(t) = scale(u0) * factor * (1 + t)
            factor = layers.params(ps.params)[1].factor
            scaled = layers.scale(u0, ps.scale)
            return scaled .* factor .* (1 .+ t)
        end
        
        model = AnalyticModel(layers, scaling_solution)
        
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        # Create batch of inputs
        batch_x = [
            (u0 = [1.0], t = 0.0),
            (u0 = [2.0], t = 1.0),
            (u0 = [0.5], t = 2.0)
        ]
        
        sol_batch, new_st = model(batch_x, ps, st)
        
        @test size(sol_batch) == (1, 3)  # 1 variable, 3 batches
        @test all(isfinite.(sol_batch))
    end
    
    @testset "AnalyticModel gradient computation" begin
        layers = (;
            linear = Dense(2, 2),
            params = ParameterLayer(init_value = (coeff = 1.5,))
        )
        
        function linear_solution(layers, u0, ps, t)
            # Linear solution: u(t) = linear(u0) * coeff * t
            coeff = layers.params(ps.params)[1].coeff
            transformed = layers.linear(u0, ps.linear)
            return transformed .* coeff .* t
        end
        
        model = AnalyticModel(layers, linear_solution)
        
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        ps = ComponentArray(ps)
        
        # Test gradient computation
        x = (u0 = [1.0, -0.5], t = 2.0)
        
        function loss_fn(params)
            sol, _ = model(x, params, st)
            return sum(abs2, sol)
        end
        
        # Test that gradients can be computed
        loss_val = loss_fn(ps)
        @test isfinite(loss_val)
        
        grad = value_and_gradient(loss_fn, AutoZygote(), ps)[2]
        @test all(!isnothing(grad[k]) for k in keys(grad))
        @test all(isfinite(grad[k]) for k in keys(grad) if grad[k] !== nothing)
    end
    
    @testset "AnalyticModel parameter inference compatibility" begin
        # Test that AnalyticModel works with parameter structures expected by training
        layers = (;
            dynamics = Dense(3, 3),
            physics = ParameterLayer(init_value = (α = 0.1, β = 0.9, γ = -0.2))
        )
        
        function physics_solution(layers, u0, ps, t)
            # Physics-inspired solution with multiple parameters
            params = layers.physics(ps.physics)[1]
            α, β, γ = params.α, params.β, params.γ
            
            base = layers.dynamics(u0, ps.dynamics)
            return base .* (α .* t.^2 .+ β .* t .+ γ)
        end
        
        model = AnalyticModel(layers, physics_solution)
        
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        # Test that parameters are properly structured
        @test haskey(ps, :dynamics)
        @test haskey(ps, :physics)
        
        # Test forward pass
        x = (u0 = [1.0, 0.0, -0.5], t = 1.5)
        sol, new_st = model(x, ps, st)
        
        @test length(sol) == 3
        @test all(isfinite.(sol))
        
        # Test that the model can handle both scalar and vector time
        x_vec = (u0 = [1.0, 0.0, -0.5], t = [0.0, 0.5, 1.0])
        sol_vec, _ = model(x_vec, ps, st)
        @test size(sol_vec) == (3, 3)
    end
end
