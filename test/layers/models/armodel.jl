using Test
using HybridModelling
using Random
using Lux
using ComponentArrays

@testset "ARModel Tests" begin
    
    @testset "Basic ARModel functionality" begin
        # Create a simple AR model with linear dynamics
        layers = (;
            predictor = Dense(2, 2),
            params = ParameterLayer(init_value = (decay = 0.9,))
        )
        
        function ar_step(layers, u, ps, t)
            # Simple autoregressive step: next = decay * predictor(current)
            pred = layers.predictor(u, ps.predictor)
            decay = layers.params(ps.params)[1].decay
            return pred .* decay
        end
        
        model = ARModel(layers, ar_step; dt = 0.1)
        
        # Test setup
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        # Test forward pass
        x = (u0 = [1.0, 0.5], t0 = 0.0, saveat = 0.0:0.1:0.5)
        sol, new_st = model(x, ps, st)
        
        @test size(sol) == (2, 6)  # 2 variables, 6 time points
        @test sol[:, 1] ≈ [1.0, 0.5]  # Initial condition preserved
        @test all(isfinite.(sol))  # All values should be finite
    end
    
    @testset "ARModel with default parameters" begin
        layers = (;
            linear = Dense(1, 1, identity),
            params = ParameterLayer(init_value = (scale = 0.8,))
        )
        
        function simple_ar(layers, u, ps, t)
            scaled = layers.linear(u, ps.linear)
            scale = layers.params(ps.params)[1].scale
            return scaled .* scale
        end
        
        # Model with default saveat and dt
        model = ARModel(layers, simple_ar; 
                       u0 = [1.0], 
                       t0 = 0.0, 
                       saveat = 0.0:0.2:1.0, 
                       dt = 0.1)
        
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        # Test using default parameters
        sol, new_st = model(ps, st)
        
        @test size(sol) == (1, 6)  # 1 variable, 6 time points
        @test sol[1, 1] ≈ 1.0  # Initial condition
    end
    
    @testset "ARModel time stepping" begin
        # Test that time stepping works correctly
        layers = (;
            identity = Dense(1, 1, identity),
            counter = ParameterLayer(init_value = (step = 0.1,))
        )
        
        function counting_ar(layers, u, ps, t)
            # Add a small increment at each step
            current = layers.identity(u, ps.identity)
            increment = layers.counter(ps.counter)[1].step
            return current .+ increment
        end
        
        model = ARModel(layers, counting_ar; dt = 0.1)
        
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        x = (u0 = [0.0], t0 = 0.0, saveat = 0.0:0.1:0.5)
        sol, new_st = model(x, ps, st)
        
        @test size(sol) == (1, 6)
        @test sol[1, 1] ≈ 0.0  # Initial condition
        
        # Each step should add approximately 0.1 (accounting for weight initialization)
        # The exact values depend on the random weights, but should be monotonic
        @test all(diff(sol[1, :]) .> 0)  # Should be increasing
    end
    
    @testset "ARModel batch processing" begin
        layers = (;
            transform = Dense(2, 2),
            decay = ParameterLayer(init_value = (rate = 0.95,))
        )
        
        function batch_ar(layers, u, ps, t)
            transformed = layers.transform(u, ps.transform)
            decay_rate = layers.decay(ps.decay)[1].rate
            return transformed .* decay_rate
        end
        
        model = ARModel(layers, batch_ar; dt = 0.1)
        
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        # Create batch of initial conditions
        batch_x = [
            (u0 = [1.0, 0.0], t0 = 0.0, saveat = 0.0:0.1:0.3),
            (u0 = [0.0, 1.0], t0 = 0.0, saveat = 0.0:0.1:0.3),
            (u0 = [0.5, 0.5], t0 = 0.0, saveat = 0.0:0.1:0.3)
        ]
        
        sol_batch, new_st = model(batch_x, ps, st)
        
        @test size(sol_batch) == (2, 4, 3)  # 2 vars, 4 time points, 3 batches
        @test sol_batch[:, 1, 1] ≈ [1.0, 0.0]  # First batch IC
        @test sol_batch[:, 1, 2] ≈ [0.0, 1.0]  # Second batch IC  
        @test sol_batch[:, 1, 3] ≈ [0.5, 0.5]  # Third batch IC
    end
    
    @testset "ARModel parameter inference compatibility" begin
        # Test that ARModel works with parameter structures expected by training
        layers = (;
            dynamics = Dense(2, 2),
            params = ParameterLayer(init_value = (α = 0.1, β = 0.9))
        )
        
        function param_ar(layers, u, ps, t)
            base = layers.dynamics(u, ps.dynamics)
            α, β = layers.params(ps.params)[1].α, layers.params(ps.params)[1].β
            return α .* base .+ β .* u
        end
        
        model = ARModel(layers, param_ar; dt = 0.05)
        
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, model)
        
        # Test that parameters are properly structured
        @test haskey(ps, :dynamics)
        @test haskey(ps, :params)
        
        # Test forward pass
        x = (u0 = [1.0, -0.5], t0 = 0.0, saveat = 0.0:0.1:0.2)
        sol, new_st = model(x, ps, st)
        
        @test size(sol) == (2, 3)
        @test all(isfinite.(sol))
    end
end
