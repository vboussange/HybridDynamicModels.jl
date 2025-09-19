using Test
using HybridDynamicModels
using Random
using Lux
using ComponentArrays

@testset "ARModel" begin
    # Create a simple AR model with linear dynamics
    layers = (;
        predictor = Dense(2, 2),
        params = ParameterLayer(init_value = (decay = [0.9],))
    )
    
    function ar_step(layers, u, ps, t)
        # Simple autoregressive step: next = decay * predictor(current)
        pred = layers.predictor(u, ps.predictor)
        decay = layers.params(ps.params).decay
        return pred .* decay
    end
    
    model = ARModel(layers, ar_step; dt = 0.1)
    
    # Test setup
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    @testset "Forward" begin
        # Test forward pass
        x = (u0 = [1.0, 0.5], tspan = (0.0, 0.5), saveat = 0.0:0.1:0.5)
        sol, new_st = model(x, ps, st)
        
        @test size(sol) == (2, 6)  # 2 variables, 6 time points
        @test sol[:, 1] ≈ [1.0, 0.5]  # Initial condition preserved
        @test all(isfinite.(sol))  # All values should be finite
    end
    
    @testset "Bathed Forward" begin
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
end
