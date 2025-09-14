using HybridDynamicModelling
using Lux
using Random
using Test

@testset "AnalyticModel Tests" begin
    # Test simple exponential growth model
    @testset "Exponential Growth" begin
        # Create layers
        params = ParameterLayer(init_value = (growth_rate = 0.1,))
        layers = (; params = params)
        
        # Define analytic solution: u(t) = u0 * exp(r * (t - t0))
        function exponential_solution(layers, u0, t0, ps, t)
            params_val = layers.params(ps.params)
            r = params_val[1].growth_rate
            return u0 .* exp.(r .* (t .- t0))
        end
        
        # Create model
        model = AnalyticModel(layers, exponential_solution)
        
        # Setup
        rng = MersenneTwister(42)
        ps, st = Lux.setup(rng, model)
        
        # Test single evaluation
        u0 = [1.0]
        t0 = 0.0
        saveat = [0.0, 1.0, 2.0]
        
        result = model((; u0 = u0, t0 = t0, saveat = saveat), ps, st)
        sol, new_st = result
        
        @test size(sol) == (1, 3)  # 1 state variable, 3 time points
        @test sol[1, 1] ≈ 1.0      # u(0) = 1.0
        @test sol[1, 2] ≈ 1.0 * exp(0.1 * 1.0)  # u(1) = exp(0.1)
        @test sol[1, 3] ≈ 1.0 * exp(0.1 * 2.0)  # u(2) = exp(0.2)
    end
    
    @testset "Multi-dimensional System" begin
        # Create layers with matrix parameters
        params = ParameterLayer(init_value = (rates = [0.1, -0.05],))
        layers = (; params = params)
        
        # Define analytic solution for linear system
        function linear_solution(layers, u0, t0, ps, t)
            params_val = layers.params(ps.params)
            rates = params_val[1].rates
            # Each component evolves independently
            result = zeros(length(u0), length(t))
            for i in 1:length(u0)
                result[i, :] = u0[i] .* exp.(rates[i] .* (t .- t0))
            end
            return result
        end
        
        model = AnalyticModel(layers, linear_solution)
        
        # Setup
        rng = MersenneTwister(123)
        ps, st = Lux.setup(rng, model)
        
        # Test with 2D system
        u0 = [2.0, 1.0]
        t0 = 0.0
        saveat = [0.0, 1.0]
        
        result = model((; u0 = u0, t0 = t0, saveat = saveat), ps, st)
        sol, new_st = result
        
        @test size(sol) == (2, 2)  # 2 state variables, 2 time points
        @test sol[1, 1] ≈ 2.0
        @test sol[2, 1] ≈ 1.0
        @test sol[1, 2] ≈ 2.0 * exp(0.1)
        @test sol[2, 2] ≈ 1.0 * exp(-0.05)
    end
    
    @testset "Batch Processing" begin
        # Simple model
        params = ParameterLayer(init_value = (rate = 0.2,))
        layers = (; params = params)
        
        function simple_solution(layers, u0, t0, ps, t)
            params_val = layers.params(ps.params)
            rate = params_val[1].rate
            return u0 .* (1.0 .+ rate .* (t .- t0))  # Linear growth
        end
        
        model = AnalyticModel(layers, simple_solution)
        
        # Setup
        rng = MersenneTwister(456)
        ps, st = Lux.setup(rng, model)
        
        # Batch inputs
        batch_inputs = [
            (; u0 = [1.0], t0 = 0.0, saveat = [0.0, 1.0]),
            (; u0 = [2.0], t0 = 0.0, saveat = [0.0, 1.0])
        ]
        
        result = model(batch_inputs, ps, st)
        sol, new_st = result
        
        @test size(sol) == (1, 2, 2)  # 1 state, 2 time points, 2 batches
        @test sol[1, 1, 1] ≈ 1.0  # First batch, t=0
        @test sol[1, 2, 1] ≈ 1.0 * (1.0 + 0.2 * 1.0)  # First batch, t=1
        @test sol[1, 1, 2] ≈ 2.0  # Second batch, t=0
        @test sol[1, 2, 2] ≈ 2.0 * (1.0 + 0.2 * 1.0)  # Second batch, t=1
    end
    
    @testset "Default Parameters" begin
        # Model with default kwargs
        params = ParameterLayer(init_value = (rate = 0.1,))
        layers = (; params = params)
        
        function default_solution(layers, u0, t0, ps, t)
            params_val = layers.params(ps.params)
            rate = params_val[1].rate
            return u0 .* exp.(rate .* (t .- t0))
        end
        
        model = AnalyticModel(layers, default_solution; 
                             u0 = [1.0], t0 = 0.0, saveat = [0.0, 1.0])
        
        # Setup
        rng = MersenneTwister(789)
        ps, st = Lux.setup(rng, model)
        
        # Test with default parameters
        result = model(ps, st)
        sol, new_st = result
        
        @test size(sol) == (1, 2)
        @test sol[1, 1] ≈ 1.0
        @test sol[1, 2] ≈ 1.0 * exp(0.1)
    end
end

println("All AnalyticModel tests passed!")
