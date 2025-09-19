using Test
using Random
using Lux, Optimisers, ComponentArrays
using HybridDynamicModels
import HybridDynamicModels: is_ics_estimated
using Optimisers
using DifferentiationInterface
using Zygote
using StableRNGs

@testset "HybridDynamicModelsLuxExt Tests" begin
    @testset "SGDBackend, IC inference: $(is_ics_estimated(infer_ics))" for infer_ics in (InferICs(false), InferICs(true))
        rng = StableRNG(1234)
        
        # Define logistic growth model parameters (ground truth)
        K_true = [10.0]  # carrying capacity
        r_true = [1.5]   # growth rate
        N0_true = [1.0 ] # initial population
        
        # Define the analytic logistic function
        function logistic_solution(layers, u0, t0, ps, t)
            # Extract parameters
            params = layers.params(ps.params)
            K = params.K
            r = params.r
            
            # Logistic growth solution: N(t) = K / (1 + (K/N0 - 1)*exp(-r*t))
            return @. K / (1 + (K/u0 - 1) * exp(-r * (t - t0)))
        end
        
        # Create the model
        tspan = (0.0, 5.0)
        tsteps = collect(0.0:0.1:5.0)

        layers = (; params = ParameterLayer(init_value = (K = [8.0], r = [0.3])))
        model = AnalyticModel(layers, logistic_solution)
        ps, st = Lux.setup(rng, model)
        
        # Generate true trajectory
        ps_true = (; params = (; K = K_true, r = r_true))
        data = model((; u0 = N0_true, saveat = tsteps, tspan), ps_true, st)[1]
        # data .+= 0.2 * randn(rng, size(data))  # Add some noise
        
        # Create SegmentedTimeSeries with segment_length = 5
        dataloader = SegmentedTimeSeries((data, tsteps); 
                                        segment_length = 5, 
                                        batchsize = 1,
                                        partial_segment = true)
                
        # Create backend
        opt = Optimisers.Adam(0.01)
        loss_fn = MSELoss()
        backend = SGDBackend(opt, 200, AutoZygote(), loss_fn)
                
        # Train the model
        result = train(backend, model, dataloader, infer_ics, rng)
        
        # Extract recovered parameters
        recovered_K = result.ps.params.K
        recovered_r = result.ps.params.r

        # Test that parameters are recovered reasonably well
        @test isapprox(recovered_K, K_true, atol=1e-3, rtol=1e-3)  # Allow some tolerance
        @test isapprox(recovered_r, r_true, atol=1e-3, rtol=1e-3)  # Allow some tolerance

        # Test that the result has expected structure
        @test haskey(result, :ps)
        @test haskey(result, :st)
        @test haskey(result, :ics)
        
        # Test prediction with recovered parameters
        test_input = (; u0 = result.ics[1].u0, saveat = tsteps, tspan = tspan)
        pred, _ = model(test_input, result.ps, result.st)
        @test size(pred) == size(data)
        @test isapprox(pred, data, atol=1e-2, rtol=1e-2)  # Predictions should be close to data
    end
end
