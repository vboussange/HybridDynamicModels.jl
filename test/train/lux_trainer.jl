using HybridDynamicModelling
using Lux
using Test
using Random
using ComponentArrays
using ADTypes
using Optimisers
using OrdinaryDiffEq
using SciMLSensitivity
using UnPack

@testset "SGDBackend Tests" begin
    # Setup test data - simple logistic growth model
    function generate_test_data(; n_segments=3, segment_length=20, noise_level=0.05)
        # True parameters
        p_true = (b = [0.5, 0.3],)
        u0_true = [1.0, 0.8]
        
        # Time points
        tsteps = 1:100
        tspan = (Float64(first(tsteps)), Float64(last(tsteps)))
        
        # ODE function
        function dudt(u, p, t)
            @unpack b = p
            return 0.1 .* u .* (1.0 .- b .* u)
        end
        
        # Generate clean data
        prob = ODEProblem(dudt, u0_true, tspan, p_true)
        sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1e-6, reltol=1e-6)
        data_clean = Array(sol)
        
        # Add noise
        rng = MersenneTwister(42)
        data_noisy = data_clean .* (1.0 .+ noise_level .* randn(rng, size(data_clean)))
        
        return data_noisy, tsteps, p_true, u0_true
    end
    
    @testset "SGDBackend Construction" begin
        # Test basic construction
        backend = SGDBackend(Adam(1e-3), 100, AutoZygote(), MSELoss())
        @test backend.opt isa Adam
        @test backend.n_epochs == 100
        @test backend.adtype isa AutoZygote
        @test backend.verbose_frequency == 10
        @test backend.callback isa Function
        
        # Test with custom options
        custom_callback = (l, m, p, s) -> (loss=l,)
        backend2 = SGDBackend(Adam(1e-2), 50, AutoForwardDiff(), MSELoss(); 
                             verbose_frequency=5, callback=custom_callback)
        @test backend2.verbose_frequency == 5
        @test backend2.callback === custom_callback
    end
    
    @testset "SGDBackend Training - Fixed ICs" begin
        # Generate test data
        data, tsteps, p_true, u0_true = generate_test_data()
        
        # Create dataloader
        dataloader = SegmentedTimeSeries(data; segmentlength=20, shift=15, batchsize=1)
        
        # Create model layers
        params = ParameterLayer(init_value = (b = [1.0, 1.0],))
        
        function dudt(layers, u, ps, t)
            p = layers.params(ps.params)
            @unpack b = p
            return 0.1 .* u .* (1.0 .- b .* u)
        end
        
        ode_model = ODEModel((; params = params), 
                           dudt,
                           alg = Tsit5(),
                           abstol = 1e-6,
                           reltol = 1e-6,
                           sensealg = ForwardDiffSensitivity())
        
        # Setup training
        backend = SGDBackend(Adam(1e-2), 10, AutoZygote(), MSELoss(); verbose_frequency=5)
        infer_ics = InferICs(false)  # Use fixed initial conditions
        
        # Train model
        rng = MersenneTwister(42)
        result = train(backend, ode_model, dataloader, infer_ics, rng)
        
        # Test results structure
        @test haskey(result, :best_model)
        @test haskey(result, :info)
        @test result.best_model isa StatefulLuxLayer
        @test result.info isa Vector
        @test length(result.info) == backend.n_epochs
        
        # Test that model can make predictions
        test_input = (u0 = [1.0, 0.8], saveat = 1:10, tspan = (1.0, 10.0))
        prediction = result.best_model(test_input)
        @test prediction isa AbstractArray
        @test size(prediction, 1) == 2  # Two state variables
        @test size(prediction, 2) == 10  # Time points
    end
    
    @testset "SGDBackend Training - Learned ICs" begin
        # Generate test data
        data, tsteps, p_true, u0_true = generate_test_data()
        
        # Create dataloader
        dataloader = SegmentedTimeSeries(data; segmentlength=15, shift=10, batchsize=1)
        
        # Create model layers
        params = ParameterLayer(init_value = (b = [1.0, 1.0],))
        
        function dudt(layers, u, ps, t)
            p = layers.params(ps.params)
            @unpack b = p
            return 0.1 .* u .* (1.0 .- b .* u)
        end
        
        ode_model = ODEModel((; params = params), 
                           dudt,
                           alg = Tsit5(),
                           abstol = 1e-6,
                           reltol = 1e-6,
                           sensealg = ForwardDiffSensitivity())
        
        # Setup training with learnable initial conditions
        backend = SGDBackend(Adam(1e-2), 15, AutoZygote(), MSELoss(); verbose_frequency=10)
        infer_ics = InferICs(true, NoConstraint())  # Learn initial conditions
        
        # Train model
        rng = MersenneTwister(123)
        result = train(backend, ode_model, dataloader, infer_ics, rng)
        
        # Test results
        @test result.best_model isa StatefulLuxLayer
        @test length(result.info) == backend.n_epochs
        
        # Test prediction with tokenized input
        n_tokens = length(tokens(tokenize(dataloader)))
        test_input = [(u0 = i, saveat = 1:5, tspan = (1.0, 5.0)) for i in 1:min(2, n_tokens)]
        prediction = result.best_model(test_input)
        @test prediction isa AbstractArray
        @test size(prediction, 3) == length(test_input)  # Batch dimension
    end
    
    @testset "SGDBackend Training - Custom Callback" begin
        # Generate minimal test data
        data, tsteps, _, _ = generate_test_data()
        dataloader = SegmentedTimeSeries(data; segmentlength=10, shift=5, batchsize=1)
        
        # Simple model
        params = ParameterLayer(init_value = (b = [1.0, 1.0],))
        function dudt(layers, u, ps, t)
            p = layers.params(ps.params)
            @unpack b = p
            return 0.1 .* u .* (1.0 .- b .* u)
        end
        ode_model = ODEModel((; params = params), dudt, alg = Euler(), dt = 0.1)
        
        # Custom callback that records information
        callback_results = []
        custom_callback = (loss, model, params, states) -> begin
            push!(callback_results, (epoch_loss = loss, param_norm = norm(vec(params))))
            return (loss = loss, param_norm = norm(vec(params)))
        end
        
        backend = SGDBackend(Adam(1e-2), 5, AutoZygote(), MSELoss(); 
                           verbose_frequency=2, callback=custom_callback)
        infer_ics = InferICs(false)
        
        # Train
        result = train(backend, ode_model, dataloader, infer_ics, MersenneTwister(42))
        
        # Test callback was called
        @test length(callback_results) == backend.n_epochs
        @test all(haskey(r, :epoch_loss) for r in callback_results)
        @test all(haskey(r, :param_norm) for r in callback_results)
        
        # Test info contains callback results
        @test length(result.info) == backend.n_epochs
        @test all(haskey(info, :loss) for info in result.info)
        @test all(haskey(info, :param_norm) for info in result.info)
    end
    
    @testset "Helper Functions" begin
        # Test parameter and state extraction
        rng = MersenneTwister(42)
        model = Dense(2, 1)
        ps, st = Lux.setup(rng, model)
        train_state = Lux.Training.TrainState(model, ps, st, Adam())
        
        extracted_ps = HybridDynamicModelling.get_parameter_values(train_state)
        extracted_st = HybridDynamicModelling.get_state_values(train_state)
        
        @test extracted_ps === train_state.parameters
        @test extracted_st === train_state.states
    end
end
