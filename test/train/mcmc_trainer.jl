using HybridDynamicModelling
using Lux
using Test
using Random
using Distributions
using Turing
using ComponentArrays
using OrdinaryDiffEq
using SciMLSensitivity
using UnPack

@testset "MCMCBackend Tests" begin
    # Setup test data generation
    function generate_bayesian_test_data(; n_segments=2, segment_length=15, noise_level=0.1)
        # True parameters
        p_true = (b = [0.4, 0.6],)
        u0_true = [1.2, 0.9]
        
        # Time points
        tsteps = 1:50
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
        
        # Add multiplicative noise (log-normal)
        rng = MersenneTwister(123)
        data_noisy = data_clean .* exp.(noise_level .* randn(rng, size(data_clean)))
        
        return data_noisy, tsteps, p_true, u0_true
    end
    
    @testset "MCMCBackend Construction" begin
        # Test basic construction
        backend = MCMCBackend(NUTS(0.65), 100, LogNormal)
        @test backend.sampler isa NUTS
        @test backend.n_iterations == 100
        @test backend.datadistrib === LogNormal
        @test backend.kwargs isa NamedTuple
        
        # Test with kwargs
        backend2 = MCMCBackend(HMC(0.01, 5), 50, Normal; progress=false, chain_type=Chains)
        @test backend2.sampler isa HMC
        @test backend2.kwargs.progress == false
        @test backend2.kwargs.chain_type === Chains
        
        # Test nameof function
        @test HybridDynamicModelling.nameof(backend) == "MCMCBackend"
    end
    
    @testset "MCMCBackend Training - Fixed ICs" begin
        # Generate test data
        data, tsteps, p_true, u0_true = generate_bayesian_test_data()
        
        # Create dataloader with fewer segments for faster testing
        dataloader = SegmentedTimeSeries(data; segmentlength=15, shift=10, batchsize=1)
        
        # Create Bayesian model
        param_priors = (b = Product([Uniform(0.1, 1.0), Uniform(0.1, 1.0)]),)
        params = BayesianLayer(
            ParameterLayer(init_value = (b = [0.5, 0.5],)), 
            param_priors
        )
        
        function dudt(layers, u, ps, t)
            p = layers.params(ps.params)
            @unpack b = p
            return 0.1 .* u .* (1.0 .- b .* u)
        end
        
        bayesian_ode = BayesianLayer(
            ODEModel((; params = params), 
                    dudt,
                    alg = Tsit5(),
                    abstol = 1e-6,
                    reltol = 1e-6,
                    sensealg = ForwardDiffSensitivity()),
            (;)  # No additional priors at ODE level
        )
        
        # Setup MCMC training
        backend = MCMCBackend(NUTS(0.65), 50, x -> LogNormal(log(x), 0.15); progress=false)
        infer_ics = InferICs(false)  # Use fixed initial conditions
        
        # Train model
        rng = MersenneTwister(42)
        result = train(backend, bayesian_ode, dataloader, infer_ics, rng)
        
        # Test results structure
        @test haskey(result, :chains)
        @test haskey(result, :st_model)
        @test result.chains isa Chains
        @test result.st_model isa StatefulLuxLayer
        
        # Test chains properties
        @test size(result.chains, 1) == backend.n_iterations  # Number of samples
        @test size(result.chains, 3) == 1  # Number of chains
        
        # Test that we can sample from posterior
        posterior_samples = sample(result.st_model, result.chains, 10)
        @test length(posterior_samples) == 10
        @test all(s isa NamedTuple for s in posterior_samples)
    end
    
    @testset "MCMCBackend Training - Learned ICs" begin
        # Generate simpler test data
        data, tsteps, p_true, u0_true = generate_bayesian_test_data()
        
        # Create smaller dataloader for faster testing
        dataloader = SegmentedTimeSeries(data; segmentlength=12, shift=8, batchsize=1)
        
        # Create Bayesian model with parameter priors only
        param_priors = (b = Product([Uniform(0.2, 0.8), Uniform(0.2, 0.8)]),)
        params = BayesianLayer(
            ParameterLayer(init_value = (b = [0.5, 0.5],)), 
            param_priors
        )
        
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
        
        # Setup MCMC training with learned ICs
        backend = MCMCBackend(NUTS(0.65), 30, x -> LogNormal(log(x), 0.2); progress=false)
        infer_ics = InferICs(true)  # Learn initial conditions
        
        # Train model
        rng = MersenneTwister(456)
        result = train(backend, ode_model, dataloader, infer_ics, rng)
        
        # Test results
        @test result.chains isa Chains
        @test result.st_model isa StatefulLuxLayer
        @test size(result.chains, 1) == backend.n_iterations
        
        # Test posterior sampling
        posterior_samples = sample(result.st_model, result.chains, 5)
        @test length(posterior_samples) == 5
    end
    
    @testset "MCMC Utility Functions" begin
        # Test vector_to_parameters function
        rng = MersenneTwister(42)
        
        # Create a simple NamedTuple structure
        original_ps = (a = rand(rng, 2, 3), b = rand(rng, 4))
        param_length = Lux.parameterlength(original_ps)
        
        # Convert to vector and back
        param_vector = rand(rng, param_length)
        reconstructed_ps = HybridDynamicModelling._vector_to_parameters(param_vector, original_ps)
        
        # Test structure preservation
        @test keys(reconstructed_ps) == keys(original_ps)
        @test size(reconstructed_ps.a) == size(original_ps.a)
        @test size(reconstructed_ps.b) == size(original_ps.b)
        @test Lux.parameterlength(reconstructed_ps) == param_length
        
        # Test parameter length for distributions
        dist = Normal(0, 1)
        @test Lux.parameterlength(dist) == 1
        
        multi_dist = Product([Normal(0, 1), Uniform(0, 1)])
        @test Lux.parameterlength(multi_dist) == 2
    end
    
    @testset "Turing Model Creation" begin
        # Test the create_turing_model function
        rng = MersenneTwister(42)
        
        # Simple model setup
        simple_layer = ParameterLayer(init_value = (a = [1.0, 2.0],))
        bayesian_layer = BayesianLayer(simple_layer, (a = Product([Normal(0, 1), Normal(0, 1)]),))
        
        ps, st = Lux.setup(rng, bayesian_layer)
        st_model = StatefulLuxLayer{true}(bayesian_layer, ps, st)
        
        priors = getpriors(bayesian_layer)
        data_distrib = Normal
        
        # Create Turing model
        turing_model_creator = HybridDynamicModelling.create_turing_model(priors, data_distrib, st_model)
        
        # Test that it creates a callable
        @test turing_model_creator isa Function
        
        # Test with dummy data
        xs = [(a = 1,), (a = 2,)]
        ys = [rand(rng, 2), rand(rng, 2)]
        turing_model = turing_model_creator(xs, ys)
        
        # Test that it creates a DynamicPPL Model
        @test turing_model isa DynamicPPL.Model
    end
    
end
