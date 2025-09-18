using Bijectors, Distributions
using HybridDynamicModels
using Lux
using Test
using Random
import Turing: arraydist

@testset "BayesianLayer Tests" begin
    nn1 = BayesianLayer(Dense(10, 5), Normal(0, 1))
    prior_distrib = getpriors(nn1)
    ps_nn1 = Lux.initialparameters(Random.default_rng(), nn1)
    @test keys(prior_distrib) == keys(ps_nn1)
    @test Lux.parameterlength(rand(Distributions.ProductNamedTupleDistribution(prior_distrib))) == Lux.parameterlength(ps_nn1)
    
    nn2 = BayesianLayer(Dense(5, 2), Normal(0, 1))
    parameters = ParameterLayer(init_value = (;a = [1., 2.]))
    bayes_param_layer = BayesianLayer(parameters, (;a = arraydist(fill(Normal(0, 1.), 2))))
    ch = Chain(nn1, nn2, bayes_param_layer)
    @test getpriors(ch) == (layer_1 = getpriors(nn1), layer_2 = getpriors(nn2), layer_3 = getpriors(bayes_param_layer))

    @test isa(getpriors([nn1]), Vector)


    @testset "Turing.sample Overload Tests" begin
        # Test the custom sample method for Lux models with MCMC chains
        
        # Create a simple Bayesian model for testing
        param_priors = (weight = Normal(0, 1), bias = Normal(0, 0.5))
        simple_model = BayesianLayer(
            ParameterLayer(init_value = (weight = 1.0, bias = 0.0)),
            param_priors
        )
        
        rng = MersenneTwister(123)
        
        # Generate some test MCMC chains
        # We'll create a synthetic chain with known parameter values
        n_samples = 50
        n_params = 2  # weight and bias
        
        # Create parameter samples manually 
        weight_samples = randn(rng, n_samples) * 0.5 .+ 0.2  # mean ≈ 0.2, std ≈ 0.5
        bias_samples = randn(rng, n_samples) * 0.3 .+ 0.1    # mean ≈ 0.1, std ≈ 0.3
        
        # Stack into matrix format expected by Chains (rows = samples, cols = params)
        param_matrix = hcat(weight_samples, bias_samples)
        
        # Create Turing Chains object
        chain_names = [:weight, :bias]
        chains = Chains(param_matrix, chain_names)
        
        @testset "Basic sample functionality" begin
            # Test the overloaded sample method
            posterior_samples = sample(rng, simple_model, chains, n_samples)
            
            # Check that we get the right number of samples
            @test length(posterior_samples) == n_samples
            
            # Check that each sample has the right structure
            @test all(s isa NamedTuple for s in posterior_samples)
            @test all(haskey(s, :weight) && haskey(s, :bias) for s in posterior_samples)
            
            # Check that sample values are approximately what we expect
            weights = [s.weight for s in posterior_samples]
            biases = [s.bias for s in posterior_samples]
            
            @test all(isfinite.(weights))
            @test all(isfinite.(biases))
            
            # Check that the samples match our input (within tolerance for floating point)
            @test weights ≈ weight_samples atol=1e-10
            @test biases ≈ bias_samples atol=1e-10
        end
        
        @testset "Sample method without RNG" begin
            # Test the version without explicit RNG
            posterior_samples = sample(simple_model, chains, n_samples)
            
            @test length(posterior_samples) == n_samples
            @test all(s isa NamedTuple for s in posterior_samples)
        end
        
        @testset "Different sample counts" begin
            # Test sampling different numbers of samples
            for n_test in [1, 5, 25, 50]
                samples = sample(rng, simple_model, chains, n_test)
                @test length(samples) == n_test
            end
        end
        
        @testset "Empty chains handling" begin
            # Test with zero samples
            empty_matrix = zeros(0, n_params)
            empty_chains = Chains(empty_matrix, chain_names)
            
            empty_samples = sample(rng, simple_model, chains, 0)
            @test length(empty_samples) == 0
            @test empty_samples isa Vector
        end
        
        @testset "Complex model structure" begin
            # Test with a more complex nested parameter structure
            complex_priors = (
                layer1 = (weight = Normal(0, 1), bias = Normal(0, 0.1)),
                layer2 = (weight = Normal(0, 0.5), bias = Normal(0, 0.1)),
                global_param = Uniform(-1, 1)
            )
            
            complex_model = BayesianLayer(
                ParameterLayer(init_value = (
                    layer1 = (weight = 1.0, bias = 0.0),
                    layer2 = (weight = 0.5, bias = 0.0), 
                    global_param = 0.0
                )),
                complex_priors
            )
            
            # Create corresponding chains (5 parameters total)
            n_complex_params = 5
            complex_matrix = randn(rng, 20, n_complex_params) * 0.1
            complex_names = [:layer1_weight, :layer1_bias, :layer2_weight, :layer2_bias, :global_param]
            complex_chains = Chains(complex_matrix, complex_names)
            
            complex_samples = sample(rng, complex_model, complex_chains, 20)
            
            @test length(complex_samples) == 20
            @test all(s isa NamedTuple for s in complex_samples)
            @test all(haskey(s, :layer1) for s in complex_samples)
            @test all(haskey(s, :layer2) for s in complex_samples)
            @test all(haskey(s, :global_param) for s in complex_samples)
            
            # Check nested structure
            first_sample = complex_samples[1]
            @test haskey(first_sample.layer1, :weight)
            @test haskey(first_sample.layer1, :bias)
            @test haskey(first_sample.layer2, :weight)
            @test haskey(first_sample.layer2, :bias)
        end
        
        @testset "StatefulLuxLayer compatibility" begin
            # Test that sample works with StatefulLuxLayer as well
            ps_init, st = Lux.setup(rng, simple_model)
            stateful_model = StatefulLuxLayer{true}(simple_model, ps_init, st)
            
            stateful_samples = sample(rng, stateful_model, chains, 10)
            
            @test length(stateful_samples) == 10
            @test all(s isa NamedTuple for s in stateful_samples)
        end
        
        @testset "Parameter length consistency" begin
            # Test that the parameter vector length matches expectations
            ps_init, _ = Lux.setup(rng, simple_model)
            expected_length = Lux.parameterlength(ps_init)
            
            # Our test chains should have the right number of parameters
            @test size(param_matrix, 2) == expected_length
            
            # Test with a mismatch (should error)
            wrong_matrix = randn(rng, 10, expected_length + 1)  # Too many parameters
            wrong_chains = Chains(wrong_matrix, [chain_names; :extra])
            
            @test_throws AssertionError sample(rng, simple_model, wrong_chains, 10)
        end
        
        @testset "Type preservation" begin
            # Test that parameter types are preserved correctly
            float32_priors = (weight = Normal(0.0f0, 1.0f0), bias = Normal(0.0f0, 0.5f0))
            float32_model = BayesianLayer(
                ParameterLayer(init_value = (weight = 1.0f0, bias = 0.0f0)),
                float32_priors
            )
            
            # Create Float32 chains
            float32_matrix = Float32.(param_matrix)
            float32_chains = Chains(float32_matrix, chain_names)
            
            float32_samples = sample(rng, float32_model, float32_chains, 5)
            
            @test all(typeof(s.weight) == Float32 for s in float32_samples)
            @test all(typeof(s.bias) == Float32 for s in float32_samples)
        end
    end

end

