using Turing, Lux, ComponentArrays, Distributions
using HybridDynamicModels
import HybridDynamicModels: is_ics_estimated, getpriors
using StableRNGs
using Test

function get_parameter_error(st_model, chain, p_true, nsamples = 100)
    nsamples = min(nsamples, size(chain, 1))
    posterior_samples = sample(st_model, chain, nsamples; replace = false)
    err = []
    for ps in posterior_samples
        ps = ps.model.params
        med_par_err = median([median(abs.(ps[k] - p_true[k]) ./ p_true[k])
                              for k in keys(ps)])
        push!(err, med_par_err)
    end
    return median(err)
end

@testset "BayesianLayer" begin
    rng = StableRNG(1234)

    nn1 = BayesianLayer(Dense(10, 5), Normal(0, 1))
    prior_distrib = getpriors(nn1)
    ps_nn1 = Lux.initialparameters(rng, nn1)
    @test keys(prior_distrib) == keys(ps_nn1)
    @test Lux.parameterlength(rand(Distributions.ProductNamedTupleDistribution(prior_distrib))) ==
          Lux.parameterlength(ps_nn1)

    nn2 = BayesianLayer(Dense(5, 2), Normal(0, 1))
    parameters = ParameterLayer(init_value = (; a = [1.0, 2.0]))
    bayes_param_layer = BayesianLayer(
        parameters, (; a = arraydist(fill(Normal(0, 1.0), 2))))
    ch = Chain(nn1, nn2, bayes_param_layer)
    @test getpriors(ch) == (layer_1 = getpriors(nn1), layer_2 = getpriors(nn2),
        layer_3 = getpriors(bayes_param_layer))
end

@testset "MCSamplingBackend, IC inference: $(is_ics_estimated(infer_ics))" for infer_ics in (
    InferICs(false), InferICs(true))
    rng = StableRNG(1234)

    # Define logistic growth model parameters (ground truth)
    K_true = [10.0]  # carrying capacity
    r_true = [1.5]   # growth rate
    N0_true = [1.0] # initial population

    # Define the analytic logistic function
    function logistic_solution(layers, u0, t0, ps, t)
        # Extract parameters
        params = layers.params(ps.params)
        K = params.K
        r = params.r

        # Logistic growth solution: N(t) = K / (1 + (K/N0 - 1)*exp(-r*t))
        return @. K / (1 + (K / u0 - 1) * exp(-r * (t - t0)))
    end

    p_true = (; K = K_true, r = r_true)

    # Create the model
    tspan = (0.0, 5.0)
    tsteps = collect(0.0:0.1:5.0)

    param_priors = (; K = product_distribution([Uniform(5.0, 15.0)]),
        r = product_distribution([Uniform(0.1, 3.0)]))
    layers = (; params = BayesianLayer(ParameterLayer(), param_priors))

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
    datadistrib = x -> Normal(x, 0.1)

    # Create backend
    sampler = NUTS(; adtype = AutoForwardDiff()) # fastest, by far
    backend = MCSamplingBackend(sampler, 1000, datadistrib)

    # Train the model
    result = train(backend, model, dataloader, infer_ics, rng)

    # Test that the result has expected structure
    @test haskey(result, :chains)
    @test haskey(result, :ics)
    @test haskey(result, :st_model)

    err = get_parameter_error(result.st_model, result.chains, p_true)
    @test err < 0.1  # Median parameter error should be less than 10%
end
