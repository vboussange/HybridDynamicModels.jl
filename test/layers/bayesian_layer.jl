using Bijectors, Distributions
using HybridModelling
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

end