using Bijectors, Distributions
using HybridModelling
using Lux
using Test
using Random
@testset "BayesianLayer Tests" begin
    nn1 = BayesianLayer(Dense(10, 5), Normal(0, 1))
    nn2 = BayesianLayer(Dense(5, 2), Normal(0, 1))
    prior_distrib = priors(nn1)
    ps_nn1 = Lux.initialparameters(Random.default_rng(), nn1)
    @test keys(prior_distrib) == keys(ps_nn1)
    @test Lux.parameterlength(rand(Distributions.ProductNamedTupleDistribution(prior_distrib))) == Lux.parameterlength(ps_nn1)
    
    ch = Chain(nn1, nn2)
    @test priors(ch) == (layer_1 = priors(nn1), layer_2 = priors(nn2))
end