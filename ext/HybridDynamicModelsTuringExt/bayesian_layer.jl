@concrete struct BayesianLayer <: HybridDynamicModels.BayesianLayer
    layers <: LuxCore.AbstractLuxLayer
    priors <: Union{Distributions.Distribution, NamedTuple}
end
HybridDynamicModels.BayesianLayer(layer, priors) = BayesianLayer(layer, priors)

HybridDynamicModels.getpriors(l::BayesianLayer{L, P}) where {L, P <: NamedTuple} = l.priors

function HybridDynamicModels.getpriors(l::BayesianLayer{
        L, D}) where {L, D <: Distributions.Distribution}
    ps = LuxCore.initialparameters(Random.default_rng(), l)
    if isa(l.priors, UnivariateDistribution)
        distrib = []
        for k in keys(ps)
            push!(distrib, k => arraydist(fill(l.priors, size(ps[k]))))
        end
    else
        for k in keys(ps)
            push!(distrib, k => l.priors)
        end
    end
    return NamedTuple(distrib)
end

HybridDynamicModels.getpriors(l::NamedTuple) = map(HybridDynamicModels.getpriors, l)
HybridDynamicModels.getpriors(::LuxCore.AbstractLuxLayer) = NamedTuple()

function HybridDynamicModels.getpriors(l::LuxCore.AbstractLuxWrapperLayer{layer}) where {layer}
    return HybridDynamicModels.getpriors(getfield(l, layer))
end

function HybridDynamicModels.getpriors(l::LuxCore.AbstractLuxContainerLayer{layers}) where {layers}
    return NamedTuple{layers}(HybridDynamicModels.getpriors.(getfield.((l,), layers)))
end

function HybridDynamicModels.getpriors(l::LuxCore.StatefulLuxLayer)
    return HybridDynamicModels.getpriors(getfield(l, :model))
end

function HybridDynamicModels.getpriors(l)
    LuxCore.contains_lux_layer(l) || throw(MethodError(HybridDynamicModels.getpriors, l))
    return LuxCore.Internal.fmap(
        HybridDynamicModels.getpriors, l; exclude = LuxCore.Internal.isleaf)
end

function Turing.sample(rng::Random.AbstractRNG,
        model::Union{LuxCore.AbstractLuxLayer, LuxCore.StatefulLuxLayer},
        chain::Turing.MCMCChains.Chains, args...; kwargs...)
    priors = HybridDynamicModels.getpriors(model)
    posterior_samples = sample(rng, chain, args...; kwargs...)
    mat = Array(posterior_samples)              # rows = draws, cols = flattened params
    n = size(mat, 1)

    # infer element type from first sample (or from a zero-length dummy)
    elty = if n > 0
        typeof(_vector_to_parameters(mat[1, :], priors))
    else
        typeof(_vector_to_parameters(zeros(Lux.parameterlength(priors)), priors))
    end

    samples = Vector{elty}(undef, n)
    for i in 1:n
        samples[i] = _vector_to_parameters(mat[i, :], priors)
    end

    return samples
end

function Turing.sample(model::Union{LuxCore.AbstractLuxLayer, LuxCore.StatefulLuxLayer},
        chain::Turing.MCMCChains.Chains, args...; kwargs...)
    return Turing.sample(Random.default_rng(), model, chain, args...; kwargs...)
end