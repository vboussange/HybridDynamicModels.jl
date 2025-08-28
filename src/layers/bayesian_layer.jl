using Distributions
import Functors: fmap
import Turing: arraydist

@concrete struct BayesianLayer <: AbstractLuxWrapperLayer{:layer}
    layer <: AbstractLuxLayer
    priors <: Union{Distributions.Distribution, NamedTuple}
end

function BayesianLayer(layer, priors)
    ps = Lux.initialparameters(Random.default_rng(), layer)
    if isa(priors, NamedTuple)
        @assert keys(priors) == keys(ps)
        @assert all(size(priors[k]) == size(ps[k]) for k in keys(priors))
    end
    BayesianLayer(layer, priors)
end

# priors(::Union{AbstractLuxLayer,Nothing}) = Uniform(-Inf, Inf)
priors(l::BayesianLayer{L, P}) where {L, P <: NamedTuple} = l.priors

function priors(l::BayesianLayer{L, D}) where {L, D<:Distributions.Distribution}
    ps = Lux.initialparameters(Random.default_rng(), l)
    distrib = []
    for k in keys(ps)
        push!(distrib, k => arraydist(fill(l.priors, size(ps[k]))))
    end
    return NamedTuple(distrib)
end

priors(l::NamedTuple) = map(priors, l)

function priors(l::AbstractLuxWrapperLayer{layer}) where {layer}
    return priors(getfield(l, layer))
end
