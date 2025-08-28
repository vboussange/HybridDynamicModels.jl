using Distributions
import Turing: arraydist
import Lux: LuxCore, AbstractLuxContainerLayer

@concrete struct BayesianLayer <: AbstractLuxWrapperLayer{:layer}
    layer <: AbstractLuxLayer
    priors <: Union{Distributions.Distribution, NamedTuple}
end

# TODO: this may be an overkill, since it imposes to define initial parameter values
# At the same time, if we do not do this, we need the user to exactly match the parameter shapes
# function BayesianLayer(layer::L, priors::P) where {L <: AbstractLuxLayer, P <: Union{Distributions.Distribution, NamedTuple}}
#     ps = Lux.initialparameters(Random.default_rng(), layer)
#     if isa(priors, NamedTuple)
#         @assert keys(priors) == keys(ps)
#         @assert all(size(priors[k]) == size(ps[k]) for k in keys(priors))

#     elseif isa(priors, MultivariateDistribution)
#         @assert length(priors) == Lux.parameterlength(ps)
#     end
#     BayesianLayer{L, P}(layer, priors)
# end

# This is a required behavior for Layers with no tunable parameters
# But since use case is marginal, we can also define it by hand
# priors(::Union{AbstractLuxLayer,Nothing}) = (;)

getpriors(l::BayesianLayer{L, P}) where {L, P <: NamedTuple} = l.priors

function getpriors(l::BayesianLayer{L, D}) where {L, D<:Distributions.Distribution}
    ps = Lux.initialparameters(Random.default_rng(), l)
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

getpriors(l::NamedTuple) = map(getpriors, l)
getpriors(::AbstractRNG, AbstractLuxLayer) = NamedTuple()

function getpriors(l::AbstractLuxWrapperLayer{layer}) where {layer}
    return getpriors(getfield(l, layer))
end

function getpriors( l::AbstractLuxContainerLayer{layers}) where {layers}
    return NamedTuple{layers}(getpriors.(getfield.((l,), layers)))
end

function getpriors(l::StatefulLuxLayer)
    return getpriors(getfield(l, :model))
end

function getpriors(l)
    LuxCore.contains_lux_layer(l) || throw(MethodError(getpriors, l))
    return LuxCore.Internal.fmap(getpriors, l; exclude=LuxCore.Internal.isleaf)
end

