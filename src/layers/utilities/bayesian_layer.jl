using Distributions
import Turing: arraydist
import LuxCore

"""
    BayesianLayer(layer, priors)

Wrapper layer that adds Bayesian priors to any Lux layer for probabilistic modeling and MCMC inference. Behaves identically to the wrapped layer, but compatible with `getpriors()` to extract prior distributions for MCMC sampling
Required for probabilistic inference with `MCSamplingBackend`. Prior structure should match the parameter structure of the wrapped layer. Use `arraydist()` for array-valued parameters to ensure proper dimensionality
## Arguments
- `layer`: Any Lux layer to be given Bayesian treatment
- `priors`: Prior distributions, either:
  - A single `Distribution` applied to all parameters
  - A `NamedTuple` with distribution for each parameter group
  - Use `arraydist()` for array-valued parameters

## Examples
```julia
# Dense layer with Normal priors on all parameters
dense_layer = Dense(10, 5)
bayesian_dense = BayesianLayer(dense_layer, Normal(0, 1))

# Parameter layer with structured priors
param_layer = ParameterLayer(init_value = (a = [1.0, 2.0], b = 0.5))
priors = (
    a = arraydist(fill(Normal(0, 1), 2)),  # Prior for vector parameter
    b = Normal(0, 0.1)                     # Prior for scalar parameter
)
bayesian_params = BayesianLayer(param_layer, priors)

# ODEModel with parameter priors
ode_model = ODEModel(layers, dudt; kwargs...)
param_priors = (
    params = (
        growth_rate = Uniform(0.1, 2.0),
        carrying_capacity = LogNormal(log(10), 0.5)
    )
)
bayesian_ode = BayesianLayer(ode_model, param_priors)

# Extract priors for MCMC
priors = getpriors(bayesian_ode)
```
"""

@concrete struct BayesianLayer <: LuxCore.AbstractLuxWrapperLayer{:layer}
    layer <: LuxCore.AbstractLuxLayer
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

"""
    getpriors(layer)

Extract prior distributions from Bayesian layers in a model hierarchy.

## Arguments
- `layer`: A composition of `BayesianLayer`s or a `StatefulLuxLayer` wrapping a composition of `BayesianLayer`s.

## Returns
- For `BayesianLayer`: The associated prior distributions
- For container layers (`Chain`, etc.): NamedTuple of priors from child layers  
- For regular layers: Empty NamedTuple `(;)`

## Examples
```julia
# Single Bayesian layer
bayesian_dense = BayesianLayer(Dense(10, 5), Normal(0, 1))
priors = getpriors(bayesian_dense)

# Complex model with multiple Bayesian layers
model = Chain(
    encoder = BayesianLayer(Dense(10, 5), Normal(0, 1)),
    ode_model = BayesianLayer(ode_component, param_priors),
    decoder = Dense(5, 1)  # No priors
)
all_priors = getpriors(model)
# Returns: (encoder = ..., ode_model = ..., decoder = (;))

# Use with MCMC training
backend = MCSamplingBackend(NUTS(), 1000, LogNormal)
priors = getpriors(model)  # Automatically used by MCMC training
```
"""
getpriors(l::BayesianLayer{L, P}) where {L, P <: NamedTuple} = l.priors

function getpriors(l::BayesianLayer{L, D}) where {L, D<:Distributions.Distribution}
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

getpriors(l::NamedTuple) = map(getpriors, l)
getpriors(::LuxCore.AbstractLuxLayer) = NamedTuple()

function getpriors(l::LuxCore.AbstractLuxWrapperLayer{layer}) where {layer}
    return getpriors(getfield(l, layer))
end

function getpriors(l::LuxCore.AbstractLuxContainerLayer{layers}) where {layers}
    return NamedTuple{layers}(getpriors.(getfield.((l,), layers)))
end

function getpriors(l::StatefulLuxLayer)
    return getpriors(getfield(l, :model))
end

function getpriors(l)
    LuxCore.contains_lux_layer(l) || throw(MethodError(getpriors, l))
    return LuxCore.Internal.fmap(getpriors, l; exclude=LuxCore.Internal.isleaf)
end


# TODO: implement test
function Turing.sample(rng::AbstractRNG, model::Union{LuxCore.AbstractLuxLayer, LuxCore.StatefulLuxLayer},
        chain::Turing.MCMCChains.Chains, args...; kwargs...)
    priors = getpriors(model)
    posterior_samples = sample(rng, chain, args...; kwargs...)
    mat = Array(posterior_samples)              # rows = draws, cols = flattened params
    n = size(mat, 1)

    # infer element type from first sample (or from a zero-length dummy)
    elty = if n > 0
        typeof(_vector_to_parameters(mat[1, :], priors))
    else
        typeof(_vector_to_parameters(zeros(LuxCore.parameterlength(priors)), priors))
    end

    samples = Vector{elty}(undef, n)
    for i in 1:n
        samples[i] = _vector_to_parameters(mat[i, :], priors)
    end

    return samples
end

function Turing.sample(model::Union{LuxCore.AbstractLuxLayer, LuxCore.StatefulLuxLayer},
        chain::Turing.MCMCChains.Chains, args...; kwargs...)
    return sample(Random.default_rng(), model, chain, args...; kwargs...)
end