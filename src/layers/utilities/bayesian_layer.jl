"""
    BayesianLayer(layer, priors)

Wrapper layer that adds Bayesian priors to Lux layers for probabilistic modeling.

## Arguments
  - `layer`: Lux layer to make Bayesian.
  - `priors`: Prior distributions for parameters.

## Inputs
- `layer`: Any Lux layer.
- `priors`: Distribution or NamedTuple of distributions.

## Outputs
- Layer with Bayesian priors for MCMC inference.

## Behavior
Enables probabilistic modeling by attaching priors to layer parameters.

## Example

```jldoctest
julia> dense_layer = Dense(10, 5)
julia> bayesian_dense = BayesianLayer(dense_layer, Normal(0, 1))
```
"""
abstract type BayesianLayer <: HybridDynamicModelsWrapperLayer end


"""
    getpriors(layer)

Extract prior distributions from Bayesian layers.

## Arguments
  - `layer`: Layer or model containing BayesianLayer components.

## Inputs
- `layer`: Bayesian or composite layer.

## Outputs
- NamedTuple of prior distributions.

## Behavior
Recursively extracts priors from Bayesian layers in model hierarchy.

## Example

```jldoctest
julia> priors = getpriors(bayesian_model)
```
"""
function getpriors end