abstract type AbstractOptimBackend end


"""
    SGDBackend(opt, n_epochs, adtype, loss_fn, callback)

Training backend using Lux.jl for mode estimation.

!!! warning `Conditional loading`
  You need to load `Optimisers`, `ComponentArrays` and `Lux` before loading `HybridDynamicModels` to use [SGDBackend](@ref).

## Arguments
  - `opt`: Optimizers.jl rule for parameter updates.
  - `n_epochs`: Number of training epochs.
  - `adtype`: Automatic differentiation backend.
  - `loss_fn`: Loss function for training.
  - `callback`: User-defined callback function.

## Inputs
- `opt`: Optimization rule (e.g., Adam(1e-3)).
- `n_epochs`: Total training epochs.
- `adtype`: AD backend (e.g., AutoZygote()).
- `loss_fn`: Loss function.
- `callback`: Optional callback.

## Outputs
- NamedTuple with trained parameters and states.

## Behavior
Uses stochastic gradient descent for maximum likelihood estimation.

## Example

```jldoctest
julia> backend = SGDBackend(Adam(1e-3), 1000, AutoZygote(), MSELoss())
```
"""
abstract type SGDBackend <: AbstractOptimBackend end


"""
    MCSamplingBackend(sampler, n_iterations, datadistrib; kwargs...)

Training backend for Bayesian inference using Monte Carlo sampling.
!!! warning `Conditional loading`
  You need to load `Turing`, `ComponentArrays` and `Lux` before loading `HybridDynamicModels` to use [MCSamplingBackend](@ref).

## Arguments
  - `sampler`: Turing.jl MCMC sampler.
  - `n_iterations`: Number of MCMC samples.
  - `datadistrib`: Data distribution for likelihood.
  - `kwargs`: Additional sampler options.

## Inputs
- `sampler`: MCMC sampling algorithm.
- `n_iterations`: Total posterior samples.
- `datadistrib`: Distribution for data likelihood.

## Outputs
- NamedTuple with MCMC chains and model state.

## Behavior
Performs Bayesian inference using MCMC sampling on models with priors.

## Example

```jldoctest
julia> backend = MCSamplingBackend(NUTS(0.65), 1000, LogNormal)
```
"""
abstract type MCSamplingBackend <: AbstractOptimBackend end

"""
    create_turing_model(ps_priors, data_distrib, st_model)

Create a Turing model for Bayesian inference from a `BayesianLayer` model.

# Arguments
- `ps_priors`: A nested structure (typically a NamedTuple) containing prior distributions for model parameters. Each leaf should be a `Distributions.Distribution`.
- `data_distrib`: A function or distribution constructor that creates the likelihood distribution for observed data points.
- `st_model`: A stateful Lux model that can be called with parameters to generate predictions.

# Returns
A function `(xs, ys) -> Turing.Model` that creates a Turing model when given input data `xs` and observed data `ys`.
"""
function create_turing_model end

# abstract type VIBackend <: AbstractOptimBackend end

abstract type AbstractSetup end

"""
    InferICs(infer::Bool, u0_constraint=NoConstraint())

Configuration for initial condition inference in training.

## Arguments
  - `infer`: Whether to treat initial conditions as learnable parameters.
  - `u0_constraint`: Constraint for initial condition optimization.

## Inputs
- `infer`: Boolean flag for inference.
- `u0_constraint`: Constraint object.

## Outputs
- InferICs configuration object.

## Behavior
Controls whether initial conditions are learned or fixed during training.

## Example

```jldoctest
julia> infer_ics = InferICs(true, NoConstraint())
```
"""
struct InferICs{T, U0} <: AbstractSetup
    u0_constraint::U0
end

InferICs(b::Bool, u0_constraint::U0 = NoConstraint()) where {U0} = InferICs{b, U0}(u0_constraint)

is_ics_estimated(::InferICs{val, U0}) where {val, U0} = val
get_u0_constraint(infer_ics::InferICs) = infer_ics.u0_constraint

"""
    train(backend::AbstractOptimBackend, model, dataloader::SegmentedTimeSeries, infer_ics::InferICs, rng=Random.default_rng(); pstype=Lux.f64)

Train a dynamical model using segmented time series data.

## Arguments
  - `backend`: Training configuration and optimization settings.
  - `model`: Lux model to train.
  - `dataloader`: Time series data split into segments.
  - `infer_ics`: Initial condition inference configuration.
  - `rng`: Random number generator.
  - `pstype`: Precision type for parameters.

## Inputs
- `backend`: Training backend (SGDBackend, MCSamplingBackend, etc.).
- `model`: Model to train.
- `dataloader`: SegmentedTimeSeries data.
- `infer_ics`: InferICs configuration.

## Outputs
- NamedTuple with training results (varies by backend).

## Behavior
Trains model using specified backend on segmented time series data.

## Example

```jldoctest
julia> backend = SGDBackend(Adam(1e-3), 1000, AutoZygote(), MSELoss())
julia> dataloader = SegmentedTimeSeries(data, segment_length=20)
julia> infer_ics = InferICs(true)
julia> result = train(backend, model, dataloader, infer_ics)
```
"""
function train end
