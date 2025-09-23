| **Documentation** | **Build Status** | **Julia** | **Testing** |
|:-----------------:|:----------------:|:---------:|:-----------:|
| [![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://vboussange.github.io/HybridDynamicModels.jl/stable/) [![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vboussange.github.io/HybridDynamicModels.jl/dev/) | [![CI](https://github.com/vboussange/HybridDynamicModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vboussange/HybridDynamicModels.jl/actions/workflows/CI.yml?query=branch%3Amain) | [![Julia](https://img.shields.io/badge/julia-v1.10+-blue.svg)](https://julialang.org/) [![Code Style: Blue](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle) | [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![codecov](https://codecov.io/gh/vboussange/HybridDynamicModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/vboussange/HybridDynamicModels.jl)

![](https://github.com/vboussange/HybridDynamicModels.jl/blob/main/docs/src/assets/logo.svg)

# HybridDynamicModels.jl

*[Lux.jl](https://github.com/LuxDL/Lux.jl) layers and utilities to build and train hybrid dynamic models.*

---

`HybridDynamicModels.jl` is a toolbox for easily building and training hybrid dynamic models which combine mechanistic and data driven components. Built on top of the deep learning framework `Lux.jl`, it enables both gradient descent optimization and Bayesian inference.

## 🚀 Key Features

### **Dynamic model layers**
- **`ICLayer`**: For initial condition inference
- **`ODEModel`**: Neural ODEs
- **`ARModel`**: Autoregressive models
- **`AnalyticModel`**: For explicit dynamical models

### **Utility layers for hybrid modeling**
- **`ParameterLayer`**: Learnable parameters, composable with optional `Constraint` layers
- **`BayesianLayer`**: Add probabilistic priors to any Lux layer

### **Data loaders**
- **`SegmentedTimeSeries`**: Time series data loader with segmentation, implementing mini-batching.

### **Training API, with following backends** 
- **`SGDBackend`**: Gradient descent optimization with [`Optimisers.jl`](https://github.com/FluxML/Optimisers.jl) and [`Lux.jl` training API](https://github.com/LuxDL/Lux.jl)
- **`MCSamplingBackend`**: Full Bayesian inference with uncertainty quantification using [`DynamicPPL.jl`](https://github.com/TuringLang/DynamicPPL.jl) and [`Turing.jl`](https://github.com/TuringLang/Turing.jl).

## 📦 Installation

```julia
using Pkg
Pkg.add("HybridDynamicModels")
```

## 🔥 Quick Start

### Autoregressive hybrid model

```julia
using Lux
using HybridDynamicModels
using Random

# Dense layer for interactions
interaction_layer = Dense(2, 2, tanh)

# Parameter layer for growth/decay rates
rate_params = ParameterLayer(init_value = (growth = [0.1], decay = [0.05]),
    constraint = NamedTupleConstraint((; growth = BoxConstraint([0.0], [1.0]),
        decay = BoxConstraint([0.0], [1.0]))
    )
)

# Simple hybrid dynamics: linear terms + neural interactions
function ar_step(layers, u, ps, t)
    # Linear terms from parameters
    params = layers.rates(ps.rates)
    growth = vcat(params.growth, -params.decay)

    # Neural network interactions
    interactions = layers.interaction(u, ps.interaction)

    return u .* (growth + interactions)
end

# Create autoregressive model
model = ARModel(
    (interaction = interaction_layer, rates = rate_params),
    ar_step;
    dt = 0.1)

# Setup and train
ps, st = Lux.setup(Random.default_rng(), model)
tsteps = range(0, stop = 10.0, step = 0.1)

preds, _ = model(
    (; u0 = [1.0, 1.0],
        tspan = (tsteps[1], tsteps[end]),
        saveat = tsteps), ps, st)
size(preds)  # (2, 101)
```
We can predict batches of time series by providing a batch of initial conditions.
```julia
x = [(; u0 = rand(2),
         tspan = (tsteps[1], tsteps[end]),
         saveat = tsteps) for _ in 1:5]
batch_preds, _ = model(x, ps, st)
size(batch_preds)  # (2, 101, 5)
```

We can bind our `model` with an additional layer predicting initial conditions from some other input data using the `ICLayer`.

```julia
ic_layer = ICLayer(Dense(10, 2, tanh))
model_with_ic = Chain(ic_layer, model)
ps, st = Lux.setup(Random.default_rng(), model_with_ic)
x = [(; u0 = rand(10),
         tspan = (tsteps[1], tsteps[end]),
         saveat = tsteps) for _ in 1:5]
batch_preds, _ = model_with_ic(x, ps, st)
size(batch_preds)  # (2, 101, 5)
```

A similar workwflow can be used with `ODEModel` and `AnalyticModel`.

### Training with `Optimisers.jl` through the `SGDBackend`

> ⚠️ The default [`train`] function is opiniated and meant for demonstration purposes. You are encouraged [to define your own training pipeline](https://vboussange.github.io/HybridDynamicModels.jl/stable/examples/customtraining_example/).

```julia
using Lux, Optimisers, ComponentArrays # conditional loading to use `SGDBackend`
using Zygote
data = rand(2, length(tsteps))
dataloader = SegmentedTimeSeries((data, tsteps); segment_length = 10, shift = 2)

backend = SGDBackend(Adam(1e-2), 100, AutoZygote(), MSELoss())
result = train(backend, model, dataloader, InferICs(false))

# Make predictions
tspan = (tsteps[1], tsteps[end])
prediction, _ = model(
    (; u0 = result.ics[1].u0,
        tspan = tspan,
        saveat = tsteps), result.ps, result.st)
```

### Bayesian inference with `Turing.jl` through the `MCSamplingBackend`

```julia
using Distributions, Turing, ComponentArrays # conditional loading to use `MCSamplingBackend`

# Add priors to rate parameters
rate_priors = (
    growth = arraydist([Normal(0.1, 0.05)]),
    decay = arraydist([Normal(0.05, 0.02)])
)
nn_priors = Normal(0, 1)  # Example prior for NN weights

# Create Bayesian model
bayesian_model = ARModel(
    (interaction = BayesianLayer(interaction_layer, nn_priors),
        rates = BayesianLayer(rate_params, rate_priors)),
    ar_step;
    dt = 0.1
)

# MCMC training
datadistrib = Normal
mcmc_backend = MCSamplingBackend(NUTS(0.65), 500, datadistrib)
result = train(mcmc_backend,
    bayesian_model,
    dataloader,
    InferICs(false))

# Sample from posterior
chains = result.chains
posterior_samples = sample(bayesian_model, chains, 50)
```

## 📚 Documentation
Check out [the tutorials](https://vboussange.github.io/HybridDynamicModels.jl/stable/examples/data_loading/) and the [API](https://vboussange.github.io/HybridDynamicModels.jl/stable/api/) in [the documentation](https://vboussange.github.io/HybridDynamicModels.jl/dev/).

## 🙏 Acknowledgments

Built on the excellent LuxDL, SciML and TuringLang ecosystems:
- [Lux.jl](https://github.com/LuxDL/Lux.jl) for neural networks
- [Turing.jl](https://github.com/TuringLang/Turing.jl) for Bayesian inference
- [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) for automatic differentiation
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) for differential equations

