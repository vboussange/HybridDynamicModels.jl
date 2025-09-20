| **Documentation** | **Build Status** | **Julia** | **Testing** |
|:-----------------:|:----------------:|:---------:|:-----------:|
| [![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://vboussange.github.io/HybridDynamicModels.jl/stable/) [![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vboussange.github.io/HybridDynamicModels.jl/dev/) | [![CI](https://github.com/vboussange/HybridDynamicModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vboussange/HybridDynamicModels.jl/actions/workflows/CI.yml?query=branch%3Amain) | [![Julia](https://img.shields.io/badge/julia-v1.10+-blue.svg)](https://julialang.org/) [![Code Style: Blue](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle) | [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![codecov](https://codecov.io/gh/vboussange/HybridDynamicModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/vboussange/HybridDynamicModels.jl)

![](https://github.com/vboussange/HybridDynamicModels.jl/blob/main/docs/src/assets/logo.svg)

# HybridDynamicModels.jl

*A comprehensive toolbox for hybrid dynamical modeling combining domain knowledge with machine learning.*

---

`HybridDynamicModels.jl` provides a unified framework for building, training, and analyzing hybrid dynamical models that seamlessly integrate traditional scientific models with machine learning layers. Built on `Lux.jl`, it enables both gradient-based optimization and Bayesian inference for uncertainty quantification.

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
- **`SGDBackend`**: Fast gradient-based optimization with automatic differentiation
- **`MCSamplingBackend`**: Full Bayesian inference with uncertainty quantification  
- **`VIBackend`**: Variational inference for scalable approximate Bayesian methods

## 📦 Installation

```julia
using Pkg
Pkg.add("HybridDynamicModels")
```

## 🔥 Quick Start

### Simple Hybrid Model

```julia
using Lux, Optimisers, ComponentArrays
using HybridDynamicModels
using Zygote
using Random

# Dense layer for nonlinear interactions
interaction_layer = Dense(2, 2)

# Parameter layer for growth/decay rates
rate_params = ParameterLayer(init_value = (growth = [0.1], decay = [0.05]))

# Simple hybrid dynamics: linear terms + neural interactions
function ar_step(layers, u, ps, t)
    # Linear terms from parameters
    params = layers.rates(ps.rates)
    linear = [params.growth[1] * u[1], -params.decay[1] * u[2]]
    
    # Neural network interactions
    neural = layers.interaction(u, ps.interaction)
    
    return linear + neural
end

# Create autoregressive model
model = ARModel(
    (interaction = interaction_layer, rates = rate_params),
    ar_step;
    dt = 0.1)

# Setup and train
ps, st = Lux.setup(Random.default_rng(), model)
tsteps = range(0, stop=10.0, step=0.1)
data = rand(2, length(tsteps))

dataloader = SegmentedTimeSeries((data, tsteps); segment_length=20, shift= 2)

backend = SGDBackend(Adam(1e-3), 100, AutoZygote(), MSELoss())
result = train(backend, model, dataloader, InferICs(false))

# Make predictions
tspan = (tsteps[1], tsteps[end])
prediction, _ = model((; u0 = result.ics[1].u0, tspan = tspan, saveat = tsteps), result.ps, result.st)
```

### Bayesian Parameter Estimation

```julia
using Distributions, Turing

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
    dt = 0.1,
)

# MCMC training
datadistrib = Normal
mcmc_backend = MCSamplingBackend(NUTS(0.65), 500, datadistrib)
result = train(mcmc_backend, bayesian_model, dataloader, InferICs(false))

# Sample from posterior
chains = result.chains
posterior_samples = sample(result.st_model, chains, 50)
```

### Learning Initial Conditions

```julia
# Learn initial conditions for each data segment
constraint_u0 = NamedTupleConstraint((; u0 = BoxConstraint([0.1, 0.1], [2.0, 2.0])))  # Reasonable bounds
infer_ics = InferICs(true, constraint_u0)

# Create model for initial condition learning
ic_model = ARModel(
    (interaction = interaction_layer, rates = rate_params),
    ar_step;
    dt = 0.1,
)

# Train with learned initial conditions
result = train(backend, ic_model, dataloader, infer_ics)

# Access learned initial conditions
for (i, ic) in enumerate(result.ics)
    println("Segment $i initial condition: ", ic.u0)
end
```

## 📚 Documentation
### Examples

- **[`data_loading.jl`](examples/data_loading.jl)**: Demonstrates how to use the `SegmentedTimeSeries` data loader for batching and segmentation of time series data.

- **[`sgd_example.jl`](examples/sgd_example.jl)**: Complete example showcasing gradient-based training with the SGD backend using real Lynx-Hare population data.

- **[`mcsampling_example.jl`](examples/mcsampling_example.jl)**: Bayesian parameter estimation example using MCMC sampling with the MCSamplingBackend.

### API
See [the documentation](https://vboussange.github.io/HybridDynamicModels.jl/dev/api/).

## 🙏 Acknowledgments

Built on the excellent LuxDL, SciML and TuringLang ecosystem, particularly:
- [Lux.jl](https://github.com/LuxDL/Lux.jl) for neural networks
- [Turing.jl](https://github.com/TuringLang/Turing.jl) for Bayesian inference
- [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) for automatic differentiation
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) for differential equations

