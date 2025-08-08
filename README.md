[![Build Status](https://github.com/vboussange/HybridModelling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vboussange/HybridModelling.jl/actions/workflows/CI.yml?query=branch%3Amain)

<div align="center">
    <img src="material/logo.svg" alt="MLJ" width="200">
</div>

# HybridModelling.jl

*A toolbox for hybridizing deep learning and traditional dynamical models.*

---
`HybridModelling.jl` is a library providing `Lux.jl` layers and utilities to build, train and evaluate hybrid dynamical models. We view a hybrid dynamical model as a domain-specific model with machine learning components. The library is designed to be hackable and extensible, allowing users to create custom models and training strategies that suit their specific needs.

Features include:
- `Lux.jl` layers
  - to handle hybrid models (`ODEModel`, `AnalyticModel`, `ARModel`)
  - to learn dynamical model initial conditions with projected gradient descent (`LearnableICs`)
  - to learn domain parameters with projected gradient descent (`LearnableParameters`)
  - These layers are of course composable with other `Lux.jl` layers.

- Segmentation utility for training and evaluation (`SegmentedTimeSeries`)

- Loss with support of custom parameter regularization (`MSELoss`)

Because it is built on top of `Lux.jl`, `HybridModelling.jl` is composable with a large ecosystem of libraries, such as `Turing.jl` and `Optimization.jl`. This enables Bayesian inference and advanced optimization strategies.

## Quick example

```julia
using HybridModelling



```

## TODO
- Make a HybridModelZool library (adapting EcoEvoModelZoo.jl)
- Remove SciMLSensitivity, Zygote, MoonCake, DifferentiationInterface, and ComponentArrays dependencies
- Resolve problem with NamedTransformation, which does not work with ComponentArrays
- Debug grad problems with AutoZygote