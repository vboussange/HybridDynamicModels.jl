[![Build Status](https://github.com/vboussange/HybridDynamicModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vboussange/HybridDynamicModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

<div align="center">
    <img src="material/logo.svg" alt="HybridDynamicModels.jl" width="200">
</div>

# HybridDynamicModels.jl

*A comprehensive toolbox for hybrid dynamical modeling combining domain knowledge with machine learning.*

---

`HybridDynamicModels.jl` provides a unified framework for building, training, and analyzing hybrid dynamical models that seamlessly integrate traditional scientific models with machine learning layers. Built on `Lux.jl`, it enables both gradient-based optimization and Bayesian inference for uncertainty quantification.

## 🚀 Key Features

### **Dynamical model layers**
- **`ICLayer`**: For initial condition inference
- **`ODEModel`**: Neural ODEs
- **`ARModel`**: Autoregressive models
- **`AnalyticModel`**: For explicit dynamical models

### **Utility layers for hybrid modelling**
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

### Basic Hybrid ODE Model

```julia
using HybridDynamicModels
using Lux, OrdinaryDiffEq, Optimisers

# Define hybrid model layers
neural_layer = Chain(Dense(2, 2, tanh), Dense(2, 2))

param_constraint = BoxConstraint([1e-2], [1e0])
param_layer = ParameterLayer(init_value = (growth_rate = [0.1, 0.2]), constraint = param_constraint)

# Hybrid dynamics combining neural network and domain knowledge
function dudt(layers, u, ps, t)
    # Domain-specific term
    domain_params = layers.params(ps.params)
    growth_term = domain_params.growth_rate .* u
    
    # Neural network term  
    neural_term = layers.neural(u, ps.neural)
    
    return growth_term + neural_term
end

# Create the ODE model
model = ODEModel(
    (neural = neural_layer, params = param_layer),
    dudt,
    alg = Tsit5(),
    abstol = 1e-6,
    reltol = 1e-6
)

# Setup training data
data = rand(2, 100)  # Your time series data
dataloader = SegmentedTimeSeries(data; segment_length=20, shift=10)

# Configure training
backend = SGDBackend(
    Adam(1e-3),           # Optimizer
    1000,                 # Number of epochs  
    AutoZygote(),         # AD backend
    MSELoss()            # Loss function
)

# Train the model
result = train(backend, model, dataloader, InferICs(false))
trained_model = result.best_model

# Make predictions
prediction = trained_model((u0 = [1.0, 0.5], tspan = (0.0, 10.0), saveat = 0:0.1:10))
```

### Bayesian Parameter Estimation

```julia
using Distributions, Turing

# Add Bayesian priors to parameters
param_priors = (growth_rate = product_distribution([Normal(0.1, 0.05), Normal(0.2, 0.05)]),)
bayesian_params = BayesianLayer(param_layer, param_priors)

# Create Bayesian model
bayesian_model = ODEModel(
    (neural = neural_layer, params = bayesian_params),
    dudt,
    alg = Tsit5()
)

# MCMC training for uncertainty quantification
mcmc_backend = MCSamplingBackend(
    NUTS(0.65),          # MCMC sampler
    1000,                # Number of samples
    LogNormal            # Data likelihood
)

# Bayesian training
result = train(mcmc_backend, bayesian_model, dataloader, InferICs(true))
chains = result.chains

# Posterior analysis
posterior_samples = sample(result.st_model, chains, 100)
```

### Learning Initial Conditions

```julia
# Configure learnable initial conditions with constraints
constraint_u0 = BoxConstraint([1e-3], [5e0])  # Positive initial conditions
infer_ics = InferICs(true, constraint_u0)

# Train with learned initial conditions
result = train(backend, model, dataloader, infer_ics)

# Access learned initial conditions for each segment
for (i, token) in enumerate(tokens(tokenize(dataloader)))
    ic_params = result.best_model.initial_conditions.ics[i]
    println("Segment $i initial condition: ", ic_params)
end
```

## 📚 Documentation
### Examples
TO COMPLETE
<!-- - **[Scientific Models](examples/)**: Population dynamics, chemical kinetics, epidemiology
- **[Neural ODEs](examples/neural_ode.jl)**: Pure and hybrid neural differential equations
- **[Bayesian Inference](examples/bayesian.jl)**: Uncertainty quantification workflows
- **[Parameter Estimation](examples/parameter_estimation.jl)**: Learning physical parameters -->

### API
Checkout the API documentation.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on the excellent LuxDL, SciML and TuringLang ecosystem, particularly:
- [Lux.jl](https://github.com/LuxDL/Lux.jl) for neural networks
- [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) for automatic differentiation
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) for differential equations
- [Turing.jl](https://github.com/TuringLang/Turing.jl) for Bayesian inference
- [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl) for parameter transformations


## ⏭️ Roadmap
- [ ] Implement ARModel
- [ ] Implement AnalyticModel
