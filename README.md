[![Build Status](https://github.com/vboussange/HybridModelling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vboussange/HybridModelling.jl/actions/workflows/CI.yml?query=branch%3Amain)

<div align="center">
    <img src="material/logo.svg" alt="HybridModelling.jl" width="200">
</div>

# HybridModelling.jl

*A comprehensive toolbox for hybrid dynamical modeling combining domain knowledge with machine learning.*

---

`HybridModelling.jl` provides a unified framework for building, training, and analyzing hybrid dynamical models that seamlessly integrate traditional scientific models with machine learning components. Built on `Lux.jl`, it enables both gradient-based optimization and Bayesian inference for uncertainty quantification.

## 🚀 Key Features

### **Flexible Model Components**
- **`ODEModel`**: Neural ODEs and hybrid differential equation models
- **`ParameterLayer`**: Learnable parameters with optional constraints
- **`InitialConditions`**: Flexible initial condition handling for segmented data
- **`BayesianLayer`**: Add probabilistic priors to any Lux layer

### **Advanced Training Backends** 
- **`LuxBackend`**: Fast gradient-based optimization with automatic differentiation
- **`MCMCBackend`**: Full Bayesian inference with uncertainty quantification  
- **`VIBackend`**: Variational inference for scalable approximate Bayesian methods

### **Data Management**
- **`SegmentedTimeSeries`**: Efficient handling of time series segments for training
- **Flexible constraints**: Physics-informed parameter bounds using `Bijectors.jl`
- **Custom loss functions**: Including `LogMSELoss` for positive-valued data

### **Ecosystem Integration**
- **`Lux.jl`**: Modern neural network framework with excellent AD support
- **`Turing.jl`**: Bayesian inference and probabilistic programming
- **`SciML`**: Differential equations and scientific machine learning
- **`Bijectors.jl`**: Transformations and constraints for parameters

## 📦 Installation

```julia
using Pkg
Pkg.add("HybridModelling")
```

## 🔥 Quick Start

### Basic Hybrid ODE Model

```julia
using HybridModelling
using Lux, OrdinaryDiffEq, Optimisers
using ComponentArrays, Random

# Define hybrid model components
neural_layer = Dense(2, 2, tanh)
param_layer = ParameterLayer(init_value = (growth_rate = [0.1, 0.2],))

# Hybrid dynamics combining neural network and domain knowledge
function hybrid_dynamics(components, u, ps, t)
    # Domain-specific term
    domain_params = components.params(ps.params)
    growth_term = domain_params.growth_rate .* u
    
    # Neural network term  
    neural_term = components.neural(u, ps.neural)
    
    return growth_term + neural_term
end

# Create the ODE model
model = ODEModel(
    (neural = neural_layer, params = param_layer),
    hybrid_dynamics,
    alg = Tsit5(),
    abstol = 1e-6,
    reltol = 1e-6
)

# Setup training data
data = rand(2, 100)  # Your time series data
dataloader = SegmentedTimeSeries(data; segmentsize=20, shift=10)

# Configure training
backend = LuxBackend(
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
param_priors = (growth_rate = Product([Normal(0.1, 0.05), Normal(0.2, 0.05)]),)
bayesian_params = BayesianLayer(param_layer, param_priors)

# Create Bayesian model
bayesian_model = ODEModel(
    (neural = neural_layer, params = bayesian_params),
    hybrid_dynamics,
    alg = Tsit5()
)

# MCMC training for uncertainty quantification
mcmc_backend = MCMCBackend(
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
using Bijectors
constraint = Constraint(bijector(Uniform(0.0, 5.0)))  # Positive initial conditions
infer_ics = InferICs(true, constraint)

# Train with learned initial conditions
result = train(backend, model, dataloader, infer_ics)

# Access learned initial conditions for each segment
for (i, token) in enumerate(tokens(tokenize(dataloader)))
    ic_params = result.best_model.initial_conditions.ics[i]
    println("Segment $i initial condition: ", ic_params)
end
```

## 📚 Documentation

### Core Components

- **[Model Layers](docs/layers.md)**: `ODEModel`, `ParameterLayer`, `InitialConditions`, `BayesianLayer`
- **[Training Backends](docs/training.md)**: `LuxBackend`, `MCMCBackend`, `VIBackend` 
- **[Data Management](docs/data.md)**: `SegmentedTimeSeries`, data preprocessing
- **[Constraints](docs/constraints.md)**: Parameter bounds and transformations

### Examples

- **[Scientific Models](examples/)**: Population dynamics, chemical kinetics, epidemiology
- **[Neural ODEs](examples/neural_ode.jl)**: Pure and hybrid neural differential equations
- **[Bayesian Inference](examples/bayesian.jl)**: Uncertainty quantification workflows
- **[Parameter Estimation](examples/parameter_estimation.jl)**: Learning physical parameters

## 🎯 Use Cases

- **Scientific Computing**: Add ML components to mechanistic models
- **Parameter Estimation**: Learn unknown parameters from data with uncertainty
- **Neural ODEs**: Differentiable programming for continuous-time models  
- **Digital Twins**: Hybrid models combining physics and data-driven components
- **Uncertainty Quantification**: Bayesian approaches for robust predictions

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on the excellent Julia SciML ecosystem, particularly:
- [Lux.jl](https://github.com/LuxDL/Lux.jl) for neural networks
- [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) for automatic differentiation
- [Turing.jl](https://github.com/TuringLang/Turing.jl) for Bayesian inference
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) for differential equations