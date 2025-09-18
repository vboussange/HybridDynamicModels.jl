abstract type AbstractOptimBackend end
abstract type AbstractSetup end

"""
    InferICs(infer::Bool, u0_constraint=NoConstraint())

Configuration for initial condition inference in segmented time series training.
Inferring initial conditions increases the number of parameters to optimize.
Constraints help ensure physically meaningful initial conditions

## Arguments
- `infer::Bool`: Whether to treat initial conditions as learnable parameters
  - `true`: Create learnable initial conditions with optional constraints
  - `false`: Use fixed initial conditions extracted from data segments
- `u0_constraint=NoConstraint()`: Constraint for initial condition optimization

## Behavior
- **When `infer=true`**: Initial conditions become trainable parameters,
  allowing the model to learn optimal starting points for each segment
- **When `infer=false`**: Initial conditions are fixed to the first data point
  of each segment

## Utility Functions
- `istrue(infer_ics)`: Returns the boolean inference flag

## Examples
```julia
# Learn unconstrained initial conditions
infer_ics = InferICs(true)

# Learn initial conditions with positivity constraint
using Bijectors, Distributions
constraint = Constraint(bijector(Uniform(0.0, 10.0)))
infer_ics = InferICs(true, constraint)

# Use fixed initial conditions from data
infer_ics = InferICs(false)

# Check configuration
if istrue(infer_ics)
    println("Will learn initial conditions")
else
    println("Will use fixed initial conditions")
end
```

## Use Cases
- **Infer=true**: When initial conditions are uncertain or when segments
  represent different experimental conditions
- **Infer=false**: When initial conditions are well-known or when you want to
  focus only on parameter learning
- **With constraints**: When initial conditions have physical bounds (e.g.,
  concentrations ≥ 0)

"""
struct InferICs{T, U0} <: AbstractSetup
    u0_constraint::U0
end

InferICs(b::Bool, u0_constraint::U0 = NoConstraint()) where {U0} = InferICs{b, U0}(u0_constraint)

is_ics_estimated(::InferICs{val, U0}) where {val, U0} = val
get_u0_constraint(infer_ics::InferICs) = infer_ics.u0_constraint

"""
    train(backend::AbstractOptimBackend, model, dataloader::SegmentedTimeSeries, infer_ics::InferICs, rng=Random.default_rng(); pstype=Lux.f64)

Train a dynamical model using a segmentation training strategy data with optional initial condition inference.

## Arguments
- `backend::AbstractOptimBackend`: Training configuration and optimization settings
- `model`: The Lux model to train (typically a `Chain` with an `ODEModel`)
- `dataloader::SegmentedTimeSeries`: Time series data split into segments for training
- `infer_ics::InferICs`: Configuration for initial condition inference
- `rng=Random.default_rng()`: Random number generator for initialization

## Keyword Arguments
- `luxtype=Lux.f64`: Precision type for model parameters (e.g., `Lux.f32`, `Lux.f64`)

## Returns
A `NamedTuple` with different inputs depending on the backend used.

## Example
```julia
using Lux # required

# Setup
backend = SGDBackend(Adam(1e-3), 1000, AutoZygote(), MSELoss())
dataloader = SegmentedTimeSeries(data, segmentlength=20, shift=5)
infer_ics = InferICs(true, NoConstraint())

# Train
result = train(backend, ode_model, dataloader, infer_ics)
```
"""
function train end
