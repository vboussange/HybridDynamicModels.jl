using LogExpFunctions

# see https://github.com/TuringLang/Bijectors.jl/blob/00b08eaaae8f5133452e38c1ec949af453d8bbe6/src/Bijectors.jl#L87
function _clamp(x, a, b)
    T = promote_type(typeof(x), typeof(a), typeof(b))
    clamped_x = ifelse(x < a, convert(T, a), ifelse(x > b, convert(T, b), x))
    return clamped_x
end

struct NoConstraint <: AbstractLuxLayer end
LuxCore.initialstates(::AbstractRNG, layer::NoConstraint) = (;)
(n::NoConstraint)(x, st) = x, st
inverse(::NoConstraint, y, st) = y, st


@concrete struct BoxConstraint <: AbstractLuxLayer
  init_state <: Function
end
LuxCore.initialstates(::AbstractRNG, layer::BoxConstraint) = layer.init_state()
BoxConstraint(lb::AbstractArray, ub::AbstractArray) = BoxConstraint(() -> (;lb, ub))

function (::BoxConstraint)(y::AbstractArray, st)
    lb = st.lb
    ub = st.ub
    # elementwise inverse: x = lb + (ub - lb) * logistic(y)
    x = _clamp.(truncated_invlink.(y, lb, ub), lb, ub)
    return x, st
end

function truncated_invlink(y, a, b)
    return a + (b - a) * LogExpFunctions.logistic(y)
end

function inverse(::BoxConstraint, x::AbstractArray, st)
    lb = st.lb
    ub = st.ub
    # elementwise transform: y = logit((x - lb) / (ub - lb))
    return truncated_link.(_clamp.(x, lb, ub), lb, ub), st
end

function truncated_link(x, a, b)
    return LogExpFunctions.logit((x - a) / (b - a))
end


@concrete struct NamedTupleConstraint <: LuxCore.AbstractLuxWrapperLayer{:constraints}
    constraints <: NamedTuple
end
(c::NamedTupleConstraint)(x, st) = applyconstraints(c.constraints, x, st)
inverse(c::NamedTupleConstraint, y, st) = inverseapplyconstraints(c.constraints, y, st)

@generated function inverseapplyconstraints(constraints::NamedTuple{fields}, x::NamedTuple, st::NamedTuple{fields}) where fields
    N = length(fields)
    x_symbols = [gensym() for _ in 1:N]
    calls = [
        :(
            $(x_symbols[i]) = @inline inverse(
                constraints.$(fields[i]), x.$(fields[i]), st.$(fields[i])
            )[1]
        ) for i in 1:N
    ]
    push!(calls, :(x_c = NamedTuple{$fields}((($(Tuple(x_symbols)...),)))))
    push!(calls, :(x = merge(x, x_c)))
    push!(calls, :(return x, st))
    return Expr(:block, calls...)
end

@generated function applyconstraints(constraints::NamedTuple{fields}, x::NamedTuple, st::NamedTuple{fields}) where fields
    N = length(fields)
    x_symbols = [gensym() for _ in 1:N]
    calls = [
        :(
            $(x_symbols[i]) = @inline constraints.$(fields[i])(x.$(fields[i]), st.$(fields[i])
            )[1]
        ) for i in 1:N
    ]
    push!(calls, :(x_c = NamedTuple{$fields}((($(Tuple(x_symbols)...),)))))
    push!(calls, :(x = merge(x, x_c)))
    push!(calls, :(return x, st))
    return Expr(:block, calls...)
end

"""
    Constraint

Abstract type representing parameter constraints in HybridDynamicModels.jl.

Constraints are used to enforce physical or mathematical bounds on parameters during optimization.
They work by transforming unconstrained parameters to constrained parameter spaces and vice versa.

# Available Constraint Types

## NoConstraint
The simplest constraint that applies no transformation. Parameters remain unconstrained.

```julia
constraint = NoConstraint()
```

## BoxConstraint
Constrains parameters to lie within specified lower and upper bounds using a sigmoid transformation.

```julia
# Constrain parameters to [0, 1]
constraint = BoxConstraint([0.0, 0.0], [1.0, 1.0])

# Constrain to different bounds for each parameter
constraint = BoxConstraint([0.0, -1.0], [10.0, 1.0])
```

## NamedTupleConstraint
Applies different constraints to different fields of a NamedTuple.

```julia
constraints = (
    decay_rate = BoxConstraint([0.0], [1.0]),  # Between 0 and 1
    amplitude = NoConstraint()                  # Unconstrained
)
constraint = NamedTupleConstraint(constraints)
```

# Usage in ParameterLayer

Constraints are primarily used with `ParameterLayer` to ensure parameters stay within valid ranges:

```julia
# Parameter with positivity constraint
param = ParameterLayer(
    constraint = BoxConstraint([0.0], [Inf]),
    init_value = (;rate = [0.1])
)
```

# Mathematical Details

Constraints work by transforming between constrained and unconstrained spaces:

- **Forward transformation**: `constrained = constraint(unconstrained)`
- **Inverse transformation**: `unconstrained = inverse(constraint, constrained)`

The inverse transformation is used during optimization to work in an unconstrained space,
while the forward transformation produces the final constrained parameters.

# Examples

## Basic usage with bounds
```julia
using HybridDynamicModels

# Create a parameter that must be positive
param = ParameterLayer(
    constraint = BoxConstraint([0.0], [Inf]),
    init_value = (;growth_rate = 0.05)
)

ps, st = LuxCore.setup(Random.default_rng(), param)
constrained_value, _ = param(ps, st)
# constrained_value.growth_rate will be >= 0
```

## Multiple constraints
```julia
# Parameters with different constraints
constraints = (
    concentration = BoxConstraint([0.0], [1.0]),    # 0 to 1
    temperature = BoxConstraint([273.0], [373.0]),  # 0°C to 100°C
    pressure = NoConstraint()                        # Unconstrained
)

param = ParameterLayer(
    constraint = NamedTupleConstraint(constraints),
    init_value = (;concentration = 0.5, temperature = 298.0, pressure = 1.0)
)
```

See also: `ParameterLayer`, `NoConstraint`, `BoxConstraint`, `NamedTupleConstraint`
"""
Constraint = Union{NoConstraint, BoxConstraint, NamedTupleConstraint}