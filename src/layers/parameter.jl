"""
    ParameterLayer(;constraint::AbstractConstraint = NoConstraint(), 
                    init_value = (;), 
                    init_state_value = (;))

A layer representing parameters, optionally with constraints.

## Arguments

  - `constraint`: An `AbstractConstraint` to transform parameters of type `init_value`.
  - `init_value`: Initial parameter values; can be a NamedTuple or an `AbstractArray` with `getproperty` defined.
  - `init_state_value`: Parameter internal state; must be a NamedTuple.

## Inputs

  - `ps`: Parameters of the layer.
  - `st`: States of the layer.

## Output
- Parameter values constrained by `constraint`, merged with states `st`.

## Example

```jldoctest
julia> param = ParameterLayer(; constraint = NoConstraint(),
                            init_value = (;a = ones(2)), 
                            init_state_value = (;b = (0.0, 1.0)))
ParameterLayer(...)
julia> ps, st = Lux.setup(Random.default_rng(), param)
julia> x, _ = param(ps, st)
julia> x == (a = [1.0, 1.0], b = (0.0, 1.0))
true
```
"""
@concrete struct ParameterLayer <: AbstractLuxLayer 
    constraint <: AbstractConstraint
    init_value <: Function
    init_state <: Function
end
isfeatureless(::Type{<:ParameterLayer}) = Val(true)
isfeatureless(::Type{<:StatefulLuxLayer{ST, M, psType, stType}}) where {ST, M <: ParameterLayer, psType, stType} = Val(true)


function ParameterLayer(;constraint = NoConstraint(), init_value = (;), init_state_value = (;))
    isa(init_state_value, NamedTuple) || throw(ArgumentError("`init_state_value` must be a `NamedTuple`"))
    # isa(init_value, NamedTuple) || isa(init_value, AbstractArray) || throw(ArgumentError("`init_value` must be a `NamedTuple`"))
    isa(constraint, AbstractConstraint) || throw(ArgumentError("`constraint` must be an `AbstractConstraint`"))

    init_values_transformed = _to_optim_space(constraint, init_value)
    return ParameterLayer(constraint, () -> deepcopy(init_values_transformed), () -> deepcopy(init_state_value))
end

# Default behavior of ParameterLayer is to merge parameter and state information
# This is useful to store e.g. t0 associated with u0 when ParameterLayer is used for initial conditions
function (pl::ParameterLayer)(ps, st)
    # we transform ps to a named tuple, as this may become a feature
    # Note: this is probably very slow...
    ps_tr = pl.constraint(NamedTuple(ps))
    x = merge(ps_tr, st)
    return (x, st)
end

(pl::ParameterLayer)(x, ps, st) = pl(ps, st) # required for compatibility with Lux Training API

# https://github.com/LuxDL/Lux.jl/blob/b467ff85e293af84d9e11d86bad7fb19e1cb561a/src/helpers/stateful.jl#L138-L142
# we place it here instead of in generics.jl as we specifiy this behavior only for parameter layer, to avoid type piracy
function (s::StatefulLuxLayer{ST, M, psType, stType} where {ST, M <: ParameterLayer, psType, stType})(ps=s.ps)
    y, st = Lux.apply(s.model, ps, get_state(s))
    set_state!(s, st)
    return y
end

Lux.initialstates(::AbstractRNG, layer::ParameterLayer) = layer.init_state()
Lux.initialparameters(::AbstractRNG, layer::ParameterLayer) = layer.init_value()