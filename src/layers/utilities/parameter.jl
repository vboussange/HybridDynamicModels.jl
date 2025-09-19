import Functors: functor
"""
    ParameterLayer(;constraint::AbstractConstraint = NoConstraint(), 
                    init_value = (;), 
                    init_state_value = (;))

A layer representing parameters, optionally with constraints.

## Arguments
  - `constraint`: Constraint to transform parameters.
  - `init_value`: Initial parameter values (NamedTuple or AbstractArray).
  - `init_state_value`: Internal state (NamedTuple).

## Inputs
- `ps`: Parameters of the layer.
- `st`: States of the layer.

## Outputs
- Constrained parameter values merged with states.

## Behavior
Applies constraints to parameters during forward pass. Parameters are transformed from unconstrained to constrained space.

## Example

```jldoctest
julia> param = ParameterLayer(; constraint = NoConstraint(),
                            init_value = (;a = ones(2)), 
                            init_state_value = (;b = (0.0, 1.0)))
julia> ps, st = Lux.setup(Random.default_rng(), param)
julia> x, _ = param(ps, st)
julia> x == (a = [1.0, 1.0], b = (0.0, 1.0))
true
```
"""
@concrete struct ParameterLayer <: LuxCore.AbstractLuxLayer 
    constraint <: Constraint
    init_value <: Function
    init_state <: Function
end

function ParameterLayer(;constraint = NoConstraint(), init_value = (;), init_state_value = (;))
    isa(init_state_value, NamedTuple) || throw(ArgumentError("`init_state_value` must be a `NamedTuple`"))
    # isa(init_value, NamedTuple) || isa(init_value, AbstractArray) || throw(ArgumentError("`init_value` must be a `NamedTuple`"))
    isa(constraint, Constraint) || throw(ArgumentError("`constraint` must be a `Constraint`"))

    _, st_constraint = LuxCore.setup(Random.default_rng(), constraint)
    init_values_transformed, _ = inverse(constraint, init_value, st_constraint)
    init_state_value = merge(init_state_value, (; constraint = st_constraint))
    return ParameterLayer(constraint, () -> deepcopy(init_values_transformed), () -> deepcopy(init_state_value))
end

function (pl::ParameterLayer)(ps, st)
    # we transform ps to a named tuple, as this may become a feature
    # Note: this is probably very slow...
    ps_tr, _ = pl.constraint(NamedTuple(ps), st.constraint)
    return (ps_tr, st)
end

(pl::ParameterLayer)(_, ps, st) = pl(ps, st) # required for compatibility with Lux Training API

LuxCore.initialstates(::AbstractRNG, layer::ParameterLayer) = layer.init_state()
LuxCore.initialparameters(::AbstractRNG, layer::ParameterLayer) = layer.init_value()

# https://github.com/LuxDL/Lux.jl/blob/b467ff85e293af84d9e11d86bad7fb19e1cb561a/src/helpers/stateful.jl#L138-L142
function (s::LuxCore.StatefulLuxLayer{ST, M, psType, stType} where {ST, M <: ParameterLayer, psType, stType})(ps=s.ps)
    y, st = s.model(ps, get_state(s))
    set_state!(s, st)
    return y
end
