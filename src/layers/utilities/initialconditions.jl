"""
    ICLayer(ics::AbstractLuxLayer)
    ICLayer(ics::<:ParameterLayer)
    ICLayer(ics::Vector{<:ParameterLayer})

Initial condition layer.

## Arguments
  - `ics`: A LuxCore layer, a `ParameterLayer` or a vector of
    `ParameterLayer`.

## Inputs
    - (`x`, `ps`, `st`) with `x` a `NamedTuple` or `AbstractVector{NamedTuple}` (batch mode), when `ics` is an `AbstractLuxLayer` or a `Vector{<:ParameterLayer}`.
    - (`ps`, `st`) when `ics` is a `ParameterLayer` (feature less).

## Outputs
    - `u0`: Initial conditions, merged with states other fields.
    - `st`: Updated states.

## Behavior
    - when applicable, `x.u0` is passed to the wrapped layer; the output is returned, merged with other fields of `x`.

!!!warning
    Undefined behavior when `ps` is not a NamedTuple
"""
@concrete struct ICLayer <: HybridDynamicModelsLayer
    layers
end

function ICLayer(ics::AbstractVector{<:AbstractLuxLayer})
    n_ics = length(ics)
    nt_ics = NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}(ics)
    ICLayer(nt_ics)
end

# batch mode
function (lics::ICLayer)(x::AbstractVector{<:NamedTuple}, ps, st)
    function step(acc, xi)
        sols, curr_st = acc
        sol, new_st = lics(xi, ps, curr_st)
        return (tuple(sols..., sol), new_st)
    end
    (sols, new_st) = reduce(step, x; init=((), st))
    return [sols...], new_st
end

(lics::ICLayer{<:AbstractLuxLayer})(ps, st) = begin 
    @assert hasproperty(ps, :u0) "Parameter must have field `u0`."
    return lics.layers((), ps, st), st
end

function (lics::ICLayer{<:AbstractLuxLayer})(x::NamedTuple, ps, st)
    @assert hasproperty(x, :u0) "Input `x` must have field `u0`."
    new_u0, new_st_u0 = lics.layers(x.u0, ps, st)
    new_x = merge(x, (;u0 = new_u0)) # merging initial conditions with other fields to carry
    return new_x, new_st_u0
end

function (lics::ICLayer{<:NamedTuple{fields}})(x::NamedTuple, ps, st) where fields
    @assert hasproperty(x, :u0) && isa(x.u0, Int) "Input `x` must have field `u0` of type Int to index initial conditions."
    k = fields[x.u0]
    _ics = getfield(lics.layers, k)
    new_u0, new_st_k = _ics((), ps[k], st[k])
    # new_st = merge(st, (k => new_st_k,)) creates mutation, hence we use the uglier form below
    new_st = merge(st, NamedTuple{(k,)}((new_st_k,))) 
    new_x = merge(x, (;new_u0...)) # merging initial conditions with other fields to carry
    return new_x, new_st
end