"""
    InitialConditions(ics::AbstractLuxLayer)
    InitialConditions(ics::<:ParameterLayer)
    InitialConditions(ics::Vector{<:ParameterLayer})

Initial condition layer.

## Arguments
  - `ics`: A Lux layer, a `ParameterLayer` or a vector of
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
@concrete struct InitialConditions <: Lux.AbstractLuxWrapperLayer{:ics}
    ics
end
# TODO: you probably want to ensure that ics returns a NamedTuple with field u0

function InitialConditions(ics::AbstractVector{<:AbstractLuxLayer})
    n_ics = length(ics)
    nt_ics = NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}(ics)
    InitialConditions(nt_ics)
end

# batch mode
function (lics::InitialConditions)(x::AbstractVector{<:NamedTuple}, ps, st)
    function step(acc, xi)
        sols, curr_st = acc
        sol, new_st = lics(xi, ps, curr_st)
        return (tuple(sols..., sol), new_st)
    end
    (sols, new_st) = reduce(step, x; init=((), st))
    return [sols...], new_st
end

function (lics::InitialConditions{<:AbstractLuxLayer})(ps, st) 
    return lics.ics((), ps, st)
end

function (lics::InitialConditions{<:AbstractLuxLayer})(x::NamedTuple, ps, st)
    new_u0, new_st_u0 = lics.ics(x.u0, ps, st)
    new_x = merge(x, (;u0 = new_u0)) # merging initial conditions with other fields to carry
    return new_x, new_st_u0
end


# function Lux.initialstates(rng::AbstractRNG, ics::InitialConditions{<:AbstractVector{<:ParameterLayer}})
#     n_ics = length(ics.ics)
#     NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}([Lux.initialstates(rng, _u0) for _u0 in ics.ics])
# end

# function Lux.initialparameters(rng::AbstractRNG, ics::InitialConditions{<:AbstractVector{<:ParameterLayer}})
#     n_ics = length(ics.ics)
#     NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}([Lux.initialparameters(rng, _u0) for _u0 in ics.ics])
# end

function (lics::InitialConditions{<:NamedTuple{fields}})(x::NamedTuple, ps, st) where fields
    @assert hasproperty(x, :u0) && isa(x.u0, Int) "Input `x` must have field `u0` of type Int to index initial conditions."
    k = fields[x.u0]
    _ics = getfield(lics.ics, k)
    new_u0, new_st_k = _ics((), ps[k], st[k])
    # new_st = merge(st, (k => new_st_k,)) creates mutation, hence we use the uglier form below
    new_st = merge(st, NamedTuple{(k,)}((new_st_k,))) 
    new_x = merge(x, (;new_u0...)) # merging initial conditions with other fields to carry
    return new_x, new_st
end

"""
    ODEModel(components::NamedTuple, dudt::Function; kwargs...)

Wraps an ODE model for simulation using Lux layers.

## Arguments
  - `components`: NamedTuple of Lux layers representing the components of the model.
  - `dudt`: Function that computes the derivative of the state, with signature `dudt(components, u, ps, t)`.
  - `kwargs`: Additional keyword arguments passed to the solver (e.g., `tspan`, `saveat`, `alg`).

## Inputs
- (`x`, `ps`, `st`)
  - `x`: a `NamedTuple` or `AbstractVector{NamedTuple}` (batch mode).
  - `ps`: Parameters of the model.
  - `st`: States of the model.
- A tuple of (`x`, `ps`, `st`): batch mode.
- (`ps`, `st`): If `x` not provided, defaults to `kwargs`.

## Outputs
- (`sol`, `st`)
  - `sol`: Solution of the ODE problem, with second dimension corresponding to time and batches stacked along the third dimension, if applicable.
  - `st`: Updated states of the model.

## Behavior
`components` are wrapped in `StatefulLuxLayer` to maintain their states. Hence, if `components = (; layer1 = Lux.Dense(10, 10, relu))`, then `dudt` should be defined as `dudt(components, ps, u, t) = components.layer1(u, ps.layer1)`.

## Example

```jldoctest
julia> components = (; layer1 = Lux.Dense(10, 10, relu))
julia> dudt(components, ps, st, u, t) = components.layer1(u, ps.layer1)[1]
julia> ode_model = ODEModel(components, dudt, tspan = (0f0, 1f0), saveat = range(0f0, stop=1f0, length=100), alg = Tsit5())
julia> ps, st = Lux.setup(Random.default_rng(), ode_model)
julia> ode_model((; u0 = ones(Float32, 10)), ps, st)
```

!!!warning
    Undefined behavior when `ps` is not a NamedTuple
"""
@concrete struct ODEModel <: Lux.AbstractLuxWrapperLayer{:components}
    components<:NamedTuple{names, <:NTuple{N, AbstractLuxLayer}} where {names, N}
    dudt # function that computes the derivative
    kwargs
end

ODEModel(components, dudt; kwargs...) = ODEModel(components, dudt, NamedTuple(kwargs))

function (m::ODEModel)(x::NamedTuple, ps, st)
    u0 = hasproperty(x, :u0) ? getproperty(x, :u0) : getproperty(m.kwargs, :u0)
    tspan = hasproperty(x, :tspan) ? getproperty(x, :tspan) : getproperty(m.kwargs, :tspan)
    saveat = hasproperty(x, :saveat) ? getproperty(x, :saveat) : getproperty(m.kwargs, :saveat)
    component_keys = keys(m.components)
    component_vals = map(k -> StatefulLuxLayer{true}(getproperty(m.components, k), getproperty(ps, k), get_state(getproperty(st, k))), component_keys)
    components = NamedTuple{component_keys}(component_vals)

    # Remove :u0, :p, and :tspan from x if present, so that when x and kwargs
    # are passed to `solve`, they don't overwrite values in prob see
    # https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/
    # kwargs = merge(m.kwargs, x) # overwriting kwargs with x
    kwargs = Base.structdiff(m.kwargs, NamedTuple{(:u0, :p, :tspan, :saveat)}) # keep kwargs that are not in x

    function __dudt(u, p, t)
        m.dudt(components, u, p, t)
    end

    prob = ODEProblem{false}(ODEFunction{false}(__dudt), 
                            u0, 
                            tspan, 
                            ps)
    alg = kwargs[:alg]


    sol = solve(prob, alg; saveat, kwargs...) |> Array
    new_st = NamedTuple{component_keys}(map(k -> get_state(getproperty(components, k)), component_keys))
    return sol, new_st
end
(m::ODEModel)(ps, st) = m(NamedTuple(), ps, st)

# Handling batches
function (m::ODEModel)(x::AbstractVector{<:NamedTuple}, ps, st)
    function step(acc, xi)
        sols, curr_st = acc
        sol, new_st = m(xi, ps, curr_st)
        return (tuple(sols..., sol), new_st)
    end
    (sols, new_st) = reduce(step, x; init=((), st))
    first_shape = size(sols[1])
    if any(size(sol) != first_shape for sol in sols)
        throw(DimensionMismatch("All solutions in batch must have the same dimensions. Got: $(map(size, sols))"))
    end
    return cat(sols...; dims=ndims(sols[1])+1), new_st
end