"""
    ParameterLayer(constraint::AbstractConstraint, init_value::NamedTuple, init_state::NamedTuple)
    ParameterLayer(init_value::NamedTuple)
    ParameterLayer(constraint::AbstractConstraint, init_value::NamedTuple)

A layer representing model parameters, optionally with constraints and initial states.

## Arguments

  - `constraint`: An `AbstractConstraint` (e.g., from Bijectors) to transform parameters.
  - `init_value`: NamedTuple of initial parameter values.
  - `init_state`: NamedTuple of initial states (optional).

## Inputs

  - `ps`: Parameters of the layer.
  - `st`: States of the layer.

## Behavior

Transforms parameters according to the constraint when used in a model. States are not transformed.

## Example

```jldoctest
julia> param = ParameterLayer(NoConstraint(), (;u0 = ones(10)), (;tspan = (0.0, 1.0)))
ParameterLayer(...)
julia> ps, st = Lux.setup(Random.default_rng(), param)
julia> kwargs, _ = Lux.apply(param, ps, st)
julia> kwargs.u0 ≈ ones(10)
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


function ParameterLayer(constraint::AbstractConstraint = NoConstraint(), init_value = (;), init_state = (;))
    init_values_transformed = _to_optim_space(constraint, init_value)
    return ParameterLayer(constraint, () -> deepcopy(init_values_transformed), () -> deepcopy(init_state))
end

function (dl::ParameterLayer)(ps, st)
    ps_tr = dl.constraint(ps)
    return (merge(ps_tr, st), st)
end

# https://github.com/LuxDL/Lux.jl/blob/b467ff85e293af84d9e11d86bad7fb19e1cb561a/src/helpers/stateful.jl#L138-L142
function (s::StatefulLuxLayer{ST, M, psType, stType} where {ST, M <: ParameterLayer, psType, stType})(ps=s.ps)
    y, st = Lux.apply(s.model, ps, get_state(s))
    set_state!(s, st)
    return y
end

# NOTE: not sure if this behavior is desired. 
# function (dl::ParameterLayer)(x, ps, st) # access only a subset of the parameters
#     ps_tr = dl.constraint.transform.bs[x](ps[x])
#     return (merge(ps_tr, st), st)
# end

Lux.initialstates(::AbstractRNG, layer::ParameterLayer) = layer.init_state()
Lux.initialparameters(::AbstractRNG, layer::ParameterLayer) = layer.init_value()

"""
    InitialConditions(ics::AbstractLuxLayer)
    InitialConditions(ics::<:ParameterLayer)
    InitialConditions(ics::Vector{<:ParameterLayer})

Wraps initial condition layers for ODE or other models.

## Arguments
  - `ics`: A Lux layer, a `ParameterLayer` or a vector of
    `ParameterLayer`. `ParameterLayer`s must have field `u0` in their parameters.

## Inputs
    - (`x`, `ps`, `st`) when `ics` is a single layer or a vector of parameter layers.
    - (`ps`, `st`) when `ics` is a single parameter layer.
"""
@concrete struct InitialConditions <: Lux.AbstractLuxWrapperLayer{:ics}
    ics
end
isfeatureless(::Type{<:InitialConditions{<:AbstractLuxLayer}}) = Val(false)
isfeatureless(::Type{<:InitialConditions{<:AbstractVector{<:ParameterLayer}}}) = Val(false)
isfeatureless(::Type{<:InitialConditions{<:ParameterLayer}}) = Val(true)

function (lics::InitialConditions{<:ParameterLayer})(ps, st)
    x_u0, new_st_u0 = lics.ics(ps, st)
    new_st = merge(st, new_st_u0)
    return merge((;u0 = x_u0), new_st), new_st
end


# TODO: better handle batch dimension
function (lics::InitialConditions{<:AbstractLuxLayer})(x, ps, st) # handling e.g. a neural net
    x_u0, new_st_u0 = lics.ics(x, ps, st)
    new_st = merge(st, new_st_u0)
    return merge((;u0 = x_u0), new_st), new_st
end

function Lux.initialstates(rng::AbstractRNG, ics::InitialConditions{<:AbstractVector{<:ParameterLayer}})
    n_ics = length(ics.ics)
    NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}([Lux.initialstates(rng, _u0) for _u0 in ics.ics])
end

function Lux.initialparameters(rng::AbstractRNG, ics::InitialConditions{<:AbstractVector{<:ParameterLayer}})
    n_ics = length(ics.ics)
    NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}([Lux.initialparameters(rng, _u0) for _u0 in ics.ics])
end

function (lics::InitialConditions{<:AbstractVector{<:ParameterLayer}})(x::Int, ps, st)
    k = keys(ps)[x]
    u0, new_st_u0 = lics.ics[x](ps[k], st[k])
    new_st = merge(st, (;k = new_st_u0))
    return u0, new_st
end

# handling batches
function (lics::InitialConditions{<:AbstractVector{<:ParameterLayer}})(x::AbstractVector{<:Int}, ps, st)
    sols = map(xi -> lics(xi, ps, st)[1], x)
    # Concatenate solutions along a new axis (e.g., batch dimension)
    return sols, st
end

"""
    ODEModel(components::NamedTuple, dudt::Function; kwargs...)

Wraps an ODE model for simulation using Lux layers.

## Arguments
  - `components`: NamedTuple of Lux layers representing the components of the model.
  - `dudt`: Function that computes the derivative of the state, with signature `dudt(components, ps, st, u, t)`.
  - `kwargs`: Additional keyword arguments passed to the solver (e.g., `tspan`, `saveat`, `alg`).

## Inputs
  - `x`: NamedTuple containing initial conditions and parameters.
  - `ps`: Parameters of the model.
  - `st`: States of the model.

## Example

```jldoctest
julia> components = (; layer1 = Lux.Dense(10, 10, relu))
julia> dudt(components, ps, st, u, t) = components.layer1(u, ps.layer1, st.layer1)[1]
julia> ode_model = ODEModel(components, dudt, tspan = (0f0, 1f0), saveat = range(0f0, stop=1f0, length=100), alg = Tsit5())
julia> ps, st = Lux.setup(Random.default_rng(), ode_model)
julia> ode_model((; u0 = ones(Float32, 10)), ps, st)
```
"""
@concrete struct ODEModel <: Lux.AbstractLuxWrapperLayer{:components}
    components<:NamedTuple{names, <:NTuple{N, AbstractLuxLayer}} where {names, N}
    dudt<:Function # function that computes the derivative
    kwargs
end

# function ODEModel(components::NamedTuple{names, <:NTuple{N, AbstractLuxLayer}} where {names, N}, dudt, kwargs)
#     # Ensure components are StatefulLuxLayers
#     return ODEModel(components, dudt, kwargs)
# end

ODEModel(components, dudt; kwargs...) = ODEModel(components, dudt, kwargs)

function (m::ODEModel)(x::NamedTuple, ps, st)
    u0 = hasproperty(x, :u0) ? getfield(x, :u0) : m.kwargs[:u0]
    tspan = hasproperty(x, :tspan) ? getfield(x, :tspan) : m.kwargs[:tspan]
    component_keys = keys(m.components)
    component_vals = map(k -> StatefulLuxLayer{true}(getproperty(m.components, k), getproperty(ps, k), get_state(getfield(st, k))), component_keys)
    components = NamedTuple{component_keys}(component_vals)

    function __dudt(u, p, t)
        m.dudt(components, u, p, t)
    end

    prob = ODEProblem{false}(ODEFunction{false}(__dudt), u0, tspan, ps)
    alg = m.kwargs[:alg]

    kwargs = merge((;m.kwargs...), x) # overwriting kwargs with x

    sol = solve(prob, alg; kwargs...) |> Array
    return sol, st
end

# Handling batches
function (m::ODEModel)(x::AbstractVector{<:NamedTuple}, ps, st)
    sols = map(xi -> m(xi, ps, st)[1], x)
    # Check that all solutions have the same size
    first_shape = size(sols[1])
    if any(size(sol) != first_shape for sol in sols)
        throw(DimensionMismatch("All solutions in batch must have the same dimensions. Got: $(map(size, sols))"))
    end
    # Concatenate solutions along a new axis (e.g., batch dimension)
    return cat(sols...; dims=ndims(sols[1])+1), st
end

# # Testing
# components = (; layer1 = Lux.Dense(10, 10, relu))
# dudt(components, ps, st, u, t) = components.layer1(u, ps.layer1, st.layer1)[1]
# ode_model = ODEModel(components, dudt, tspan = (0.0, 1.0), saveat = range(0.0, stop=1.0, length=100), alg = Tsit5())

# using Random
# rng = MersenneTwister(1234)
# ps, st = Lux.setup(rng, ode_model)

# ode_model((; u0 = ones(10)), ps, st)

# ics = ParameterLayer(NoConstraint(), () -> (;u0 = ones(10)), () -> (;tspan = (0.0, 1.0), saveat = range(0.0, stop=1.0, length=100)))
# ps, st = Lux.setup(rng, ics)

# Lux.apply(ics, ps, st)
# xs, st = ics(ps, st)
# ics(nothing, ps, st)

# wrapped_ics = Chain(ics)
# ps, st = Lux.setup(rng, wrapped_ics)
# wrapped_ics(nothing, ps, st)
# wrapped_ics(ps, st)


# ode_model_with_ics = Chain(initial_conditions = ics, model = ode_model)
# ps, st = Lux.setup(rng, ode_model_with_ics)


# ode_model_with_ics(ps, st)
