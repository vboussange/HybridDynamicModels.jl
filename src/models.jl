Lux.apply(layer::AbstractLuxLayer, ps, st) = layer(ps, st) # This defines the behavior of Lux.Chain

"""
    AbstractParamLayer <: AbstractLuxLayer

Abstract type for layers representing parameters without feature computation.
"""
abstract type AbstractParamLayer <: AbstractLuxLayer end # Defines abstract feature computation free layers

"""
    Parameter(constraint::AbstractConstraint, init_value::NamedTuple, init_state::NamedTuple)
    Parameter(init_value::NamedTuple)
    Parameter(constraint::AbstractConstraint, init_value::NamedTuple)

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
julia> param = Parameter(NoConstraint(), (;u0 = ones(10)), (;tspan = (0.0, 1.0)))
Parameter(...)
julia> ps, st = Lux.setup(Random.default_rng(), param)
julia> kwargs, _ = Lux.apply(param, ps, st)
julia> kwargs.u0 ≈ ones(10)
true
```
"""
@concrete struct Parameter <: AbstractParamLayer 
    constraint <: AbstractConstraint
    init_value
    init_state
end

function Parameter(constraint::AbstractConstraint, init_value::NamedTuple, init_state::NamedTuple) 
    init_values_transformed = _to_optim_space(constraint, init_value)
    return Parameter(constraint, () -> deepcopy(init_values_transformed), () -> deepcopy(init_state))
end

function Parameter(init_value::NamedTuple)
    return Parameter(NoConstraint(), init_value, (;))
end

function Parameter(constraint::AbstractConstraint, init_value::NamedTuple)
    return Parameter(constraint, init_value, (;))
end

function (dl::Parameter)(ps, st)
    ps_tr = dl.constraint(ps)
    return (merge(ps_tr, st), st)
end

# NOTE: not sure if this behavior is desired. 
# function (dl::Parameter)(x, ps, st) # access only a subset of the parameters
#     ps_tr = dl.constraint.transform.bs[x](ps[x])
#     return (merge(ps_tr, st), st)
# end

LuxCore.initialstates(::AbstractRNG, layer::Parameter) = layer.init_state()
LuxCore.initialparameters(::AbstractRNG, layer::Parameter) = layer.init_value()

"""
    InitialConditions(ics::AbstractLuxLayer)
    InitialConditions(ics::<:AbstractParamLayer)
    InitialConditions(ics::Vector{<:AbstractParamLayer})

Wraps initial condition layers for ODE or other models.

## Arguments
  - `ics`: A Lux layer, an `AbstractParamLayer` or a vector of
    `AbstractParamLayer`. `AbstractParamLayer`s have field `u0`.

## Inputs
    - (`x`, `ps`, `st`) when `ics` is a single layer or a vector of parameter layers.
    - (`ps`, `st`) when `ics` is a single parameter layer.
"""
@concrete struct InitialConditions <: LuxCore.AbstractLuxWrapperLayer{:ics}
    ics
end

function (lics::InitialConditions{<:AbstractParamLayer})(ps, st)
    x_u0, new_st_u0 = lics.ics(ps, st)
    new_st = merge(st, new_st_u0)
    return merge((;u0 = x_u0), new_st), new_st
end

function (lics::InitialConditions{<:AbstractLuxLayer})(x, ps, st) # handling e.g. a neural net
    x_u0, new_st_u0 = lics.ics(x, ps, st)
    new_st = merge(st, new_st_u0)
    return merge((;u0 = x_u0), new_st), new_st
end

function LuxCore.initialstates(rng::AbstractRNG, ics::InitialConditions{<:AbstractVector{<:AbstractParamLayer}})
    n_ics = length(ics.ics)
    NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}([LuxCore.initialstates(rng, _u0) for _u0 in ics.ics])
end

function LuxCore.initialparameters(rng::AbstractRNG, ics::InitialConditions{<:AbstractVector{<:AbstractParamLayer}})
    n_ics = length(ics.ics)
    NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}([LuxCore.initialparameters(rng, _u0) for _u0 in ics.ics])
end

function (lics::InitialConditions{<:AbstractVector{<:AbstractParamLayer}})(x, ps, st)
    k = keys(ps)[x]
    u0, new_st_u0 = lics.ics[x](ps[k], st[k])
    new_st = merge(st, (;k = new_st_u0))
    return u0, new_st
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
@concrete struct ODEModel <: LuxCore.AbstractLuxWrapperLayer{:components}
    components<:NamedTuple # we should enforce to be a namedtuple of Lux layers
    dudt<:Function # function that computes the derivative
    kwargs
end


ODEModel(components, dudt; kwargs...) = ODEModel(components, dudt, kwargs)

function (m::ODEModel)(x::NamedTuple, ps, st)
    u0 = hasproperty(x, :u0) ? getfield(x, :u0) : m.kwargs[:u0]
    tspan = hasproperty(x, :tspan) ? getfield(x, :tspan) : m.kwargs[:tspan]

    function __dudt(u, _, t)
        m.dudt(m.components, ps, st, u, t)
    end

    prob = ODEProblem(__dudt, u0, tspan)
    alg = m.kwargs[:alg]

    kwargs = merge((;m.kwargs...), x) # overwriting kwargs with x

    sol = solve(prob, alg; kwargs...)
    return (sol, st)
end

# # Testing
# components = (; layer1 = Lux.Dense(10, 10, relu))
# dudt(components, ps, st, u, t) = components.layer1(u, ps.layer1, st.layer1)[1]
# ode_model = ODEModel(components, dudt, tspan = (0.0, 1.0), saveat = range(0.0, stop=1.0, length=100), alg = Tsit5())

# using Random
# rng = MersenneTwister(1234)
# ps, st = Lux.setup(rng, ode_model)

# ode_model((; u0 = ones(10)), ps, st)

# ics = Parameter(NoConstraint(), () -> (;u0 = ones(10)), () -> (;tspan = (0.0, 1.0), saveat = range(0.0, stop=1.0, length=100)))
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
