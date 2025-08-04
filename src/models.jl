Lux.apply(layer::AbstractLuxLayer, ps, st) = layer(ps, st) # This defines the behavior of Lux.Chain

abstract type AbstractParamLayer <: AbstractLuxLayer end # Defines abstract feature computation free layers

"""
    TODO: Place signatures here

# Example
```jldoctest
julia> using Bijectors
initial_ics = [(a = rand(3), b = randn(3)) for _ in 1:5] # a should be in [0, 1], b has no constraints
transform = NamedTransform((
    a = bijector(Uniform(0., 1.0)),
    b = identity)
)
constraint = Constraint(transform)

julia> InitialConditions(initial_ics, constraint)
```
"""
@concrete struct Parameter <: AbstractParamLayer 
    constraint <: AbstractConstraint
    init_value
    init_state
end

# TODO: define default behavior for init_value = AbstractArray or Tuple
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

@concrete struct InitialConditions <: LuxCore.AbstractLuxWrapperLayer{:ics}
    ics
end

function (lics::InitialConditions{<:AbstractLuxLayer})(ps, st)
    x_u0, new_st_u0 = lics.ics(ps, st)
    new_st = merge(st, new_st_u0)
    return merge((;u0 = x_u0), new_st), new_st
end

function (lics::InitialConditions{<:AbstractLuxLayer})(x, ps, st) # handling e.g. a neural net
    x_u0, new_st_u0 = lics.ics(x, ps, st)
    new_st = merge(st, new_st_u0)
    return merge((;u0 = x_u0), new_st), new_st
end

function LuxCore.initialstates(rng::AbstractRNG, ics::InitialConditions{<:AbstractVector{<:AbstractLuxLayer}})
    n_ics = length(ics.ics)
    NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}([LuxCore.initialstates(rng, _u0) for _u0 in ics.ics])
end

function LuxCore.initialparameters(rng::AbstractRNG, ics::InitialConditions{<:AbstractVector{<:AbstractLuxLayer}})
    n_ics = length(ics.ics)
    NamedTuple{ntuple(i -> Symbol(:u0_, i), n_ics)}([LuxCore.initialparameters(rng, _u0) for _u0 in ics.ics])
end

function (lics::InitialConditions{<:AbstractVector{<:AbstractLuxLayer}})(x, ps, st)
    k = keys(ps)[x]
    u0, new_st_u0 = lics.ics[x](ps[k], st[k])
    new_st = merge(st, (;k = new_st_u0))
    return u0, new_st
end

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
