using ConcreteStructs: @concrete
using Lux
import Lux: IntegerType
using OrdinaryDiffEq
using Random

abstract type AbstractModelFeatures end

@concrete struct ModelFeatures{T}
    u0
    t0
    tsteps
end

abstract type AbstractParamLayer <: AbstractLuxLayer end

function (dl::AbstractParamLayer)(ps, st)
    ps_tr = dl.constraint(ps)
    return (merge(ps_tr, st), st)
end

(dl::AbstractParamLayer)(::Nothing, ps, st) = dl(ps, st)

# function Lux.apply(dl::AbstractParamLayer, idx, ps, st)
#     ps_tr = dl.constraint(getfield(ps, idx))
#     return (ps_tr, st)
# end

LuxCore.initialstates(::AbstractRNG, layer::L) where {L <: AbstractParamLayer} = layer.init_states()
LuxCore.initialparameters(::AbstractRNG, layer::L) where {L <: AbstractParamLayer} = layer.init_values()

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

julia> LearnableICs(initial_ics, constraint)
```
"""
@concrete struct ParamLayer <: AbstractParamLayer 
    # var_dims <: Union{IntegerType,Tuple{Vararg{IntegerType}}} # Number of learnt initial conditions
    constraint <: AbstractConstraint
    init_values
    init_states
end

# NOTE: We may want to define two different structs, or parametric ones,
# depending on whether we have multiple initial conditions or just one.
# Something like LearnableICs{N} where N is the number of initial conditions.
# function LearnableICs((var_dims, n_ics)::Pair)
#     LearnableICs(var_dims, n_ics, identity, () -> NamedTuple{ntuple(i -> Symbol(:u0s_, i), n_ics)}(zeros(var_dims...) for _ in 1:n_ics))
# end
# TODO: we should define a ParamLayer which is parametrized by its number of dimensions
# and then define const LearnableICs{N} = ParamLayer{N}
# and const Parameters = LearnableICs{0}

function BatchedLearnableICs(ics_vec::AbstractVector{T}, constraint = NoConstraint()) where T
    for ic in ics_vec
        y = _to_optim_space(constraint, ic)
        x = constraint(y)
        @assert all([isapprox(x[k], ic[k]) for k in keys(x)]) "Values and constraints do not seem compatible."
    end

    # project each initial condition into the optimization space and wrap them in a NamedTuple
    ics_trs_tup = NamedTuple{ntuple(i -> Symbol(:u0s_, i), n_ics)}([_to_optim_space(constraint, ic) for ic in ics_vec])

    ParamLayer(constraint, () -> ics_trs_tup)
end

function LearnableICs(ics, constraint = NoConstraint())
    y = _to_optim_space(constraint, ics)
    x = constraint(y)
    @assert all([isapprox(x[k], y[k]) for k in keys(x)]) "Values and constraints do not seem compatible."

    ParamLayer(constraint, () -> (u0s = _to_optim_space(constraint, ics),))
end

@concrete struct ODEModel <: Lux.AbstractLuxWrapperLayer{:components}
    components # we should enforce to be a namedtuple of Lux layers
    dudt
    kwargs
end

ODEModel(components, dudt; kwargs...) = ODEModel(components, dudt, kwargs)

function (m::ODEModel)(x::NamedTuple, ps, st)
    u0 = hasproperty(x, :u0) ? getfield(x, :u0) : m.kwargs[:u0]
    tspan = hasproperty(x, :tspan) ? getfield(x, :tspan) : m.kwargs[:tspan]

    function __dudt(u, _, t)
        dudt(m.components, ps, st, u, t)
    end

    prob = ODEProblem(__dudt, u0, tspan)
    alg = m.kwargs[:alg]
    @show m.kwargs
    @show __dudt(u0, nothing, nothing)
    sol = solve(prob, alg; m.kwargs...)
    return (sol, st)
end

Lux.apply(layer::AbstractLuxLayer, ps, st) = layer(nothing, ps, st)
(l::AbstractLuxLayer)(ps, st) = l(nothing, ps, st)


# Testing
components = (; layer1 = Lux.Dense(10, 10, relu))
dudt(components, ps, st, u, t) = components.layer1(u, ps.layer1, st.layer1)[1]
ode_model = ODEModel(components, dudt, tspan = (0.0, 1.0), saveat = range(0.0, stop=1.0, length=100), alg = Tsit5())

using Random
rng = MersenneTwister(1234)
ps, st = Lux.setup(rng, ode_model)

ode_model((; u0 = ones(10)), ps, st)

ics = ParamLayer(NoConstraint(), () -> (;u0 = ones(10)), () -> (;tspan = (0.0, 1.0), saveat = range(0.0, stop=1.0, length=100)))
ps, st = Lux.setup(rng, ics)

Lux.apply(ics, ps, st)
xs, st = ics(ps, st)
ics(nothing, ps, st)

wrapped_ics = Chain(ics)
ps, st = Lux.setup(rng, wrapped_ics)
wrapped_ics(nothing, ps, st)
wrapped_ics(ps, st)


ode_model_with_ics = Chain(initial_conditions = ics, model = ode_model)
ps, st = Lux.setup(rng, ode_model_with_ics)


ode_model_with_ics(ps, st)
