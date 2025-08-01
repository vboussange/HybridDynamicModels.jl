using ConcreteStructs: @concrete
using Lux
import Lux: IntegerType

abstract type AbstractModelFeatures end

@concrete struct ModelFeatures{T}
    u0
    t0
    tsteps
end

abstract type AbstractParamLayer <: AbstractLuxLayer end

function Lux.apply(dl::AbstractParamLayer, ps, st)
    ps_tr = dl.constraint(ps)
    return (ps_tr, st)
end

LuxCore.initialstates(::AbstractRNG, ::AbstractParamLayer) = NamedTuple()

function LuxCore.initialparameters(rng::AbstractRNG, layer::L) where {L <: AbstractParamLayer}
    return layer.init_values()
end


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
@concrete struct LearnableICs <: AbstractParamLayer 
    # var_dims <: Union{IntegerType,Tuple{Vararg{IntegerType}}} # Number of learnt initial conditions
    n_ics <: IntegerType # Number of initial conditions
    constraint <: AbstractConstraint
    init_values
end

# NOTE: We may want to define two different structs, or parametric ones,
# depending on whether we have multiple initial conditions or just one.
# Something like LearnableICs{N} where N is the number of initial conditions.
# function LearnableICs((var_dims, n_ics)::Pair)
#     LearnableICs(var_dims, n_ics, identity, () -> NamedTuple{ntuple(i -> Symbol(:u0s_, i), n_ics)}(zeros(var_dims...) for _ in 1:n_ics))
# end

function LearnableICs(ics_vec::Vector, constraint = NoConstraint())
    for ic in ics_vec
        y = _to_optim_space(constraint, ic)
        x = constraint(y)
        @show x, y
        @assert all([isapprox(x[k], ic[k]) for k in keys(x)]) "Initial conditions and constraints do not seem compatible."
    end

    # TODO: this should be checked with a fmap that walks through each element of ics_vec, to ensure they all have the same dimensions
    # @assert all([same_names_and_dims(ic, first(ics_vec)) for ic in ics_vec[2:end]]) "Initial conditions do not have the same dimensions."

    n_ics = length(ics_vec)

    # project each initial condition into the optimization space and wrap them in a NamedTuple
    ics_trs_tup = NamedTuple{ntuple(i -> Symbol(:u0s_, i), n_ics)}([_to_optim_space(constraint, ic) for ic in ics_vec])

    LearnableICs(n_ics, constraint, () -> ics_trs_tup)
end

function LearnableICs(ics, constraint = NoConstraint())
    y = _to_optim_space(constraint, ics)
    x = constraint(y)
    @assert all([isapprox(x[k], y[k]) for k in keys(x)]) "Initial conditions and constraints do not seem compatible."

    n_ics = 1

    LearnableICs(n_ics, constraint, () -> (u0s = _to_optim_space(constraint, ics),))
end

function Lux.apply(lics::LearnableICs, idx, ps, st)
    ps_tr = [lics.constraint(ps[i]) for i in idx]
    return (ps_tr, st)
end

