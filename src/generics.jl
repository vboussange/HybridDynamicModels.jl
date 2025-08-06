const LuxStruct = Union{AbstractLuxLayer, StatefulLuxLayer}

# featureless Trait definition
# See here: https://docs.julialang.org/en/v1/manual/types/#%22Value-types%22
# By default, all Lux objects are *not* featureless, and must be called with (x, ps, st)
# Featureless objects can be called with (ps, st).
# A featureless object may also be called with (x, ps, st), should it has the method implemented, but not the opposite.
isfeatureless(::T) where {T <: LuxStruct} = isfeatureless(T)
isfeatureless(::Type{<:LuxStruct}) = Val(false)

# Overwriting the apply function to handle featureless layers
function Lux.apply(m::M, ps, st) where {M <: LuxStruct} 
    return Lux.apply(isfeatureless(M), m, ps, st)
end

function Lux.apply(m::M, x, ps, st) where {M <: LuxStruct} 
    return Lux.apply(isfeatureless(M), x, m, ps, st)
end

function Lux.apply(::Val{false}, m::M, ps, st) where {M <: LuxStruct}
    throw(ErrorException("$M must be provided with a feature `x`."))
end

# function Lux.apply(::Val{true}, m::M, x, ps, st) where {M <: LuxStruct}
#     throw(ErrorException("$M is feature less, behavior is undefined when called with a feature `x`."))
# end

@stable default_mode = "disable" function Lux.apply(::Val{true}, m::M, ps, st) where {M <: AbstractLuxLayer}
    return m(ps, st)
end

@stable default_mode = "disable" function Lux.apply(::Val{false}, m::M, x, ps, st) where {M <: AbstractLuxLayer}
    return m(x, ps, st)
end

@stable default_mode = "disable" function Lux.apply(::Val{true}, m::M, ps) where {M <: StatefulLuxLayer}
    y, st = Lux.apply(s.model, ps, get_state(s))
    set_state!(s, st)
    return y
end

@stable default_mode = "disable" function Lux.apply(::Val{false}, m::M, x, ps) where {M <: StatefulLuxLayer}
    y, st = Lux.apply(s.model, x, ps, get_state(s))
    set_state!(s, st)
    return y
end

# Overloading Lux.chain to work with featureless layers
function (c::Lux.Chain)(ps, st)
    layers = c.layers
    keys_all = keys(layers)
    isempty(keys_all) && return nothing, st  # Handle empty chain

    first_key = first(keys_all)
    # Call first layer without input: layer(ps, st)
    x, new_first_st = layers[first_key](ps[first_key], st[first_key])

    remaining_keys = Base.tail(keys_all)
    isempty(remaining_keys) && return x, NamedTuple{(first_key,)}((new_first_st,))

    # Prepare remaining layers, parameters, and state
    remaining_layers = layers[remaining_keys]
    remaining_ps = ps[remaining_keys]
    remaining_st = st[remaining_keys]

    # Process remaining layers with applychain
    y, new_remaining_st = Lux.applychain(remaining_layers, x, remaining_ps, remaining_st)

    # Combine updated states from first and remaining layers
    new_st = merge((; first_key => new_first_st), new_remaining_st)
    return y, new_st
end