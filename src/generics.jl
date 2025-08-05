const LuxStruct = Union{AbstractLuxLayer, StatefulLuxLayer}

# featureless Trait definition
# See here: https://docs.julialang.org/en/v1/manual/types/#%22Value-types%22
# By default, all Lux objects are *not* featureless, and must be called with (x, ps, st)
isfeatureless(::T) where {T <: LuxStruct} = isfeatureless(T)
isfeatureless(::Type{<:LuxStruct}) = Val(false)

# Overwriting the apply function to handle featureless layers
function Lux.apply(m::M, ps, st) where {M <: LuxStruct} 
    return Lux.apply(isfeatureless(M), m, ps, st)
end

function Lux.apply(m::M, x, ps, st) where {M <: LuxStruct} 
    return Lux.apply(isfeatureless(M), x, m, ps, st)
end

function Lux.apply(::Val{false}, m::M, ps, st) where {M <: AbstractLuxLayer}
    throw(ErrorException("$M must be provided with a feature `x`."))
end

function Lux.apply(::Val{true}, m::M, x, ps, st) where {M <: LuxStruct}
    throw(ErrorException("$M is feature less, behavior is undefined when called with a feature `x`."))
end

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