using LogExpFunctions

# see https://github.com/TuringLang/Bijectors.jl/blob/00b08eaaae8f5133452e38c1ec949af453d8bbe6/src/Bijectors.jl#L87
function _clamp(x, a, b)
    T = promote_type(typeof(x), typeof(a), typeof(b))
    clamped_x = ifelse(x < a, convert(T, a), ifelse(x > b, convert(T, b), x))
    return clamped_x
end

struct NoConstraint <: AbstractLuxLayer end
Lux.initialstates(::AbstractRNG, layer::NoConstraint) = (;)
(n::NoConstraint)(x, st) = x, st
inverse(::NoConstraint, y, st) = y, st


@concrete struct BoxConstraint <: AbstractLuxLayer
  init_state <: Function
end
Lux.initialstates(::AbstractRNG, layer::BoxConstraint) = layer.init_state()
BoxConstraint(lb::AbstractArray, ub::AbstractArray) = BoxConstraint(() -> (;lb, ub))

"""
    _to_optim_space(constraint::BoxConstraint, x::AbstractArray)

Maps x from parameter space [lower_bound, upper_bound] to optimization space (-Inf, Inf) using a scaled logit transform.
Works elementwise for arrays or scalars.
"""
function (::BoxConstraint)(y::AbstractArray, st)
    lb = st.lb
    ub = st.ub
    # elementwise inverse: x = lb + (ub - lb) * logistic(y)
    x = _clamp.(truncated_invlink.(y, lb, ub), lb, ub)
    return x, st
end

function truncated_invlink(y, a, b)
    return a + (b - a) * LogExpFunctions.logistic(y)
end

"""
    _to_param_space(constraint::BoxConstraint, y::AbstractArray)

Inverse of _to_optim_space: maps y from optimization space (-Inf, Inf) to parameter space [lower_bound, upper_bound].
Works elementwise for arrays or scalars.
"""
function inverse(::BoxConstraint, x::AbstractArray, st)
    lb = st.lb
    ub = st.ub
    # elementwise transform: y = logit((x - lb) / (ub - lb))
    return truncated_link.(_clamp.(x, lb, ub), lb, ub), st
end


function truncated_link(x, a, b)
    return LogExpFunctions.logit((x - a) / (b - a))
end


@concrete struct NamedTupleConstraint <: LuxCore.AbstractLuxWrapperLayer{:constraints}
    constraints <: NamedTuple
end
(c::NamedTupleConstraint)(x, st) = applyconstraints(c.constraints, x, st)
inverse(c::NamedTupleConstraint, y, st) = inverseapplyconstraints(c.constraints, y, st)

@generated function inverseapplyconstraints(constraints::NamedTuple{fields}, x::NamedTuple, st::NamedTuple{fields}) where fields
    N = length(fields)
    x_symbols = [gensym() for _ in 1:N]
    calls = [
        :(
            $(x_symbols[i]) = @inline inverse(
                constraints.$(fields[i]), x.$(fields[i]), st.$(fields[i])
            )[1]
        ) for i in 1:N
    ]
    push!(calls, :(x_c = NamedTuple{$fields}((($(Tuple(x_symbols)...),)))))
    push!(calls, :(x = merge(x, x_c)))
    push!(calls, :(return x, st))
    return Expr(:block, calls...)
end

@generated function applyconstraints(constraints::NamedTuple{fields}, x::NamedTuple, st::NamedTuple{fields}) where fields
    N = length(fields)
    x_symbols = [gensym() for _ in 1:N]
    calls = [
        :(
            $(x_symbols[i]) = @inline constraints.$(fields[i])(x.$(fields[i]), st.$(fields[i])
            )[1]
        ) for i in 1:N
    ]
    push!(calls, :(x_c = NamedTuple{$fields}((($(Tuple(x_symbols)...),)))))
    push!(calls, :(x = merge(x, x_c)))
    push!(calls, :(return x, st))
    return Expr(:block, calls...)
end

Constraint = Union{NoConstraint, BoxConstraint, NamedTupleConstraint}