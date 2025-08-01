abstract type AbstractConstraint end
using Bijectors

"""
    Constraint(transform)

`transform(ps)` should project a named tuple of parameters `ps` defined over a
certain parameter space into the Euclidean space (optimization space).
`inverse(transform)(ys)` should map the optimization space back to the parameter
space.
"""
struct Constraint{T} <: AbstractConstraint
    transform::T
end

(constraint::Constraint)(x) = inverse(constraint.transform)(x)
_to_optim_space(constraint::Constraint, x) = constraint.transform(x)

const NoConstraint() = Constraint(identity)

# Below is a custom implementation of a box constaint, but it is not as nice 
# as the solution from Bijectors.jl, since Bijectors.jl works with named tuples of transforms
# struct NoConstraint <: AbstractConstraint end
# (n::NoConstraint)(x) = x
# _to_param_space(n::NoConstraint, x) = x

# @concrete struct BoxConstraint
#     lower_bound
#     upper_bound
# end

# """
#     _to_optim_space(constraint::BoxConstraint, x::AbstractArray)

# Maps x from parameter space [lower_bound, upper_bound] to optimization space (-Inf, Inf) using a scaled logit transform.
# Works elementwise for arrays or scalars.
# """
# function (constraint::BoxConstraint)(x::AbstractArray)
#     lb = constraint.lower_bound
#     ub = constraint.upper_bound
#     # elementwise transform: y = logit((x - lb) / (ub - lb))
#     return LogExpFunctions.logit.((x .- lb) ./ (ub .- lb))
# end

# """
#     _to_param_space(constraint::BoxConstraint, y::AbstractArray)

# Inverse of _to_optim_space: maps y from optimization space (-Inf, Inf) to parameter space [lower_bound, upper_bound].
# Works elementwise for arrays or scalars.
# """
# function _to_param_space(constraint::BoxConstraint, y::AbstractArray)
#     lb = constraint.lower_bound
#     ub = constraint.upper_bound
#     # elementwise inverse: x = lb + (ub - lb) * logistic(y)
#     return lb .+ (ub .- lb) .* LogExpFunctions.logistic.(y)
# end
