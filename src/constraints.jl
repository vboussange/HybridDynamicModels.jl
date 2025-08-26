abstract type AbstractConstraint end
using Bijectors
import Bijectors: NamedTransform, transform, logabsdetjac

"""
    Constraint(transform)

Defines a constraint for projected gradient descent.

## Arguments

  - `transform`: A function that maps parameters in the optimization space to the parameter space.
    It should be invertible, i.e., `inverse(transform)(ps)` should map the
    parameter space to the optimization space.

## Inputs
  - `x`: Parameters in the optimization space.

## Outputs
  - `y`: Parameters in the parameter space.
"""
@concrete struct Constraint <: AbstractConstraint
    transform
end

(constraint::Constraint)(x) = inverse(constraint.transform)(x)
_to_optim_space(constraint::Constraint, x) = constraint.transform(x)

const NoConstraint() = Constraint(identity)

# Below is an attempt to support ComponentArray with NamedTransform; for now, it fails
# fieldnames(T <: Type{<:ComponentVector}) = keys(getaxes(T)[1])

# @generated function transform(
#     b::NamedTransform{names1}, x::T
# ) where {names1, T}
#     exprs = []
#     for n in fieldnames(T)
#         if n in names1
#             # Use processed value
#             push!(exprs, :($n = b.bs.$n(x.$n)))
#         else
#             # Use existing value
#             push!(exprs, :($n = x.$n))
#         end
#     end
#     return :($(exprs...),)
# end

# @generated function logabsdetjac(b::NamedTransform{names}, x) where {names}
#     exprs = [:(logabsdetjac(b.bs.$n, x.$n)) for n in names]
#     return :(+($(exprs...)))
# end

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
