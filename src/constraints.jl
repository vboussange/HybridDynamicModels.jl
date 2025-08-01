using ConcreteStructs: @concrete
using LogExpFunctions

abstract type AbstractConstraint end

@concrete struct BoxConstraint
    lower_bound
    upper_bound
end


"""
    _to_optim_space(constraint::BoxConstraint, x::AbstractArray)

Maps x from parameter space [lower_bound, upper_bound] to optimization space (-Inf, Inf) using a scaled logit transform.
Works elementwise for arrays or scalars.
"""
function _to_optim_space(constraint::BoxConstraint, x::AbstractArray)
    lb = constraint.lower_bound
    ub = constraint.upper_bound
    # elementwise transform: y = logit((x - lb) / (ub - lb))
    return LogExpFunctions.logit.((x .- lb) ./ (ub .- lb))
end

"""
    _to_param_space(constraint::BoxConstraint, y::AbstractArray)

Inverse of _to_optim_space: maps y from optimization space (-Inf, Inf) to parameter space [lower_bound, upper_bound].
Works elementwise for arrays or scalars.
"""
function _to_param_space(constraint::BoxConstraint, y::AbstractArray)
    lb = constraint.lower_bound
    ub = constraint.upper_bound
    # elementwise inverse: x = lb + (ub - lb) * logistic(y)
    return lb .+ (ub .- lb) .* LogExpFunctions.logistic.(y)
end