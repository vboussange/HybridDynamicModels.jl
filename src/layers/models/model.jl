abstract type AbstractDynamicalModel  <: Lux.AbstractLuxWrapperLayer{:layers} end


# Handling batches
function (m::AbstractDynamicalModel)(x::AbstractVector{<:NamedTuple}, ps, st)
    function step(acc, xi)
        sols, curr_st = acc
        sol, new_st = m(xi, ps, curr_st)
        return (tuple(sols..., sol), new_st)
    end
    (sols, new_st) = reduce(step, x; init=((), st))
    first_shape = size(sols[1])
    if any(size(sol) != first_shape for sol in sols)
        throw(DimensionMismatch("All solutions in batch must have the same dimensions. Got: $(map(size, sols))"))
    end
    return cat(sols...; dims=ndims(sols[1])+1), new_st
end


(m::AbstractDynamicalModel)(ps, st) = m(NamedTuple(), ps, st)