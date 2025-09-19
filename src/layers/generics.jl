abstract type HybridDynamicModelsWrapperLayer  <: LuxCore.AbstractLuxWrapperLayer{:layers} end

# Handling batches
function (m::HybridDynamicModelsWrapperLayer)(x::AbstractVector{<:NamedTuple}, ps, st)
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

(m::HybridDynamicModelsWrapperLayer)(ps, st) = m(NamedTuple(), ps, st)

abstract type HybridDynamicModel <: HybridDynamicModelsWrapperLayer end

function LuxCore.initialstates(rng::AbstractRNG, m::HybridDynamicModel)
    st = LuxCore.initialstates(rng, m.layers)
    st = merge(st, (; kwargs = m.init_state()))
    return st
end

function (m::HybridDynamicModel)(x::NamedTuple, ps, st)
    u0 = hasproperty(x, :u0) ? getproperty(x, :u0) : getproperty(st.kwargs, :u0)
    tspan = hasproperty(x, :tspan) ? getproperty(x, :tspan) : getproperty(st.kwargs, :tspan)
    saveat = hasproperty(x, :saveat) ? getproperty(x, :saveat) : getproperty(st.kwargs, :saveat)
    component_keys = keys(m.layers)
    component_vals = map(k -> LuxCore.StatefulLuxLayer{true}(getproperty(m.layers, k), getproperty(ps, k), get_state(getproperty(st, k))), component_keys)
    layers = NamedTuple{component_keys}(component_vals)

    # Remove :u0, :p, and :tspan from x if present, so that when x and kwargs
    # are passed to `solve`, they don't overwrite values in prob see
    # https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/
    # kwargs = merge(st.kwargs, x) # overwriting kwargs with x
    kwargs = Base.structdiff(st.kwargs, NamedTuple{(:u0, :p, :tspan, :saveat)}) # keep kwargs that are not in x

    sol = forward(m, layers, u0, tspan, saveat, ps; kwargs...)

    new_st = NamedTuple{component_keys}(map(k -> get_state(getproperty(layers, k)), component_keys))
    new_st = merge(new_st, (; kwargs = st.kwargs))
    return sol, new_st
end

(m::HybridDynamicModel)(ps, st) = m(NamedTuple(), ps, st)
