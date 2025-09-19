
"""
    AnalyticModel(layers::NamedTuple, fun::Function; kwargs...)

Wraps an analytic model for direct evaluation using LuxCore layers.

## Arguments
  - `layers`: NamedTuple of LuxCore layers representing the layers of the model.
  - `fun`: Function that computes the analytic solution, with signature `fun(layers, u0, t0, ps, t)`.
  - `kwargs`: Additional keyword arguments (e.g., default values for `u0`, `tspan`, `saveat`).

## Inputs
- (`x`, `ps`, `st`)
  - `x`: a `NamedTuple` or `AbstractVector{NamedTuple}` (batch mode).
  - `ps`: Parameters of the model.
  - `st`: States of the model.
- A tuple of (`x`, `ps`, `st`): batch mode.
- (`ps`, `st`): If `x` not provided, defaults to `kwargs`.

## Outputs
- (`sol`, `st`)
  - `sol`: Solution array evaluated at specified time points, with second dimension corresponding to time and batches stacked along the third dimension, if applicable.
  - `st`: Updated states of the model.

## Behavior
`layers` are wrapped in `StatefulLuxLayer` to maintain their states. The analytic function `fun` should be defined as `fun(layers, u0, t0, ps, t)` where `t` can be a vector of time points and `t0` is extracted from `tspan`.

## Example

```jldoctest
julia> layers = (; params = ParameterLayer(init_value = (a = 1.0, b = 0.5)))
julia> analytic_solution(layers, u0, t0, ps, t) = u0 .* exp.(layers.params(ps.params)[1].a .* (t .- t0))
julia> model = AnalyticModel(layers, analytic_solution; u0 = [1.0], tspan = (0.0, 1.0), saveat = 0:0.1:1.0)
julia> ps, st = Lux.setup(Random.default_rng(), model)
julia> model((; u0 = [1.0]), ps, st)
```

!!!warning
    Undefined behavior when `ps` is not a NamedTuple
"""
@concrete struct AnalyticModel <: HybridDynamicModel
    layers<:NamedTuple{names, <:NTuple{N, AbstractLuxLayer}} where {names, N}
    fun # function that computes the analytic solution
    init_state <: Function
end

function AnalyticModel(layers, fun; kwargs...)
    AnalyticModel(layers, fun, () -> NamedTuple(kwargs))
end

function forward(m::AnalyticModel, layers, u0, tspan, saveat, ps; kwargs...)
    t0 = first(tspan)
    
    # Evaluate the analytic function at specified time points
    sol_arr = m.fun.(Ref(layers), Ref(u0), Ref(t0), Ref(ps), saveat)
    return reduce(hcat, sol_arr)
end
