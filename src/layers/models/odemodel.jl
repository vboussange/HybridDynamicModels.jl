"""
    ODEModel(layers::NamedTuple, dudt::Function; kwargs...)

Wraps an ODE model for simulation using Lux layers.

## Arguments
  - `layers`: NamedTuple of Lux layers representing the layers of the model.
  - `dudt`: Function that computes the derivative of the state, with signature `dudt(layers, u, ps, t)`.
  - `kwargs`: Additional keyword arguments passed to the solver (e.g., `tspan`, `saveat`, `alg`).

## Inputs
- (`x`, `ps`, `st`)
  - `x`: a `NamedTuple` or `AbstractVector{NamedTuple}` (batch mode).
  - `ps`: Parameters of the model.
  - `st`: States of the model.
- A tuple of (`x`, `ps`, `st`): batch mode.
- (`ps`, `st`): If `x` not provided, defaults to `kwargs`.

## Outputs
- (`sol`, `st`)
  - `sol`: Solution of the ODE problem, with second dimension corresponding to time and batches stacked along the third dimension, if applicable.
  - `st`: Updated states of the model.

## Behavior
`layers` are wrapped in `StatefulLuxLayer`s to maintain their states. Hence, if `layers = (; layer1 = Lux.Dense(10, 10, relu))`, then `dudt` should be defined as `dudt(layers, ps, u, t) = layers.layer1(u, ps.layer1)`.

## Example

```jldoctest
julia> layers = (; layer1 = Lux.Dense(10, 10, relu))
julia> dudt(layers, ps, st, u, t) = layers.layer1(u, ps.layer1)[1]
julia> ode_model = ODEModel(layers, dudt, tspan = (0f0, 1f0), saveat = range(0f0, stop=1f0, length=100), alg = Tsit5())
julia> ps, st = Lux.setup(Random.default_rng(), ode_model)
julia> ode_model((; u0 = ones(Float32, 10)), ps, st)
```

!!!warning
    Undefined behavior when `ps` is not a NamedTuple
"""
@concrete struct ODEModel <: HybridDynamicModel
    layers<:NamedTuple{names, <:NTuple{N, AbstractLuxLayer}} where {names, N}
    dudt # function that computes the derivative
    init_state <: Function
end

function ODEModel(layers, dudt; kwargs...)
  ODEModel(layers, dudt, () -> NamedTuple(kwargs))
end

function LuxCore.initialstates(rng::AbstractRNG, m::ODEModel)
    st = LuxCore.initialstates(rng, m.layers)
    st = merge(st, (; kwargs = m.init_state()))
    return st
end

function forward(m::ODEModel, layers, u0, tspan, saveat, ps; kwargs...)
    function __dudt(u, p, t)
        m.dudt(layers, u, p, t)
    end

    prob = ODEProblem{false}(ODEFunction{false}(__dudt), 
                            u0, 
                            tspan, 
                            ps)
    alg = kwargs[:alg] === nothing ? Tsit5() : kwargs[:alg]

    sol = solve(prob, alg; saveat, kwargs...) |> Array
    return sol
end