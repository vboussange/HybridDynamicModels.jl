"""
    ARModel(layers::NamedTuple, fun::Function; kwargs...)

Wraps an autoregressive (AR) model.

## Arguments
  - `layers`: NamedTuple of Lux layers representing the layers associated with the model.
  - `fun`: Function that computes the next time step, with signature `fun(layers, u, ps, t)`.
  - `kwargs`: Additional keyword arguments (e.g., default values for `u0`, `tspan`, `saveat`, `dt`).

## Inputs
- (`x`, `ps`, `st`)
  - `x`: a `NamedTuple` or `AbstractVector{NamedTuple}` (batch mode).
  - `ps`: Parameters of the model.
  - `st`: States of the model.
- A tuple of (`x`, `ps`, `st`): batch mode.
- (`ps`, `st`): If `x` not provided, defaults to `kwargs`.

## Outputs
- (`sol`, `st`)
  - `sol`: Solution array with iterative predictions, with second dimension corresponding to time and batches stacked along the third dimension, if applicable.
  - `st`: Updated states of the model.

## Behavior
`layers` are wrapped in `StatefulLuxLayer` to maintain their states. The AR function `fun` should be defined as `fun(layers, u, ps, t)` where:
- `u` is the current state
- `t` is the current time
- The function returns the next state

The model iteratively applies the function to generate a time series from initial conditions.

## Example

```jldoctest
julia> using HybridDynamicModels, Lux, Random

julia> layers = (; 
           predictor = Dense(2, 2), 
           params = ParameterLayer(init_value = (; decay = [95f-2],))
       );

julia> ar_step(layers, u, ps, t) = layers.predictor(u, ps.predictor) .* layers.params(ps.params).decay;

julia> model = ARModel(layers, ar_step; dt = 1f-1, u0 = [1f0, 5f-1], tspan = (0f0, 1f0), saveat = 0:1f-1:1f0);

julia> ps, st = Lux.setup(Random.default_rng(), model);

julia> x = (; u0 = [1f0, 5f-1]);

julia> y, st = model(x, ps, st);

julia> size(y) # 2 state variables, 11 time points
(2, 11) 
```

!!!warning
    Undefined behavior when `x` is not a NamedTuple
"""
@concrete struct ARModel <: HybridDynamicModel
    layers<:NamedTuple{names, <:NTuple{N, AbstractLuxLayer}} where {names, N}
    fun # function that computes the next time step
    init_state <: Function
end

function ARModel(layers, fun; kwargs...)
    ARModel(layers, fun, () -> NamedTuple(kwargs))
end

function forward(m::ARModel, layers, u0, tspan, saveat, ps; dt)
    t0 = first(tspan)
    
    # Helper function to build solution immutably
    function build_sol(current_u, current_t, remaining_saveat, sol_so_far)
        if isempty(remaining_saveat)
            return sol_so_far
        end
        target_t = first(remaining_saveat)
        while current_t < target_t
            step_size = min(dt, target_t - current_t)
            next_u = m.fun(layers, current_u, ps, current_t)
            current_u = next_u
            current_t += step_size
        end
        new_sol = (sol_so_far..., copy(current_u))
        build_sol(current_u, current_t, remaining_saveat[2:end], new_sol)
    end
    
    # Determine initial sol and remaining saveat
    if !isempty(saveat) && saveat[1] ≈ t0
        initial_sol = (copy(u0),)
        remaining = saveat[2:end]
    else
        initial_sol = ()
        remaining = saveat
    end
    
    # Build the solution tuple
    sol_tuple = build_sol(copy(u0), t0, remaining, initial_sol)
    
    return reduce(hcat, sol_tuple)
end
