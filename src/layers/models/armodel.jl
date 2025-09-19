"""
    ARModel(layers::NamedTuple, fun::Function; kwargs...)

Wraps an autoregressive (AR) model for iterative time series prediction using Lux layers.

## Arguments
  - `layers`: NamedTuple of Lux layers representing the layers of the model.
  - `fun`: Function that computes the next time step, with signature `fun(layers, u, ps, t)`.
  - `kwargs`: Additional keyword arguments (e.g., default values for `u0`, `t0`, `saveat`, `dt`).

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
julia> layers = (; 
           predictor = Dense(2, 2), 
           params = ParameterLayer(init_value = (decay = 0.95,))
       )
julia> ar_step(layers, u, ps, t) = layers.predictor(u, ps.predictor) .* layers.params(ps.params)[1].decay
julia> model = ARModel(layers, ar_step; dt = 0.1)
julia> ps, st = Lux.setup(Random.default_rng(), model)
julia> model((; u0 = [1.0, 0.5], t0 = 0.0, saveat = 0:0.1:1.0), ps, st)
```

## Use Cases
- Discrete-time dynamical systems
- Time series forecasting with neural networks
- Recurrent model architectures
- Hybrid models combining neural networks with autoregressive dynamics

!!!warning
    Undefined behavior when `ps` is not a NamedTuple
"""
@concrete struct ARModel <: HybridDynamicModelsWrapperLayer
    layers<:NamedTuple{names, <:NTuple{N, AbstractLuxLayer}} where {names, N}
    fun # function that computes the next time step
    kwargs
end

ARModel(layers, fun; kwargs...) = ARModel(layers, fun, NamedTuple(kwargs))

function (m::ARModel)(x::NamedTuple, ps, st)
    u0 = hasproperty(x, :u0) ? getproperty(x, :u0) : getproperty(m.kwargs, :u0)
    t0 = hasproperty(x, :t0) ? getproperty(x, :t0) : getproperty(m.kwargs, :t0, 0.0)
    saveat = hasproperty(x, :saveat) ? getproperty(x, :saveat) : getproperty(m.kwargs, :saveat)
    dt = hasproperty(x, :dt) ? getproperty(x, :dt) : getproperty(m.kwargs, :dt, 1.0)
    
    layer_keys = keys(m.layers)
    layer_vals = map(k -> StatefulLuxLayer{true}(getproperty(m.layers, k), getproperty(ps, k), get_state(getproperty(st, k))), layer_keys)
    layers = NamedTuple{layer_keys}(layer_vals)

    # Initialize solution storage
    n_steps = length(saveat)
    n_vars = length(u0)
    sol = zeros(eltype(u0), n_vars, n_steps)
    
    # Set initial condition
    current_u = copy(u0)
    current_t = t0
    
    # Find the index of the first save point
    save_idx = 1
    
    # Store initial condition if t0 is in saveat
    if saveat[1] ≈ t0
        sol[:, save_idx] = current_u
        save_idx += 1
    end
    
    # Iterate through time
    while save_idx <= n_steps
        target_t = saveat[save_idx]
        
        # Step forward until we reach or exceed the target time
        while current_t < target_t
            # Take a step (or partial step to land exactly on target)
            step_size = min(dt, target_t - current_t)
            
            # Apply the AR function
            next_u = m.fun(layers, current_u, ps, current_t)
            
            # Update state and time
            current_u = next_u
            current_t += step_size
        end
        
        # Store the result at the save point
        sol[:, save_idx] = current_u
        save_idx += 1
    end
    
    new_st = NamedTuple{layer_keys}(map(k -> get_state(getproperty(layers, k)), layer_keys))
    return sol, new_st
end
