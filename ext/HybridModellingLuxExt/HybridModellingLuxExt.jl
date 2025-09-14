module HybridModellingLuxExt

import Lux
const Optimisers = Lux.Optimisers
import Lux: Training
using ADTypes
using ConcreteStructs: @concrete
import HybridDynamicModelling: SegmentedTimeSeries

function _default_callback(l, epoch, ts)
    if epoch % 10 == 0
        @info "Epoch $epoch: Loss = $l"
    end
end

"""
    SGDBackend(opt, n_epochs, adtype, loss_fn; verbose_frequency=10, callback=(l,m,p,s)->nothing)

Training backend relying on Lux.jl training API. Use for mode estimation.

## Fields
- `opt`: Optimizers.jl rule for parameter updates
- `n_epochs`: Number of training epochs
- `adtype`: Automatic differentiation backend from ADTypes.jl
- `loss_fn`: Loss function for training
- `callback`: User-defined callback function called each epoch

## Arguments
- `opt`: Optimization rule (e.g., `Adam(1e-3)`)
- `n_epochs`: Total number of training epochs
- `adtype`: AD backend (e.g., `AutoZygote()`, `AutoForwardDiff()`)
- `loss_fn`: Loss function compatible with Lux training
- `callback=(l::AbstractFloat, epoch::Int, ts::Lux.TrainingState)->nothing`: Called at each epoch. Refer to [Lux.Training.TrainState](https://lux.csail.mit.edu/stable/api/Lux/utilities#Training-API) for fields of `ts`.

## Returns
When provided to `train`, the function returns a named tuple with the following fields:
- `ps`: The best model parameters found during training.
- `st`: Associated states.
- `ics`: A vector of named tuple where `ics[i].u0` contains estimated initial conditions for segment `i`, indexed with `ics[i].t0`

## Example
```julia
backend = SGDBackend(Adam(1e-3), 1000, AutoZygote(), MSELoss())
result = train(backend, model, dataloader, infer_ics)
```
"""
@concrete struct SGDBackend <: AbstractOptimBackend
    opt::Optimisers.AbstractRule
    n_epochs::Int
    adtype::ADTypes.AbstractADType
    loss_fn::Any
    callback::Any
end

SGDBackend(opt, n_epochs, adtype, loss_fn) = SGDBackend(opt, n_epochs, adtype, loss_fn, _default_callback)

function _feature_wrapper((token, tsteps_batch))
    return [(; u0 = token[i],
                saveat = tsteps_batch[:, i],
                tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
            )
            for i in eachindex(token)]
end

function _get_ics(dataloader, infer_ics::InferICs)
    function _fun(tok)
        segment_data, _ = dataloader[tok]
        u0 = segment_data[:, 1]
        ParameterLayer(;constraint = infer_ics.u0_constraint,
                    init_value = (; u0))
    end
    ics_list = [ _fun(tok) for tok in tokens(dataloader)]
    return InitialConditions(ics_list)
end

function train(backend::SGDBackend,
        model::AbstractLuxLayer,
        dataloader::SegmentedTimeSeries,
        infer_ics::InferICs,
        rng = Random.default_rng(),
        luxtype = Lux.f64)

    dataloader = luxtype(dataloader)
    dataloader = tokenize(dataloader)

    ics = _get_ics(dataloader, infer_ics)

    if !istrue(infer_ics)
        ics = Lux.Experimental.FrozenLayer(ics)
    end

    ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(_feature_wrapper),
        initial_conditions = ics, model = model)

    ps, st = luxtype(Lux.setup(rng, ode_model_with_ics))
    ps = ComponentArray(ps) # We transforms ps to support all sensealg from SciMLSensitivity

    train_state = Training.TrainState(ode_model_with_ics, ps, st, backend.opt)
    best_ps = ps
    best_st = st
    best_loss = luxtype(Inf)
    for epoch in 1:(backend.n_epochs)
        tot_loss = luxtype(0.0) 
        for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader
            _, loss, _, train_state = Training.single_train_step!(
                backend.adtype,
                backend.loss_fn,
                ((batched_tokens, batched_tsteps), batched_segments),
                train_state)
            tot_loss += loss
        end
        if tot_loss < best_loss
            best_ps = get_parameter_values(train_state)
            best_st = get_state_values(train_state)
            best_loss = tot_loss
        end
        backend.callback(tot_loss, epoch, train_state)
    end
    segment_ics = []
    for i in tokens(dataloader)
        _, segment_tsteps = dataloader[i]
        t0 = segment_tsteps[1]
        push!(segment_ics, merge(ics((; u0 = i), best_ps.initial_conditions, best_st.initial_conditions)[1], (; t0)))
    end
    segment_ics = vcat(segment_ics...)

    return (; ps = best_ps.model, st = best_st.model, ics = segment_ics)
end

get_parameter_values(train_state::Training.TrainState) = train_state.parameters
get_state_values(train_state::Training.TrainState) = train_state.states

end