using ComponentArrays
using Optimisers
using ADTypes
using ConcreteStructs: @concrete
import HybridModelling: SegmentedTimeSeries

"""
    LuxBackend(opt, n_epochs, adtype, loss_fn; verbose_frequency=10, callback=(l,m,p,s)->nothing)

Training backend relying on Lux.jl training API. Use for mode estimation.

## Fields
- `opt`: Optimizers.jl rule for parameter updates
- `n_epochs`: Number of training epochs
- `adtype`: Automatic differentiation backend from ADTypes.jl
- `loss_fn`: Loss function for training
- `verbose_frequency`: Frequency for printing training progress
- `callback`: User-defined callback function called each epoch

## Arguments
- `opt`: Optimization rule (e.g., `Adam(1e-3)`)
- `n_epochs`: Total number of training epochs
- `adtype`: AD backend (e.g., `AutoZygote()`, `AutoForwardDiff()`)
- `loss_fn`: Loss function compatible with Lux training

## Keyword Arguments
- `verbose_frequency=10`: Print loss every N epochs
- `callback=(l,m,p,s)->nothing`: Called with (loss, model, params, state) each epoch

## Returns
When provided to `train`, the function returns a named tuple with the following fields:
- `best_model`: The best model found during training
- `best_ics`: Best initial conditions found during training.
- `info`: container obtained from `callback`

## Example
```julia
backend = LuxBackend(Adam(1e-3), 1000, AutoZygote(), MSELoss())
result = train(backend, model, dataloader, infer_ics)
```
"""
@concrete struct LuxBackend <: AbstractOptimBackend
    opt::Optimisers.AbstractRule
    n_epochs::Int
    adtype::ADTypes.AbstractADType
    loss_fn::Any
    verbose_frequency::Any
    callback::Any
end

function LuxBackend(opt, n_epochs, adtype, loss_fn; verbose_frequency = 10,
        callback = (l, m, p, s) -> nothing)
    return LuxBackend(opt, n_epochs, adtype, loss_fn, verbose_frequency, callback)
end

# TODO: implement test
function train(backend::LuxBackend,
        model::AbstractLuxLayer,
        dataloader::SegmentedTimeSeries,
        infer_ics::InferICs,
        rng = Random.default_rng();
        luxtype = Lux.f64)
    dataloader = tokenize(dataloader)

    ic_list = ParameterLayer[]
    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        t0 = segment_tsteps[1]
        if istrue(infer_ics)
            push!(ic_list,
                ParameterLayer(constraint = infer_ics.u0_constraint,
                    init_value = (; u0), init_state_value = (; t0)))
        else
            push!(ic_list, ParameterLayer(init_value = (;), init_state_value = (; t0, u0)))
        end
    end
    ics = InitialConditions(ic_list)

    function feature_wrapper((token, tsteps_batch))
        return [(; u0 = token[i],
                    saveat = tsteps_batch[:, i],
                    tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
                )
                for i in eachindex(token)]
    end

    ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(feature_wrapper),
        initial_conditions = ics, model = model)

    ps, st = Lux.setup(rng, ode_model_with_ics)
    ps = ps |> luxtype |> ComponentArray # We transforms ps to support all sensealg from SciMLSensitivity

    train_state = Training.TrainState(ode_model_with_ics, ps, st, backend.opt)
    best_ps = ps
    best_loss = Inf
    info = []
    for epoch in 1:(backend.n_epochs)
        tot_loss = 0.0
        for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader
            _, loss, _, train_state = Training.single_train_step!(
                backend.adtype,
                backend.loss_fn,
                ((batched_tokens, batched_tsteps), batched_segments),
                train_state)
            tot_loss += loss
        end
        if epoch % backend.verbose_frequency == 0
            println("Epoch $epoch: Total Loss = ", tot_loss)
        end
        if tot_loss < best_loss
            best_ps = get_parameter_values(train_state)
            best_loss = tot_loss
        end
        push!(info,
            backend.callback(
                tot_loss, ode_model_with_ics, get_parameter_values(train_state),
                get_state_values(train_state)))
    end

    # TODO: separate ics from model; return ics as a vector of named tuples

    best_model = StatefulLuxLayer{true}(ode_model_with_ics, best_ps, st)
    return (; best_model, info)
end

get_parameter_values(train_state::Training.TrainState) = train_state.parameters
get_state_values(train_state::Training.TrainState) = train_state.states
