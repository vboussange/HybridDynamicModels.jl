module HybridDynamicModelsLuxExt
import LuxCore
import Lux: Lux, Chain, Training
import Optimisers
import ComponentArrays: ComponentArray
using ADTypes
using ConcreteStructs: @concrete
using HybridDynamicModels: HybridDynamicModels,
                           SegmentedTimeSeries,
                           AbstractOptimBackend,
                           InferICs,
                           tokenize,
                           tokens,
                           get_u0_constraint,
                           ParameterLayer,
                           ICLayer,
                           is_ics_estimated

function _default_callback(l, epoch, ts)
    if epoch % 10 == 0
        @info "Epoch $epoch: Loss = $l"
    end
end

@concrete struct SGDBackend <: HybridDynamicModels.SGDBackend
    opt::Optimisers.AbstractRule
    n_epochs::Int
    adtype::ADTypes.AbstractADType
    loss_fn::Any
    callback::Any
end

function HybridDynamicModels.SGDBackend(opt, n_epochs, adtype, loss_fn)
    return SGDBackend(opt, n_epochs, adtype, loss_fn, _default_callback)
end

function _feature_wrapper((token, tsteps_batch))
    return [(; u0 = token[i],
                saveat = tsteps_batch[:, i],
                tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
            )
            for i in eachindex(token)]
end

function _get_ic_layer(dataloader, experimental_setup)
    function _fun(tok)
        segment_data, _ = dataloader[tok]
        u0 = segment_data[:, 1]
        return ParameterLayer(; constraint = get_u0_constraint(experimental_setup),
            init_value = (; u0))
    end
    ics_list = [_fun(tok) for tok in tokens(dataloader)]
    return ICLayer(ics_list)
end

function _get_ic_values(dataloader, ic_layer, ps, st)
    segment_ics = []
    for i in tokens(dataloader)
        _, segment_tsteps = dataloader[i]
        t0 = segment_tsteps[1]
        push!(segment_ics, merge(ic_layer((; u0 = i), ps, st)[1], (; t0)))
    end
    segment_ics = vcat(segment_ics...)
    return segment_ics
end

function HybridDynamicModels.train(backend::SGDBackend,
        model::LuxCore.AbstractLuxLayer,
        dataloader::SegmentedTimeSeries,
        infer_ics::InferICs,
        rng = Random.default_rng(),
        luxtype = Lux.f64)
    dataloader = luxtype(dataloader)
    dataloader = tokenize(dataloader)

    ics_layer = _get_ic_layer(dataloader, infer_ics)

    if !is_ics_estimated(infer_ics)
        ics_layer = Lux.Experimental.FrozenLayer(ics_layer)
    end

    ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(_feature_wrapper),
        initial_conditions = ics_layer, model = model)

    ps, st = luxtype(Lux.setup(rng, ode_model_with_ics))
    ps = ComponentArray(ps) # We transforms ps to support all sensealg from SciMLSensitivity

    train_state = Training.TrainState(ode_model_with_ics, ps, st, backend.opt)
    best_ps = ps
    best_st = st
    best_loss = luxtype(Inf)
    for epoch in 1:(backend.n_epochs)
        train_loss = luxtype(0.0)
        for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader
            _, loss, _, train_state = Training.single_train_step!(
                backend.adtype,
                backend.loss_fn,
                ((batched_tokens, batched_tsteps), batched_segments),
                train_state)
            train_loss += loss
        end
        if train_loss < best_loss
            best_ps = get_parameter_values(train_state)
            best_st = get_state_values(train_state)
            best_loss = train_loss
        end
        backend.callback(train_loss, epoch, train_state)
    end

    segment_ics = _get_ic_values(dataloader, ics_layer, best_ps.initial_conditions, best_st.initial_conditions)

    return (; ps = best_ps.model, st = best_st.model, ics = segment_ics)
end

get_parameter_values(train_state::Training.TrainState) = train_state.parameters
get_state_values(train_state::Training.TrainState) = train_state.states

end