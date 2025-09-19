using ComponentArrays
using Optimisers
using ADTypes
using ConcreteStructs: @concrete

function _default_callback(l, epoch, ts)
    if epoch % 10 == 0
        @info "Epoch $epoch: Loss = $l"
    end
end

@concrete struct LuxBackend <: AbstractOptimBackend
    opt::Optimisers.AbstractRule
    n_epochs::Int
    adtype::ADTypes.AbstractADType
    loss_fn::Any
    callback::Any
end

function LuxBackend(opt, n_epochs, adtype, loss_fn)
    return LuxBackend(opt, n_epochs, adtype, loss_fn, _default_callback)
end

nameof(::LuxBackend) = "LuxBackend"

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
    return InitialConditions(ics_list)
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

# TODO: this function is type unstable
function train(backend::LuxBackend,
        model::AbstractLuxLayer,
        dataloader::SegmentedTimeSeries,
        experimental_setup::InferICs,
        rng = Random.default_rng(),
        luxtype = Lux.f64)
    dataloader = luxtype(dataloader)
    dataloader = tokenize(dataloader)

    ics_layer = _get_ic_layer(dataloader, experimental_setup)

    if !is_ics_estimated(experimental_setup)
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

    segment_ics = _get_ic_values(dataloader, ics_layer, best_ps, best_st)

    return (; ps = best_ps.model, st = best_st.model, ics = segment_ics)
end

get_parameter_values(train_state::Training.TrainState) = train_state.parameters
get_state_values(train_state::Training.TrainState) = train_state.states

function train(backend::LuxBackend,
        model::AbstractLuxLayer,
        dataloader_train::SegmentedTimeSeries,
        experimental_setup::WithValidation,
        rng = Random.default_rng(),
        luxtype = Lux.f64)
    dataloader_train = luxtype(tokenize(dataloader_train))
    dataloader_valid = luxtype(tokenize(experimental_setup.dataloader))

    @assert length(tokens(dataloader_train))==length(tokens(dataloader_valid)) "The training and validation dataloaders must have the same number of segments"

    ics_layer = _get_ic_layer(dataloader_train, experimental_setup)

    if !is_ics_estimated(experimental_setup)
        ics_layer = Lux.Experimental.FrozenLayer(ics_layer)
    end

    ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(_feature_wrapper),
        initial_conditions = ics_layer, model = model)

    ps, st = luxtype(Lux.setup(rng, ode_model_with_ics))
    ps = ComponentArray(ps) # We transforms ps to support all sensealg from SciMLSensitivity

    # Initialize training state
    train_state = Training.TrainState(ode_model_with_ics, ps, st, backend.opt)
    best_ps = ps.model
    best_st = st.model
    segment_ics = _get_ic_values(dataloader_train, ics_layer, ps.initial_conditions, st.initial_conditions)
    best_loss = luxtype(Inf)
    for epoch in 1:(backend.n_epochs)
        train_loss = luxtype(0.0)
        for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader_train
            _, loss, _, train_state = Training.single_train_step!(
                backend.adtype,
                backend.loss_fn,
                ((batched_tokens, batched_tsteps), batched_segments),
                train_state)
            train_loss += loss
        end

        valid_loss = 0.0
        ps, st = get_parameter_values(train_state), get_state_values(train_state)
        segment_ics = _get_ic_values(dataloader_train, ics_layer, ps.initial_conditions, st.initial_conditions)
        for tok in tokens(dataloader_valid)
            u0 = segment_ics[tok].u0
            _, segment_tsteps_train = dataloader_train[tok]
            t0 = segment_tsteps_train[1]
            data_valid, saveat = dataloader_valid[tok]
            tspan = (t0, saveat[end])
            data_pred = model((; u0, saveat, tspan), ps.model, st.model)[1][
                :, :, 1
            ]
            valid_loss += backend.loss_fn(data_pred, data_valid)
        end
        @debug "Train loss: $train_loss"
        @debug "Validation loss: $valid_loss"
        if valid_loss < best_loss
            best_ps = ps.model
            best_st = st.model
            best_loss = train_loss
        end
        backend.callback(train_loss, epoch, train_state)
    end

    return (; ps = best_ps, st = best_st, ics = segment_ics)
end
