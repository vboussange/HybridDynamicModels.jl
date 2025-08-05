function train(model, dataloader, optimizer::Optimisers.AbstractRule, loss_fn, epochs, device=Lux.cpu_device(), rng=Random.GLOBAL_RNG)
    dev = device()
    ps, st = dev(Lux.setup(model))

    train_state = Training.TrainState(model, ps, st, optimizer)

    vjp = dev isa ReactantDevice ? AutoEnzyme() : AutoZygote()

    if dev isa ReactantDevice
        x_ra = first(test_dataloader)[1]
        model_compiled = Reactant.with_config(;
            dot_general_precision=PrecisionConfig.HIGH,
            convolution_precision=PrecisionConfig.HIGH,
        ) do
            @compile model(x_ra, ps, Lux.testmode(st))
        end
    else
        model_compiled = model
    end

    for epoch in 1:epochs
        for (x, y) in dataloader
            # Forward pass
            _, _, _, train_state = Training.single_train_step!(
                vjp, loss_fn, (x, y), train_state
            )
        end
    end

    return train_state.parameters, train_state.states
end
