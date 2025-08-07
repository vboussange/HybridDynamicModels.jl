using OrdinaryDiffEq
using Bijectors
import Lux
using Lux: MSELoss, Chain, Training
using HybridModelling
using SciMLSensitivity
using UnPack
using Plots
import Optimisers: Adam
using Random
using Printf


p_true = (;b = [0.23, 0.5],)

loss_fn = MSELoss()

params = ParameterLayer(#constraint = Constraint(transform), 
                        constraint = NoConstraint(),
                        # init_value = ComponentArray(;b = [0.1, 0.3])
                        init_value = (;b = [10., 11.]),
                        )
mychain = Chain(params = params, wrapper = Lux.WrappedFunction(x -> x.b))
ps, st = Lux.setup(Random.default_rng(), mychain)
train_state = Training.TrainState(mychain, ps, st, Adam(1f-1))

n_epochs = 1000
for epoch in 1:n_epochs
    # @show tokens
    # @show size(segment_data)
    _, loss, _, train_state = Training.single_train_step!(
        Lux.AutoZygote(), 
        loss_fn, 
        ((), p_true.b),
        train_state)
    println("Epoch $epoch: Loss = ", loss)
end

params(train_state.parameters.params, (;)) # This works

