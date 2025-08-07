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
using ComponentArrays

function dudt(u, p, t)
    @unpack b = p
    return 0.1 .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:1:100.
tspan = (tsteps[1], tsteps[end])

p_true = (;b = [0.23, 0.5],)
u0 = ones(2)
prob = ODEProblem(ODEFunction{false}(dudt), u0, tspan, p_true)
data = solve(prob, 
            Tsit5(), 
            saveat=tsteps,
            abstol = 1e-6,
            reltol = 1e-6,) |> Array

plot(tsteps, data')

batchsize = 1
dataloader = SegmentedTimeSeries((data, tsteps), 
                                segmentsize = 5, 
                                shift = 4,
                                batchsize = batchsize)

# TODO: bijectors.NamedTransform only works for tuples, and not for ComponentArrays
# Hence, when we convert ps = ComponentArray(ps) for compat with ODEModel, the transform fails
# A quick fix is to use a simple Bijector, which applies to a vector, which is compatible with ComponentArrays
# But ideall, we need to have something working with ComponentArrays
loss_fn = MSELoss()
transform = bijector(Uniform(1e-3, 5e0))
params = ParameterLayer(constraint = Constraint(transform), 
                        # constraint = NoConstraint(),
                        init_value = ComponentArray(;b = [0.1, 0.3])
                        # init_value = ComponentArray(p_true),
                        )

function dudt(components, u, ps, t)
    p = components.params(ps.params)
    @unpack b = p
    return 0.1 .* u .* ( 1. .- b .* u) 
end

ode_model = ODEModel((;params = params), 
                    dudt,
                    alg = BS3(),
                    abstol = 1e-3,
                    reltol = 1e-3,
                    u0 = u0,
                    tspan = tspan,
                    saveat = tsteps,
                    # sensealg = ForwardDiffSensitivity() # works
                    sensealg = InterpolatingAdjoint(; autojacvec=ZygoteVJP()))

ps, st = Lux.setup(Random.default_rng(), ode_model) 

all(params(ps.params, (;))[1].b .≈ p_true.b) # should be true

function feature_wrapper((batched_segments, tsteps_batch))
    return [
        (;u0 = batched_segments[:, 1, i],
        saveat = tsteps_batch[:, i], 
        tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
        )
        for i in 1:batchsize
    ]
end

mychain = Chain(wrapper = Lux.WrappedFunction(feature_wrapper), model = ode_model)
ps, st = Lux.setup(Random.default_rng(), mychain)
all(params(ps.model.params, (;))[1].b .≈ p_true.b) # should be true

colors = [:blue, :red]

function plot_segments(dataloader, mychain, ps, st)
    plt = plot()
    tok = 1
    for (batched_segments, batched_tsteps) in dataloader

        batched_pred = mychain((batched_segments, batched_tsteps), ps, st)[1]

        for (segment_tsteps, segment_data, pred) in zip(eachslice(batched_tsteps, dims=ndims(batched_tsteps)), 
                                                                    eachslice(batched_segments, dims=ndims(batched_segments)), 
                                                                    eachslice(batched_pred, dims=ndims(batched_pred)))
            color = colors[mod1(tok, 2)]
            plot!(plt, segment_tsteps, segment_data', label=(tok == 1 ? "Data" : ""), color=color, linestyle=:solid)
            plot!(plt, segment_tsteps, pred', label=(tok == 1 ? "Predicted" : ""), color=color, linestyle=:dash)
            tok += 1
        end
    end
    return plt
end

display(plot_segments(dataloader, mychain, ps, st))

ps = ComponentArray(ps)
train_state = Training.TrainState(mychain, ps, st, Adam(1e-2))

n_epochs = 200
for epoch in 1:n_epochs
    tot_loss = 0.
    for (batched_segments, batched_tsteps) in dataloader
        _, loss, _, train_state = Training.single_train_step!(
            Lux.AutoZygote(),
            loss_fn, 
            ((batched_segments, batched_tsteps), batched_segments),
            train_state)
        tot_loss += loss
    end
    if epoch % 10 == 0
        println("Epoch $epoch: Total Loss = ", tot_loss)
        display(plot_segments(dataloader, mychain, train_state.parameters, st))
    end
end

params(train_state.parameters.model.params, (;)) # this will be a ComponentArray

plot_segments(dataloader, mychain, train_state.parameters, st)
