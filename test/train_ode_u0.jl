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

function dudt(u, p, t)
    @unpack b = p
    return 0.1 .* u .* ( 1. .- b .* u) 
end

tsteps = 1.:1:100.
tspan = (tsteps[1], tsteps[end])

p_true = (;b = [0.23, 0.5],)
u0 = ones(2)
prob = ODEProblem(ODEFunction{false}(dudt), u0, tspan, p_true)
data = solve(prob, Tsit5(), saveat=tsteps) |> Array
data_with_noise = data .* exp.(0.1 .* randn(size(data)))

plot(tsteps, data')

batchsize = 1
dataloader = SegmentedTimeSeries((data_with_noise, tsteps), 
                                segmentsize = 20, 
                                shift = 10, 
                                batchsize = batchsize)

dataloader = tokenize(dataloader)

ic_list = ParameterLayer[]
for tok in tokens(dataloader)
    segment_data, segment_tsteps = dataloader[tok]
    u0 = segment_data[:, 1]
    push!(ic_list, ParameterLayer(constraint = NoConstraint(), init_value = (;u0)))
end

loss_fn = MSELoss()
# transform = Bijectors.NamedTransform((; b = bijector(Uniform(1e-3, 5e0))))
params = ParameterLayer(constraint = NoConstraint(), 
                        init_value = (;b = [1., 2.]))

function dudt(components, u, ps, t)
    p = components.params(ps.params)
    @unpack b = p
    return 0.1 .* u .* ( 1. .- b .* u) 
end

ode_model = ODEModel((;params = params), 
                    dudt,
                    alg = Tsit5(),
                    abstol = 1e-6,
                    reltol = 1e-6,
                    sensealg = ForwardDiffSensitivity())
ics = InitialConditions(ic_list)

function feature_wrapper((token, tsteps_batch))
    return [
        (;u0 = token[i],
        saveat = tsteps_batch[:, i], 
        tspan = (tsteps_batch[1, i], tsteps_batch[end, i])
        )
        for i in 1:batchsize
    ]
end

ode_model_with_ics = Chain(wrapper = Lux.WrappedFunction(feature_wrapper), initial_conditions = ics, model = ode_model)
ps, st = Lux.setup(Random.default_rng(), ode_model_with_ics)
ps = ComponentArray(ps)

n_segments = length(tokens(dataloader))
colors = [:blue, :red]

function plot_segments(dataloader, ode_model_with_ics, ps, st)
    plt = plot()
    for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader

        batched_pred = ode_model_with_ics((batched_tokens, batched_tsteps), ps, st)[1]
        for (tok, segment_tsteps, segment_data, pred) in zip(batched_tokens, 
                                                            eachslice(batched_tsteps, dims=ndims(batched_tsteps)), 
                                                            eachslice(batched_segments, dims=ndims(batched_segments)), 
                                                            eachslice(batched_pred, dims=ndims(batched_pred)))
            color = colors[mod1(tok, 2)]
            plot!(plt, segment_tsteps, segment_data', label=(tok == 1 ? "Data" : ""), color=color, linestyle=:solid)
            plot!(plt, segment_tsteps, pred', label=(tok == 1 ? "Predicted" : ""), color=color, linestyle=:dash)
        end
    end

    display(plt)
    return plt
end

plot_segments(dataloader, ode_model_with_ics, ps, st)

# function eval_loss(model, ps, st, dataloader)
#     total_loss = 0.0
#     n_batches = 0
#     for tok in tokens(dataloader)
#         segment_data, segment_tsteps = dataloader[tok]
#         u0 = tok
#         saveat = segment_tsteps
#         tspan = (segment_tsteps[1], segment_tsteps[end])
#         predicted = ode_model_with_ics((;u0, saveat, tspan), ps, st)[1]
#         loss = loss_fn(predicted, segment_data)
#         total_loss += loss
#         n_batches += 1
#     end
#     return total_loss / n_batches
# end

# println("Initial Loss: ", eval_loss(ode_model_with_ics, ps, st, dataloader))

train_state = Training.TrainState(ode_model_with_ics, ps, st, Adam(1f-2))

n_epochs = 100
for epoch in 1:n_epochs
    loss = Inf
    for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader
        # @show tokens
        # @show size(segment_data)
        _, loss, _, train_state = Training.single_train_step!(
            Lux.AutoZygote(), 
            loss_fn, 
            ((batched_tokens, batched_tsteps), batched_segments),
            train_state)
    end
    if epoch % 10 == 0
        println("Epoch $epoch: Loss = ", loss)
    end
end

plot_segments(dataloader, ode_model_with_ics, train_state.parameters, st)
