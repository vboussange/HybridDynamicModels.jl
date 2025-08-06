using OrdinaryDiffEq
using ComponentArrays
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

tsteps = 1.:0.5:100.5
tspan = (tsteps[1], tsteps[end])

p_true = (;b = [0.23, 0.5],)
u0 = ones(2)
prob = ODEProblem(dudt, u0, tspan, p_true)
data = solve(prob, Tsit5(), saveat=tsteps) |> Array

plot(tsteps, data')

dataloader = SegmentedTimeSeries((data, tsteps), 
                                segmentsize = 20, 
                                shift = 10, 
                                batchsize = 1, 
                                shuffle = true,)

dataloader = tokenize(dataloader)

ic_list = ParameterLayer[]
for tok in tokens(dataloader)
    segment_data, segment_tsteps = dataloader[tok]
    u0 = segment_data[:, 1]
    tspan = (segment_tsteps[1], segment_tsteps[end])
    saveat = segment_tsteps
    push!(ic_list, ParameterLayer(NoConstraint(), (;u0), (;tspan, saveat)))
end

loss_fn = MSELoss()
transform = Bijectors.NamedTransform((; b = bijector(Uniform(1e-3, 5e0))))
params = ParameterLayer(Constraint(transform), (;b = [1., 2.]))

# TODO: we should only support stateful components, but check Diffrax
function dudt(components, u, t)
    p = components.params()
    @unpack b = p
    return 0.1 .* u .* ( 1. .- b .* u) 
end

ode_model = ODEModel((;params = params), 
                    dudt, alg = Tsit5(),
                    sensealg = InterpolatingAdjoint(; autojacvec=ZygoteVJP()),)
ics = InitialConditions(ic_list)
ode_model_with_ics = Chain(initial_conditions = ics, model = ode_model)

ps, st = Lux.setup(Random.default_rng(), ode_model_with_ics)

n_segments = length(tokens(dataloader))
colors = [:blue, :red]

plt = plot()
for tok in tokens(dataloader)
    segment_data, segment_tsteps = dataloader[tok]
    predicted = ode_model_with_ics(tok, ps, st)[1]

    color = colors[mod1(tok, 2)]
    plot!(plt, segment_tsteps, segment_data', label=(tok == 1 ? "Data" : ""), color=color, linestyle=:solid)
    plot!(plt, segment_tsteps, predicted', label=(tok == 1 ? "Predicted" : ""), color=color, linestyle=:dash)
end

display(plt)

function eval_loss(model, ps, st, dataloader)
    total_loss = 0.0
    n_batches = 0
    for (tokens, (segment_data, _)) in dataloader
        y_pred, _ = model(tokens, ps, st)
        total_loss += loss_fn(y_pred, segment_data)
        n_batches += 1
    end
    return total_loss / n_batches
end

println("Initial Loss: ", eval_loss(ode_model_with_ics, ps, st, dataloader))

train_state = Training.TrainState(ode_model_with_ics, ps, st, Adam(3.0f-4))

n_epochs = 100
for epoch in 1:n_epochs
    for (tokens, (segment_data, segment_tsteps)) in dataloader
        # @show tokens
        # @show size(segment_data)
        train_state = Training.single_train_step!(
            Lux.AutoZygote(), 
            loss_fn, 
            (tokens, segment_data),
            train_state)
    end
    if epoch % 10 == 0
        println("Epoch $epoch: Loss = ", eval_loss(ode_model_with_ics, train_state.parameters, train_state.states, dataloader))
    end
end