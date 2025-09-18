#=
Fitting initial conditions and parameters with the simple logistic model
=#

using OrdinaryDiffEq
using Bijectors
import Lux
using Lux: MSELoss, Chain, Training
using HybridDynamicModels
using SciMLSensitivity
using UnPack
using Plots
import Optimisers: Adam
using Random
using Test
using ComponentArrays
using Distributions
import Turing: arraydist

const σ = 0.1

function get_data()
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
    data_with_noise = rand(arraydist(LogNormal.(log.(data), σ)))
    return data_with_noise, tsteps
end
data_with_noise, tsteps = get_data()
plot(tsteps, data_with_noise')

batchsize = 1 #TODO: SegmentedTimeSeries should default to full batch
dataloader = SegmentedTimeSeries((data_with_noise, tsteps), 
                                segment_length = 20, 
                                shift = 5,
                                batchsize=batchsize)

dataloader = tokenize(dataloader)

ic_list = ParameterLayer[]
for tok in tokens(dataloader)
    segment_data, segment_tsteps = dataloader[tok]
    u0 = segment_data[:, 1]
    push!(ic_list, ParameterLayer(constraint = NoConstraint(), init_value = (;u0)))
end

loss_fn = MSELoss() # TODO: change to a log loss
# transform = Bijectors.NamedTransform((; b = bijector(Uniform(1e-3, 5e0))))
params = ParameterLayer(constraint = NoConstraint(), 
                        init_value = (;b = [1., 2.]))

function dudt(layers, u, ps, t)
    p = layers.params(ps.params)
    @unpack b = p
    return 0.1 .* u .* ( 1. .- b .* u) 
end

ode_model = ODEModel((;params = params), 
                    dudt,
                    alg = BS3(),
                    abstol = 1e-3,
                    reltol = 1e-3,
                    sensealg = ForwardDiffSensitivity()
                    )
ics = ICLayer(ic_list)

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

train_state = Training.TrainState(ode_model_with_ics, ps, st, Adam(1e-3))

n_epochs = 1000
@time for epoch in 1:n_epochs
    tot_loss = 0.
    for (batched_tokens, (batched_segments, batched_tsteps)) in dataloader
        # @show tokens
        # @show size(segment_data)
        _, loss, _, train_state = Training.single_train_step!(
            Lux.AutoZygote(), 
            loss_fn, 
            ((batched_tokens, batched_tsteps), batched_segments),
            train_state)
        tot_loss += loss
    end
    if epoch % 10 == 0
        println("Epoch $epoch: Total Loss = ", tot_loss)
        display(plot_segments(dataloader, ode_model_with_ics, train_state.parameters, st))
    end
end
# approx 18 secs, converges after 10 its

@test isapprox(params(train_state.parameters.model.params, (;))[1].b, p_true.b, rtol = 1e-2) # should be true

plot_segments(dataloader, ode_model_with_ics, train_state.parameters, st)
