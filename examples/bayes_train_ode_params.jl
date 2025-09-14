#=
Fitting parameters only with the simple logistic model.
=#

using OrdinaryDiffEq
import Lux
using Lux: StatefulLuxLayer, Chain
using HybridDynamicModelling
using SciMLSensitivity
using UnPack
using Plots, StatsPlots # for plotting chain
using Random
using ComponentArrays
using Turing
using Distributions
using DifferentiationInterface
using Zygote

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

@model function bayes_fit(xs, ys)
    b ~ Product(fill(Uniform(0.1, 1e0),2))
    ps = ComponentArray(;params = (;b))
    for i in eachindex(xs)
        preds = st_model(xs[i], ps)
        ys[i] ~ arraydist(LogNormal.(log.(preds), σ))
    end
    return nothing
end

data_with_noise, tsteps = get_data()
plot(tsteps, data_with_noise')

params = ParameterLayer(init_value = (;b = [1., 2.]))

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
                    u0 = data_with_noise[:, 1],
                    tspan = (tsteps[1], tsteps[end]),
                    saveat = tsteps,
                    sensealg = ForwardDiffSensitivity()
                    )

ps_init, st = Lux.setup(Random.default_rng(), ode_model) 
ps_init = ComponentArray(ps_init)
st_model = StatefulLuxLayer{true}(ode_model, ps_init, st)
preds = st_model((;u0 = data_with_noise[:, 1]), ps_init)
Plots.plot(tsteps, preds', label = "Predicted", color = 1)
Plots.plot!(tsteps, data_with_noise', label = "Data", color = 2)

dataloader = SegmentedTimeSeries((data_with_noise, tsteps), 
                                segmentlength = 50)
dataloader = tokenize(dataloader)

xs = []
ys = []
for tok in tokens(dataloader)
    segment_data, segment_tsteps = dataloader[tok]
    push!(xs, (;u0 = segment_data[:, 1], saveat = segment_tsteps, tspan = (segment_tsteps[1], segment_tsteps[end])))
    push!(ys, segment_data)
end


N = 5000
ch = sample(bayes_fit(xs, ys), HMC(0.05, 4; adtype=AutoForwardDiff()), N)
describe(ch)
Plots.plot(ch[3000:end])


posterior_samples = sample(ch, 300; replace=false) |> Array
ps = deepcopy(ps_init)
plt = plot()
for (j, ps_vec) in enumerate(eachrow(posterior_samples))
    ps .=  ps_vec
    for i in eachindex(xs)
        preds = st_model(xs[i], ps)
        segment_tsteps = xs[i].saveat
        segment_data = ys[i]
        j == 1 && scatter!(plt, segment_tsteps, segment_data', label="", color= i%2, linestyle=:solid)
        plot!(plt, segment_tsteps, preds', label="", alpha=0.1, color="#BBBBBB")
    end
end
display(plt)
