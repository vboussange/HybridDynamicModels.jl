#=
Fitting parameters only with the simple logistic model.
=#

using OrdinaryDiffEq
import Lux
using Lux: StatefulLuxLayer, Chain, fmap
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

function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

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

@model function bayes_fit(xs, ys, ps)

    # Order for defining parameters matters !
    # when converting back to named tuples
    # should be declared in the same order as ps
    u0s = Vector{Vector}(undef, length(xs))
    for i in eachindex(xs)
        _u0 = ys[i][:, 1]
        u0s[i] ~ arraydist(LogNormal.(log.(_u0), σ))
    end

    b ~ Product(fill(Uniform(0.1, 1e0),2))

    parameters = vcat(u0s..., b)
    ps_tur = vector_to_parameters(parameters, ps)

    for i in eachindex(xs)
        preds = st_model(xs[i], ps_tur)
        ys[i] ~ arraydist(LogNormal.(log.(preds), σ))
    end
    return nothing
end

data_with_noise, tsteps = get_data()
plot(tsteps, data_with_noise')


dataloader = SegmentedTimeSeries((data_with_noise, tsteps), 
                                segmentlength = 20, 
                                shift = 5,
                                batchsize=1)
dataloader = tokenize(dataloader)

xs = []
ys = []
ic_list = ParameterLayer[]

for tok in tokens(dataloader)
    segment_data, segment_tsteps = dataloader[tok]
    u0 = segment_data[:, 1]
    push!(xs, (;u0 = tok, saveat = segment_tsteps, tspan = (segment_tsteps[1], segment_tsteps[end])))
    push!(ys, segment_data)
    push!(ic_list, ParameterLayer(constraint = NoConstraint(), init_value = (;u0)))
end
ics = ICLayer(ic_list)

function dudt(layers, u, ps, t)
    p = layers.params(ps.params)
    @unpack b = p
    return 0.1 .* u .* ( 1. .- b .* u) 
end
myparams = ParameterLayer(init_value = (;b = [1., 2.]))

ode_model = ODEModel((;params = myparams), 
                    dudt,
                    alg = BS3(),
                    abstol = 1e-3,
                    reltol = 1e-3,
                    u0 = data_with_noise[:, 1],
                    tspan = (tsteps[1], tsteps[end]),
                    saveat = tsteps,
                    sensealg = ForwardDiffSensitivity())

ode_model_with_ics = Chain(initial_conditions = ics, model = ode_model)

ps_init, st = Lux.setup(Random.default_rng(), ode_model_with_ics) 
st_model = StatefulLuxLayer{true}(ode_model_with_ics, ps_init, st)

N = 1000
@time ch = sample(bayes_fit(xs, ys, ps_init), HMC(0.05, 4; adtype=AutoForwardDiff()), N)
# 18 secs, poor results
describe(ch)
Plots.plot(ch[3000:end])


posterior_samples = sample(ch, 300; replace=false) |> Array
ps = deepcopy(ps_init)
plt = plot()
for (j, ps_vec) in enumerate(eachrow(posterior_samples))
    ps_tur = vector_to_parameters(ps_vec, ps_init)
    for i in eachindex(xs)
        preds = st_model(xs[i], ps_tur)
        segment_tsteps = xs[i].saveat
        segment_data = ys[i]
        j == 1 && scatter!(plt, segment_tsteps, segment_data', label="", color= i%2, linestyle=:solid)
        plot!(plt, segment_tsteps, preds', label="", alpha=0.1, color="#BBBBBB")
    end
end
display(plt)