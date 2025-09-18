using HybridDynamicModelsExperiments
using HybridDynamicModels
using Lux
using SciMLSensitivity
import OrdinaryDiffEqTsit5: Tsit5
using Random
using BenchmarkTools
using DifferentiationInterface
using ComponentArrays

p_true = (H = [1.24, 2.5],
    q = [4.98, 0.8],
    r = [1.0, -0.4, -0.08],
    A = [1.0])
u0_true = [0.77, 0.060, 0.945]
tsteps = range(500e0, step = 4, length = 111)
tspan = (0e0, tsteps[end])

alg = Tsit5()
adtype = AutoZygote()
abstol = 1e-4
reltol = 1e-4
tsteps
verbose = false
maxiters = 50_000
sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP(true))
rng = Random.MersenneTwister(1234)


parameters = ParameterLayer(init_value = p_true)
lux_true_model = ODEModel(
    (; parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

ps, st = Lux.setup(rng, lux_true_model)
ps = ComponentArray(ps)
synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)

dataloader = tokenize(SegmentedTimeSeries((synthetic_data, tsteps); 
                                segmentlength = 4, 
                                partial_batch = true,
                                batchsize = typemax(Int64)))

xs = []
for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        t0 = segment_tsteps[1]
        push!(xs,
            (; u0, saveat = segment_tsteps,
                tspan = (segment_tsteps[1], segment_tsteps[end])))
end
xs = vcat(xs...)


stats_noseg = @benchmark lux_true_model((; u0 = synthetic_data[:, 1], saveat = tsteps, tspan = (tsteps[1], tsteps[end])), ps, st)[1]
stats_seg = @benchmark lux_true_model(xs, ps, st)[1]


f_noseg(ps) = sum(lux_true_model((; u0 = synthetic_data[:, 1], saveat = tsteps, tspan = (tsteps[1], tsteps[end])), ps, st)[1])
f_seg(ps) = sum(lux_true_model(xs, ps, st)[1])

diff_stats_noseg = @benchmark value_and_gradient(f_noseg, AutoZygote(), ps) # returns (5.0, [2.0, 4.0]) with Zygote.jl
diff_stats_seg = @benchmark value_and_gradient(f_seg, AutoZygote(), ps) # returns (5.0, [2.0, 4.0]) with Zygote.jl

println("forward (no segments)"); display(stats_noseg)
println("forward (segments)"); display(stats_seg)
println("value_and_gradient (no segments)"); display(diff_stats_noseg)
println("value_and_gradient (segments)"); display(diff_stats_seg)