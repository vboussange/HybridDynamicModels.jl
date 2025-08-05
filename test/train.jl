using OrdinaryDiffEq
using ComponentArrays
using Bijectors
using Lux: MSELoss
using HybridModelling
using SciMLSensitivity

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

using Plots
plot(tsteps, data')

dataloader = SegmentedTimeSeries(1:length(tsteps), 
                                segmentsize = 20, 
                                shift = 10, 
                                batchsize = 1, 
                                shuffle = true,)


loss_fn = MSELoss()
transform = Bijectors.NamedTransform((; b = bijector(Uniform(1e-3, 5e0))))
p_init = Parameter(Constraint(transform), (;b = [1., 2.]))


ode_model = ODEModel((;parms = Parameter), 
                    dudt, alg = Tsit5(),
                    tspan = tspan, 
                    saveat = tsteps, 
                    sensealg = ForwardDiffSensitivity(),
                    )

ics = [Parameter(NoConstraint(), (;u0 = ones(2),(;tspan = tsteps)))]
initial_conditions = InitialConditions()