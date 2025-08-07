using OrdinaryDiffEq
import Lux
import Lux: AbstractLuxLayer, StatefulLuxLayer
using Lux: MSELoss, Chain, Training
using SciMLSensitivity
using UnPack
using Random
using Printf
using ComponentArrays
using DifferentiationInterface
using ConcreteStructs

@concrete struct ParameterLayer <: AbstractLuxLayer 
    init_value <: Function
end

function (pl::ParameterLayer)(x, ps, st)
    ps_tr = NamedTuple(ps) # we transform it to a named tuple, as this may become a feature
    return (ps_tr, st)
end
Lux.initialparameters(::AbstractRNG, layer::ParameterLayer) = layer.init_value()

struct StatefulNeuralODE{M<:Lux.AbstractLuxLayer,So,T,K} <: Lux.AbstractLuxWrapperLayer{:params}
    params::M
    solver::So
    tspan::T
    kwargs::K
end

function StatefulNeuralODE(
    model::Lux.AbstractLuxLayer; solver=Tsit5(), tspan=(0.0, 1.0), kwargs...
)
    return StatefulNeuralODE(model, solver, tspan, kwargs)
end

function (n::StatefulNeuralODE)(_, ps, st)
    st_params = StatefulLuxLayer{true}(n.params, ps, st)
    function dudt(u, ps, t)
        p = st_params((), ps)
        @unpack b = p
        return 0.1 .* u .* ( 1. .- b .* u) 
    end
    prob = ODEProblem{false}(ODEFunction{false}(dudt), ones(2), n.tspan, ps)
    return solve(prob, n.solver; n.kwargs...) |> Array, st_params.st
end

params = ParameterLayer(() -> (;b = [1., 2.]))
ps, st = Lux.setup(Random.default_rng(), params)
p = params((), ps, st)

function get_model(sensealg)
    ode_model = StatefulNeuralODE(params, 
                        abstol = 1e-6,
                        reltol = 1e-5,
                        saveat = 0:0.1:1.0,
                        sensealg = sensealg
                        # sensealg = InterpolatingAdjoint(; autojacvec=ZygoteVJP()) # fails
                        )

    ps, st = Lux.setup(Random.default_rng(), ode_model)
    return ode_model, ps, st
end

function loss_fn(model, ps, st)
    return sum(model((), ps, st)[1])
end


sensealg = ForwardDiffSensitivity()
ode_model, ps, st = get_model(sensealg)
loss_fn(ode_model, ps, st)
ps = ComponentArray(ps)
println("ForwardDiffSensitivity + AutoZygote gradient: ", value_and_gradient(ps -> loss_fn(ode_model, ps, st), AutoZygote(), ps)[2].b)
println("ForwardDiffSensitivity + AutoForwardDiff gradient: ", value_and_gradient(ps -> loss_fn(ode_model, ps, st), AutoForwardDiff(), ps)[2].b)

sensealg = GaussAdjoint()
ode_model, ps, st = get_model(sensealg)
println("GaussAdjoint + AutoZygote gradient: ", value_and_gradient(ps -> loss_fn(ode_model, ps, st), AutoZygote(), ps)[2].b)
ps = ComponentArray(ps)
println("GaussAdjoint + AutoForwardDiff gradient: ", value_and_gradient(ps -> loss_fn(ode_model, ps, st), AutoForwardDiff(), ps)[2].b)
