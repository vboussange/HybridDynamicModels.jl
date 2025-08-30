
using ComponentArrays
using ConcreteStructs: @concrete
import HybridModelling: SegmentedTimeSeries
import Turing: @model, arraydist, q_meanfield_gaussian, vi
import Lux

"""
    VIBackend <: AbstractOptimBackend

Training backend for Variational Inference (VI) of Bayesian models using Turing.jl.

## Constructor
```julia
VIBackend()
```

## Notes
This backend is currently under development and may have incomplete functionality.
Use for variational approximation of posterior distributions when MCMC is too slow.
"""
@concrete struct VIBackend <: AbstractOptimBackend end

function train(::VIBackend,
                model::AbstractLuxLayer,
                dataloader::SegmentedTimeSeries,
                experimental_setup::InferICs;
                rng=Random.default_rng(),
                q_init = q_meanfield_gaussian,
                n_iterations, 
                kwargs...)

    dataloader = tokenize(dataloader)

    xs = []
    ys = []
    ic_list = ParameterLayer[]
    u0_priors = []

    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        t0 = segment_tsteps[1]
        push!(xs, (;u0 = tok, saveat = segment_tsteps, tspan = (segment_tsteps[1], segment_tsteps[end])))
        push!(ys, segment_data)
        if isa(experimental_setup, InferICs{true})
            push!(ic_list, ParameterLayer(init_state_value = (;t0)))
            push!(u0_priors, (;u0 = arraydist(datadistrib.(u0))))
        elseif isa(experimental_setup, InferICs{false})
            push!(ic_list, ParameterLayer(init_state_value = (;t0, u0)))
            push!(u0_priors, (;))
        end
    end
    ics = InitialConditions(ic_list)
    u0_priors = NamedTuple{ntuple(i -> Symbol(:u0_, i), length(ic_list))}(u0_priors)

    ode_model_with_ics = Chain(initial_conditions = ics, model = model)
    priors = (initial_conditions = u0_priors, model = model_priors)

    ps_init, st = Lux.setup(rng, ode_model_with_ics)
    st_model = StatefulLuxLayer{true}(ode_model_with_ics, ps_init, st)

    turing_fit = create_turing_model(priors, datadistrib, st_model)
    turing_model = turing_fit(xs, ys)
    q_avg, q_last, info, state = vi(rng, turing_model, q_init(rng, turing_model), n_iterations; kwargs...)
    # best_ps = get_best_parameters(chains, ps_init)
    # best_model = StatefulLuxLayer{true}(model, best_ps, st)
    return (;q_avg, q_last, info, state)
end