import Turing: @model, NUTS, sample, Chains, arraydist, q_meanfield_gaussian, vi
import DynamicPPL
import DynamicPPL: @varname, VarName
using Distributions
import Lux
import Lux: fmap
import Functors: @leaf, fmap_with_path
using ComponentArrays
using ConcreteStructs: @concrete
import HybridModelling: SegmentedTimeSeries
import Turing

@concrete struct MCMCBackend <: AbstractOptimBackend
    sampler::Any
    n_iterations::Int
    datadistrib::Any
    kwargs::Any
end

nameof(::MCMCBackend) = "MCMCBackend"

function MCMCBackend(sampler,
        n_iterations,
        datadistrib,
        ; kwargs...)
    return MCMCBackend(sampler, n_iterations, datadistrib, kwargs)
end

# TODO: implement test
function Turing.sample(rng::AbstractRNG, model::Union{AbstractLuxLayer, StatefulLuxLayer},
        chain::Turing.MCMCChains.Chains, args...; kwargs...)
    priors = getpriors(model)
    posterior_samples = sample(rng, chain, args...; kwargs...)
    mat = Array(posterior_samples)              # rows = draws, cols = flattened params
    n = size(mat, 1)

    # infer element type from first sample (or from a zero-length dummy)
    elty = if n > 0
        typeof(_vector_to_parameters(mat[1, :], priors))
    else
        typeof(_vector_to_parameters(zeros(Lux.parameterlength(priors)), priors))
    end

    samples = Vector{elty}(undef, n)
    for i in 1:n
        samples[i] = _vector_to_parameters(mat[i, :], priors)
    end

    return samples
end

function Turing.sample(model::Union{AbstractLuxLayer, StatefulLuxLayer},
        chain::Turing.MCMCChains.Chains, args...; kwargs...)
    return sample(Random.default_rng(), model, chain, args...; kwargs...)
end

function _vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

# required for handling prior distributions in NamedTuples
Lux.parameterlength(dist::Distributions.Distribution) = length(dist)
Base.vec(dist::Product) = dist.v
@leaf Distributions.Distribution

function create_turing_model(ps_priors, data_distrib, st_model)
    function generated_model(model, varinfo, xs, ys)
        # Use a Ref to allow updating varinfo inside the fmap_with_path closure
        varinfo_ref = Ref(varinfo)

        # Function to handle each node in the param_prior structure
        function handle_node(path, node::Distributions.Distribution)
            # Generate variable name from path
            varname = Symbol(join(path, "_"))
            # Sample parameter and update varinfo
            value, new_varinfo = DynamicPPL.tilde_assume!!(
                model.context, node, VarName{varname}(), varinfo_ref[])
            varinfo_ref[] = new_varinfo
            return value
        end

        handle_node(path, node) = (;)

        # Apply fmap_with_path to sample all parameters and maintain structure
        # convert to ComponentArray for compatibility with all SciMLSensitivity sensealg
        ps = fmap_with_path(handle_node, ps_priors) |> ComponentArray

        # Update varinfo after sampling all parameters
        varinfo = varinfo_ref[]
        # Observe data points
        for i in eachindex(xs)
            preds = st_model(xs[i], ps)
            dists = data_distrib.(preds)
            _retval, varinfo = DynamicPPL.tilde_observe!!(
                model.context, arraydist(dists), ys[i], @varname(ys[i]), varinfo
            )
        end

        return nothing, varinfo
    end

    return (xs, ys) -> DynamicPPL.Model(generated_model, (; xs, ys))
end

function train(backend::MCMCBackend,
        model::AbstractLuxLayer,
        dataloader::SegmentedTimeSeries,
        experimental_setup::AbstractSetup,
        rng = Random.default_rng())

    dataloader = tokenize(dataloader)

    xs = []
    ys = []
    ic_list = []

    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        push!(xs,
            (; u0 = tok, saveat = segment_tsteps,
                tspan = (segment_tsteps[1], segment_tsteps[end])))
        push!(ys, segment_data)
        push!(ic_list, ParameterLayer(init_value = (; u0)))
    end
    if is_ics_estimated(experimental_setup)
        bics = []
        for ic in ic_list
            ps, st = Lux.setup(rng, ic)
            u0, _ = ic(ps, st)
            push!(bics, BayesianLayer(ic, (;u0 = arraydist(backend.datadistrib.(u0.u0)))))
        end
        ics = InitialConditions(vcat(bics...))
    else
        # Both work:
        # ics = InitialConditions(Lux.Experimental.FrozenLayer.(ic_list))
        ics = Lux.Experimental.FrozenLayer(InitialConditions(vcat(ic_list...)))
    end

    ode_model_with_ics = Chain(initial_conditions = ics, model = model)
    priors = getpriors(ode_model_with_ics)

    ps_init, st = Lux.setup(rng, ode_model_with_ics)
    st_model = StatefulLuxLayer{true}(ode_model_with_ics, ps_init, st)

    turing_fit = create_turing_model(priors, backend.datadistrib, st_model)

    chains = sample(
        rng, turing_fit(xs, ys), backend.sampler, backend.n_iterations; backend.kwargs...)
    segment_ics = []
    for i in tokens(dataloader)
        _, segment_tsteps = dataloader[i]
        t0 = segment_tsteps[1]
        push!(segment_ics, (; t0))
    end
    segment_ics = vcat(segment_ics...)


    return (; chains, st_model, ics=segment_ics)
end