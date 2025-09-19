@concrete struct MCSamplingBackend <: HybridDynamicModels.MCSamplingBackend
    sampler::Any
    n_iterations::Int
    datadistrib::Any
    kwargs::Any
end

function HybridDynamicModels.MCSamplingBackend(sampler,
        n_iterations,
        datadistrib,
        ; kwargs...)
    return MCSamplingBackend(sampler, n_iterations, datadistrib, kwargs)
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
LuxCore.parameterlength(dist::Distributions.Distribution) = length(dist)
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

function HybridDynamicModels.train(backend::MCSamplingBackend,
        model::LuxCore.AbstractLuxLayer,
        dataloader::SegmentedTimeSeries,
        infer_ics::InferICs,
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
    if is_ics_estimated(infer_ics)
        bics = []
        for ic in ic_list
            ps, st = LuxCore.setup(rng, ic)
            u0, _ = ic(ps, st)
            push!(bics, BayesianLayer(ic, (;u0 = arraydist(backend.datadistrib.(u0.u0)))))
        end
        ics = ICLayer(vcat(bics...))
    else
        # Both work:
        # ics = ICLayer(LuxCore.Experimental.FrozenLayer.(ic_list))
        ics = Lux.Experimental.FrozenLayer(ICLayer(vcat(ic_list...)))
    end

    ode_model_with_ics = Chain(initial_conditions = ics, model = model)
    priors = HybridDynamicModels.getpriors(ode_model_with_ics)

    ps_init, st = LuxCore.setup(rng, ode_model_with_ics)
    st_model = LuxCore.StatefulLuxLayer{true}(ode_model_with_ics, ps_init, st)

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