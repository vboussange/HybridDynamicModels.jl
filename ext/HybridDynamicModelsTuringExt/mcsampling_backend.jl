

"""
    MCSamplingBackend <: AbstractOptimBackend

Training backend for Bayesian inference of Lux.jl models using Markov Chain Monte Carlo (MCMC). Requires models with Bayesian priors (use `BayesianLayer` wrappers)

## Fields
- `sampler`: Turing.jl MCMC sampler (e.g., `NUTS()`, `HMC()`)
- `n_iterations`: Number of MCMC samples to draw
- `datadistrib`: Data distribution for likelihood computation
- `kwargs`: Additional keyword arguments passed to the sampler

## Arguments
- `sampler`: MCMC sampling algorithm from Turing.jl
- `n_iterations`: Total number of posterior samples to generate
- `datadistrib`: Distribution constructor for data likelihood (e.g., `Normal`, `LogNormal`)

## Keyword Arguments
- `kwargs...`: Additional sampler-specific options (e.g., `progress=true`, `drop_warmup=true`)

## Example
```julia
# Gaussian data likelihood
backend = MCSamplingBackend(NUTS(0.65), 1000, Normal)

# Log-normal data for positive-valued observations
backend = MCSamplingBackend(NUTS(0.65), 2000, x -> LogNormal(log(x), 0.1))

# Train with MCMC
result = train(backend, bayesian_model, dataloader, infer_ics)
chains = result.chains
```
"""
@concrete struct MCSamplingBackend <: AbstractOptimBackend
    sampler::Any
    n_iterations::Int
    datadistrib::Any
    kwargs::Any
end

"""
    MCSamplingBackend(sampler, n_iterations, datadistrib; kwargs...)

Construct an MCSamplingBackend for Bayesian training of Lux.jl models.
See [`MCSamplingBackend`](@ref) for detailed documentation.
"""
function MCSamplingBackend(sampler,
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


"""
    train(backend::MCSamplingBackend, model, dataloader, infer_ics, rng=Random.default_rng())

Perform Bayesian inference on a hybrid dynamical model using MCMC sampling.

## Arguments
- `backend::MCSamplingBackend`: MCMC configuration and sampling settings
- `model::AbstractLuxLayer`: Bayesian dynamical model with priors (use `BayesianLayer` wrappers)
- `dataloader::SegmentedTimeSeries`: Time series data split into segments
- `infer_ics::InferICs`: Configuration for initial condition inference
- `rng=Random.default_rng()`: Random number generator for sampling

## Returns
A NamedTuple with:
- `chains`: Turing.jl MCMC chains containing posterior samples
- `st_model`: StatefulLuxLayer for forward predictions with posterior samples

## Behavior
The function:
1. **Tokenizes** the dataloader and extracts segment data
2. **Creates Bayesian initial conditions** based on `infer_ics`
3. **Constructs a Turing model** from the Lux model and priors
4. **Runs MCMC sampling** to approximate the posterior distribution
5. **Returns chains** for posterior analysis and uncertainty quantification

## Initial Condition Handling
- If `InferICs{true}`: Creates Bayesian initial conditions with data-informed priors
- If `InferICs{false}`: Uses fixed initial conditions from data

## Example
```julia
# Setup Bayesian model with priors
bayesian_model = BayesianLayer(ode_model, parameter_priors)
backend = MCSamplingBackend(NUTS(0.65), 1000, LogNormal)
infer_ics = InferICs(true)

# Bayesian training
result = train(backend, bayesian_model, dataloader, infer_ics)

# Analyze posterior
chains = result.chains
posterior_samples = sample(result.st_model, chains, 100)
```

## Notes
- Requires models with proper Bayesian priors via `BayesianLayer`
- MCMC provides full posterior uncertainty quantification
- Use `sample(st_model, chains, n)` to generate posterior predictive samples
- Choose appropriate `datadistrib` based on your data characteristics
"""
function train(backend::MCSamplingBackend,
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
    if istrue(infer_ics)
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
    priors = getpriors(ode_model_with_ics)

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