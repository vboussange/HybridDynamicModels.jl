module HybridModellingTuringExt
import Turing: NUTS, sample, Chains, arraydist, q_meanfield_gaussian
const DynamicPPL = Turing.DynamicPPL
import DynamicPPL: @varname, VarName
using Distributions
import Functors: @leaf, fmap_with_path, fmap
using ConcreteStructs: @concrete
import HybridModelling: SegmentedTimeSeries
import Turing



"""
    MCMCBackend <: AbstractOptimBackend

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
backend = MCMCBackend(NUTS(0.65), 1000, Normal)

# Log-normal data for positive-valued observations
backend = MCMCBackend(NUTS(0.65), 2000, x -> LogNormal(log(x), 0.1))

# Train with MCMC
result = train(backend, bayesian_model, dataloader, experimental_setup)
chains = result.chains
```
"""
@concrete struct MCMCBackend <: AbstractOptimBackend
    sampler::Any
    n_iterations::Int
    datadistrib::Any
    kwargs::Any
end

"""
    MCMCBackend(sampler, n_iterations, datadistrib; kwargs...)

Construct an MCMCBackend for Bayesian training of Lux.jl models.
See [`MCMCBackend`](@ref) for detailed documentation.
"""
function MCMCBackend(sampler,
        n_iterations,
        datadistrib,
        ; kwargs...)
    return MCMCBackend(sampler, n_iterations, datadistrib, kwargs)
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

function create_turing_model(ps_priors, data_distrib, st_model, pstype)
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
        ps = fmap_with_path(handle_node, ps_priors) |> pstype

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
    train(backend::MCMCBackend, model, dataloader, experimental_setup, rng=Random.default_rng())

Perform Bayesian inference on a hybrid dynamical model using MCMC sampling.

## Arguments
- `backend::MCMCBackend`: MCMC configuration and sampling settings
- `model::AbstractLuxLayer`: Bayesian dynamical model with priors (use `BayesianLayer` wrappers)
- `dataloader::SegmentedTimeSeries`: Time series data split into segments
- `experimental_setup::InferICs`: Configuration for initial condition inference
- `rng=Random.default_rng()`: Random number generator for sampling

## Returns
A NamedTuple with:
- `chains`: Turing.jl MCMC chains containing posterior samples
- `st_model`: StatefulLuxLayer for forward predictions with posterior samples

## Behavior
The function:
1. **Tokenizes** the dataloader and extracts segment data
2. **Creates Bayesian initial conditions** based on `experimental_setup`
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
backend = MCMCBackend(NUTS(0.65), 1000, LogNormal)
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
function train(backend::MCMCBackend,
        model::AbstractLuxLayer,
        dataloader::SegmentedTimeSeries,
        experimental_setup::InferICs,
        rng = Random.default_rng();
        pstype = Lux.f64)

    dataloader = tokenize(dataloader)

    xs = []
    ys = []
    ic_list = BayesianLayer[]

    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        u0 = segment_data[:, 1]
        t0 = segment_tsteps[1]
        push!(xs,
            (; u0 = tok, saveat = segment_tsteps,
                tspan = (segment_tsteps[1], segment_tsteps[end])))
        push!(ys, segment_data)
        if isa(experimental_setup, InferICs{true})
            push!(ic_list,
                BayesianLayer(
                    ParameterLayer(init_value = (; u0), init_state_value = (; t0)),
                    (; u0 = arraydist(backend.datadistrib.(u0)))))
        elseif isa(experimental_setup, InferICs{false})
            push!(
                ic_list, BayesianLayer(ParameterLayer(init_state_value = (; t0, u0)),
                    (;)))
        end
    end
    ics = InitialConditions(ic_list)

    ode_model_with_ics = Chain(initial_conditions = ics, model = model)
    priors = getpriors(ode_model_with_ics)

    ps_init, st = Lux.setup(rng, ode_model_with_ics)

    # TODO: separate ics from model; return ics as a vector of named tuples
    st_model = StatefulLuxLayer{true}(ode_model_with_ics, ps_init, st)

    turing_fit = create_turing_model(priors, backend.datadistrib, st_model, pstype)

    chains = sample(
        rng, turing_fit(xs, ys), backend.sampler, backend.n_iterations; backend.kwargs...)

    return (; chains, st_model)
end

end