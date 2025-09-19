# # Bayesian Learning of Initial Conditions with HybridDynamicModels.jl
#
# This example demonstrates Bayesian inference for learning initial conditions using
# the Turing.jl backend in HybridDynamicModels.jl. We'll use MCMC to quantify
# uncertainty in the estimated initial conditions.

using HybridDynamicModels
using Lux
using OrdinaryDiffEq
using Plots, StatsPlots
using Random
using ComponentArrays
using Turing
using Distributions
using DifferentiationInterface
using Zygote

# ## Step 1: Generate Synthetic Data
#
# Create data from segments with different (unknown) initial conditions.

# True ODE parameters (known)
r_true = 0.15
K_true = 2.5
p_true = (; r = r_true, K = K_true)

# True initial conditions for each segment (what we want to learn)
true_ics = [0.3, 0.7, 1.1, 1.8]

function logistic_dudt(u, p, t)
    r, K = p.r, p.K
    return r * u * (1 - u / K)
end

function generate_data(true_ics, p_true)
    tsteps = 0.0:0.5:8.0
    segments = []

    for u0 in true_ics
        prob = ODEProblem(logistic_dudt, [u0], (tsteps[1], tsteps[end]), p_true)
        sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1e-6, reltol=1e-6)
        data_clean = Array(sol)

        # Add log-normal noise (appropriate for positive quantities)
        σ_noise = 0.1
        data_noisy = rand.(LogNormal.(log.(data_clean), σ_noise))
        push!(segments, data_noisy)
    end

    return segments, tsteps
end

segments, tsteps = generate_data(true_ics, p_true)

println("Generated $(length(segments)) data segments")
println("True initial conditions: ", true_ics)

# ## Step 2: Prepare Data for Bayesian Inference
#
# Set up the data structures for MCMC sampling.

# Concatenate segments for SegmentedTimeSeries
all_data = hcat(segments...)
all_times = repeat(tsteps, 1, length(segments))

# Create tokenized data loader
dataloader = SegmentedTimeSeries((all_data, all_times);
                                segment_length = 8,  # Full segments
                                shift = 8,            # No overlap
                                batchsize = 1)

tokenized_dataloader = tokenize(dataloader)

# Prepare inputs and outputs for Turing model
xs = []
ys = []

for token in tokens(tokenized_dataloader)
    segment_data, segment_tsteps = tokenized_dataloader[token]
    push!(xs, (; u0 = token,
               saveat = segment_tsteps,
               tspan = (segment_tsteps[1], segment_tsteps[end])))
    push!(ys, segment_data)
end

println("Prepared $(length(xs)) segments for Bayesian inference")

# ## Step 3: Define the Bayesian Model
#
# Create a Turing model that learns initial conditions probabilistically.

# Observation noise
const σ_obs = 0.1

@model function bayesian_ic_fit(xs, ys, ps_template)
    # Priors for initial conditions (one per segment)
    u0_priors = []
    for i in 1:length(xs)
        # Prior centered around observed first point but with uncertainty
        observed_start = ys[i][1, 1]
        push!(u0_priors, LogNormal(log(observed_start), 0.5))
    end

    # Sample initial conditions
    u0_samples = [u0_priors[i]() for i in 1:length(xs)]

    # Convert to parameter format expected by the model
    # Create a vector of all parameters: [u0s..., other_params...]
    all_params = u0_samples  # Only learning ICs, parameters are fixed

    # Convert vector back to NamedTuple structure
    ps_current = vector_to_namedtuple(all_params, ps_template)

    # Likelihood: data given model predictions
    for i in 1:length(xs)
        preds = st_model(xs[i], ps_current)
        # Log-normal likelihood for positive observations
        ys[i] ~ arraydist(LogNormal.(log.(preds), σ_obs))
    end
end

# Helper function to convert parameter vector back to NamedTuple
function vector_to_namedtuple(params_vec, template)
    # For this example, we're only learning ICs
    # In more complex cases, you'd need to handle the full parameter structure
    u0s = params_vec[1:length(xs)]

    # Create IC layers with sampled values
    ic_layers = [ParameterLayer(init_value = (; u0 = [u0s[i]])) for i in 1:length(xs)]
    ic_layer = ICLayer(ic_layers)

    # Return a complete parameter structure
    # This would need to be adapted based on your specific model structure
    return (; initial_conditions = ic_layer, model = (; params = template.model.params))
end

# ## Step 4: Set Up the Hybrid Model
#
# Create the ODE model with fixed parameters.

# Fixed parameters
param_layer = ParameterLayer(init_value = (; r = r_true, K = K_true))

function dudt_model(layers, u, ps, t)
    params = layers.params(ps.params)
    r, K = params.r, params.K
    return r * u * (1 - u / K)
end

ode_model = ODEModel(
    (; params = param_layer),
    dudt_model;
    alg = Tsit5(),
    abstol = 1e-6,
    reltol = 1e-6
)

# Create IC layers (will be updated during sampling)
ic_layers = [ParameterLayer(init_value = (; u0 = [0.5])) for _ in 1:length(xs)]
ic_layer = ICLayer(ic_layers)

model = Chain(
    initial_conditions = ic_layer,
    model = ode_model
)

# Initialize and create stateful model
ps_init, st = Lux.setup(Random.default_rng(), model)
st_model = Lux.StatefulLuxLayer{true}(model, ps_init, st)

# ## Step 5: Run MCMC Sampling
#
# Use Turing to perform Bayesian inference.

println("Running MCMC sampling...")

# MCMC settings
n_samples = 1000
n_chains = 1

# Sample from posterior
@time chains = sample(
    bayesian_ic_fit(xs, ys, ps_init),
    HMC(0.05, 4; adtype = AutoForwardDiff()),
    n_samples
)

println("MCMC sampling completed")
describe(chains)

# ## Step 6: Analyze Posterior Samples
#
# Extract and analyze the posterior distributions of initial conditions.

# Extract posterior samples for initial conditions
posterior_u0 = Array(chains)[:, 1:length(xs)]

println("\nPosterior Analysis of Initial Conditions:")
println("Segment | True IC | Mean | Std | 95% CI")
println("-" ^ 50)

for i in 1:length(true_ics)
    true_val = true_ics[i]
    samples = posterior_u0[:, i]
    mean_val = mean(samples)
    std_val = std(samples)
    ci_lower = quantile(samples, 0.025)
    ci_upper = quantile(samples, 0.975)

    println(@sprintf("%7d | %7.3f | %4.3f | %4.3f | [%.3f, %.3f]",
                    i, true_val, mean_val, std_val, ci_lower, ci_upper))
end

# ## Step 7: Visualize Results
#
# Plot posterior distributions and model predictions.

# Plot posterior distributions
plt_posterior = plot()
for i in 1:length(true_ics)
    density!(plt_posterior, posterior_u0[:, i],
             label = "Segment $i",
             fill = true, alpha = 0.5)
    vline!(plt_posterior, [true_ics[i]],
           label = "True IC $i",
           linestyle = :dash,
           linewidth = 2)
end

xlabel!(plt_posterior, "Initial Condition")
ylabel!(plt_posterior, "Posterior Density")
title!(plt_posterior, "Posterior Distributions of Initial Conditions")

display(plt_posterior)

# ## Step 8: Plot Model Predictions with Uncertainty
#
# Show model predictions using posterior samples.

function plot_predictions_with_uncertainty(xs, ys, posterior_u0, n_samples_plot=50)
    plt = plot()

    # Sample from posterior
    sample_indices = rand(1:size(posterior_u0, 1), n_samples_plot)

    colors = [:blue, :red, :green, :orange]

    for seg_idx in 1:length(xs)
        color = colors[mod1(seg_idx, length(colors))]

        # Plot data
        segment_tsteps = xs[seg_idx].saveat
        segment_data = ys[seg_idx]
        scatter!(plt, segment_tsteps, segment_data[:, 1],
                label = seg_idx == 1 ? "Data" : "",
                color = color, alpha = 0.7)

        # Plot posterior predictions
        for sample_idx in sample_indices
            u0_sample = posterior_u0[sample_idx, seg_idx]

            # Create parameter set with this IC
            ic_layers_sample = [ParameterLayer(init_value = (; u0 = [posterior_u0[sample_idx, j]]))
                               for j in 1:length(xs)]
            ic_layer_sample = ICLayer(ic_layers_sample)
            ps_sample = (; initial_conditions = ic_layer_sample, model = ps_init.model)

            # Make prediction
            preds = st_model(xs[seg_idx], ps_sample)

            plot!(plt, segment_tsteps, preds[:, 1],
                 label = "", color = color, alpha = 0.1, linewidth = 1)
        end

        # Plot median prediction
        median_u0 = median(posterior_u0[:, seg_idx])
        ic_layers_median = [ParameterLayer(init_value = (; u0 = [median_u0]))
                           for j in 1:length(xs)]
        ic_layer_median = ICLayer(ic_layers_median)
        ps_median = (; initial_conditions = ic_layer_median, model = ps_init.model)
        preds_median = st_model(xs[seg_idx], ps_median)

        plot!(plt, segment_tsteps, preds_median[:, 1],
             label = seg_idx == 1 ? "Median Prediction" : "",
             color = color, linewidth = 3, linestyle = :dash)
    end

    xlabel!(plt, "Time")
    ylabel!(plt, "Population")
    title!(plt, "Bayesian Initial Condition Inference")

    return plt
end

plt_predictions = plot_predictions_with_uncertainty(xs, ys, posterior_u0)
display(plt_predictions)

# ## Step 9: Convergence Diagnostics
#
# Check MCMC convergence using diagnostic plots.

# Trace plots
plt_trace = plot()
for i in 1:length(true_ics)
    plot!(plt_trace, posterior_u0[:, i],
          label = "IC $i",
          linewidth = 1)
    hline!(plt_trace, [true_ics[i]],
           label = "True IC $i",
           linestyle = :dash,
           linewidth = 2)
end

xlabel!(plt_trace, "MCMC Iteration")
ylabel!(plt_trace, "Initial Condition")
title!(plt_trace, "MCMC Trace Plots")

display(plt_trace)