# # Bayesian Parameter Estimation with HybridDynamicModels.jl
#
# This example demonstrates Bayesian inference for ODE parameters using the Turing.jl
# backend in HybridDynamicModels.jl. We'll use MCMC to estimate parameters with
# uncertainty quantification.

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
# Create data from a logistic growth model with known parameters.

# True parameters (what we want to infer)
r_true = 0.12
K_true = 3.0
p_true = (; r = r_true, K = K_true)

# Known initial condition
u0_true = [0.5]

function logistic_dudt(u, p, t)
    r, K = p.r, p.K
    return r * u * (1 - u / K)
end

function generate_data(p_true, u0_true)
    tsteps = 0.0:0.5:15.0
    tspan = (tsteps[1], tsteps[end])

    prob = ODEProblem(logistic_dudt, u0_true, tspan, p_true)
    sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1e-6, reltol=1e-6)
    data_clean = Array(sol)

    # Add log-normal noise (appropriate for positive quantities)
    σ_noise = 0.08
    data_noisy = rand.(LogNormal.(log.(data_clean), σ_noise))

    return data_noisy, tsteps
end

data, tsteps = generate_data(p_true, u0_true)

println("Generated $(length(tsteps)) time points of logistic growth data")
println("True parameters: r = $r_true, K = $K_true")

# ## Step 2: Create Data Segments
#
# Use SegmentedTimeSeries to create multiple overlapping segments for robust inference.

batchsize = 2
segment_length = 8

dataloader = SegmentedTimeSeries((data, tsteps);
                                segment_length = segment_length,
                                shift = 4,  # 50% overlap
                                batchsize = batchsize)

println("Created data loader with $(length(dataloader)) batches")

# Prepare data for Turing model
xs = []
ys = []

for (data_batch, tsteps_batch) in dataloader
    for i in 1:size(data_batch, 3)
        push!(xs, (; u0 = data_batch[:, 1, i],
                   saveat = tsteps_batch[:, i],
                   tspan = (tsteps_batch[1, i], tsteps_batch[end, i])))
        push!(ys, data_batch[:, :, i])
    end
end

println("Prepared $(length(xs)) data segments for Bayesian inference")

# ## Step 3: Define the Bayesian Model
#
# Create a Turing model for parameter inference.

const σ_obs = 0.08  # Observation noise

@model function bayesian_parameter_fit(xs, ys)
    # Priors for parameters
    r ~ Uniform(0.01, 0.5)   # Growth rate prior
    K ~ Uniform(1.0, 10.0)   # Carrying capacity prior

    # Parameter vector for the model
    ps = ComponentArray(; params = (; r = r, K = K))

    # Likelihood
    for i in eachindex(xs)
        preds = st_model(xs[i], ps)
        ys[i] ~ arraydist(LogNormal.(log.(preds), σ_obs))
    end
end

# ## Step 4: Set Up the Hybrid Model
#
# Create the ODE model that will be used in the Bayesian inference.

param_layer = ParameterLayer(init_value = (; r = 0.1, K = 2.0))

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

# Initialize and create stateful model
ps_init, st = Lux.setup(Random.default_rng(), ode_model)
st_model = Lux.StatefulLuxLayer{true}(ode_model, ps_init, st)

# ## Step 5: Run MCMC Sampling
#
# Use Turing to perform Bayesian inference on the parameters.

println("Running MCMC sampling for parameter estimation...")

n_samples = 1500
n_chains = 2

@time chains = sample(
    bayesian_parameter_fit(xs, ys),
    HMC(0.05, 4; adtype = AutoForwardDiff()),
    MCMCThreads(),
    n_samples,
    n_chains
)

println("MCMC sampling completed")
describe(chains)

# ## Step 6: Analyze Posterior Distributions
#
# Examine the inferred parameter distributions.

# Extract posterior samples
posterior_r = Array(chains[:, :r, :])
posterior_K = Array(chains[:, :K, :])

println("\nParameter Estimation Results:")
println("Parameter | True | Mean | Std | 95% CI")
println("-" ^ 45)

for (param_name, true_val, posterior) in [("r", r_true, posterior_r), ("K", K_true, posterior_K)]
    samples = vec(posterior)
    mean_val = mean(samples)
    std_val = std(samples)
    ci_lower = quantile(samples, 0.025)
    ci_upper = quantile(samples, 0.975)

    println(@sprintf("%9s | %4.3f | %4.3f | %4.3f | [%.3f, %.3f]",
                    param_name, true_val, mean_val, std_val, ci_lower, ci_upper))
end

# ## Step 7: Visualize Posterior Distributions
#
# Plot marginal and joint posterior distributions.

# Marginal posteriors
plt_marginal = plot(layout = (1, 2), size = (800, 300))

plot!(plt_marginal[1], posterior_r, seriestype = :density,
      xlabel = "Growth Rate (r)", ylabel = "Density", title = "Posterior: r",
      fill = true, alpha = 0.5, color = :blue)
vline!(plt_marginal[1], [r_true], color = :red, linewidth = 2, label = "True value")

plot!(plt_marginal[2], posterior_K, seriestype = :density,
      xlabel = "Carrying Capacity (K)", ylabel = "Density", title = "Posterior: K",
      fill = true, alpha = 0.5, color = :green)
vline!(plt_marginal[2], [K_true], color = :red, linewidth = 2, label = "True value")

display(plt_marginal)

# Joint posterior
plt_joint = scatter(posterior_r, posterior_K,
                   xlabel = "Growth Rate (r)",
                   ylabel = "Carrying Capacity (K)",
                   title = "Joint Posterior Distribution",
                   alpha = 0.6, color = :purple, legend = false)
scatter!(plt_joint, [r_true], [K_true],
         color = :red, markersize = 8, marker = :star, label = "True values")

display(plt_joint)

# ## Step 8: Predictive Checks
#
# Generate predictions using posterior samples to check model fit.

function plot_posterior_predictions(xs, ys, chains, n_pred_samples=100)
    plt = plot()

    # Plot data
    for i in 1:length(xs)
        segment_tsteps = xs[i].saveat
        segment_data = ys[i]
        scatter!(plt, segment_tsteps, segment_data[:, 1],
                label = i == 1 ? "Data" : "",
                color = :blue, alpha = 0.6)
    end

    # Sample from posterior and plot predictions
    sample_indices = rand(1:size(chains, 1), n_pred_samples)

    for sample_idx in sample_indices
        r_sample = chains[sample_idx, :r, 1]  # Using first chain
        K_sample = chains[sample_idx, :K, 1]

        ps_sample = ComponentArray(; params = (; r = r_sample, K = K_sample))

        # Predict for first segment (as example)
        preds = st_model(xs[1], ps_sample)
        plot!(plt, xs[1].saveat, preds[:, 1],
             label = "", color = :orange, alpha = 0.1, linewidth = 1)
    end

    # Plot median prediction
    r_median = median(posterior_r)
    K_median = median(posterior_K)
    ps_median = ComponentArray(; params = (; r = r_median, K = K_median))
    preds_median = st_model(xs[1], ps_median)

    plot!(plt, xs[1].saveat, preds_median[:, 1],
         label = "Median Prediction", color = :red, linewidth = 3, linestyle = :dash)

    xlabel!(plt, "Time")
    ylabel!(plt, "Population")
    title!(plt, "Posterior Predictive Distribution")

    return plt
end

plt_pred = plot_posterior_predictions(xs, ys, chains)
display(plt_pred)

# ## Step 9: Convergence Diagnostics
#
# Check MCMC convergence.

# Trace plots
plt_trace = plot(layout = (2, 1), size = (800, 400))

plot!(plt_trace[1], posterior_r, xlabel = "Iteration", ylabel = "r",
      title = "Trace: Growth Rate", legend = false, linewidth = 1)
hline!(plt_trace[1], [r_true], color = :red, linestyle = :dash)

plot!(plt_trace[2], posterior_K, xlabel = "Iteration", ylabel = "K",
      title = "Trace: Carrying Capacity", legend = false, linewidth = 1)
hline!(plt_trace[2], [K_true], color = :red, linestyle = :dash)

display(plt_trace)

# ## Step 10: Full Trajectory Prediction
#
# Use inferred parameters to predict the full trajectory.

function predict_full_trajectory(posterior_r, posterior_K, u0, tspan, tsteps, n_samples=100)
    sample_indices = rand(1:length(posterior_r), n_samples)

    plt = plot()

    # Plot true trajectory
    prob_true = ODEProblem(logistic_dudt, u0, tspan, p_true)
    sol_true = solve(prob_true, Tsit5(), saveat=tsteps)
    plot!(plt, tsteps, Array(sol_true)[:, 1],
          label = "True Trajectory", color = :red, linewidth = 3, linestyle = :solid)

    # Plot posterior predictions
    for idx in sample_indices
        r_sample = posterior_r[idx]
        K_sample = posterior_K[idx]

        p_sample = (; r = r_sample, K = K_sample)
        prob_sample = ODEProblem(logistic_dudt, u0, tspan, p_sample)
        sol_sample = solve(prob_sample, Tsit5(), saveat=tsteps)

        plot!(plt, tsteps, Array(sol_sample)[:, 1],
             label = "", color = :blue, alpha = 0.1, linewidth = 1)
    end

    # Plot median trajectory
    r_median = median(posterior_r)
    K_median = median(posterior_K)
    p_median = (; r = r_median, K = K_median)
    prob_median = ODEProblem(logistic_dudt, u0, tspan, p_median)
    sol_median = solve(prob_median, Tsit5(), saveat=tsteps)

    plot!(plt, tsteps, Array(sol_median)[:, 1],
         label = "Median Prediction", color = :green, linewidth = 3, linestyle = :dash)

    xlabel!(plt, "Time")
    ylabel!(plt, "Population")
    title!(plt, "Full Trajectory Prediction with Uncertainty")

    return plt
end

extended_tsteps = 0.0:0.2:20.0
extended_tspan = (0.0, 20.0)

plt_full = predict_full_trajectory(vec(posterior_r), vec(posterior_K),
                                  u0_true, extended_tspan, extended_tsteps)
display(plt_full)
