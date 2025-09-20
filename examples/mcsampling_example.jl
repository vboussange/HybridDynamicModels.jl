#=
Example showcasing Bayesian inference with the MCSamplingBackend.
Demonstrates uncertainty quantification and posterior analysis.
=#
using Distributions, Turing, ComponentArrays, Lux
using HybridDynamicModels
using Random
using Plots, StatsPlots
using DataFrames, DelimitedFiles, HTTP

const luxtype = Lux.f64

# Loading data
url = "http://people.whitman.edu/~hundledr/courses/M250F03/LynxHare.txt"
data = readdlm(IOBuffer(HTTP.get(url).body), ' ') |> luxtype
df_data = DataFrame(Year = data[:, 1], Hare = data[:, 2], Lynx = data[:, 3])
plt_data = plot(df_data.Year, df_data.Hare, label = "Hare", xlabel = "Year",
    ylabel = "Population", title = "Lynx-Hare Data")
plot!(plt_data, df_data.Year, df_data.Lynx, label = "Lynx")
display(plt_data)

# Setup data loader
data = Array(df_data[:, Not(:Year)])' |> luxtype
# Normalize
data ./= maximum(data)
tsteps = Vector(df_data.Year) |> luxtype


# Define model components (same as README)
hlsize = 2^3
interaction_layer = Chain(Dense(2, hlsize, relu), 
                    Dense(hlsize, hlsize, relu),
                    Dense(hlsize, 2))
rate_params = ParameterLayer(init_value = (growth = [0.7], decay = [0.4]))

# Simple hybrid dynamics
function ar_step(layers, u, ps, t)
    params = layers.rates(ps.rates)
    linear = [params.growth[1] * u[1], -params.decay[1] * u[2]]
    neural = layers.interaction(u, ps.interaction)
    return linear + neural
end


# Define priors for Bayesian inference
rate_priors = (
    growth = Normal(0.7, 0.2),    # Prior around initial value
    decay = Normal(0.4, 0.1)     # Prior around initial value
)

interaction_priors = (
    interaction = Normal(0.0, 1.0)  # Prior for interaction layer weights
)

# Create Bayesian model with priors
bayesian_model = ARModel(
    (interaction = BayesianLayer(interaction_layer, interaction_priors), rates = BayesianLayer(rate_params, rate_priors)),
    ar_step;
    dt = tsteps[2] - tsteps[1],
)

# Held-out data for test
forecast_length = 20
test_idx = size(data, 2)- forecast_length:size(data, 2)

dataloader = SegmentedTimeSeries(
    (data[:, Not(test_idx)], tsteps[Not(test_idx)]); segment_length = 4, shift = 2, batchsize = 20)

# Setup MCMC backend
sampler = NUTS(0.65)  # No-U-Turn Sampler
data_distrib = LogNormal  # Likelihood distribution
backend = MCSamplingBackend(sampler, 500, data_distrib)

# Train with Bayesian inference
@info "Starting Bayesian training..."
result = train(backend, bayesian_model, dataloader, InferICs(false))

# Analyze MCMC chains
chains = result.chains
@info "MCMC Summary:"
display(chains)

# Plot parameter traces
trace_plot = plot(chains, title="Parameter Traces")
display(trace_plot)

# Plot posterior distributions
posterior_plot = plot(chains, seriestype=:histogram, title="Posterior Distributions")
display(posterior_plot)

# Sample from posterior for uncertainty quantification
n_samples = 50
posterior_samples = sample(result.st_model, chains, n_samples)

# Generate predictions with uncertainty
function predict_with_uncertainty(model, posterior_samples, dataloader, result, n_samples)
    predictions = []
    dataloader_tokenized = tokenize(dataloader)
    
    for tok in tokens(dataloader_tokenized)
        segment_data, segment_tsteps = dataloader_tokenized[tok]
        _ics = result.ics[tok].u0
        
        tok_predictions = []
        for i in 1:n_samples
            pred, _ = model(
                (; u0 = _ics, saveat = segment_tsteps,
                    tspan = (segment_tsteps[1], segment_tsteps[end])),
                posterior_samples[i],
                result.st)
            push!(tok_predictions, pred)
        end
        push!(predictions, (segment_tsteps, segment_data, tok_predictions))
    end
    return predictions
end

predictions = predict_with_uncertainty(result.st_model, posterior_samples, dataloader, result, n_samples)

# Plot predictions with uncertainty bands
# Plot predictions with uncertainty bands
function plot_uncertainty(predictions, dataloader)
    plt = plot(title = "Bayesian Predictions with Uncertainty", xlabel = "Year",
        ylabel = "Population", legend = :topright)

    # Colors: blue for hare, red for lynx
    hare_color = :blue
    lynx_color = :red

    dataloader_tokenized = tokenize(dataloader)

    for (i, (segment_tsteps, segment_data, tok_predictions)) in enumerate(predictions)
        # Plot data as scatter points
        scatter!(plt, segment_tsteps, segment_data[1, :],
            label = (i == 1 ? "Hare Data" : ""),
            color = hare_color, markersize = 4, alpha = 0.7)
        scatter!(plt, segment_tsteps, segment_data[2, :],
            label = (i == 1 ? "Lynx Data" : ""),
            color = lynx_color, markersize = 4, alpha = 0.7)

        # Calculate mean and std across posterior samples
        pred_array = cat(tok_predictions..., dims=3)
        pred_mean_hare = mean(pred_array[1, :, :], dims=2)[:]
        pred_std_hare = std(pred_array[1, :, :], dims=2)[:]
        pred_mean_lynx = mean(pred_array[2, :, :], dims=2)[:]
        pred_std_lynx = std(pred_array[2, :, :], dims=2)[:]

        # Plot mean predictions
        plot!(plt, segment_tsteps, pred_mean_hare,
            label = (i == 1 ? "Hare Predicted (mean)" : ""),
            color = hare_color, linewidth = 2, linestyle = :solid)
        plot!(plt, segment_tsteps, pred_mean_lynx,
            label = (i == 1 ? "Lynx Predicted (mean)" : ""),
            color = lynx_color, linewidth = 2, linestyle = :solid)

        # Plot uncertainty bands (mean ± 2σ)
        plot!(plt, segment_tsteps, pred_mean_hare .+ 2*pred_std_hare,
              fillrange = pred_mean_hare .- 2*pred_std_hare,
              alpha = 0.3, color = hare_color, label = (i == 1 ? "Hare uncertainty (95%)" : ""))
        plot!(plt, segment_tsteps, pred_mean_lynx .+ 2*pred_std_lynx,
              fillrange = pred_mean_lynx .- 2*pred_std_lynx,
              alpha = 0.3, color = lynx_color, label = (i == 1 ? "Lynx uncertainty (95%)" : ""))
    end
    return plt
end

uncertainty_plot = plot_uncertainty(predictions, dataloader)
display(uncertainty_plot)

# Analyze parameter posteriors
# Note: Parameter names may vary depending on the exact model structure
# Let's examine what parameters are available in the chains
@info "Available parameters in chains:"
for param_name in names(chains)
    @info "Parameter: $param_name"
end

# Try to extract growth and decay parameters (names may vary)
growth_param_name = ""
decay_param_name = ""

for param_name in names(chains)
    if occursin("growth", string(param_name))
        growth_param_name = string(param_name)
    elseif occursin("decay", string(param_name))
        decay_param_name = string(param_name)
    end
end

if !isempty(growth_param_name) && !isempty(decay_param_name)
    growth_posterior = vec(chains[growth_param_name])
    decay_posterior = vec(chains[decay_param_name])

    @info "Posterior Statistics:"
    @info "Growth rate - Mean: $(mean(growth_posterior)), Std: $(std(growth_posterior))"
    @info "Decay rate - Mean: $(mean(decay_posterior)), Std: $(std(decay_posterior))"
    @info "Initial values - Growth: 0.7, Decay: 0.4"

    # Parameter recovery plot
    param_plot = plot(title="Parameter Posterior Distributions", xlabel="Value", ylabel="Density")
    histogram!(growth_posterior, label="Growth rate", color=:blue, alpha=0.7, normalize=:pdf)
    histogram!(decay_posterior, label="Decay rate", color=:red, alpha=0.7, normalize=:pdf)
    vline!([0.7], label="Initial growth", color=:blue, linestyle=:dash, linewidth=2)
    vline!([0.4], label="Initial decay", color=:red, linestyle=:dash, linewidth=2)
    display(param_plot)
else
    @info "Could not find growth/decay parameters in chains. Available parameters:"
    for param_name in names(chains)
        @info "  $param_name"
    end
end

# Forecast with uncertainty on test data
function forecast_with_uncertainty(model, posterior_samples, tsteps_test, data_test, u0, n_samples)
    forecasts = []
    
    for i in 1:n_samples
        pred, _ = model((; u0 = u0, tspan = (tsteps_test[1], tsteps_test[end]), saveat = tsteps_test), 
                       posterior_samples[i], result.st)
        push!(forecasts, pred)
    end
    
    return forecasts
end

tsteps_test = tsteps[test_idx]
data_test = data[:, test_idx]
u0, t0 = result.ics[end]

forecast_predictions = forecast_with_uncertainty(result.st_model, posterior_samples, tsteps_test, data_test, u0, n_samples)

# Plot forecast with uncertainty
forecast_plot = plot(title="Bayesian Forecast with Uncertainty", xlabel="Year", ylabel="Population", legend=:topright)

# Training data (last part)
train_end_idx = size(data, 2) - forecast_length
scatter!(forecast_plot, tsteps[1:train_end_idx], data[1, 1:train_end_idx], 
         label="Hare Training", color=:blue, markersize=4, alpha=0.7)
scatter!(forecast_plot, tsteps[1:train_end_idx], data[2, 1:train_end_idx], 
         label="Lynx Training", color=:red, markersize=4, alpha=0.7)

# Test data
scatter!(forecast_plot, tsteps_test, data_test[1, :], 
         label="Hare Test", color=:blue, markershape=:diamond, markersize=6, alpha=0.7)
scatter!(forecast_plot, tsteps_test, data_test[2, :], 
         label="Lynx Test", color=:red, markershape=:diamond, markersize=6, alpha=0.7)

# Forecast uncertainty
forecast_array = cat(forecast_predictions..., dims=3)
forecast_mean_hare = mean(forecast_array[1, :, :], dims=2)[:]
forecast_std_hare = std(forecast_array[1, :, :], dims=2)[:]
forecast_mean_lynx = mean(forecast_array[2, :, :], dims=2)[:]
forecast_std_lynx = std(forecast_array[2, :, :], dims=2)[:]

plot!(forecast_plot, tsteps_test, forecast_mean_hare, 
      label="Hare Forecast (mean)", color=:blue, linewidth=2, linestyle=:dash)
plot!(forecast_plot, tsteps_test, forecast_mean_lynx, 
      label="Lynx Forecast (mean)", color=:red, linewidth=2, linestyle=:dash)

plot!(forecast_plot, tsteps_test, forecast_mean_hare .+ 2*forecast_std_hare,
      fillrange=forecast_mean_hare .- 2*forecast_std_hare, 
      alpha=0.3, color=:blue, label="Hare uncertainty (95%)")
plot!(forecast_plot, tsteps_test, forecast_mean_lynx .+ 2*forecast_std_lynx,
      fillrange=forecast_mean_lynx .- 2*forecast_std_lynx, 
      alpha=0.3, color=:red, label="Lynx uncertainty (95%)")

vline!([tsteps[train_end_idx]], label="Training end", color=:black, linestyle=:dot, linewidth=2)
display(forecast_plot)