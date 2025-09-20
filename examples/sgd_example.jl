#=
Example showcasing the training API with a simple hybrid model.
Demonstrates the use of callbacks, parameter schedulers, and forecasting.
=#
using Lux, Optimisers, ComponentArrays
using Zygote
using HybridDynamicModels
using ParameterSchedulers
using Random
using Plots
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

# Define model components (same as README)
hlsize = 2^4
interaction_layer = Chain(Dense(2, hlsize, relu), 
                    Dense(hlsize, hlsize, relu),
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

# Create autoregressive model
model = ARModel(
    (interaction = interaction_layer, rates = rate_params),
    ar_step;
    dt = tsteps[2] - tsteps[1],
)

# Setup data loader
data = Array(df_data[:, Not(:Year)])' |> luxtype
# Normalize
data ./= maximum(data)
tsteps = Vector(df_data.Year) |> luxtype

# Held-out data for test
forecast_length = 20
test_idx = size(data, 2)- forecast_length:size(data, 2)

dataloader = SegmentedTimeSeries(
    (data[:, Not(test_idx)], tsteps[Not(test_idx)]); segment_length = 4, shift = 2, batchsize = 20)

# Learning rate scheduler
lr_schedule = Step(1e-2, 0.9, 200)  # Start at 1e-2, decay by 0.9 every 100 epochs

# Callback function with scheduler
function callback(loss, epoch, ts)
    if epoch % 20 == 0
        current_lr = lr_schedule(epoch)
        @info "Epoch $epoch: Loss = $loss, LR = $current_lr"
        Optimisers.adjust!(ts.optimizer_state, current_lr)
    end
end

# Training configuration
backend = SGDBackend(
    AdamW(eta = 1e-2, lambda = 1e-4),           # Initial optimizer
    2000,                  # Number of epochs
    AutoZygote(),         # AD backend
    MSELoss(),           # Loss function
    callback             # Callback with scheduler
)

# Train the model
@info "Starting training..."
result = train(backend, model, dataloader, InferICs(true))

# Plot training results
function plot_results(dataloader, result, model)
    plt = plot(title = "Training Results", xlabel = "Year",
        ylabel = "Population", legend = :topright)

    # Colors: blue for hare, red for lynx
    hare_color = :blue
    lynx_color = :red

    dataloader_tokenized = tokenize(dataloader)

    for tok in tokens(dataloader_tokenized)
        segment_data, segment_tsteps = dataloader_tokenized[tok]
        _ics = result.ics[tok].u0
        pred, _ = model(
            (; u0 = _ics, saveat = segment_tsteps,
                tspan = (segment_tsteps[1], segment_tsteps[end])),
            result.ps,
            result.st)

        # Plot data as scatter points
        scatter!(plt, segment_tsteps, segment_data[1, :],
            label = (tok == 1 ? "Hare Data" : ""),
            color = hare_color, markersize = 4, alpha = 0.7)
        scatter!(plt, segment_tsteps, segment_data[2, :],
            label = (tok == 1 ? "Lynx Data" : ""),
            color = lynx_color, markersize = 4, alpha = 0.7)

        # Plot predictions as solid lines
        plot!(plt, segment_tsteps, pred[1, :],
            label = (tok == 1 ? "Hare Predicted" : ""),
            color = hare_color, linewidth = 2, linestyle = :solid)
        plot!(plt, segment_tsteps, pred[2, :],
            label = (tok == 1 ? "Lynx Predicted" : ""),
            color = lynx_color, linewidth = 2, linestyle = :solid)
    end
    return plt
end

plt = plot_results(dataloader, result, model)
display(plt)

tsteps_test = tsteps[test_idx]
data_test = data[:, test_idx]
u0, t0 = result.ics[end]

preds = model((; u0 = u0, tspan = (t0, tsteps_test[end]), saveat = tsteps_test), result.ps, result.st)[1]
plt_test = plot(title="Test Predictions", xlabel="Year", ylabel="Population", legend=:topright)
scatter!(plt_test, tsteps_test, data_test[1, :], label="Hare Data", color=:blue, markersize=4, alpha=0.7)
scatter!(plt_test, tsteps_test, data_test[2, :], label="Lynx Data", color=:red, markersize=4, alpha=0.7)
plot!(plt_test, tsteps_test, preds[1, :], label="Hare Predicted", color=:blue, linewidth=2, linestyle=:solid)
plot!(plt_test, tsteps_test, preds[2, :], label="Lynx Predicted", color=:red, linewidth=2, linestyle=:solid)
display(plt_test)