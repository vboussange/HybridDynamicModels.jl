# # Learning Initial Conditions with HybridDynamicModels.jl
#
# This example demonstrates how to learn initial conditions (ICs) for time series segments
# using the `ICLayer` in HybridDynamicModels.jl. This is useful when you have multiple
# time series segments with unknown starting points.

using HybridDynamicModels
using Lux
using Optimisers
using OrdinaryDiffEq
using Plots
using Random
using ComponentArrays
using Distributions
using Test

# ## Step 1: Generate Synthetic Data with Multiple Segments
#
# We'll create data from a logistic growth model with different initial conditions
# for different segments.

# True ODE parameters
r_true = 0.1
K_true = 2.0
p_true = (; r = r_true, K = K_true)

# Logistic growth ODE
function logistic_dudt(u, p, t)
    r, K = p.r, p.K
    return r * u * (1 - u / K)
end

# Generate multiple segments with different initial conditions
function generate_segmented_data()
    tsteps = 0.0:0.5:10.0
    segments = []

    # Different initial conditions for each segment
    u0_values = [0.2, 0.5, 0.8, 1.2]

    for u0 in u0_values
        prob = ODEProblem(logistic_dudt, [u0], (tsteps[1], tsteps[end]), p_true)
        sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1e-6, reltol=1e-6)
        data_clean = Array(sol)

        # Add noise
        noise_level = 0.05
        rng = Random.MersenneTwister(42)
        data_noisy = data_clean .* (1 .+ noise_level * randn(rng, size(data_clean)))

        push!(segments, data_noisy)
    end

    return segments, tsteps
end

segments, tsteps = generate_segmented_data()

println("Generated $(length(segments)) segments with different initial conditions")
println("Time points per segment: $(length(tsteps))")

# ## Step 2: Create SegmentedTimeSeries DataLoader
#
# Combine all segments into a single time series and create a segmented data loader.

# Concatenate all segments
all_data = hcat(segments...)
all_times = repeat(tsteps, 1, length(segments))

# Create data loader
batchsize = 2
dataloader = SegmentedTimeSeries((all_data, all_times);
                                segment_length = 10,
                                shift = 5,
                                batchsize = batchsize)

# Tokenize for easier handling of segments
tokenized_dataloader = tokenize(dataloader)

println("Created tokenized data loader with $(length(tokens(tokenized_dataloader))) segments")

# ## Step 3: Set Up Initial Condition Layers
#
# Create an ICLayer with learnable initial conditions for each segment.

# Create initial condition layers for each segment
ic_layers = []
for token in tokens(tokenized_dataloader)
    segment_data, _ = tokenized_dataloader[token]
    # Initialize ICs with small random values (different from true)
    u0_init = 0.1 + 0.1 * rand(Random.MersenneTwister(token))
    push!(ic_layers, ParameterLayer(
        init_value = (; u0 = [u0_init]),
        constraint = BoxConstraint([1e-3], [3.0])  # Positive constraint
    ))
end

ic_layer = ICLayer(ic_layers)

# ## Step 4: Define the ODE Model
#
# Create an ODE model with fixed parameters (we'll learn ICs, not parameters).

# Fixed parameters (known)
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

# ## Step 5: Create Feature Wrapper
#
# Prepare inputs for the model from tokenized data.

function feature_wrapper((tokens_batch, tsteps_batch))
    return [
        (; u0 = tokens_batch[i],
          saveat = tsteps_batch[:, i],
          tspan = (tsteps_batch[1, i], tsteps_batch[end, i]))
        for i in 1:batchsize
    ]
end

# ## Step 6: Build the Complete Model
#
# Chain the wrapper, IC layer, and ODE model.

model = Chain(
    wrapper = Lux.WrappedFunction(feature_wrapper),
    initial_conditions = ic_layer,
    model = ode_model
)

# ## Step 7: Set Up Training
#
# Use the high-level training API with IC inference enabled.

backend = SGDBackend(
    Adam(5e-3),       # Higher learning rate for ICs
    300,              # More epochs for IC learning
    Lux.AutoZygote(),
    MSELoss()
)

# Enable initial condition inference
infer_ics = InferICs(true)

# ## Step 8: Train the Model
#
# Train to learn the initial conditions for each segment.

println("Starting training to learn initial conditions...")
result = train(backend, model, tokenized_dataloader, infer_ics)

# ## Step 9: Analyze Learned Initial Conditions
#
# Compare learned ICs with true values.

println("\nInitial Condition Recovery:")
println("Segment | True IC | Learned IC | Error")
println("-" ^ 40)

learned_ics = []
for (i, ic) in enumerate(result.ics)
    learned_u0 = ic.u0[1]
    true_u0 = segments[i][1]
    error = abs(learned_u0 - true_u0)
    rel_error = error / true_u0 * 100
    println(@sprintf("%7d | %7.3f | %9.3f | %5.1f%%", i, true_u0, learned_u0, rel_error))
    push!(learned_ics, learned_u0)
end

# ## Step 10: Visualize Results
#
# Plot the data segments and model predictions with learned ICs.

function plot_results(dataloader, model, ps, st, segments, tsteps)
    plt = plot()

    colors = [:blue, :red, :green, :orange, :purple, :brown]

    segment_idx = 1
    for (tokens_batch, (data_batch, tsteps_batch)) in dataloader
        preds = model((tokens_batch, tsteps_batch), ps, st)[1]

        for i in 1:size(data_batch, 3)
            color = colors[mod1(segment_idx, length(colors))]

            # Plot data
            plot!(plt, tsteps_batch[:, i], data_batch[:, 1, i],
                  label = segment_idx == 1 ? "Data" : "",
                  color = color, linewidth = 2)

            # Plot predictions
            plot!(plt, tsteps_batch[:, i], preds[:, 1, i],
                  label = segment_idx == 1 ? "Learned IC Model" : "",
                  color = color, linestyle = :dash, linewidth = 2)

            # Mark learned initial condition
            learned_ic = result.ics[segment_idx].u0[1]
            scatter!(plt, [tsteps_batch[1, i]], [learned_ic],
                    label = segment_idx == 1 ? "Learned IC" : "",
                    color = color, markersize = 6, marker = :star)

            segment_idx += 1
        end
    end

    xlabel!(plt, "Time")
    ylabel!(plt, "Population")
    title!(plt, "Learning Initial Conditions for Multiple Segments")

    return plt
end

plt = plot_results(tokenized_dataloader, model, result.ps, result.st, segments, tsteps)
display(plt)

# ## Step 11: Test Generalization
#
# Test how well the learned ICs generalize to the full time series.

function plot_full_trajectories(segments, learned_ics, tsteps, p_true)
    plt = plot()

    colors = [:blue, :red, :green, :orange]

    for (i, (segment, learned_ic)) in enumerate(zip(segments, learned_ics))
        color = colors[mod1(i, length(colors))]

        # Plot observed segment data
        plot!(plt, tsteps, segment[:, 1],
              label = "Segment $i Data",
              color = color, linewidth = 2)

        # Plot full trajectory starting from learned IC
        extended_tsteps = 0.0:0.2:15.0
        prob = ODEProblem(logistic_dudt, [learned_ic], (extended_tsteps[1], extended_tsteps[end]), p_true)
        sol = solve(prob, Tsit5(), saveat=extended_tsteps)
        plot!(plt, extended_tsteps, Array(sol)[:, 1],
              label = "Segment $i Full Trajectory",
              color = color, linestyle = :dash, linewidth = 2)

        # Mark the learned IC
        scatter!(plt, [tsteps[1]], [learned_ic],
                color = color, markersize = 8, marker = :star)
    end

    xlabel!(plt, "Time")
    ylabel!(plt, "Population")
    title!(plt, "Full Trajectories Using Learned Initial Conditions")

    return plt
end

plt_full = plot_full_trajectories(segments, learned_ics, tsteps, p_true)
display(plt_full)
