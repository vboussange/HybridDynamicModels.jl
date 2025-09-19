# # Fitting ODE Parameters with HybridDynamicModels.jl
#
# This example demonstrates how to fit parameters in ordinary differential equations (ODEs)
# using HybridDynamicModels.jl. We'll use a logistic growth model as an example.

using HybridDynamicModels
using Lux
using Optimisers
using OrdinaryDiffEq
using Plots
using Random
using ComponentArrays
using Test

# ## Step 1: Define the True ODE Model
#
# We'll use the logistic growth equation: du/dt = r * u * (1 - u/K)
# where r is the growth rate and K is the carrying capacity.

# True parameters
r_true = 0.1
K_true = 2.0
p_true = (; r = r_true, K = K_true)

# Initial condition
u0_true = [0.5]

# Logistic growth ODE function
function logistic_dudt(u, p, t)
    r, K = p.r, p.K
    return r * u * (1 - u / K)
end

# ## Step 2: Generate Synthetic Data
#
# Create time series data by solving the ODE with true parameters.

# Time span and points
tspan = (0.0, 20.0)
tsteps = 0.0:0.5:20.0

# Solve the ODE
prob = ODEProblem(logistic_dudt, u0_true, tspan, p_true)
sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1e-6, reltol=1e-6)
data_clean = Array(sol)

# Add some noise to simulate real data
rng = Random.MersenneTwister(42)
noise_level = 0.05
data_noisy = data_clean .* (1 .+ noise_level * randn(rng, size(data_clean)))

println("Generated $(length(tsteps)) time points of logistic growth data")
println("True parameters: r = $r_true, K = $K_true")

# ## Step 3: Create Data Loader
#
# Use SegmentedTimeSeries to create mini-batches for training.

batchsize = 2
segment_length = 10

dataloader = SegmentedTimeSeries((data_noisy, tsteps);
                                segment_length = segment_length,
                                shift = 5,
                                batchsize = batchsize)

println("Created data loader with $(length(dataloader)) batches")

# ## Step 4: Define the Hybrid ODE Model
#
# Create an ODEModel with learnable parameters.

# Parameter layer with constraints (positive parameters)
param_layer = ParameterLayer(
    init_value = (; r = 0.05, K = 1.5),  # Initial guesses
    constraint = BoxConstraint([1e-3, 1e-3], [1.0, 10.0])  # Bounds
)

# ODE function that uses the parameter layer
function dudt_model(layers, u, ps, t)
    params = layers.params(ps.params)
    r, K = params.r, params.K
    return r * u * (1 - u / K)
end

# Create the ODE model
ode_model = ODEModel(
    (; params = param_layer),
    dudt_model;
    alg = Tsit5(),
    abstol = 1e-6,
    reltol = 1e-6
)

# ## Step 5: Create Feature Wrapper
#
# The feature wrapper prepares inputs for the ODE model from batched data.

function feature_wrapper((batched_segments, tsteps_batch))
    return [
        (; u0 = batched_segments[:, 1, i],
          saveat = tsteps_batch[:, i],
          tspan = (tsteps_batch[1, i], tsteps_batch[end, i]))
        for i in 1:batchsize
    ]
end

# Combine wrapper and model
model = Chain(
    wrapper = Lux.WrappedFunction(feature_wrapper),
    model = ode_model
)

# ## Step 6: Set Up Training
#
# Configure the training backend and loss function.

# Training backend
backend = SGDBackend(
    Adam(1e-3),      # Optimizer
    200,             # Number of epochs
    Lux.AutoZygote(), # AD backend
    MSELoss()        # Loss function
)

# ## Step 7: Train the Model
#
# Train using the high-level training API.

println("Starting training...")
result = train(backend, model, dataloader, InferICs(false))

# ## Step 8: Analyze Results
#
# Check how well we recovered the true parameters.

learned_params = param_layer(result.ps.model.params, (;))
println("\nParameter Recovery:")
println("True r: $r_true, Learned r: $(learned_params.r)")
println("True K: $K_true, Learned K: $(learned_params.K)")
println("Relative error r: $(abs(learned_params.r - r_true)/r_true * 100)%")
println("Relative error K: $(abs(learned_params.K - K_true)/K_true * 100)%")

# ## Step 9: Visualize Results
#
# Plot the data and model predictions.

# Function to make predictions with the trained model
function predict_trajectory(model, ps, st, u0, tspan, tsteps)
    input = (; u0 = u0, tspan = tspan, saveat = tsteps)
    pred, _ = model.model(input, ps.model, st.model)
    return pred
end

# Plot comparison
plt = plot(tsteps, data_noisy', label="Noisy Data", color=:blue, linewidth=2)
plot!(plt, tsteps, data_clean', label="True Solution", color=:red, linewidth=2, linestyle=:dash)

# Predict with learned parameters
pred = predict_trajectory(model, result.ps, result.st, u0_true, tspan, tsteps)
plot!(plt, tsteps, pred', label="Learned Model", color=:green, linewidth=2, linestyle=:dot)

xlabel!(plt, "Time")
ylabel!(plt, "Population")
title!(plt, "Logistic Growth: Parameter Fitting Results")

display(plt)

# ## Step 10: Test on New Initial Conditions
#
# Test the learned model on different initial conditions.

u0_test = [0.1, 0.8, 1.5]
test_tspan = (0.0, 15.0)
test_tsteps = 0.0:0.2:15.0

plt_test = plot()
for (i, u0) in enumerate(u0_test)
    # True solution
    prob_test = ODEProblem(logistic_dudt, [u0], test_tspan, p_true)
    sol_test = solve(prob_test, Tsit5(), saveat=test_tsteps)
    plot!(plt_test, test_tsteps, Array(sol_test)', label="True (u0=$u0)", color=i, linestyle=:solid)

    # Learned model prediction
    pred_test = predict_trajectory(model, result.ps, result.st, [u0], test_tspan, test_tsteps)
    plot!(plt_test, test_tsteps, pred_test', label="Learned (u0=$u0)", color=i, linestyle=:dash)
end

xlabel!(plt_test, "Time")
ylabel!(plt_test, "Population")
title!(plt_test, "Model Generalization to Different Initial Conditions")

display(plt_test)
