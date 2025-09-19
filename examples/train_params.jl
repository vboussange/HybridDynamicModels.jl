# # Parameter Fitting with HybridDynamicModels.jl
#
# This example demonstrates how to fit parameters using the `ParameterLayer` in HybridDynamicModels.jl.
# We'll learn a simple parameter vector through gradient-based optimization.

using HybridDynamicModels
using Lux
using Optimisers
using Random
using Plots

# ## Step 1: Define the True Parameters and Generate Data
#
# Let's create some synthetic data by defining true parameters and a simple function.

# True parameters we want to learn
p_true = (; b = [0.23, 0.5])

# Simple function that uses these parameters
function simple_function(params)
    return params.b
end

# Generate "observed" data (in practice, this would be your real data)
observed_data = simple_function(p_true)

println("True parameters: ", p_true)
println("Observed data: ", observed_data)

# ## Step 2: Create a ParameterLayer
#
# The `ParameterLayer` is a learnable layer that holds parameters we want to optimize.

# Create a parameter layer with initial values
params_layer = ParameterLayer(init_value = (; b = [1.0, 2.0]))  # Different from true values

# ## Step 3: Build the Model
#
# We create a simple model that uses our parameter layer.

# Create a Lux chain that applies our parameter layer
model = Chain(params = params_layer, wrapper = Lux.WrappedFunction(x -> x.b))

# Initialize the model parameters and state
rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

println("Initial parameters: ", params_layer(ps.params, (;)))

# ## Step 4: Set Up Training
#
# Configure the optimizer, loss function, and training state.

# Optimizer
optimizer = Adam(0.1f0)

# Loss function (Mean Squared Error)
loss_fn = MSELoss()

# Training state
train_state = Lux.Training.TrainState(model, ps, st, optimizer)

# ## Step 5: Training Loop
#
# Train the model by minimizing the loss between predictions and observed data.

n_epochs = 100
losses = Float64[]

for epoch in 1:n_epochs
    # Single training step
    _, loss, _, train_state = Lux.Training.single_train_step!(
        Lux.AutoZygote(),
        loss_fn,
        ((), observed_data),  # Input and target
        train_state
    )

    push!(losses, loss)

    if epoch % 10 == 0
        println("Epoch $epoch: Loss = $(round(loss, digits=4))")
    end
end

# ## Step 6: Check Results
#
# Compare the learned parameters with the true values.

learned_params = params_layer(train_state.parameters.params, (;))
println("\nLearned parameters: ", learned_params)
println("True parameters: ", p_true)
println("Parameter error: ", abs.(learned_params - p_true.b))

# ## Step 7: Visualization
#
# Plot the training loss over time.

plot(1:n_epochs, losses,
     xlabel = "Epoch",
     ylabel = "Loss",
     title = "Training Loss",
     legend = false,
     linewidth = 2)

# ## Step 8: Make Predictions
#
# Use the trained model to make predictions.

prediction = model((), train_state.parameters, st)[1]
println("\nPrediction: ", prediction)
println("Target: ", observed_data)
println("Prediction error: ", abs.(prediction - observed_data))

