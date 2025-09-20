# # Different Model Types in HybridDynamicModels.jl
#
# This example demonstrates the three main model types available in HybridDynamicModels.jl:
# - `ODEModel`: For ordinary differential equations
# - `ARModel`: For autoregressive models
# - `AnalyticModel`: For explicit analytical solutions
#
# We'll show how to implement the same logistic growth dynamics using each approach.
using Lux, Optimisers, ComponentArrays
using HybridDynamicModels
using Zygote
using OrdinaryDiffEq, SciMLSensitivity # for ODEModel and sensitivity analysis
using Plots
using Random
using ComponentArrays
using StatsBase

rng = MersenneTwister(42)
# ## Step 1: Define the Problem
#
# We'll model logistic growth: du/dt = r*u*(1 - u/K)
# with known analytical solution: u(t) = K*u0 / (u0 + (K - u0)*exp(-r*t))

# True parameters
r_true = [0.2]
K_true = [3.0]
u0_true = [0.5]

# Time settings
tspan = (0.0, 10.0)
tsteps = 0.0:0.2:10.0

# Generate reference data
function analytical_solution(u0, r, K, t)
    return @. K * u0 / (u0 + (K - u0) * exp(-r * t))
end

reference_solution = reduce(hcat, [analytical_solution(u0_true, r_true, K_true, t) for t in tsteps])

println("Modeling logistic growth with r=$(r_true[]), K=$(K_true[]), u0=$(u0_true[])")

# ## Step 2: ODEModel Implementation
#
# Use ODEModel for the differential equation approach.

println("\n=== ODEModel ===")
param_layer_ode = ParameterLayer(init_value = (; r = [0.15], K = [2.5]))

function logistic_ode(layers, u, ps, t)
    T = eltype(u)
    params = layers.params(ps.params)
    r, K = params.r, params.K
    return @. r * u * (one(T) - u / K)
end

ode_model = ODEModel(
    (; params = param_layer_ode),
    logistic_ode;
    alg = Tsit5(),
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()),
    abstol = 1e-6,
    reltol = 1e-6
)

# Test ODE model
ps_ode, st_ode = Lux.setup(rng, ode_model)
ps_ode = ComponentArray(ps_ode)

input_ode = (; u0 = u0_true, saveat = tsteps, tspan = tspan)
pred_ode, _ = ode_model(input_ode, ps_ode, st_ode)

println("ODE model prediction shape: ", size(pred_ode))

# ## Step 3: ARModel Implementation
#
# Use ARModel for an autoregressive discretization.

println("\n=== ARModel ===")

# For AR model, we need to define the step function
param_layer_ar = ParameterLayer(init_value = (; r = [0.15], K = [2.5]))

function logistic_ar_step(layers, u, ps, t)
    T = eltype(u)
    params = layers.params(ps.params)
    r, K = params.r, params.K
    dt = 0.2  # Time step (should match data spacing)
    # Simple Euler discretization of the ODE
    return @. u + dt * r * u * (one(T) - u / K)
end

ar_model = ARModel(
    (; params = param_layer_ar),
    logistic_ar_step;
    tspan = tspan,
    saveat = tsteps,
    dt = 0.2
)

# Test AR model
ps_ar, st_ar = Lux.setup(rng, ar_model)
ps_ar = ComponentArray(ps_ar)

input_ar = (; u0 = u0_true, saveat = tsteps, tspan = tspan)
pred_ar, _ = ar_model(input_ar, ps_ar, st_ar)

println("AR model prediction shape: ", size(pred_ar))

# ## Step 4: AnalyticModel Implementation
#
# Use AnalyticModel with the exact analytical solution.

println("\n=== AnalyticModel ===")

param_layer_analytic = ParameterLayer(init_value = (; r = [0.15], K = [2.5]))

function logistic_analytic(layers, u0, t0, ps, t)
    params = layers.params(ps.params)
    r, K = params.r, params.K
    # Analytical solution for logistic growth
    return @. K * u0 / (u0 + (K - u0) * exp(-r * (t - t0)))
end

analytic_model = AnalyticModel(
    (; params = param_layer_analytic),
    logistic_analytic;
    tspan = tspan,
    saveat = tsteps
)

# Test analytic model
ps_analytic, st_analytic = Lux.setup(rng, analytic_model)
ps_analytic = ComponentArray(ps_analytic)

input_analytic = (; u0 = u0_true, saveat = tsteps, tspan = tspan)
pred_analytic, _ = analytic_model(input_analytic, ps_analytic, st_analytic)

println("Analytic model prediction shape: ", size(pred_analytic))

# ## Step 5: Compare Model Outputs
#
# Plot and compare the outputs from all three models.

plt_comparison = plot(
    tsteps, reference_solution',
    label = "True Analytical",
    linewidth = 3,
    color = :black,
    linestyle = :solid,
    title = "Model Type Comparison: Logistic Growth",
    xlabel = "Time",
    ylabel = "Population"
)

plot!(plt_comparison, tsteps, pred_ode',
      label = "ODE Model",
      linewidth = 2,
      color = :blue,
      linestyle = :dash)

plot!(plt_comparison, tsteps, pred_ar',
      label = "AR Model",
      linewidth = 2,
      color = :red,
      linestyle = :dot)

plot!(plt_comparison, tsteps, pred_analytic',
      label = "Analytic Model",
      linewidth = 2,
      color = :green,
      linestyle = :dashdot)

display(plt_comparison)

# ## Step 6: Accuracy Analysis
#
# Compare the accuracy of each model type.

function compute_rmse(pred, true_val)
    return sqrt(mean((pred .- true_val).^2))
end

rmse_ode = compute_rmse(pred_ode', reference_solution)
rmse_ar = compute_rmse(pred_ar', reference_solution)
rmse_analytic = compute_rmse(pred_analytic', reference_solution)

println("\nAccuracy Comparison (RMSE):")
println("ODE Model: ", round(rmse_ode, digits=6))
println("AR Model: ", round(rmse_ar, digits=6))
println("Analytic Model: ", round(rmse_analytic, digits=6))

# ## Step 7: Training Comparison
#
# Show how each model type can be trained.

# Generate noisy training data
noise_level = 0.05
rng = Random.MersenneTwister(42)
training_data = reference_solution .* (1 .+ noise_level * randn(rng, length(reference_solution)))

# Create data loader
dataloader = SegmentedTimeSeries((training_data', tsteps);
                                segment_length = 10,
                                shift = 5,
                                batchsize = 1)

function train_model(model, model_name)
    backend = SGDBackend(
        Adam(1e-2),
        100,  # Short training for demo
        Lux.AutoZygote(),
        MSELoss()
    )

    println("Training $model_name...")
    result = train(backend, model, dataloader, InferICs(false))

    # Get final predictions
    ps_final = result.ps
    st_final = result.st
    final_pred, _ = model((; u0 = [u0_true], saveat = tsteps, tspan = tspan), ps_final, st_final)

    return final_pred[:, 1], result.losses
end

# Train each model
pred_ode_trained, losses_ode = train_model(ode_model, "ODE Model")
pred_ar_trained, losses_ar = train_model(ar_model, "AR Model")
pred_analytic_trained, losses_analytic = train_model(analytic_model, "Analytic Model")

# ## Step 8: Training Performance Visualization
#
# Plot the training progress and final results.

# Training loss comparison
plt_training = plot(
    1:length(losses_ode), losses_ode,
    label = "ODE Model",
    linewidth = 2,
    title = "Training Loss Comparison",
    xlabel = "Epoch",
    ylabel = "Loss"
)

plot!(plt_training, 1:length(losses_ar), losses_ar,
      label = "AR Model", linewidth = 2)

plot!(plt_training, 1:length(losses_analytic), losses_analytic,
      label = "Analytic Model", linewidth = 2)

display(plt_training)

# Final trained predictions
plt_final = plot(
    tsteps, training_data,
    label = "Training Data",
    color = :gray,
    alpha = 0.7,
    title = "Final Trained Predictions",
    xlabel = "Time",
    ylabel = "Population"
)

plot!(plt_final, tsteps, reference_solution,
      label = "True Solution",
      color = :black,
      linewidth = 2)

plot!(plt_final, tsteps, pred_ode_trained,
      label = "Trained ODE",
      linewidth = 2,
      color = :blue)

plot!(plt_final, tsteps, pred_ar_trained,
      label = "Trained AR",
      linewidth = 2,
      color = :red)

plot!(plt_final, tsteps, pred_analytic_trained,
      label = "Trained Analytic",
      linewidth = 2,
      color = :green)

display(plt_final)

# ## Step 9: Computational Performance
#
# Compare the computational cost of each model type.

using BenchmarkTools

function benchmark_model(model, input, ps, st, model_name)
    pred = @benchmark $model($input, $ps, $st)
    return minimum(pred.times) / 1e6  # Convert to milliseconds
end

time_ode = benchmark_model(ode_model, input_ode, ps_ode, st_ode, "ODE")
time_ar = benchmark_model(ar_model, input_ar, ps_ar, st_ar, "AR")
time_analytic = benchmark_model(analytic_model, input_analytic, ps_analytic, st_analytic, "Analytic")

println("\nComputational Performance (minimum time in ms):")
println("ODE Model: ", round(time_ode, digits=3), " ms")
println("AR Model: ", round(time_ar, digits=3), " ms")
println("Analytic Model: ", round(time_analytic, digits=3), " ms")

# ## Step 10: When to Use Each Model Type
#
# Provide guidance on model selection.

println("\n=== Model Type Selection Guide ===")
println()
println("ODE Model:")
println("  ✓ Most general - can represent any continuous dynamics")
println("  ✓ Automatic differentiation works well")
println("  ✓ Good for complex, nonlinear systems")
println("  ✗ Requires numerical ODE solver (slower)")
println("  ✗ May have numerical errors")
println()
println("AR Model:")
println("  ✓ Fast evaluation (no ODE solving)")
println("  ✓ Good for discrete-time modeling")
println("  ✓ Easy to implement custom dynamics")
println("  ✗ Requires manual discretization")
println("  ✗ May accumulate discretization errors")
println()
println("Analytic Model:")
println("  ✓ Exact solutions (no numerical errors)")
println("  ✓ Fastest evaluation")
println("  ✓ Mathematically interpretable")
println("  ✗ Requires analytical solution to exist")
println("  ✗ Limited to specific equation forms")
println()
println("Recommendation: Start with ODEModel for general problems,")
println("use AnalyticModel when you have closed-form solutions,")
println("and consider ARModel for performance-critical applications.")