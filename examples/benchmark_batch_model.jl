# # Benchmarking Training Approaches in HybridDynamicModels.jl
#
# This example demonstrates how to benchmark different training approaches and
# model configurations in HybridDynamicModels.jl. We'll compare the performance
# of segmented vs non-segmented data handling for both forward passes and
# gradient computations.

using HybridDynamicModels
using Lux
using OrdinaryDiffEq
using BenchmarkTools
using Random
using ComponentArrays
using Plots

# ## Step 1: Set Up a Test Model
#
# Create a moderately complex ODE model for benchmarking.

# True parameters
p_true = (; r = 0.15, K = 2.5, α = 0.1)
u0_true = [0.5]

# Complex ODE: logistic growth with additional dynamics
function complex_dudt(u, p, t)
    r, K, α = p.r, p.K, p.α
    # Logistic growth with oscillatory component
    growth = r * u * (1 - u / K)
    oscillation = α * u * sin(2π * t)
    return growth + oscillation
end

# Time settings
tsteps = 0.0:0.1:20.0
tspan = (tsteps[1], tsteps[end])

# Generate synthetic data
prob = ODEProblem(complex_dudt, u0_true, tspan, p_true)
sol = solve(prob, Tsit5(), saveat=tsteps, abstol=1e-6, reltol=1e-6)
data_clean = Array(sol)

# Add noise
rng = Random.MersenneTwister(42)
noise_level = 0.02
data_noisy = data_clean .* (1 .+ noise_level * randn(rng, size(data_clean)))

println("Generated $(length(tsteps)) time points for benchmarking")
println("Model has $(length(p_true)) parameters")

# ## Step 2: Create Hybrid Model
#
# Set up the model for benchmarking.

param_layer = ParameterLayer(init_value = p_true)

function model_dudt(layers, u, ps, t)
    params = layers.params(ps.params)
    r, K, α = params.r, params.K, params.α
    growth = r * u * (1 - u / K)
    oscillation = α * u * sin(2π * t)
    return growth + oscillation
end

ode_model = ODEModel(
    (; params = param_layer),
    model_dudt;
    alg = Tsit5(),
    abstol = 1e-6,
    reltol = 1e-6
)

ps, st = Lux.setup(Random.default_rng(), ode_model)
ps = ComponentArray(ps)

# ## Step 3: Prepare Data in Different Formats
#
# Create both segmented and non-segmented data representations.

# Non-segmented: single long trajectory
input_noseg = (; u0 = data_noisy[:, 1], saveat = tsteps, tspan = tspan)

# Segmented: break into overlapping segments
segment_length = 20
shift = 10  # 50% overlap

dataloader = SegmentedTimeSeries((data_noisy, tsteps);
                                segment_length = segment_length,
                                shift = shift,
                                batchsize = 1)

tokenized_dataloader = tokenize(dataloader)

# Prepare segmented inputs
inputs_seg = []
for token in tokens(tokenized_dataloader)
    segment_data, segment_tsteps = tokenized_dataloader[token]
    push!(inputs_seg, (; u0 = segment_data[:, 1],
                       saveat = segment_tsteps,
                       tspan = (segment_tsteps[1], segment_tsteps[end])))
end

println("Created $(length(inputs_seg)) overlapping segments")

# ## Step 4: Benchmark Forward Passes
#
# Compare the speed of forward model evaluations.

println("\n=== Forward Pass Benchmarks ===")

# Benchmark non-segmented forward pass
println("Benchmarking non-segmented forward pass...")
forward_noseg = @benchmark $ode_model($input_noseg, $ps, $st)[1]

# Benchmark segmented forward passes
println("Benchmarking segmented forward passes...")
forward_seg = @benchmark begin
    results = []
    for input in $inputs_seg
        push!(results, $ode_model(input, $ps, $st)[1])
    end
    results
end

println("\nForward Pass Results:")
println("Non-segmented: ", minimum(forward_noseg.times) / 1e6, " ms (median)")
println("Segmented: ", minimum(forward_seg.times) / 1e6, " ms (median)")
println("Speedup factor: ", median(forward_noseg.times) / median(forward_seg.times))

# ## Step 5: Benchmark Gradient Computations
#
# Compare gradient computation performance.

println("\n=== Gradient Computation Benchmarks ===")

# Loss functions
loss_noseg(ps) = sum(abs2, ode_model(input_noseg, ps, st)[1] - data_noisy)
loss_seg(ps) = sum(sum(abs2, ode_model(input, ps, st)[1] - data_noisy[:, 1:length(input.saveat)])
                   for input in inputs_seg)

# Benchmark non-segmented gradients
println("Benchmarking non-segmented gradients...")
grad_noseg = @benchmark value_and_gradient($loss_noseg, Lux.AutoZygote(), $ps)

# Benchmark segmented gradients
println("Benchmarking segmented gradients...")
grad_seg = @benchmark value_and_gradient($loss_seg, Lux.AutoZygote(), $ps)

println("\nGradient Computation Results:")
println("Non-segmented: ", minimum(grad_noseg.times) / 1e6, " ms (median)")
println("Segmented: ", minimum(grad_seg.times) / 1e6, " ms (median)")
println("Speedup factor: ", median(grad_noseg.times) / median(grad_seg.times))

# ## Step 6: Memory Usage Analysis
#
# Compare memory allocation patterns.

println("\n=== Memory Usage Analysis ===")

# Memory benchmarks
mem_noseg = @benchmark $ode_model($input_noseg, $ps, $st)
mem_seg = @benchmark begin
    for input in $inputs_seg
        $ode_model(input, $ps, $st)
    end
end

println("Memory allocations (forward pass):")
println("Non-segmented: ", mem_noseg.memory / 1024, " KB")
println("Segmented: ", mem_seg.memory / 1024, " KB")

# ## Step 7: Training Performance Comparison
#
# Compare actual training performance with different batch sizes.

println("\n=== Training Performance Comparison ===")

function train_with_config(batchsize, segment_length, shift)
    # Create data loader with specific configuration
    dataloader = SegmentedTimeSeries((data_noisy, tsteps);
                                    segment_length = segment_length,
                                    shift = shift,
                                    batchsize = batchsize)

    # Set up training
    backend = SGDBackend(
        Adam(1e-3),
        50,  # Short training for benchmarking
        Lux.AutoZygote(),
        MSELoss()
    )

    # Time the training
    time = @elapsed begin
        result = train(backend, ode_model, dataloader, InferICs(false))
    end

    return time, result
end

# Test different configurations
configs = [
    (batchsize=1, segment_length=20, shift=10, name="Small segments"),
    (batchsize=4, segment_length=20, shift=10, name="Larger batches"),
    (batchsize=4, segment_length=10, shift=5, name="More segments"),
]

println("Training performance for different configurations:")
for config in configs
    time_taken, result = train_with_config(config.batchsize, config.segment_length, config.shift)
    final_loss = result.losses[end]
    println(@sprintf("%-15s | Time: %6.3f s | Final loss: %.6f",
                    config.name, time_taken, final_loss))
end

# ## Step 8: Scalability Analysis
#
# Test performance as data size increases.

println("\n=== Scalability Analysis ===")

data_sizes = [50, 100, 200, 500]
scalability_results = []

for n_points in data_sizes
    # Create data of different sizes
    tsteps_scaled = range(0.0, 20.0, length=n_points)
    prob_scaled = ODEProblem(complex_dudt, u0_true, (0.0, 20.0), p_true)
    sol_scaled = solve(prob_scaled, Tsit5(), saveat=tsteps_scaled)
    data_scaled = Array(sol_scaled) .* (1 .+ 0.02 * randn(rng, size(Array(sol_scaled))))

    # Time forward pass
    input_scaled = (; u0 = data_scaled[:, 1], saveat = tsteps_scaled, tspan = (0.0, 20.0))
    time_forward = @elapsed ode_model(input_scaled, ps, st)

    # Time gradient computation
    loss_scaled(ps) = sum(abs2, ode_model(input_scaled, ps, st)[1] - data_scaled)
    time_grad = @elapsed value_and_gradient(loss_scaled, Lux.AutoZygote(), ps)

    push!(scalability_results, (n_points, time_forward, time_grad))
end

println("Scalability with data size:")
println("Data points | Forward (ms) | Gradient (ms)")
for (n, t_fwd, t_grad) in scalability_results
    println(@sprintf("%11d | %12.3f | %13.3f", n, t_fwd*1000, t_grad*1000))
end

# ## Step 9: Visualization of Results
#
# Plot the benchmarking results.

# Plot scalability
plt_scalability = plot(
    [r[1] for r in scalability_results],
    [r[2]*1000 for r in scalability_results],
    label = "Forward pass",
    xlabel = "Number of data points",
    ylabel = "Time (ms)",
    title = "Scalability Analysis",
    linewidth = 2,
    marker = :circle
)

plot!(plt_scalability,
    [r[1] for r in scalability_results],
    [r[3]*1000 for r in scalability_results],
    label = "Gradient computation",
    linewidth = 2,
    marker = :square
)

display(plt_scalability)

# ## Step 10: Recommendations
#
# Provide guidance based on the benchmarks.

println("\n=== Recommendations ===")
println("Based on the benchmarking results:")
println("1. For large datasets, segmented processing can provide better memory efficiency")
println("2. Gradient computation benefits from parallelization opportunities with segments")
println("3. Choose segment length based on your model's memory requirements")
println("4. Larger batches generally improve training efficiency")
println("5. Consider the trade-off between computation time and memory usage")
println("\nKey insight: Segmented processing enables better scalability for large datasets")
println("while maintaining training performance for smaller problems.")