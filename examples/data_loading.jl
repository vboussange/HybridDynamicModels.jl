# # Data Loading with SegmentedTimeSeries in HybridDynamicModels.jl
#
# This example demonstrates how to use the `SegmentedTimeSeries` data loader
# for efficient batching and segmentation of time series data. This is essential
# for training hybrid dynamical models on time series data.

using HybridDynamicModels
using Plots
using Random
using Statistics

# ## Step 1: Generate Example Time Series Data
#
# Create synthetic time series data for demonstration.

# Generate multiple time series with different characteristics
function generate_time_series()
    t = 0.0:0.1:20.0
    series = []

    # Series 1: Logistic growth
    r1, K1 = 0.2, 3.0
    u0_1 = 0.5
    series1 = K1 * u0_1 ./ (u0_1 .+ (K1 .- u0_1) .* exp.(-r1 .* t))
    push!(series, series1)

    # Series 2: Oscillatory decay
    A2, ω2, γ2 = 2.0, 0.5, 0.05
    u0_2 = 1.0
    series2 = u0_2 .+ A2 .* exp.(-γ2 .* t) .* cos.(ω2 .* t)
    push!(series, series2)

    # Series 3: Step function with noise
    series3 = 1.0 .+ 0.5 .* (t .> 10.0) .+ 0.1 .* randn(length(t))
    push!(series, series3)

    return series, t
end

time_series, t = generate_time_series()

println("Generated $(length(time_series)) time series with $(length(t)) time points each")

# ## Step 2: Basic SegmentedTimeSeries Usage
#
# Demonstrate the basic functionality of SegmentedTimeSeries.

println("\n=== Basic SegmentedTimeSeries ===")

# Use the first time series for demonstration
data = time_series[1]'
time_data = t

# Create a basic segmented time series
segment_length = 20
shift = 10  # 50% overlap

dataloader = SegmentedTimeSeries((data, time_data);
                                segment_length = segment_length,
                                shift = shift,
                                batchsize = 1)

println("Created SegmentedTimeSeries with:")
println("  - Segment length: $segment_length")
println("  - Shift: $shift")
println("  - Total segments: $(length(dataloader))")

# Examine the first few segments
println("\nFirst 3 segments:")
for (i, (segment_data, segment_times)) in enumerate(dataloader)
    if i > 3
        break
    end
    println("Segment $i:")
    println("  Time range: $(segment_times[1]) to $(segment_times[end])")
    println("  Data shape: $(size(segment_data))")
    println("  Start value: $(segment_data[1, 1, 1])")
end

# ## Step 3: Tokenization for Advanced Features
#
# Use tokenization to access segments by index and enable advanced features.

println("\n=== Tokenization ===")

tokenized_dataloader = tokenize(dataloader)

println("Tokenized dataloader has $(length(tokens(tokenized_dataloader))) segments")

# Access specific segments by token
tokens_list = collect(tokens(tokenized_dataloader))
println("Available tokens: ", tokens_list[1:min(5, length(tokens_list))])

# Access a specific segment
token = tokens_list[1]
segment_data, segment_times = tokenized_dataloader[token]
println("\nSegment $token details:")
println("  Time points: $(length(segment_times))")
println("  Data values: $(segment_data[1, :, 1])")

# ## Step 4: Batching
#
# Demonstrate how to use different batch sizes.

println("\n=== Batching ===")

batch_sizes = [1, 2, 4]

for batch_size in batch_sizes
    dataloader_batch = SegmentedTimeSeries((data, time_data);
                                          segment_length = segment_length,
                                          shift = shift,
                                          batchsize = batch_size)

    println("Batch size $batch_size:")
    println("  Number of batches: $(length(dataloader_batch))")

    if length(dataloader_batch) > 0
        first_batch_data, first_batch_times = first(dataloader_batch)
        println("  First batch shape: $(size(first_batch_data))")
        println("  Time points per segment: $(size(first_batch_times, 1))")
    end
    println()
end

# ## Step 5: Handling Multiple Time Series
#
# Show how to work with multiple time series simultaneously.

println("\n=== Partial Segments ===")

# With partial segments allowed
dataloader_partial = SegmentedTimeSeries((data, time_data);
                                        segment_length = segment_length,
                                        shift = shift,
                                        batchsize = 1,
                                        partial_segment = true)

dataloader_no_partial = SegmentedTimeSeries((data, time_data);
                                           segment_length = segment_length,
                                           shift = shift,
                                           batchsize = 1,
                                           partial_segment = false)

println("With partial segments: $(length(dataloader_partial)) segments")
println("Without partial segments: $(length(dataloader_no_partial)) segments")

# Show the last segment with partial segments
tokenized_partial = tokenize(dataloader_partial)
last_token = maximum(tokens(tokenized_partial))
last_segment_data, last_segment_times = tokenized_partial[last_token]

println("\nLast segment (token $last_token):")
println("  Length: $(length(last_segment_times)) (requested: $segment_length)")
println("  Time range: $(last_segment_times[1]) to $(last_segment_times[end])")

# ## Step 7: Shuffling and Randomization
#
# Demonstrate shuffling for training.

println("\n=== Shuffling ===")

rng = Random.MersenneTwister(42)

dataloader_shuffled = SegmentedTimeSeries((data, time_data);
                                         segment_length = segment_length,
                                         shift = shift,
                                         batchsize = 1,
                                         shuffle = true,
                                         rng = rng)

tokenized_shuffled = tokenize(dataloader_shuffled)

println("Shuffled dataloader tokens: ", collect(tokens(tokenized_shuffled))[1:10])

# Compare with unshuffled
dataloader_unshuffled = SegmentedTimeSeries((data, time_data);
                                           segment_length = segment_length,
                                           shift = shift,
                                           batchsize = 1,
                                           shuffle = false)

tokenized_unshuffled = tokenize(dataloader_unshuffled)
println("Unshuffled dataloader tokens: ", collect(tokens(tokenized_unshuffled))[1:10])

# ## Step 8: Visualization of Segmentation
#
# Create visualizations to understand how segmentation works.

function plot_segmentation(data, times, dataloader, title_str)
    plt = plot(times, data[1, :],
               label = "Full time series",
               linewidth = 2,
               title = title_str,
               xlabel = "Time",
               ylabel = "Value")

    # Plot segments
    colors = [:red, :blue, :green, :orange, :purple, :brown]
    tokenized = tokenize(dataloader)

    for (i, token) in enumerate(tokens(tokenized))
        if i > 6  # Limit to first 6 segments for clarity
            break
        end
        segment_data, segment_times = tokenized[token]
        color = colors[mod1(i, length(colors))]

        plot!(plt, segment_times, segment_data[1, :, 1],
              label = "Segment $token",
              linewidth = 2,
              color = color,
              linestyle = :dash)
    end

    return plt
end

# Plot different segmentation strategies
plt_basic = plot_segmentation(data, time_data, dataloader,
                             "Basic Segmentation (50% overlap)")

# Different overlap
dataloader_no_overlap = SegmentedTimeSeries((data, time_data);
                                           segment_length = segment_length,
                                           shift = segment_length,  # No overlap
                                           batchsize = 1)

plt_no_overlap = plot_segmentation(data, time_data, dataloader_no_overlap,
                                  "No Overlap Segmentation")

# High overlap
dataloader_high_overlap = SegmentedTimeSeries((data, time_data);
                                             segment_length = segment_length,
                                             shift = 5,  # High overlap
                                             batchsize = 1)

plt_high_overlap = plot_segmentation(data, time_data, dataloader_high_overlap,
                                    "High Overlap Segmentation (75%)")

display(plt_basic)
display(plt_no_overlap)
display(plt_high_overlap)

# ## Step 9: Memory and Performance Considerations
#
# Demonstrate memory usage patterns.

println("\n=== Memory and Performance ===")

# Compare memory usage for different configurations
configs = [
    (segment_length = 10, shift = 5, batchsize = 1, name = "Small segments"),
    (segment_length = 20, shift = 10, batchsize = 1, name = "Medium segments"),
    (segment_length = 50, shift = 25, batchsize = 1, name = "Large segments"),
    (segment_length = 20, shift = 10, batchsize = 4, name = "Batched"),
]

println("Configuration comparison:")
println("Configuration | Segments | Memory (est.)")
for config in configs
    test_dataloader = SegmentedTimeSeries((data, time_data);
                                         segment_length = config.segment_length,
                                         shift = config.shift,
                                         batchsize = config.batchsize)

    n_segments = length(test_dataloader)
    # Rough memory estimate (bytes per segment)
    mem_per_segment = config.segment_length * 8 * 2  # 8 bytes per float64, 2 arrays
    total_mem = n_segments * mem_per_segment

    println(@sprintf("%-15s | %8d | %8.1f KB",
                    config.name, n_segments, total_mem / 1024))
end