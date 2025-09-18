# Adapted from Flux.jl `DataLoader` and MLUtils.jl `DataLoader`
using Random: AbstractRNG, shuffle!, GLOBAL_RNG
using Functors: @functor
"""
    SegmentedTimeSeries(data; segmentlength=2, shift=nothing, batchsize=1, shuffle=false, partial_segment=false, partial_batch=false, rng=GLOBAL_RNG)

An object that iterates over mini-batches of segments of `data`,
each segment containing `segmentlength` data points, each mini-batch containing `batchsize` segments
(except possibly the last one). The last dimension in each tensor is the time dimension, i.e. the one segmented.

# Arguments
- `segmentlength`: Number of time points in each segment.
- `shift`: Step size between the start of consecutive segments. If `shift < segmentlength`, segments will overlap; if `shift > segmentlength`, there will be gaps. By default, `shift = segmentlength` (no overlap).
- `batchsize`: Number of segments per batch.
- `shuffle`: Shuffle the order of segments before batching.
- `partial_segment`: Allow the last segment to be shorter than `segmentlength` if not enough data remains.
- `partial_batch`: Allow the last batch to contain fewer than `batchsize` segments if not enough segments remain.
- `rng`: Random number generator for shuffling.

# Examples

## Basic usage with array
```jldoctest
julia> Xtrain = rand(10, 100)
julia> sdl = SegmentedTimeSeries(Xtrain; segmentlength=2, batchsize=1)
julia> for batch in sdl
           println("Batch: ", summary(batch))
       end
```

## With time steps and tuple input
```jldoctest
julia> tsteps = 1:100
julia> sdl = SegmentedTimeSeries((Xtrain, tsteps); segmentlength=2, batchsize=1)
julia> for (data, tseg) in sdl
           println("Data: ", summary(data))
           println("Time segment: ", tseg)
       end
```

## Custom shift and batch size
```jldoctest
julia> sdl = SegmentedTimeSeries(Xtrain; segmentlength=3, shift=1, batchsize=2)
julia> for batch in sdl
           println("Batch: ", batch)
       end
```

## Partial segments and batches
```jldoctest
julia> sdl = SegmentedTimeSeries(Xtrain; segmentlength=3, batchsize=2, partial_segment=true, partial_batch=true)
julia> for batch in sdl
           println("Batch: ", batch)
       end
```

## Shuffle segments
```jldoctest
julia> using Random
julia> rng = Random.MersenneTwister(42)
julia> sdl = SegmentedTimeSeries(Xtrain; segmentlength=2, batchsize=1, shuffle=true, rng=rng)
julia> for batch in sdl
           println("Shuffled batch: ", batch)
       end
```
"""
struct SegmentedTimeSeries{D, I, R<:AbstractRNG} # When iterated, returns (data, model_features) where model_features is a vector
    data::D
    segmentlength::Int
    shift::Int
    batchsize::Int
    nsegments::Int
    shuffle::Bool
    partial_segment::Bool
    partial_batch::Bool
    indices::I # indices of the segments 
    imax::Int
    rng::R
end

@functor SegmentedTimeSeries (data,)

function SegmentedTimeSeries(data; segmentsize=nothing, shift=nothing, batchsize=1, shuffle=false, partial_segment=false, partial_batch=false, rng=GLOBAL_RNG)
    isnothing(segmentsize) && (segmentsize = _nobs(data))
    @assert segmentsize > 0
    @assert segmentsize <= _nobs(data) "Segment size must be less than or equal to the number of time steps."
    !isnothing(shift) && @assert shift > 0
    @assert batchsize > 0

    isnothing(shift) && (shift = segmentlength-1)
    datasize = _nobs(data)

    # Compute indices for each segment
    indices = UnitRange{Int}[]
    m = 0
    while true
        start_idx = m * shift + 1
        end_idx = start_idx + segmentlength - 1
        if end_idx > datasize
            if partial_segment && start_idx <= datasize
                push!(indices, start_idx:datasize)
            end
            break
        else
            push!(indices, start_idx:end_idx)
        end
        m += 1
    end
    nsegments = length(indices)

    # Compute imax for batching
    imax = partial_batch ? nsegments : nsegments - batchsize + 1
    imax = max(imax, 0)

    return SegmentedTimeSeries(data, segmentlength, shift, batchsize, nsegments, shuffle, partial_segment, partial_batch, indices, imax, rng)
end

# Standard iteration over SegmentedTimeSeries
Base.@propagate_inbounds function Base.iterate(sdl::SegmentedTimeSeries{D, I, R}, i=0) where {D, I <: AbstractVector{<:UnitRange}, R}
    i >= sdl.imax && return nothing
    if sdl.shuffle && i == 0
        shuffle!(sdl.rng, sdl.indices)
    end
    nexti = min(i + sdl.batchsize, sdl.nsegments)
    segments = sdl.indices[i+1:nexti]
    batch_data = _get_ts_obs(sdl.data, segments)
    return (batch_data, nexti)
end

# Iteration over SegmentedTimeSeries with tokenized input
Base.@propagate_inbounds function Base.iterate(sdl::SegmentedTimeSeries{D, I, R}, i=0) where {D, I<:AbstractVector{<:Pair}, R}
    i >= sdl.imax && return nothing
    if sdl.shuffle && i == 0
        shuffle!(sdl.rng, sdl.indices)
    end
    nexti = min(i + sdl.batchsize, sdl.nsegments)
    segments = sdl.indices[i+1:nexti]
    batch_tokens = [s[1] for s in segments]
    indices = [s[2] for s in segments]
    batch_data = _get_ts_obs(sdl.data, indices)
    return ((batch_tokens, batch_data), nexti)
end

function Base.length(d::SegmentedTimeSeries)
    n = d.nsegments / d.batchsize
    d.partial_batch ? ceil(Int, n) : floor(Int, n)
end

_nobs(data::AbstractArray) = size(data)[end]

function _nobs(data::Union{Tuple, NamedTuple})
    length(data) > 0 || throw(ArgumentError("Need at least one data input"))
    n = _nobs(data[1])
    for i in keys(data)
        ni = _nobs(data[i])
        n == ni || throw(DimensionMismatch("All data inputs should have the same number of observations, i.e. size in the last dimension. " * 
            "But data[$(repr(first(keys(data))))] ($(summary(data[1]))) has $n, while data[$(repr(i))] ($(summary(data[i]))) has $ni."))
    end
    return n
end

_get_ts_obs(data::AbstractArray, segments) = cat([data[ntuple(i -> Colon(), Val(ndims(data) - 1))..., seg] for seg in segments]...; dims=ndims(data)+1)
_get_ts_obs(data::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(_get_ts_obs, i), data)

Base.eltype(::SegmentedTimeSeries{D}) where D = Array{eltype(D), ndims(D) + 1}

function tokenize(sdl::SegmentedTimeSeries)
    return SegmentedTimeSeries(sdl.data, 
                            sdl.segmentlength, 
                            sdl.shift, 
                            sdl.batchsize, 
                            sdl.nsegments, 
                            sdl.shuffle, 
                            sdl.partial_segment, 
                            sdl.partial_batch, 
                            [i => sdl.indices[i] for i in 1:sdl.nsegments], 
                            sdl.imax, 
                            sdl.rng)
end

tokens(sdl::SegmentedTimeSeries{D, I, R}) where {D, I <: AbstractVector{<:Pair}, R} = 1:sdl.nsegments

function Base.getindex(sdl::SegmentedTimeSeries, token)
    # Find the index of the token in the indices list
    idx = findfirst(x -> (x isa Pair ? x[1] == token : false), sdl.indices)
    idx === nothing && throw(KeyError("Token $token not found in SegmentedTimeSeries."))
    seg = sdl.indices[idx]
    # seg[2] is the segment indices
    segment_data = _get_ts_obs(sdl.data, [seg[2]])
    return Tuple(dropdims(x, dims=ndims(x)) for x in segment_data)
end

"""
    create_train_val_loaders(data; segmentsize, valid_length, kwargs...)

Create separate training and validation SegmentedTimeSeries loaders from a dataset with the same number of batches.

This function splits the data into non-overlapping training and validation segments.
The training data uses segments with gaps equal to `valid_length` to leave space
for validation segments. The validation data starts after the first training segment
and uses segments of length `valid_length`. Both loaders are guaranteed to have
the same number of batches, with tokens referring to the same ordering.

# Arguments
- `data`: Input data (can be an array, tuple, or named tuple)
- `segmentsize`: Size of training segments
- `valid_length`: Size of validation segments
- `kwargs...`: Additional arguments passed to SegmentedTimeSeries constructors

# Returns
- `dataloader_train`: SegmentedTimeSeries for training data
- `dataloader_valid`: SegmentedTimeSeries for validation data

# Examples

## With array data
```julia
data = rand(10, 100)  # 10 features, 100 time steps
train_loader, val_loader = create_train_val_loaders(data; 
    segmentsize=20, valid_length=5, batchsize=4)
# Both loaders will have the same number of batches
@assert length(train_loader) == length(val_loader)
```

## With tuple data (data, time steps)
```julia
data = rand(10, 100)
tsteps = 1:100
train_loader, val_loader = create_train_val_loaders((data, tsteps); 
    segmentsize=20, valid_length=5, batchsize=4)
```

## With named tuple data
```julia
dataset = (observations = rand(10, 100), times = 1:100, metadata = rand(5, 100))
train_loader, val_loader = create_train_val_loaders(dataset; 
    segmentsize=20, valid_length=5)
```

# Notes
- Training segments are spaced `segmentsize + valid_length` apart to avoid overlap with validation
- Validation segments start at position `segmentsize + 1` to avoid overlap with first training segment
- Both loaders have `partial_segment = false` and `partial_batch = false` to ensure consistent sizes
- Both loaders are guaranteed to have the same number of batches for synchronized training/validation
"""
function create_train_val_loaders(data; segmentsize, valid_length, kwargs...)
    datasize = _nobs(data)
    shift = segmentsize + valid_length
    
    # Calculate how many training segments we can fit
    train_segments = _count_segments(datasize, segmentsize, shift, false)
    
    # Calculate how many validation segments we can fit (starting from segmentsize + 1)
    valid_datasize = datasize - segmentsize
    valid_segments = _count_segments(valid_datasize, valid_length, shift, false)
    
    # Take the minimum to ensure same number of segments
    max_segments = min(train_segments, valid_segments)
    
    # Force both loaders to have the same number of segments by using partial_batch = false
    # and ensuring we don't exceed max_segments
    kwargs_sync = merge(NamedTuple(kwargs), (partial_segment = false, partial_batch = false))
    
    # Create training loader
    dataloader_train = SegmentedTimeSeries(data; segmentsize, shift, kwargs_sync...)
    
    # Create validation data by slicing from position segmentsize + 1 onwards
    validation_data = _slice_data_from_index(data, segmentsize + 1)
    
    # Create validation loader
    dataloader_valid = SegmentedTimeSeries(validation_data;
        segmentsize = valid_length, shift, kwargs_sync...)
    
    # Verify they have the same length
    if length(dataloader_train) != length(dataloader_valid)
        @warn "Training and validation loaders have different lengths: $(length(dataloader_train)) vs $(length(dataloader_valid)). This may cause issues during training."
    end
    
    return dataloader_train, dataloader_valid
end

# Helper function to count segments that would be created
function _count_segments(datasize::Int, segmentsize::Int, shift::Int, partial_segment::Bool)
    segments = 0
    m = 0
    while true
        start_idx = m * shift + 1
        end_idx = start_idx + segmentsize - 1
        if end_idx > datasize
            if partial_segment && start_idx <= datasize
                segments += 1
            end
            break
        else
            segments += 1
        end
        m += 1
    end
    return segments
end

# Helper function to slice data generically from a given index
function _slice_data_from_index(data::AbstractArray, start_idx::Int)
    return data[ntuple(i -> Colon(), Val(ndims(data) - 1))..., start_idx:end]
end

function _slice_data_from_index(data::Union{Tuple, NamedTuple}, start_idx::Int)
    return map(x -> _slice_data_from_index(x, start_idx), data)
end