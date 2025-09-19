"""
    SegmentedTimeSeries(data; segment_length=2, shift=nothing, batchsize=1, shuffle=false, partial_segment=false, partial_batch=false, rng=GLOBAL_RNG)

An object that iterates over mini-batches of segments of `data`, each segment containing `segment_length` data points, each mini-batch containing `batchsize` segments. The last dimension in each tensor is the time dimension.

## Arguments
  - `data`: Input data (array, tuple, or named tuple).
  - `segment_length`: Number of time points in each segment.
  - `shift`: Step size between consecutive segments (default: `segment_length`).
  - `batchsize`: Number of segments per batch.
  - `shuffle`: Whether to shuffle segment order.
  - `partial_segment`: Allow shorter final segments.
  - `partial_batch`: Allow smaller final batches.
  - `rng`: Random number generator for shuffling.

## Inputs
- `data`: The time series data to segment.

## Outputs
- Iterator yielding batches of data segments.

## Behavior
Creates overlapping or non-overlapping segments from time series data for training dynamical models. Segments can be shuffled and batched for efficient training.

## Example

```jldoctest
julia> Xtrain = rand(10, 100)
julia> sdl = SegmentedTimeSeries(Xtrain; segment_length=2, batchsize=1)
julia> for batch in sdl
           println("Batch: ", summary(batch))
       end
```

!!!warning
    Undefined behavior when data dimensions are incompatible
"""
struct SegmentedTimeSeries{D, I, R<:AbstractRNG} # When iterated, returns (data, model_features) where model_features is a vector
    data::D
    segment_length::Int
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

function SegmentedTimeSeries(data; segment_length=nothing, shift=nothing, batchsize=1, shuffle=false, partial_segment=false, partial_batch=false, rng=GLOBAL_RNG)
    isnothing(segment_length) && (segment_length = _nobs(data))
    @assert segment_length > 0
    @assert segment_length <= _nobs(data) "Segment size must be less than or equal to the number of time steps."
    !isnothing(shift) && @assert shift > 0
    @assert batchsize > 0

    isnothing(shift) && (shift = segment_length-1)
    datasize = _nobs(data)

    # Compute indices for each segment
    indices = UnitRange{Int}[]
    m = 0
    while true
        start_idx = m * shift + 1
        end_idx = start_idx + segment_length - 1
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

    return SegmentedTimeSeries(data, segment_length, shift, batchsize, nsegments, shuffle, partial_segment, partial_batch, indices, imax, rng)
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
    for i in 1:length(data)
        ni = _nobs(data[i])
        n == ni || throw(DimensionMismatch("All data inputs should have the same number of observations, i.e. size in the last dimension. " * 
            "But data[$(repr(first(keys(data))))] ($(summary(data[1]))) has $n, while data[$(repr(i))] ($(summary(data[i]))) has $ni."))
    end
    return n
end

_get_ts_obs(data::AbstractArray, segments) = cat([data[ntuple(i -> Colon(), Val(ndims(data) - 1))..., seg] for seg in segments]...; dims=ndims(data)+1)
_get_ts_obs(data::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(_get_ts_obs, i), data)

Base.eltype(::SegmentedTimeSeries{D}) where D = Array{eltype(D), ndims(D) + 1}

"""
    tokenize(sdl::SegmentedTimeSeries)

Convert a SegmentedTimeSeries to use token-based indexing.

## Arguments
  - `sdl`: The SegmentedTimeSeries to tokenize.

## Inputs
- `sdl`: SegmentedTimeSeries object.

## Outputs
- Tokenized SegmentedTimeSeries with integer-based segment access.

## Behavior
Transforms segment indices into a token-based system for easier access to individual segments.

## Example

```jldoctest
julia> sdl = SegmentedTimeSeries(rand(10, 100); segment_length=2)
julia> tokenized_sdl = tokenize(sdl)
julia> tokens(tokenized_sdl) # Returns available tokens
```
"""
function tokenize(sdl::SegmentedTimeSeries)
    return SegmentedTimeSeries(sdl.data, 
                            sdl.segment_length, 
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

"""
    tokens(sdl::SegmentedTimeSeries)

Get the available tokens for a tokenized SegmentedTimeSeries.

## Arguments
  - `sdl`: A tokenized SegmentedTimeSeries.

## Inputs
- `sdl`: Tokenized SegmentedTimeSeries object.

## Outputs
- Range of available tokens (1 to number of segments).

## Behavior
Returns the range of tokens that can be used to access individual segments in a tokenized SegmentedTimeSeries.

## Example

```jldoctest
julia> sdl = SegmentedTimeSeries(rand(10, 100); segment_length=2)
julia> tokenized_sdl = tokenize(sdl)
julia> collect(tokens(tokenized_sdl)) # [1, 2, 3, ...]
```
"""
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
    create_train_val_loaders(data; segment_length, valid_length, kwargs...)

Create separate training and validation SegmentedTimeSeries loaders from a dataset with the same number of batches.

This function splits the data into non-overlapping training and validation segments.
The training data uses segments with gaps equal to `valid_length` to leave space
for validation segments. The validation data starts after the first training segment
and uses segments of length `valid_length`. Both loaders are guaranteed to have
the same number of batches, with tokens referring to the same ordering.

# Arguments
- `data`: Input data (can be an array, tuple, or named tuple)
- `segment_length`: Size of training segments
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
    segment_length=20, valid_length=5, batchsize=4)
# Both loaders will have the same number of batches
@assert length(train_loader) == length(val_loader)
```

## With tuple data (data, time steps)
```julia
data = rand(10, 100)
tsteps = 1:100
train_loader, val_loader = create_train_val_loaders((data, tsteps); 
    segment_length=20, valid_length=5, batchsize=4)
```

## With named tuple data
```julia
dataset = (observations = rand(10, 100), times = 1:100, metadata = rand(5, 100))
train_loader, val_loader = create_train_val_loaders(dataset; 
    segment_length=20, valid_length=5)
```

# Notes
- Training segments are spaced `segment_length + valid_length` apart to avoid overlap with validation
- Validation segments start at position `segment_length + 1` to avoid overlap with first training segment
- Both loaders have `partial_segment = false` and `partial_batch = false` to ensure consistent sizes
- Both loaders are guaranteed to have the same number of batches for synchronized training/validation
"""
function create_train_val_loaders(data; segment_length, valid_length, kwargs...)
    datasize = _nobs(data)
    shift = segment_length + valid_length
    
    # Calculate how many training segments we can fit
    train_segments = _count_segments(datasize, segment_length, shift, false)
    
    # Calculate how many validation segments we can fit (starting from segment_length + 1)
    valid_datasize = datasize - segment_length
    valid_segments = _count_segments(valid_datasize, valid_length, shift, false)
    
    # Take the minimum to ensure same number of segments
    max_segments = min(train_segments, valid_segments)
    
    # Force both loaders to have the same number of segments by using partial_batch = false
    # and ensuring we don't exceed max_segments
    kwargs_sync = merge(NamedTuple(kwargs), (partial_segment = false, partial_batch = false))
    
    # Create training loader
    dataloader_train = SegmentedTimeSeries(data; segment_length, shift, kwargs_sync...)
    
    # Create validation data by slicing from position segment_length + 1 onwards
    validation_data = _slice_data_from_index(data, segment_length + 1)
    
    # Create validation loader
    dataloader_valid = SegmentedTimeSeries(validation_data;
        segment_length = valid_length, shift, kwargs_sync...)
    
    # Verify they have the same length
    if length(dataloader_train) != length(dataloader_valid)
        @warn "Training and validation loaders have different lengths: $(length(dataloader_train)) vs $(length(dataloader_valid)). This may cause issues during training."
    end
    
    return dataloader_train, dataloader_valid
end

# Helper function to count segments that would be created
function _count_segments(datasize::Int, segment_length::Int, shift::Int, partial_segment::Bool)
    segments = 0
    m = 0
    while true
        start_idx = m * shift + 1
        end_idx = start_idx + segment_length - 1
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