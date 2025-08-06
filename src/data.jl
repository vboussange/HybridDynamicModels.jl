# Adapted from Flux.jl `DataLoader` and MLUtils.jl `DataLoader`
using Random: AbstractRNG, shuffle!, GLOBAL_RNG
"""
    SegmentedTimeSeries(data; segmentsize=2, shift=nothing, batchsize=1, shuffle=false, partial_segment=false, partial_batch=false, rng=GLOBAL_RNG)

An object that iterates over mini-batches of segments of `data`,
each segment containing `segmentsize` data points, each mini-batch containing `batchsize` segments
(except possibly the last one). The last dimension in each tensor is the time dimension, i.e. the one segmented.

# Arguments
- `segmentsize`: Number of time points in each segment.
- `shift`: Step size between the start of consecutive segments. If `shift < segmentsize`, segments will overlap; if `shift > segmentsize`, there will be gaps. By default, `shift = segmentsize` (no overlap).
- `batchsize`: Number of segments per batch.
- `shuffle`: Shuffle the order of segments before batching.
- `partial_segment`: Allow the last segment to be shorter than `segmentsize` if not enough data remains.
- `partial_batch`: Allow the last batch to contain fewer than `batchsize` segments if not enough segments remain.
- `rng`: Random number generator for shuffling.

# Examples

## Basic usage with array
```jldoctest
julia> Xtrain = rand(10, 100)
julia> sdl = SegmentedTimeSeries(Xtrain; segmentsize=2, batchsize=1)
julia> for batch in sdl
           println("Batch: ", summary(batch))
       end
```

## With time steps and tuple input
```jldoctest
julia> tsteps = 1:100
julia> sdl = SegmentedTimeSeries((Xtrain, tsteps); segmentsize=2, batchsize=1)
julia> for (data, tseg) in sdl
           println("Data: ", summary(data))
           println("Time segment: ", tseg)
       end
```

## Custom shift and batch size
```jldoctest
julia> sdl = SegmentedTimeSeries(Xtrain; segmentsize=3, shift=1, batchsize=2)
julia> for batch in sdl
           println("Batch: ", batch)
       end
```

## Partial segments and batches
```jldoctest
julia> sdl = SegmentedTimeSeries(Xtrain; segmentsize=3, batchsize=2, partial_segment=true, partial_batch=true)
julia> for batch in sdl
           println("Batch: ", batch)
       end
```

## Shuffle segments
```jldoctest
julia> using Random
julia> rng = Random.MersenneTwister(42)
julia> sdl = SegmentedTimeSeries(Xtrain; segmentsize=2, batchsize=1, shuffle=true, rng=rng)
julia> for batch in sdl
           println("Shuffled batch: ", batch)
       end
```
"""
struct SegmentedTimeSeries{D, I, R<:AbstractRNG} # When iterated, returns (data, model_features) where model_features is a vector
    data::D
    segmentsize::Int
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

function SegmentedTimeSeries(data; segmentsize=2, shift=nothing, batchsize=1, shuffle=false, partial_segment=false, partial_batch=false, rng=GLOBAL_RNG)
    @assert segmentsize > 1
    @assert batchsize > 0

    isnothing(shift) && (shift = segmentsize)
    datasize = _nobs(data)

    # Compute indices for each segment
    indices = UnitRange{Int}[]
    m = 0
    while true
        start_idx = m * shift + 1
        end_idx = start_idx + segmentsize - 1
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

    return SegmentedTimeSeries(data, segmentsize, shift, batchsize, nsegments, shuffle, partial_segment, partial_batch, indices, imax, rng)
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
                            sdl.segmentsize, 
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
    return Tuple(x for x in segment_data)
end
