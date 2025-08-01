# Adapted from Flux.jl `DataLoader` and MLUtils.jl `DataLoader`
using Random: AbstractRNG, shuffle!, GLOBAL_RNG
"""
    SegmentedTimeSeries(data, tsteps; segmentsize=2, batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)
An object that iterates over mini-batches of segments of `data`, 
each segment containing `segmentsize` data points, each mini-batch containing `batchsize` segments
(except possibly the last one).

The last dimension in each tensor is the observation dimension, i.e. the one

# Examples

```jldoctest
julia> Xtrain = rand(10, 100);

julia> tsteps = 1:100;

julia> sdl = SegmentedTimeSeries(Xtrain, tsteps; segmentsize=2, batchsize=1);

julia> for (data, model_features) in sdl
           println("Data: ", summary(data))
           println("Features: ", summary(model_features))
       end
```
"""
struct SegmentedTimeSeries{D, R<:AbstractRNG} # When iterated, returns (data, model_features) where model_features is a vector
    data::D
    segmentsize::Int
    shift::Int
    batchsize::Int
    nsegments::Int
    shuffle::Bool
    partial_segment::Bool
    partial_batch::Bool
    indices::Vector{UnitRange{Int}} # indices of the segments
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

Base.@propagate_inbounds function Base.iterate(sdl::SegmentedTimeSeries, i=0)
    i >= sdl.imax && return nothing
    if sdl.shuffle && i == 0
        shuffle!(sdl.rng, sdl.indices)
    end
    nexti = min(i + sdl.batchsize, sdl.nsegments)
    segments = sdl.indices[i+1:nexti]
    batch_data = _get_ts_obs(sdl.data, segments)
    return (batch_data, nexti)
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

_get_ts_obs(data::AbstractArray, segments) = [data[ntuple(i -> Colon(), Val(ndims(data) - 1))..., seg] for seg in segments]
_get_ts_obs(data::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(_get_ts_obs, i), data)

Base.eltype(::SegmentedTimeSeries{D}) where D = Vector{D}