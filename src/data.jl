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
struct SegmentedTimeSeries{D, T, R<:AbstractRNG} # When iterated, returns (data, model_features) where model_features is a vector
    data::D # Original data, should be an array
    tsteps::T # tsteps associated with the data
    segmentsize::Int
    shift::Int
    batchsize::Int
    nsegments::Int
    shuffle::Bool
    partial::Bool
    indices::Vector{UnitRange{Int}} # indices of the segments
    imax::Int
    rng::R
end

function SegmentedTimeSeries(data, tsteps; segmentsize=2, shift=nothing, batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)
    @assert segmentsize > 1
    @assert batchsize > 0
    @assert _nobs(data) == length(tsteps)
    isnothing(shift) && (shift = segmentsize)
    datasize = _nobs(data)
    # Compute number of segments (M = floor((K - S) / R) + 1)
    nsegments = max(0, fld(datasize - segmentsize, shift) + 1)
    # Indices for each segment: {ymR : mR+S−1} for m in 0:M-1
    indices = [((m * shift + 1):(min(datasize, m * shift + segmentsize))) for m in 0:(nsegments-1)]
    imax = partial ? nsegments : nsegments - batchsize + 1
    return SegmentedTimeSeries(data, tsteps, segmentsize, shift, batchsize, nsegments, shuffle, partial, indices, imax, rng)
end

Base.@propagate_inbounds function Base.iterate(sdl::SegmentedTimeSeries, i=0)
    i >= sdl.imax && return nothing
    if sdl.shuffle && i == 0
        shuffle!(sdl.rng, sdl.indices)
    end
    nexti = min(i + sdl.batchsize, sdl.nsegments)
    segments = sdl.indices[i+1:nexti]
    batch_data = _get_ts_obs(sdl.data, segments)
    batch_tsteps = _get_ts_obs(sdl.tsteps, segments)
    return ((batch_data, batch_tsteps), nexti)
end

function Base.length(d::SegmentedTimeSeries)
    n = d.nsegments / d.batchsize
    d.partial ? ceil(Int, n) : floor(Int, n)
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

Base.eltype(::SegmentedTimeSeries{D, T}) where {D, T} = Tuple{Vector{D}, Vector{T}}