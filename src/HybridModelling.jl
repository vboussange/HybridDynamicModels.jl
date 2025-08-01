module HybridModelling

using Lux
using Random
using ConcreteStructs: @concrete

export SegmentedTimeSeries, ModelFeatures, LearnableICs, initialparameters
export BoxConstraint
include("data.jl")
include("constraints.jl")

end
