module HybridModelling

# using Lux
using Random
using ConcreteStructs: @concrete

# Write your package code here.
export SegmentedTimeSeries, ModelFeatures, LearnableICs, initialparameters

include("data.jl")

end
