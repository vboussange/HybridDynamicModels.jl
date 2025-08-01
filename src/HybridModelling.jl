module HybridModelling

using Lux
using Random
using ConcreteStructs: @concrete

export SegmentedTimeSeries, ModelFeatures, LearnableICs, initialparameters
export Constraint, NoConstraint
export LearnableICs

include("data.jl")
include("constraints.jl")
include("models.jl")

end
