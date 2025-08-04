module HybridModelling

using Lux
import Lux: IntegerType
using Random
using ConcreteStructs: @concrete
using OrdinaryDiffEq
using Random

export SegmentedTimeSeries
export Constraint, NoConstraint
export Parameter, InitialConditions, ODEModel

include("data.jl")
include("constraints.jl")
include("models.jl")

end
