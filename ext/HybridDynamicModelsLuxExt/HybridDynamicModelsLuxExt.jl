module HybridDynamicModelsLuxExt

import Lux
const Optimisers = Lux.Optimisers
import Lux: Training
using ADTypes
using ConcreteStructs: @concrete
import HybridDynamicModels: SegmentedTimeSeries

include("sgdbackend.jl")
# include("viluxbackend.jl")
include("loss_fn.jl")
end