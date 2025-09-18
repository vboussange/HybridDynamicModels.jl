module HybridDynamicModelsLuxExt
import LuxCore
import Lux
import Optimisers
import Lux: Training
using ADTypes
using ConcreteStructs: @concrete
import HybridDynamicModels: SegmentedTimeSeries, AbstractOptimBackend, InferICs, AbstractSetup

include("sgd_backend.jl")
include("loss_fn.jl")
end