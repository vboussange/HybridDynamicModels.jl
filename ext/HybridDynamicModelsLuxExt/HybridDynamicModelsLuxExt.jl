module HybridDynamicModelsLuxExt
import LuxCore
import Lux
import Optimisers
import Lux: Training
using ADTypes
using ConcreteStructs: @concrete
using HybridDynamicModels: HybridDynamicModels, 
                            SegmentedTimeSeries, 
                            AbstractOptimBackend,
                            InferICs

include("sgd_backend.jl")
include("loss_fn.jl")
end