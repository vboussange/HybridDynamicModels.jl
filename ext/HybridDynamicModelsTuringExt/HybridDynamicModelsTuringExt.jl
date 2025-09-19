module HybridDynamicModelsTuringExt

import Turing
const DynamicPPL = Turing.DynamicPPL
import DynamicPPL: @varname, VarName
import Functors: @leaf, fmap_with_path, fmap
using ConcreteStructs: @concrete
using HybridDynamicModels: HybridDynamicModels,
                           SegmentedTimeSeries,
                           AbstractOptimBackend,
                           InferICs,
                           AbstractSetup
import Lux # for ForzenLayer
using Distributions
import LuxCore

include("bayesian_layer.jl")
include("mcsampling_backend.jl")
# include("vi_backend.jl")
end