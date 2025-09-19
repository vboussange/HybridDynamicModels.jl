module HybridDynamicModelsTuringExt

using Turing: Turing, arraydist
const DynamicPPL = Turing.DynamicPPL
import DynamicPPL: @varname, VarName
import Functors: @leaf, fmap_with_path, fmap
using ConcreteStructs: @concrete
using HybridDynamicModels: HybridDynamicModels,
                           SegmentedTimeSeries,
                           InferICs,
                           tokenize,
                           tokens,
                           ParameterLayer,
                           ICLayer,
                           is_ics_estimated
import Lux: Lux, Chain, Training
using Distributions
import LuxCore
using ComponentArrays
import Random

include("bayesian_layer.jl")
include("mcsampling_backend.jl")
# include("vi_backend.jl")
end