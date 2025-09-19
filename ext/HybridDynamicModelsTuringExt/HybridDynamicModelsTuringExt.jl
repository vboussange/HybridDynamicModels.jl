module HybridDynamicModelsTuringExt

using Turing: Turing, arraydist
const DynamicPPL = Turing.DynamicPPL
const Distributions = Turing.Distributions

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
import LuxCore
using ComponentArrays
import Random

include("bayesian_layer.jl")
include("mcsampling_backend.jl")
# include("vi_backend.jl")
end