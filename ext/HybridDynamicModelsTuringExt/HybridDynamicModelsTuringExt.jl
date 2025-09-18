module HybridDynamicModelsTuringExt
import Turing
import Turing: NUTS, sample, Chains, arraydist, q_meanfield_gaussian
const DynamicPPL = Turing.DynamicPPL
import DynamicPPL: @varname, VarName
using Distributions
import Functors: @leaf, fmap_with_path, fmap
using ConcreteStructs: @concrete
import HybridDynamicModels: SegmentedTimeSeries, AbstractOptimBackend, InferICs, AbstractSetup
import Lux # for ForzenLayer

include("bayesian_layer.jl")
include("mcsampling_backend.jl")
# include("vi_backend.jl")
end