module HybridDynamicModels

using LuxCore
import LuxCore.StatefulLuxLayerImpl: get_state
using Random
using ConcreteStructs: @concrete
using DiffEqBase
using DispatchDoctor: @stable

include("data.jl")
include("constraints.jl")

include("layers/generics.jl")

include("layers/utilities/parameter.jl")
include("layers/utilities/initialconditions.jl")
include("layers/utilities/bayesian_layer.jl")

include("layers/models/odemodel.jl")
include("layers/models/analyticmodel.jl")
include("layers/models/armodel.jl")

include("train.jl")


export apply
export SegmentedTimeSeries, tokenize, tokens
export NoConstraint, BoxConstraint, NamedTupleConstraint
export ParameterLayer, ICLayer
export ICLayer, ODEModel, AnalyticModel, ARModel
export BayesianLayer, getpriors
export LogMSELoss, PoissonLoss

export train, InferICs

export SGDBackend
export MCSamplingBackend
export VIBackend

end
