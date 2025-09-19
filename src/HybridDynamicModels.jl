module HybridDynamicModels

using LuxCore
import LuxCore.StatefulLuxLayerImpl: get_state, set_state!
using Random
using ConcreteStructs: @concrete
using DiffEqBase
using DispatchDoctor: @stable
using Functors: @functor

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


export SegmentedTimeSeries, tokenize, tokens, create_train_val_loaders
export ParameterLayer
export NoConstraint, BoxConstraint, NamedTupleConstraint
export ODEModel, AnalyticModel, ARModel
export ICLayer

export train, InferICs

export SGDBackend

export MCSamplingBackend
export BayesianLayer, getpriors
# export VIBackend

end
