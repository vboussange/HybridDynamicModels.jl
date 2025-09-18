module HybridDynamicModels

using LuxCore
using Random
using ConcreteStructs: @concrete
using DiffEqBase
using DispatchDoctor: @stable

export apply
export SegmentedTimeSeries, tokenize, tokens
export NoConstraint, BoxConstraint, NamedTupleConstraint
export ParameterLayer
export ICLayer, ODEModel, AnalyticModel, ARModel
export BayesianLayer, getpriors
export LogMSELoss, PoissonLoss

export train, InferICs

# This must be changed, with
ext = Base.get_extension(@__MODULE__, :HybridDynamicModelsTuringExt)
if !isnothing(ext)
    export ext: MCSamplingBackend
    export ext: VIBackend
end

ext = Base.get_extension(@__MODULE__, :HybridDynamicModelsLuxExt)
if !isnothing(ext)
    export ext: SGDBackend
end
# export SGDBackend, MCSamplingBackend, VIBackend

include("data.jl")
include("constraints.jl")

include("layers/utilities/parameter.jl")
include("layers/utilities/bayesian_layer.jl")

include("layers/models/model.jl")
include("layers/models/odemodel.jl")
include("layers/models/analyticmodel.jl")
include("layers/models/armodel.jl")

include("train.jl")

end
