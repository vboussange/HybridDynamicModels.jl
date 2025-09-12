module HybridModelling

using LuxCore
using Random
using ConcreteStructs: @concrete
using DiffEqBase
using DispatchDoctor: @stable

export apply
export SegmentedTimeSeries, tokenize, tokens
export NoConstraint, BoxConstraint, NamedTupleConstraint
export ParameterLayer
export InitialConditions, ODEModel, AnalyticModel, ARModel
export BayesianLayer, getpriors
export LogMSELoss, PoissonLoss

export train, InferICs

# This must be changed, with
ext = Base.get_extension(@__MODULE__, :HybridModellingTuringExt)
if !isnothing(ext)
    export ext: MCMCBackend
    export ext: VIBackend
end

ext = Base.get_extension(@__MODULE__, :HybridModellingLuxExt)
if !isnothing(ext)
    export ext: SGDBackend
end
# export SGDBackend, MCMCBackend, VIBackend

include("data.jl")
include("constraints.jl")
include("loss_fn.jl")

include("layers/utilities/parameter.jl")
include("layers/utilities/bayesian_layer.jl")

include("layers/models/model.jl")
include("layers/models/odemodel.jl")
include("layers/models/analyticmodel.jl")
include("layers/models/armodel.jl")

include("train.jl")

end
