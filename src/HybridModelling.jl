module HybridModelling

import Lux
import Lux: LuxCore, IntegerType, AbstractLuxLayer, AbstractLuxWrapperLayer, AbstractLuxContainerLayer, StatefulLuxLayer, get_state, set_state!
using Random
using ConcreteStructs: @concrete
using OrdinaryDiffEq
import Optimisers
using DispatchDoctor: @stable

export apply
export SegmentedTimeSeries, tokenize, tokens
export Constraint, NoConstraint
export ParameterLayer
export InitialConditions, ODEModel, AnalyticModel, ARModel
export BayesianLayer, getpriors
export LogMSELoss

export train, InferICs
export LuxBackend, MCMCBackend, VIBackend

include("data.jl")
include("constraints.jl")
include("loss_fn.jl")

include("layers/utilities/parameter.jl")
include("layers/utilities/bayesian_layer.jl")

include("layers/models/model.jl")
include("layers/models/odemodel.jl")
include("layers/models/analyticmodel.jl")
include("layers/models/armodel.jl")

include("train/generics.jl")
include("train/lux_trainer.jl")
include("train/mcmc_trainer.jl")
include("train/vi_trainer.jl")

end
