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
export InitialConditions, ODEModel
export BayesianLayer, getpriors
export LogMSELoss

export train, InferICs
export LuxBackend, MCMCBackend, VIBackend

include("data.jl")
include("constraints.jl")
include("layers/parameter.jl")
include("layers/odemodel.jl")
include("layers/bayesian_layer.jl")
include("loss_fn.jl")

include("train/lux_trainer.jl")
include("train/mcmc_trainer.jl")
include("train/vi_trainer.jl")

end
