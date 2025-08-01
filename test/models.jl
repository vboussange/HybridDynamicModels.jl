using Bijectors, Distributions
using HybridModelling
using Lux
using Random

initial_ics = [(a = rand(3), b = randn(3)) for _ in 1:5] # a should be in [0, 1], b has no constraints
transform = Bijectors.NamedTransform((
    a = bijector(Uniform(0., 1.0)),
    b = identity)
)
constraint = Constraint(transform)

lics = LearnableICs(initial_ics, constraint)

ps, st = Lux.setup(Random.default_rng(), lics)

Lux.apply(lics, ps, st) # expected to work, returns all initial conditions

Lux.apply(lics, [1, 2],  ps, st) # expected to work, returns intitial conditions associated to indices