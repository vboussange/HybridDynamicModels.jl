using Bijectors, Distributions
using HybridModelling
using Lux
using Random
using Test

######## Parameter tests ########
@testset "Parameter Layer Tests" begin
        
    # first test, no constraint
    param = Parameter(NoConstraint(), 
                        (;u0 = ones(10)),
                        (;tspan = (0.0, 1.0), saveat = range(0.0, stop=1.0, length=100)))

    ps, st = Lux.setup(Random.default_rng(), param)
    kwargs, _ = Lux.apply(param, ps, st)
    @test all(kwargs.u0 .≈ ones(10))

    # test with a constraint
    transform = Bijectors.NamedTransform((
        a = bijector(Uniform(0., 3.0)),
        b = identity)
    )
    constraint = Constraint(transform)
    param = Parameter(constraint, (;a = ones(3), b = randn(3)))

    ps, st = Lux.setup(Random.default_rng(), param)
    @test !any(ps.a .≈ ones(3)) # should be transformed
    kwargs, _ = Lux.apply(param, ps, st)
    @test all(kwargs.a .≈ ones(3))

    # overwriting the parameter; is there a case where this is desired? This is an alternative to freezing the parameter
    st = merge(st, (a = 3 * ones(3),)) # states are not transformed
    kwargs, _ = Lux.apply(param, ps, st)
    @test all(kwargs.a .≈ 3.)
end


######## Initial Conditions tests ########
@testset "Initial Conditions Layer Tests" begin
    lics = InitialConditions(Parameter((a = rand(3), b = randn(3)))) # initial conditions with no constraints
    ps, st = Lux.setup(Random.default_rng(), lics)
    u0, _, = Lux.apply(lics, ps, st) # expected to work, returns all initial conditions
    @test haskey(u0, :u0)

    initial_ics = [Parameter((u0 = rand(10),)) for _ in 1:5] # a should be in [0, 1], b has no constraints
    lics = InitialConditions(initial_ics)
    ps, st = Lux.setup(Random.default_rng(), lics)
    @test haskey(ps, :u0_1)
    u0, _ = Lux.apply(lics, 1,  ps, st) # expected to work, returns intitial conditions associated to indices
    @test haskey(u0, :u0)

    # Testing Chain
    many_ics = Chain(lics)
    ps, st = Lux.setup(Random.default_rng(), many_ics)
    u0, _ = Lux.apply(many_ics, 1, ps, st)
    @test haskey(u0, :u0)

    # testing Neural net
    initial_ics = Dense(1, 10)
    lics = InitialConditions(initial_ics)
    ps, st = Lux.setup(Random.default_rng(), lics)
    u0, _ = Lux.apply(lics, [1.],  ps, st) # expected to work, returns intitial conditions associated to indices
    @test haskey(u0, :u0)
end

######### ODEModel tests ########
using OrdinaryDiffEq
components = (; layer1 = Lux.Dense(10, 10, relu))
dudt(components, ps, st, u, t) = components.layer1(u, ps.layer1, st.layer1)[1]
ode_model = ODEModel(components, dudt, tspan = (0.0, 1.0), saveat = range(0.0, stop=1.0, length=100), alg = Tsit5())


# Neural net as initial conditions
initial_ics = Dense(1, 10)
lics = InitialConditions(initial_ics)

rng = MersenneTwister(1234)
model_with_ics = Chain(lics, ode_model)

ps, st = Lux.setup(rng, model_with_ics)
model_with_ics([1.], ps, st)


# multiple initial conditions
initial_ics = [Parameter(NoConstraint(), (;u0 = rand(10)), (;saveat= range(0.0, stop=1.0, length=i+1))) for i in 1:5]
lics = InitialConditions(initial_ics)
model_with_ics = Chain(lics, ode_model)
ps, st = Lux.setup(rng, model_with_ics)
x, _ = model_with_ics(3, ps, st)
length(x.u) == 4