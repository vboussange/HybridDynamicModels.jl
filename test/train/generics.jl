using HybridDynamicModels
using Test
using Distributions

@testset "Training Generics Tests" begin
    
    @testset "InferICs Tests" begin
        # Test basic construction
        infer_true = InferICs(true)
        infer_false = InferICs(false)
        
        @test infer_true isa InferICs{true}
        @test infer_false isa InferICs{false}
        @test infer_true.u0_constraint isa NoConstraint
        @test infer_false.u0_constraint isa NoConstraint
        
        # Test with constraints
        transform = bijector(Uniform(0.0, 10.0))
        constraint = Constraint(transform)
        infer_constrained = InferICs(true, constraint)
        
        @test infer_constrained.u0_constraint === constraint
        @test infer_constrained isa InferICs{true}
        
        # Test istrue function
        @test istrue(infer_true) == true
        @test istrue(infer_false) == false
        @test istrue(infer_constrained) == true
        
        # Test type stability
        @test typeof(infer_true) == InferICs{true, NoConstraint}
        @test typeof(infer_false) == InferICs{false, NoConstraint}
        @test typeof(infer_constrained) == InferICs{true, Constraint{typeof(transform)}}
    end
    
    @testset "InferICs with Different Constraints" begin
        # Test with various constraint types
        
        # Box constraint
        box_transform = bijector(Uniform(0.0, 5.0))
        box_constraint = Constraint(box_transform)
        infer_box = InferICs(true, box_constraint)
        @test istrue(infer_box) == true
        @test infer_box.u0_constraint isa Constraint
        
        # Named transform (more complex)
        named_transform = Bijectors.NamedTransform((
            a = bijector(Uniform(0.0, 1.0)),
            b = bijector(Uniform(-1.0, 1.0))
        ))
        named_constraint = Constraint(named_transform)
        infer_named = InferICs(true, named_constraint)
        @test istrue(infer_named) == true
        @test infer_named.u0_constraint isa Constraint
        
        # No constraint (explicit)
        no_constraint = NoConstraint()
        infer_none = InferICs(false, no_constraint)
        @test istrue(infer_none) == false
        @test infer_none.u0_constraint isa NoConstraint
    end
    
    @testset "InferICs Type Behavior" begin
        # Test that type parameters work correctly
        infer1 = InferICs(true)
        infer2 = InferICs(true)
        infer3 = InferICs(false)
        
        @test typeof(infer1) == typeof(infer2)
        @test typeof(infer1) != typeof(infer3)
        
        # Test that the boolean value is encoded in the type
        @test istrue(infer1) == istrue(infer2)
        @test istrue(infer1) != istrue(infer3)
        
        # Test with different constraint types
        constraint1 = NoConstraint()
        constraint2 = Constraint(identity)
        
        infer_c1 = InferICs(true, constraint1)
        infer_c2 = InferICs(true, constraint2)
        
        @test typeof(infer_c1) != typeof(infer_c2)  # Different constraint types
        @test istrue(infer_c1) == istrue(infer_c2)  # Same inference behavior
    end
    
    @testset "AbstractOptimBackend Interface" begin
        # Test that AbstractOptimBackend is properly defined
        @test AbstractOptimBackend isa Type
        @test AbstractOptimBackend <: Any
        
        # Test that our backends are subtypes
        @test SGDBackend <: AbstractOptimBackend
        @test MCSamplingBackend <: AbstractOptimBackend
        @test VIBackend <: AbstractOptimBackend
        
        # Test that we can create instances
        lux_backend = SGDBackend(Adam(1e-3), 100, AutoZygote(), MSELoss())
        mcmc_backend = MCSamplingBackend(NUTS(0.65), 100, Normal)
        vi_backend = VIBackend()
        
        @test lux_backend isa AbstractOptimBackend
        @test mcmc_backend isa AbstractOptimBackend
        @test vi_backend isa AbstractOptimBackend
    end
    
    @testset "Train Function Interface" begin
        # Test that train function exists with correct methods
        @test hasmethod(train, (AbstractOptimBackend, Any, SegmentedTimeSeries, InferICs))
        @test hasmethod(train, (SGDBackend, AbstractLuxLayer, SegmentedTimeSeries, InferICs, AbstractRNG))
        @test hasmethod(train, (MCSamplingBackend, AbstractLuxLayer, SegmentedTimeSeries, InferICs, AbstractRNG))
        @test hasmethod(train, (VIBackend, AbstractLuxLayer, SegmentedTimeSeries, InferICs))
        
        # Test that train is a generic function
        @test train isa Function
        
        # Test method counting (should have multiple methods)
        methods_count = length(methods(train))
        @test methods_count >= 3  # At least one for each backend type
    end
    
    @testset "Integration with Constraints" begin
        # Test that InferICs works properly with the constraint system
        
        # Create a constraint that should work with parameters
        test_params = (u0 = [1.0, 2.0],)
        transform = Bijectors.NamedTransform((u0 = bijector(Uniform(0.0, 10.0)),))
        constraint = Constraint(transform)
        
        # Test that constraint can be applied
        transformed = HybridDynamicModels._to_optim_space(constraint, test_params)
        @test transformed isa NamedTuple
        @test haskey(transformed, :u0)
        
        # Test round-trip
        recovered = constraint(transformed)
        @test recovered isa NamedTuple
        @test haskey(recovered, :u0)
        @test all(recovered.u0 .≈ test_params.u0)
        
        # Test with InferICs
        infer_with_constraint = InferICs(true, constraint)
        @test infer_with_constraint.u0_constraint === constraint
        @test istrue(infer_with_constraint) == true
    end
    
    @testset "Edge Cases and Error Handling" begin
        # Test with extreme values
        infer_extreme = InferICs(true, NoConstraint())
        @test istrue(infer_extreme) == true
        
        # Test type consistency
        T1 = InferICs{true, NoConstraint}
        T2 = InferICs{false, NoConstraint}
        @test T1 != T2
        
        # Test that constraint field is accessible
        infer_test = InferICs(false)
        @test hasfield(typeof(infer_test), :u0_constraint)
        @test infer_test.u0_constraint isa NoConstraint
    end
end
