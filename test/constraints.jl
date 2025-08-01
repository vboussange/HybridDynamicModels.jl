using Test
using HybridModelling

@testset "BoxConstraint" begin
    # Scalar bounds
    bc = BoxConstraint(0.0, 1.0)
    x = [0.0, 0.5, 1.0]
    y = HybridModelling._to_optim_space(bc, x)
    x_recovered = HybridModelling._to_param_space(bc, y)
    @test isapprox(x, x_recovered; atol=1e-8)

    # Array bounds
    lb = [0.0, -1.0]
    ub = [1.0, 1.0]
    bc_arr = BoxConstraint(lb, ub)
    x_arr = [0.5, 0.0]
    y_arr = HybridModelling._to_optim_space(bc_arr, x_arr)
    x_arr_recovered = HybridModelling._to_param_space(bc_arr, y_arr)
    @test isapprox(x_arr, x_arr_recovered; atol=1e-8)

    # Test edge cases
    @test isapprox(HybridModelling._to_optim_space(bc, [0.0])[1], -Inf; atol=1e-8)
    @test isapprox(HybridModelling._to_optim_space(bc, [1.0])[1], Inf; atol=1e-8)
    @test isapprox(HybridModelling._to_param_space(bc, [-Inf])[1], 0.0; atol=1e-8)
    @test isapprox(HybridModelling._to_param_space(bc, [Inf])[1], 1.0; atol=1e-8)
end
