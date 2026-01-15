using Test
using DistributionsHEP

# Access internal functions via the module
_integral = DistributionsHEP._integral
_integral_inversion = DistributionsHEP._integral_inversion
_compute_standard_tail_constants = DistributionsHEP._compute_standard_tail_constants

@testset "CrystalBallTail integral functions" verbose = true begin

    # Create a left tail (L_x0 > 0)
    μ = 0.0
    σ = 1.0
    α = 2.0  # positive, so L_x0 = α > 0
    n = 3.2
    tail = _compute_standard_tail_constants(μ, σ, α, n)

    @test tail.L_x0 > 0  # Verify it's a left tail

    # Test 1: _integral_inversion(_integral(t, a)) ≈ a
    @testset "Round-trip: integral then inversion" begin
        # For left tail, a must be < x0 (transition point)
        # x0 = μ - α*σ = 0 - 2*1 = -2.0
        # So we test points less than -2.0
        test_points = [-5.0, -4.0, -3.0, -2.5, -2.1]
        for a in test_points
            integral_val = _integral(tail, a)
            a_recovered = _integral_inversion(tail, integral_val)
            @test isapprox(a_recovered, a; atol=1e-10, rtol=1e-10)
        end
    end

    # Test 2: _integral(t, _integral_inversion(integral)) ≈ integral
    @testset "Round-trip: inversion then integral" begin
        # Test with various integral values (should be in [0, const_tail])
        const_tail = DistributionsHEP._norm_const(tail)
        test_integrals = [0.01 * const_tail, 0.1 * const_tail, 0.3 * const_tail,
            0.5 * const_tail, 0.7 * const_tail, 0.9 * const_tail,
            0.99 * const_tail]

        for integral_val in test_integrals
            a = _integral_inversion(tail, integral_val)
            integral_recovered = _integral(tail, a)
            @test isapprox(integral_recovered, integral_val; atol=1e-10, rtol=1e-10)
        end
    end

    # Test 3: Right tail behavior (L < 0)
    @testset "Right tail (L < 0) behavior" begin
        # Create a right tail (L_x0 < 0)
        right_tail = DistributionsHEP.CrystalBallTail(exp(-4.0 / 2), 3.2, -2.0, 2.0)  # L_x0 = -2.0 < 0

        # For right tail, _integral should work (returns -integral from [a, +Inf])
        # Test with points a > x0 (valid for right tail)
        test_points = [3.0, 5.0, 10.0]
        for a in test_points
            integral_val = _integral(right_tail, a)
            # Should be negative (since it's -integral from [a, +Inf])
            @test integral_val < 0
        end

        # Test round-trip for right tail with negative values
        @testset "Right tail round-trip" begin
            for a in test_points
                integral_val = _integral(right_tail, a)
                a_recovered = _integral_inversion(right_tail, integral_val)
                @test isapprox(a_recovered, a; atol=1e-10, rtol=1e-10)
            end
        end
    end
end
