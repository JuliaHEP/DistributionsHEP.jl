using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test

# Test distribution with different parameters for left and right tails
d = DoubleCrystalBall(0.0, 1.0, 1.5, 2.0, 2.0, 3.0)

@testset "DoubleCrystalBall parameter validation" begin
    @test_throws ErrorException DoubleCrystalBall(0.0, -1.0, 1.0, 2.0, 1.0, 2.0)  # negative σ
    @test_throws ErrorException DoubleCrystalBall(0.0, 1.0, -1.0, 2.0, 1.0, 2.0)  # negative αL
    @test_throws ErrorException DoubleCrystalBall(0.0, 1.0, 1.0, 0.5, 1.0, 2.0)  # nL ≤ 1
    @test_throws ErrorException DoubleCrystalBall(0.0, 1.0, 1.0, 2.0, -1.0, 2.0)  # negative αR
    @test_throws ErrorException DoubleCrystalBall(0.0, 1.0, 1.0, 2.0, 1.0, 0.5)  # nR ≤ 1
end

@testset "DoubleCrystalBall PDF continuity at transition points" begin
    # Left transition point
    x_left_merge = -d.αL * d.σ + d.μ
    pdf_value_left = pdf(d, x_left_merge - 1e-6)  # just below the transition point
    pdf_value_right = pdf(d, x_left_merge + 1e-6) # just above the transition point
    @test isapprox(pdf_value_left, pdf_value_right; atol = 1e-5)

    # Right transition point
    x_right_merge = d.αR * d.σ + d.μ
    pdf_value_left = pdf(d, x_right_merge - 1e-6)  # just below the transition point
    pdf_value_right = pdf(d, x_right_merge + 1e-6) # just above the transition point
    @test isapprox(pdf_value_left, pdf_value_right; atol = 1e-5)
end

@testset "DoubleCrystalBall PDF properties" begin
    # PDF should be positive everywhere
    x_test_points = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    for x in x_test_points
        @test pdf(d, x) > 0
    end

    # PDF should be finite everywhere
    for x in x_test_points
        @test isfinite(pdf(d, x))
    end
end

@testset "DoubleCrystalBall normalization" begin
    # Check that the integral of the PDF equals 1 (with reasonable tolerance)
    numerical_integral = quadgk(x -> pdf(d, x), -Inf, Inf)[1]
    @test isapprox(numerical_integral, 1.0; atol = 1e-6)
end

@testset "DoubleCrystalBall CDF continuity at transition points" begin
    # Left transition point
    x_left_merge = -d.αL * d.σ + d.μ
    cdf_value_left = cdf(d, x_left_merge - 1e-6)
    cdf_value_right = cdf(d, x_left_merge + 1e-6)
    @test isapprox(cdf_value_left, cdf_value_right; atol = 1e-5)
    @test cdf_value_left < cdf_value_right

    # Right transition point
    x_right_merge = d.αR * d.σ + d.μ
    cdf_value_left = cdf(d, x_right_merge - 1e-6)
    cdf_value_right = cdf(d, x_right_merge + 1e-6)
    @test isapprox(cdf_value_left, cdf_value_right; atol = 1e-5)
    @test cdf_value_left < cdf_value_right
end

@testset "DoubleCrystalBall CDF basic properties" begin
    # CDF should be between 0 and 1
    x_values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    for x in x_values
        cdf_val = cdf(d, x)
        @test cdf_val >= 0
        @test cdf_val <= 1
    end
end

@testset "DoubleCrystalBall quantile basic properties" begin
    # Test quantile in different regions
    x_left = -2.0
    p_left = cdf(d, x_left)
    @test quantile(d, p_left) ≈ x_left atol = 1e-9

    x_core = 0.5
    p_core = cdf(d, x_core)
    @test quantile(d, p_core) ≈ x_core atol = 1e-9

    x_right = 3.0
    p_right = cdf(d, x_right)
    @test quantile(d, p_right) ≈ x_right atol = 1e-9

    # Test edge cases
    @test quantile(d, 0.0) == -Inf
    @test quantile(d, 1.0) == Inf
    @test_throws DomainError quantile(d, -0.1)
    @test_throws DomainError quantile(d, 1.1)
end

@testset "DoubleCrystalBall symmetry test" begin
    # Test with symmetric parameters
    d_sym = DoubleCrystalBall(0.0, 1.0, 1.5, 2.0, 1.5, 2.0)

    # PDF should be symmetric around the mean
    x_test = 1.0
    @test isapprox(pdf(d_sym, x_test), pdf(d_sym, -x_test); atol = 1e-10)

    # CDF should satisfy: CDF(x) + CDF(-x) ≈ 1 for symmetric distribution
    @test isapprox(cdf(d_sym, x_test) + cdf(d_sym, -x_test), 1.0; atol = 1e-8)
end

@testset "DoubleCrystalBall type stability" begin
    # Test that the distribution works with different numeric types
    d_float32 = DoubleCrystalBall(0.0f0, 1.0f0, 1.5f0, 2.0f0, 2.0f0, 3.0f0)
    @test pdf(d_float32, 0.0f0) isa Float32
    @test cdf(d_float32, 0.0f0) isa Float32
    @test quantile(d_float32, 0.5f0) isa Float32
end

d_float64 = DoubleCrystalBall(0.0, 1.0, 1.5, 2.0, 2.0, 3.0)
d_float32 = DoubleCrystalBall(0.0f0, 1.0f0, 1.5f0, 2.0f0, 2.0f0, 3.0f0)
@testset "DoubleCrystalBall maximum/minimum interfaces" begin
    @test maximum(d_float64) == Inf
    @test maximum(d_float32) == Inf32
    @test minimum(d_float64) == -Inf
    @test minimum(d_float32) == -Inf32
    @test minimum(d_float64) == support(d_float64).lb
    @test maximum(d_float64) == support(d_float64).ub
end
