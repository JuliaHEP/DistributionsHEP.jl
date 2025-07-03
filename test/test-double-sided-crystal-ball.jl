using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test

# Helper to check quantile accuracy for different σ values
function check_quantile_accuracy(d, ps; atol = 1e-8)
    for p in ps
        q = quantile(d, p)
        cdf_val = cdf(d, q)
        @test isapprox(cdf_val, p; atol = atol)
        if !isapprox(cdf_val, p; atol = atol)
            @warn "Quantile test failed" p q cdf_val
        end
    end
end

# Test distribution with σ = 1 (standard case)
d = DoubleCrystalBall(0.0, 1.0, 1.5, 2.0, 2.0, 3.0)

@testset "DoubleCrystalBall parameter validation" begin
    @test_throws ErrorException DoubleCrystalBall(0.0, -1.0, 1.0, 2.0, 1.0, 2.0)  # negative σ
    @test_throws ErrorException DoubleCrystalBall(0.0, 1.0, -1.0, 2.0, 1.0, 2.0)  # negative αL
    @test_throws ErrorException DoubleCrystalBall(0.0, 1.0, 1.0, 0.5, 1.0, 2.0)  # nL ≤ 1
    @test_throws ErrorException DoubleCrystalBall(0.0, 1.0, 1.0, 2.0, -1.0, 2.0)  # negative αR
    @test_throws ErrorException DoubleCrystalBall(0.0, 1.0, 1.0, 2.0, 1.0, 0.5)  # nR ≤ 1
end

@testset "DoubleCrystalBall PDF properties" begin
    # PDF should be positive and finite everywhere
    x_test_points = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    for x in x_test_points
        @test pdf(d, x) > 0
        @test isfinite(pdf(d, x))
    end

    # PDF should be continuous at transition points
    x_left_merge = -d.αL * d.σ + d.μ
    pdf_value_left = pdf(d, x_left_merge - 1e-6)
    pdf_value_right = pdf(d, x_left_merge + 1e-6)
    @test isapprox(pdf_value_left, pdf_value_right; atol = 1e-5)

    x_right_merge = d.αR * d.σ + d.μ
    pdf_value_left = pdf(d, x_right_merge - 1e-6)
    pdf_value_right = pdf(d, x_right_merge + 1e-6)
    @test isapprox(pdf_value_left, pdf_value_right; atol = 1e-5)

    # PDF should integrate to 1
    numerical_integral = quadgk(x -> pdf(d, x), -Inf, Inf)[1]
    @test isapprox(numerical_integral, 1.0; atol = 1e-6)
end

@testset "DoubleCrystalBall CDF properties" begin
    # CDF should be between 0 and 1
    x_values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    for x in x_values
        cdf_val = cdf(d, x)
        @test cdf_val >= 0
        @test cdf_val <= 1
    end

    # CDF should approach 0 and 1 at extremes
    @test cdf(d, -100) < 0.1
    @test cdf(d, d.μ + 5 * d.σ) > 0.99  # Should be very close to 1

    # CDF should be continuous at transition points
    x_left_merge = -d.αL * d.σ + d.μ
    cdf_value_left = cdf(d, x_left_merge - 1e-6)
    cdf_value_right = cdf(d, x_left_merge + 1e-6)
    @test isapprox(cdf_value_left, cdf_value_right; atol = 1e-5)

    x_right_merge = d.αR * d.σ + d.μ
    cdf_value_left = cdf(d, x_right_merge - 1e-6)
    cdf_value_right = cdf(d, x_right_merge + 1e-6)
    @test isapprox(cdf_value_left, cdf_value_right; atol = 1e-5)
end

@testset "DoubleCrystalBall quantile properties" begin
    # Quantile should be inverse of CDF in different regions
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

@testset "DoubleCrystalBall sigma scaling" begin
    # Test quantile accuracy for different σ values
    test_cases = [
        (0.0, 0.5, 0.5, 1.5, 3.0, 5.0),  # σ < 1
        (0.0, 1.0, 0.5, 1.5, 3.0, 5.0),  # σ = 1
        (0.0, 5.0, 0.5, 1.5, 3.0, 5.0),  # σ > 1
    ]

    ps = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]

    for (μ, σ, αL, nL, αR, nR) in test_cases
        d_test = DoubleCrystalBall(μ, σ, αL, nL, αR, nR)

        # Test quantile accuracy
        check_quantile_accuracy(d_test, ps)

        # Test CDF normalization (should approach 1 for large x)
        cdf_large = cdf(d_test, 1000)
        @test cdf_large > 0.999

        # Test PDF normalization via numerical integration
        # The PDF should integrate to 1 for all σ values
        numerical_integral = quadgk(x -> pdf(d_test, x), -Inf, Inf)[1]
        @test isapprox(numerical_integral, 1.0; atol = 1e-6) ||
              @warn "PDF normalization failed for σ = $σ" numerical_integral
    end
end

@testset "DoubleCrystalBall symmetry" begin
    # Test with symmetric parameters
    d_sym = DoubleCrystalBall(0.0, 1.0, 1.5, 2.0, 1.5, 2.0)

    # PDF should be symmetric around the mean
    x_test = 1.0
    @test isapprox(pdf(d_sym, x_test), pdf(d_sym, -x_test); atol = 1e-10)

    # CDF should satisfy: CDF(x) + CDF(-x) ≈ 1 for symmetric distribution
    @test isapprox(cdf(d_sym, x_test) + cdf(d_sym, -x_test), 1.0; atol = 1e-8)
end

@testset "DoubleCrystalBall type stability" begin
    d_float64 = DoubleCrystalBall(0.0, 1.0, 1.5, 2.0, 2.0, 3.0)
    d_float32 = DoubleCrystalBall(0.0f0, 1.0f0, 1.5f0, 2.0f0, 2.0f0, 3.0f0)

    @test pdf(d_float64, 0.0) isa Float64
    @test pdf(d_float32, 0.0f0) isa Float32
    @test cdf(d_float64, 0.0) isa Float64
    @test cdf(d_float32, 0.0f0) isa Float32
    @test quantile(d_float64, 0.5) isa Float64
    @test quantile(d_float32, 0.5f0) isa Float32
end

@testset "DoubleCrystalBall support interface" begin
    d_float64 = DoubleCrystalBall(0.0, 1.0, 1.5, 2.0, 2.0, 3.0)
    d_float32 = DoubleCrystalBall(0.0f0, 1.0f0, 1.5f0, 2.0f0, 2.0f0, 3.0f0)

    @test maximum(d_float64) == Inf
    @test maximum(d_float32) == Inf32
    @test minimum(d_float64) == -Inf
    @test minimum(d_float32) == -Inf32
    @test minimum(d_float64) == support(d_float64).lb
    @test maximum(d_float64) == support(d_float64).ub
end
