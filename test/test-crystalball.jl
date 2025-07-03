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
        @test isapprox(cdf_val, p; atol = atol) || @warn "Quantile test failed" p q cdf_val
    end
end

# Test distribution with σ = 1 (standard case)
d = CrystalBall(0.0, 1.0, 1.0, 1.6)

@testset "CrystalBall parameter validation" begin
    @test_throws ErrorException CrystalBall(0.0, -1.0, 1.0, 1.6)  # negative σ
    @test_throws ErrorException CrystalBall(0.0, 1.0, -1.0, 1.6)  # negative α
    @test_throws ErrorException CrystalBall(0.0, 1.0, 1.0, 0.5)   # n ≤ 1
end

@testset "CrystalBall PDF properties" begin
    # PDF should be positive and finite everywhere
    x_test_points = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    for x in x_test_points
        @test pdf(d, x) > 0
        @test isfinite(pdf(d, x))
    end

    # PDF should be continuous at transition point
    x_merge = -d.α * d.σ + d.μ
    pdf_value_left = pdf(d, x_merge + 1e-5)  # just above the transition point
    pdf_value_right = pdf(d, x_merge - 1e-5) # just below the transition point
    @test isapprox(pdf_value_left, pdf_value_right; atol = 1e-5)

    # PDF should integrate to 1
    numerical_integral = quadgk(x -> pdf(d, x), -Inf, Inf)[1]
    @test isapprox(numerical_integral, 1.0; atol = 1e-7)
end

@testset "CrystalBall CDF properties" begin
    # CDF should be between 0 and 1
    x_values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    for x in x_values
        cdf_val = cdf(d, x)
        @test cdf_val >= 0
        @test cdf_val <= 1
    end

    # CDF should approach 0 and 1 at extremes
    @test cdf(d, -100) < 0.1
    @test isapprox(cdf(d, d.μ + 5 * d.σ), 1; atol = 1e-5)

    # CDF should be continuous at transition point
    x_merge = -d.α * d.σ + d.μ
    cdf_value_left = cdf(d, x_merge + 1e-6)
    cdf_value_right = cdf(d, x_merge - 1e-6)
    @test isapprox(cdf_value_left, cdf_value_right; atol = 1e-5)
end

@testset "CrystalBall quantile properties" begin
    # Quantile should be inverse of CDF
    x_test = 0.9
    p_test = cdf(d, x_test)
    @test quantile(d, p_test) ≈ x_test atol = 1e-9

    # Test edge cases
    @test quantile(d, 0.0) == -Inf
    @test quantile(d, 1.0) == Inf
    @test_throws DomainError quantile(d, -0.1)
    @test_throws DomainError quantile(d, 1.1)
end

@testset "CrystalBall sigma scaling" begin
    # Test quantile accuracy for different σ values
    test_cases = [
        (0.0, 0.5, 2.0, 3.2),  # σ < 1
        (0.0, 1.0, 2.0, 3.2),  # σ = 1
        (0.0, 5.0, 2.0, 3.2),  # σ > 1
    ]

    ps = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]

    for (μ, σ, α, n) in test_cases
        d_test = CrystalBall(μ, σ, α, n)

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

@testset "CrystalBall type stability" begin
    d_float64 = CrystalBall(0.0, 1.0, 2.0, 3.0)
    d_float32 = CrystalBall(0.0f0, 1.0f0, 2.0f0, 3.0f0)

    @test pdf(d_float64, 0.0) isa Float64
    @test pdf(d_float32, 0.0f0) isa Float32
    @test cdf(d_float64, 0.0) isa Float64
    @test cdf(d_float32, 0.0f0) isa Float32
    @test quantile(d_float64, 0.5) isa Float64
    @test quantile(d_float32, 0.5f0) isa Float32
end

@testset "CrystalBall support interface" begin
    d_float64 = CrystalBall(0.0, 1.0, 2.0, 3.0)
    d_float32 = CrystalBall(0.0f0, 1.0f0, 2.0f0, 3.0f0)

    @test maximum(d_float64) == Inf
    @test maximum(d_float32) == Inf32
    @test minimum(d_float64) == -Inf
    @test minimum(d_float32) == -Inf32
    @test minimum(d_float64) == support(d_float64).lb
    @test maximum(d_float64) == support(d_float64).ub
end
