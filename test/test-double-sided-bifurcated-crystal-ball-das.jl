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
d = DoubleSidedBifurcatedCrystalBallDas(0.0, 1.0, 0.25, 0.5, 1.25, 0.75)  # μ, σ, ψ, αL, nL, αR

@testset "DoubleSidedBifurcatedCrystalBallDas Distribution" verbose=true begin
@testset "Parameter validation" begin
    @test_throws ErrorException DoubleSidedBifurcatedCrystalBallDas(0.0, -1.0, 0.25, 0.5, 1.25, 0.75)  # negative σ
    @test_throws ErrorException DoubleSidedBifurcatedCrystalBallDas(0.0, 1.0, 0.25, -0.5, 1.25, 0.75)  # negative αL
    @test_throws ErrorException DoubleSidedBifurcatedCrystalBallDas(0.0, 1.0, 0.25, 0.5, -1.25, 0.75)  # negative nL
    @test_throws ErrorException DoubleSidedBifurcatedCrystalBallDas(0.0, 1.0, 0.25, 0.5, 1.25, -0.75)  # negative αR
end

@testset "PDF properties" begin
    # PDF should be positive and finite everywhere
    x_test_points = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    for x in x_test_points
        @test pdf(d, x) > 0
        @test isfinite(pdf(d, x))
    end

    # PDF should be continuous at transition points
    pdf_x_left_merge = d.left_tail.x0
    pdf_value_left1 = pdf(d, pdf_x_left_merge - 1e-6)
    pdf_value_right1 = pdf(d, pdf_x_left_merge + 1e-6)

    pdf_x_right_merge = d.right_tail.x0
    pdf_value_left2 = pdf(d, pdf_x_right_merge - 1e-6)
    pdf_value_right2 = pdf(d, pdf_x_right_merge + 1e-6)
    @test isapprox(pdf_value_left1, pdf_value_right1; atol = 1e-5)
    @test isapprox(pdf_value_left2, pdf_value_right2; atol = 1e-5)

    # PDF should integrate to 1
    numerical_integral = quadgk(x -> pdf(d, x), -Inf, Inf)[1]
    @test isapprox(numerical_integral, 1.0; atol = 1e-6)
end

@testset "CDF properties" begin
    # CDF should be between 0 and 1
    x_values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    for x in x_values
        cdf_val = cdf(d, x)
        @test cdf_val >= 0
        @test cdf_val <= 1
    end

    # CDF should approach 0 and 1 at extremes
    @test cdf(d, -1000.0) < 0.1  # with das, normalization is dominated by exponential right tail
    @test cdf(d, d.BifGauss.μ + 100.0 * d.BifGauss.σ) > 0.99  # Should be very close to 1

    # CDF should be continuous at transition point
    x_left_merge = d.left_tail.x0
    cdf_value_left1 = cdf(d, x_left_merge - 1e-6)
    cdf_value_right1 = cdf(d, x_left_merge + 1e-6)

    x_mu = d.BifGauss.μ
    cdf_value_left = cdf(d, x_mu - 1e-6)
    cdf_value_right = cdf(d, x_mu + 1e-6)

    x_right_merge = d.right_tail.x0
    cdf_value_left2 = cdf(d, x_right_merge - 1e-6)
    cdf_value_right2 = cdf(d, x_right_merge + 1e-6)
    @test isapprox(cdf_value_left1, cdf_value_right1; atol = 1e-5)
    @test isapprox(cdf_value_left2, cdf_value_right2; atol = 1e-5)
    @test isapprox(cdf_value_left, cdf_value_right; atol = 1e-5)
end

@testset "Quantile properties" begin
    # Quantile should be inverse of CDF in different regions
    x_left = -2.0
    p_left = cdf(d, x_left)
    @test quantile(d, p_left) ≈ x_left atol = 1e-9

    x_core = 0.1
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

@testset "Sigma scaling" begin
    # Test quantile accuracy for different σ values
    test_cases = [
        (0.0, 0.5, 0.25, 0.5, 1.25, 0.75),  # σ < 1
        (0.0, 1.0, 0.25, 0.5, 1.25, 0.75),  # σ = 1
        (0.0, 5.0, 0.25, 0.5, 1.25, 0.75),  # σ > 1
    ]

    ps = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]

    for (μ, σ, ψ, αL, nL, αR) in test_cases
        d_test = DoubleSidedBifurcatedCrystalBallDas(μ, σ, ψ, αL, nL, αR)

        # Test quantile accuracy
        check_quantile_accuracy(d_test, ps)

        # Test CDF normalization (should approach 1 for large x)
        cdf_large = cdf(d_test, 1000.0)
        @test cdf_large > 0.99

        # Test PDF normalization via numerical integration
        # The PDF should integrate to 1 for all σ values
        numerical_integral = quadgk(x -> pdf(d_test, x), -Inf, Inf)[1]
        @test isapprox(numerical_integral, 1.0; atol = 1e-6) ||
              @warn "PDF normalization failed for σ = $σ" numerical_integral
    end
end

# Symmetry not tested as DoubleSidedBifurcatedCrystalBallDas is inherently asymmetric

@testset "Type stability" begin
    d_float64 = DoubleSidedBifurcatedCrystalBallDas(0.0, 1.0, 0.25, 0.5, 1.25, 0.75)
    d_float32 = DoubleSidedBifurcatedCrystalBallDas(0.0f0, 1.0f0, 0.25f0, 0.5f0, 1.25f0, 0.75f0)

    @test pdf(d_float64, 0.0) isa Float64
    @test pdf(d_float32, 0.0f0) isa Float32
    @test cdf(d_float64, 0.0) isa Float64
    @test cdf(d_float32, 0.0f0) isa Float32
    @test quantile(d_float64, 0.5) isa Float64
    @test quantile(d_float32, 0.5f0) isa Float32
end

@testset "Support interface" begin
    d_float64 = DoubleSidedBifurcatedCrystalBallDas(0.0, 1.0, 0.25, 0.5, 1.25, 0.75)
    d_float32 = DoubleSidedBifurcatedCrystalBallDas(0.0f0, 1.0f0, 0.25f0, 0.5f0, 1.25f0, 0.75f0)

    @test maximum(d_float64) == Inf
    @test maximum(d_float32) == Inf32
    @test minimum(d_float64) == -Inf
    @test minimum(d_float32) == -Inf32
    @test minimum(d_float64) == support(d_float64).lb
    @test maximum(d_float64) == support(d_float64).ub
end
end