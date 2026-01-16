using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test

# Test distribution with σ = 1 (standard case)
d = BifurcatedGaussian(0.0, 1.0, 0.5)  # μ, σ, ψ

@testset "BifurcatedGaussian Distribution" verbose = true begin
    @testset "Parameter validation" begin
        @test_throws ErrorException BifurcatedGaussian(0.0, -1.0, 0.5)  # negative σ
    end

    @testset "PDF properties" begin
        # PDF should be positive and finite everywhere
        x_test_points = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
        for x in x_test_points
            @test pdf(d, x) > 0
            @test isfinite(pdf(d, x))
        end

        # PDF should be continuous at transition points
        x_left_merge = d.μ
        pdf_value_left = pdf(d, x_left_merge - 1e-6)
        pdf_value_right = pdf(d, x_left_merge + 1e-6)
        @test isapprox(pdf_value_left, pdf_value_right; atol=1e-5)

        # PDF should integrate to 1
        numerical_integral = quadgk(x -> pdf(d, x), -Inf, Inf)[1]
        @test isapprox(numerical_integral, 1.0; atol=1e-6)
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
        @test cdf(d, -100) < 0.1
        @test cdf(d, d.μ + 5 * d.σ) > 0.99  # Should be very close to 1

        # CDF should be continuous at transition point
        x_left_merge = d.μ
        cdf_value_left = cdf(d, x_left_merge - 1e-6)
        cdf_value_right = cdf(d, x_left_merge + 1e-6)
        @test isapprox(cdf_value_left, cdf_value_right; atol=1e-5)
    end

    @testset "Quantile properties" begin
        # Test CDF and quantile are inverse functions (both directions)
        # Direction 1: quantile(cdf(x)) ≈ x
        for x in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
            p = cdf(d, x)
            @test quantile(d, p) ≈ x atol = 1e-9
        end

        # Direction 2: cdf(quantile(p)) ≈ p
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
            x = quantile(d, p)
            @test cdf(d, x) ≈ p atol = 1e-9
        end

        # Test edge cases
        @test quantile(d, 0.0) == -Inf
        @test quantile(d, 1.0) == Inf
        @test_throws DomainError quantile(d, -0.1)
        @test_throws DomainError quantile(d, 1.1)
    end

    @testset "Sigma scaling" begin
        # Test quantile accuracy for different σ values
        test_cases = [
            (0.0, 0.5, 0.5),  # σ < 1
            (0.0, 1.0, 0.5),  # σ = 1
            (0.0, 5.0, 0.5),  # σ > 1
        ]

        ps = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]

        for (μ, σ, ψ) in test_cases
            d_test = BifurcatedGaussian(μ, σ, ψ)

            # Test quantile accuracy
            for p in ps
                q = quantile(d_test, p)
                cdf_val = cdf(d_test, q)
                @test isapprox(cdf_val, p; atol=1e-8)
            end

            # Test CDF normalization (should approach 1 for large x)
            cdf_large = cdf(d_test, 1000)
            @test cdf_large > 0.999

            # Test PDF normalization via numerical integration
            # The PDF should integrate to 1 for all σ values
            numerical_integral = quadgk(x -> pdf(d_test, x), -Inf, Inf)[1]
            @test isapprox(numerical_integral, 1.0; atol=1e-6) ||
                  @warn "PDF normalization failed for σ = $σ" numerical_integral
        end
    end

    @testset "Symmetry" begin
        # Test with symmetric parameters
        d_sym = BifurcatedGaussian(0.0, 1.0, 0.0)  # μ, σ, ψ = 0 for symmetry

        # PDF should be symmetric around the mean
        x_test = 1.0
        @test isapprox(pdf(d_sym, x_test), pdf(d_sym, -x_test); atol=1e-10)

        # CDF should satisfy: CDF(x) + CDF(-x) ≈ 1 for symmetric distribution
        @test isapprox(cdf(d_sym, x_test) + cdf(d_sym, -x_test), 1.0; atol=1e-8)
    end

    @testset "Moments and Statistics" begin
        # Test mean, variance, and skewness formulas numerically using quadgk
        test_cases = [
            (0.0, 1.0, 0.0),   # Symmetric (skewness should be 0)
            (0.0, 1.0, 0.5),   # Asymmetric
            (0.0, 1.0, -0.5),  # Asymmetric (opposite direction)
            (0.0, 2.0, 0.3),   # Different σ
            (2.0, 1.5, 0.4),   # Different μ and σ
        ]

        for (μ, σ, ψ) in test_cases
            d = BifurcatedGaussian(μ, σ, ψ)

            # Compute mean numerically: E[X] = ∫ x * pdf(x) dx
            numerical_mean, _ = quadgk(x -> x * pdf(d, x), -Inf, Inf, rtol=1e-8)
            analytical_mean = mean(d)
            @test isapprox(
                numerical_mean,
                analytical_mean;
                atol=1e-6,
                rtol=1e-5,
            ) || @warn "Mean mismatch" μ σ ψ numerical_mean analytical_mean

            # Compute variance numerically: Var = E[(X - E[X])²]
            μ_actual = numerical_mean  # Use numerical mean for consistency
            numerical_var, _ = quadgk(x -> (x - μ_actual)^2 * pdf(d, x), -Inf, Inf, rtol=1e-8)
            analytical_var = var(d)
            @test isapprox(
                numerical_var,
                analytical_var;
                atol=1e-6,
                rtol=1e-5,
            ) || @warn "Variance mismatch" μ σ ψ numerical_var analytical_var

            # Compute moments about μ for skewness calculation
            m1, _ = quadgk(x -> (x - μ) * pdf(d, x), -Inf, Inf, rtol=1e-8)
            m2, _ = quadgk(x -> (x - μ)^2 * pdf(d, x), -Inf, Inf, rtol=1e-8)
            m3, _ = quadgk(x -> (x - μ)^3 * pdf(d, x), -Inf, Inf, rtol=1e-8)

            # Central moments about the mean: Δ = m1
            Δ = m1
            μ₂ = m2 - Δ^2
            μ₃ = m3 - 3 * Δ * m2 + 2 * Δ^3

            # Skewness = μ₃ / μ₂^(3/2)
            numerical_skewness = μ₃ / (μ₂^(3 / 2))
            analytical_skewness = skewness(d)
            @test isapprox(
                numerical_skewness,
                analytical_skewness;
                atol=1e-6,
                rtol=1e-5,
            ) || @warn "Skewness mismatch" μ σ ψ numerical_skewness analytical_skewness

            # Compute fourth moment about μ for kurtosis
            m4, _ = quadgk(x -> (x - μ)^4 * pdf(d, x), -Inf, Inf, rtol=1e-8)

            # Central fourth moment about the mean: μ₄ = m₄ - 4Δm₃ + 6Δ²m₂ - 3Δ⁴
            μ₄ = m4 - 4 * Δ * m3 + 6 * Δ^2 * m2 - 3 * Δ^4

            # Excess kurtosis = μ₄ / μ₂² - 3
            numerical_kurtosis = μ₄ / (μ₂^2) - 3
            analytical_kurtosis = kurtosis(d)
            @test isapprox(
                numerical_kurtosis,
                analytical_kurtosis;
                atol=1e-5,
                rtol=1e-4,
            ) || @warn "Kurtosis mismatch" μ σ ψ numerical_kurtosis analytical_kurtosis
        end

        # Test that symmetric case has zero skewness and zero excess kurtosis
        d_sym = BifurcatedGaussian(0.0, 1.0, 0.0)
        @test isapprox(skewness(d_sym), 0.0; atol=1e-10)
        @test isapprox(kurtosis(d_sym), 0.0; atol=1e-10)  # Excess kurtosis = 0 for normal
        @test isapprox(mean(d_sym), 0.0; atol=1e-10)  # Mean should equal μ for symmetric case
    end

    @testset "Type stability" begin
        d_float64 = BifurcatedGaussian(0.0, 1.0, 0.5)
        d_float32 = BifurcatedGaussian(0.0f0, 1.0f0, 0.5f0)

        @test pdf(d_float64, 0.0) isa Float64
        @test pdf(d_float32, 0.0f0) isa Float32
        @test cdf(d_float64, 0.0) isa Float64
        @test cdf(d_float32, 0.0f0) isa Float32
        @test quantile(d_float64, 0.5) isa Float64
        @test quantile(d_float32, 0.5f0) isa Float32
    end

    @testset "Support interface" begin
        d_float64 = BifurcatedGaussian(0.0, 1.0, 0.5)
        d_float32 = BifurcatedGaussian(0.0f0, 1.0f0, 0.5f0)

        @test maximum(d_float64) == Inf
        @test maximum(d_float32) == Inf32
        @test minimum(d_float64) == -Inf
        @test minimum(d_float32) == -Inf32
        @test minimum(d_float64) == support(d_float64).lb
        @test maximum(d_float64) == support(d_float64).ub
    end
end
