using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test
using Random

# Create test objects outside of test statements
d_standard = HyperbolicSecant(0.0, 1.0)  # Standard distribution μ=0, σ=1
d_shifted = HyperbolicSecant(2.0, 1.0)  # Shifted distribution μ=2, σ=1
d_scaled = HyperbolicSecant(0.0, 2.0)  # Scaled distribution μ=0, σ=2
d_general = HyperbolicSecant(1.5, 0.8)  # General case μ=1.5, σ=0.8

@testset "HyperbolicSecant Distribution" verbose = true begin
    @testset "Basic Functionality" begin
        # Test that basic functions work
        @test pdf(d_standard, 0.0) ≈ 0.5
        @test cdf(d_standard, 0.0) ≈ 0.5
        @test quantile(d_standard, 0.5) ≈ 0.0 atol = 1e-15

        # Test that sech function works
        @test sech(0.0) ≈ 1.0
        @test sech(1.0) ≈ 0.6480542736638855
    end

    @testset "Construction" begin
        # Test basic construction
        @test d_standard.μ == 0.0
        @test d_standard.σ == 1.0
        @test d_shifted.μ == 2.0
        @test d_shifted.σ == 1.0
        @test d_scaled.μ == 0.0
        @test d_scaled.σ == 2.0
        @test d_general.μ == 1.5
        @test d_general.σ == 0.8

        # Test convenience constructors
        @test HyperbolicSecant(1, 2) == HyperbolicSecant(1.0, 2.0)

        # Test parameter validation
        @test_throws DomainError HyperbolicSecant(0.0, 0.0)  # σ must be positive
        @test_throws DomainError HyperbolicSecant(0.0, -1.0)  # σ must be positive
    end

    @testset "Support and Properties" begin
        # Test support
        @test minimum(d_standard) == -Inf
        @test maximum(d_standard) == Inf
        @test minimum(d_general) == -Inf
        @test maximum(d_general) == Inf

        # Test parameter extraction
        @test location(d_general) == 1.5
        @test scale(d_general) == 0.8
        @test params(d_general) == (1.5, 0.8)
        @test partype(d_general) == Float64
    end

    @testset "Basic Statistics" begin
        # Test moments for standard distribution
        @test mean(d_standard) ≈ 0.0
        @test var(d_standard) ≈ 1.0
        @test std(d_standard) ≈ 1.0
        @test skewness(d_standard) ≈ 0.0  # Symmetric distribution
        @test kurtosis(d_standard) ≈ 5.0  # Theoretical kurtosis
        @test mode(d_standard) ≈ 0.0
        @test median(d_standard) ≈ 0.0

        # Test moments for general distribution
        @test mean(d_general) ≈ 1.5
        @test var(d_general) ≈ 0.8^2
        @test std(d_general) ≈ 0.8
        @test skewness(d_general) ≈ 0.0  # Skewness is invariant under location-scale
        @test kurtosis(d_general) ≈ 5.0  # Kurtosis is invariant under location-scale
        @test mode(d_general) ≈ 1.5
        @test median(d_general) ≈ 1.5
    end

    @testset "PDF and LogPDF" begin
        # Test PDF at mode for standard distribution
        @test pdf(d_standard, 0.0) ≈ 0.5  # At mode, sech(0) = 1, so pdf = 1/2

        # Test PDF symmetry for standard distribution
        for x in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
            @test pdf(d_standard, x) ≈ pdf(d_standard, -x)
        end

        # Test logpdf consistency
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]
            @test logpdf(d_standard, x) ≈ log(pdf(d_standard, x))
            @test logpdf(d_general, x) ≈ log(pdf(d_general, x))
        end

        # Test that PDF integrates to 1 (numerical integration)
        integral, _ = quadgk(x -> pdf(d_standard, x), -10, 10, rtol = 1e-8)
        @test integral ≈ 1.0 atol = 1e-6

        # Test PDF for general distribution
        @test pdf(d_general, 1.5) ≈ 1 / (2 * 0.8)  # At mode
    end

    @testset "CDF and Quantile" begin
        # Test CDF at median
        @test cdf(d_standard, 0.0) ≈ 0.5
        @test cdf(d_general, 1.5) ≈ 0.5

        # Test CDF symmetry for standard distribution
        for x in [0.5, 1.0, 1.5, 2.0]
            @test cdf(d_standard, x) ≈ 1 - cdf(d_standard, -x)
        end

        # Test quantile function
        @test quantile(d_standard, 0.5) ≈ 0.0 atol = 1e-15
        @test quantile(d_general, 0.5) ≈ 1.5

        # Test quantile and CDF are inverse functions
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]
            x = quantile(d_standard, p)
            @test cdf(d_standard, x) ≈ p atol = 1e-12

            x = quantile(d_general, p)
            @test cdf(d_general, x) ≈ p atol = 1e-12
        end

        # Test specific quartiles for standard distribution
        # From theory: Q1 ≈ -0.561, Q3 ≈ 0.561
        @test quantile(d_standard, 0.25) ≈ -2 / π * log(1 + √2) atol = 1e-10
        @test quantile(d_standard, 0.75) ≈ 2 / π * log(1 + √2) atol = 1e-10
    end

    @testset "Random Number Generation" begin
        # Test that rand produces reasonable values
        Random.seed!(42)
        samples = [rand(d_standard) for _ in 1:1000]

        # Basic sanity checks
        @test length(samples) == 1000
        @test all(isfinite.(samples))

        # Test empirical mean and variance (should be close to theoretical)
        empirical_mean = mean(samples)
        empirical_var = var(samples)

        @test abs(empirical_mean - mean(d_standard)) < 0.1
        @test abs(empirical_var - var(d_standard)) < 0.2

        # Test with RNG
        Random.seed!(123)
        sample1 = rand(Random.MersenneTwister(123), d_standard)
        sample2 = rand(Random.MersenneTwister(123), d_standard)
        @test sample1 == sample2  # Same seed should give same result
    end

    @testset "Special Functions" begin
        # Test moment generating function (MGF)
        # MGF should be defined for |t| < π/(2σ)
        d = HyperbolicSecant(0.0, 1.0)

        # Test MGF at t=0 (should be 1)
        @test DistributionsHEP.mgf(d, 0.0) ≈ 1.0

        # Test MGF for small values
        for t in [0.1, 0.5, 1.0]
            @test DistributionsHEP.mgf(d, t) ≈ sec(t)  # For standard case μ=0, σ=1
            @test DistributionsHEP.mgf(d, -t) ≈ sec(t)  # Should be even function for μ=0
        end

        # Test that MGF throws error for large t
        @test_throws DomainError DistributionsHEP.mgf(d, 2.0)  # |t| >= π/2
        @test_throws DomainError DistributionsHEP.mgf(d, -2.0)  # |t| >= π/2

        # Test characteristic function
        @test DistributionsHEP.cf(d, 0.0) ≈ 1.0

        # Test characteristic function for real inputs (should be real for symmetric dist at μ=0)
        @test real(DistributionsHEP.cf(d, 1.0)) ≈ sech(1.0)
        @test imag(DistributionsHEP.cf(d, 1.0)) ≈ 0.0 atol = 1e-15
    end

    @testset "Mathematical Properties" begin
        # Test that PDF has correct maximum at mode
        μ, σ = 1.0, 2.0
        d = HyperbolicSecant(μ, σ)

        # Check that PDF is maximized at the mode
        mode_pdf = pdf(d, μ)
        for x in [μ - 1, μ + 1, μ - 0.5, μ + 0.5]
            @test pdf(d, x) ≤ mode_pdf
        end

        # Test the theoretical formula for PDF at mode
        @test pdf(d, μ) ≈ 1 / (2 * σ)

        # Test inflection points (approximately at μ ± 0.561σ)
        # At inflection points, second derivative should be zero
        inflection_distance = 2 / π * log(√2 + 1)
        @test inflection_distance ≈ 0.5615 atol = 1e-3
    end

    @testset "Edge Cases" begin
        # Test behavior near boundaries of quantile function
        @test_throws ArgumentError quantile(d_standard, 0.0)
        @test_throws ArgumentError quantile(d_standard, 1.0)
        @test_throws ArgumentError quantile(d_standard, -0.1)
        @test_throws ArgumentError quantile(d_standard, 1.1)

        # Test CDF approaches 0 and 1 at extremes
        @test cdf(d_standard, -10.0) ≈ 0.0 atol = 1e-6
        @test cdf(d_standard, 10.0) ≈ 1.0 atol = 1e-6

        # Test PDF approaches 0 at extremes
        @test pdf(d_standard, -10.0) ≈ 0.0 atol = 1e-6
        @test pdf(d_standard, 10.0) ≈ 0.0 atol = 1e-6
    end

    @testset "Type Stability" begin
        # Test with different number types
        d_float32 = HyperbolicSecant(0.0f0, 1.0f0)
        d_float64 = HyperbolicSecant(0.0, 1.0)

        @test typeof(pdf(d_float32, 0.0f0)) == Float32
        @test typeof(pdf(d_float64, 0.0)) == Float64

        @test typeof(cdf(d_float32, 0.0f0)) == Float32
        @test typeof(cdf(d_float64, 0.0)) == Float64

        @test typeof(quantile(d_float32, 0.5f0)) == Float32
        @test typeof(quantile(d_float64, 0.5)) == Float64
    end
end
