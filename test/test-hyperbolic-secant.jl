using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test

@testset "HyperbolicSecant Distribution" verbose=true begin

# Helper to check quantile accuracy for different σ values
function check_quantile_accuracy(d, ps; atol = 1e-8)
    for p in ps
        q = quantile(d, p)
        cdf_val = cdf(d, q)
        @test isapprox(cdf_val, p; atol = atol) || @warn "Quantile test failed" p q cdf_val
    end
end

# Test distribution with standard parameters (μ=0, σ=1)
d = HyperbolicSecant(0.0, 1.0)

@testset "Parameter validation" begin
    @test_throws ErrorException HyperbolicSecant(0.0, -1.0)  # negative σ
    @test_throws ErrorException HyperbolicSecant(0.0, 0.0)   # zero σ
    
    # Valid constructions
    @test isa(HyperbolicSecant(0.0, 1.0), HyperbolicSecant)
    @test isa(HyperbolicSecant(1.0), HyperbolicSecant)       # μ only
    @test isa(HyperbolicSecant(), HyperbolicSecant)          # no parameters
end

@testset "Constructor methods" begin
    d1 = HyperbolicSecant()
    @test d1.μ == 0.0
    @test d1.σ == 1.0
    
    d2 = HyperbolicSecant(2.0)
    @test d2.μ == 2.0
    @test d2.σ == 1.0
    
    d3 = HyperbolicSecant(1.0, 2.0)
    @test d3.μ == 1.0
    @test d3.σ == 2.0
end

@testset "PDF properties" begin
    # PDF should be positive and finite everywhere
    x_test_points = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    for x in x_test_points
        @test pdf(d, x) > 0
        @test isfinite(pdf(d, x))
    end

    # PDF should be symmetric around the mean
    @test isapprox(pdf(d, d.μ + 1.0), pdf(d, d.μ - 1.0); atol = 1e-10)
    @test isapprox(pdf(d, d.μ + 2.0), pdf(d, d.μ - 2.0); atol = 1e-10)
    
    # PDF should have maximum at the mean (mode)
    @test pdf(d, d.μ) > pdf(d, d.μ + 0.5)
    @test pdf(d, d.μ) > pdf(d, d.μ - 0.5)

    # PDF should integrate to 1
    numerical_integral = quadgk(x -> pdf(d, x), -Inf, Inf)[1]
    @test isapprox(numerical_integral, 1.0; atol = 1e-7)
end

@testset "CDF properties" begin
    # CDF should be between 0 and 1
    x_values = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    for x in x_values
        cdf_val = cdf(d, x)
        @test cdf_val >= 0
        @test cdf_val <= 1
    end

    # CDF should be monotonically increasing
    x_sorted = sort(x_values)
    cdf_values = [cdf(d, x) for x in x_sorted]
    for i in 2:length(cdf_values)
        @test cdf_values[i] >= cdf_values[i-1]
    end

    # CDF should approach 0 and 1 at extremes
    @test cdf(d, -100) < 0.01
    @test cdf(d, 100) > 0.99

    # CDF should be 0.5 at the mean (due to symmetry)
    @test isapprox(cdf(d, d.μ), 0.5; atol = 1e-10)
end

@testset "Quantile properties" begin
    # Quantile should be inverse of CDF
    x_test = 0.8
    p_test = cdf(d, x_test)
    @test isapprox(quantile(d, p_test), x_test; atol = 1e-9)

    # Test edge cases
    @test quantile(d, 0.0) == -Inf
    @test quantile(d, 1.0) == Inf
    @test_throws DomainError quantile(d, -0.1)
    @test_throws DomainError quantile(d, 1.1)
    
    # Quantile at 0.5 should be the mean
    @test isapprox(quantile(d, 0.5), d.μ; atol = 1e-10)
    
    # Test symmetry in quantiles
    @test isapprox(quantile(d, 0.25) + quantile(d, 0.75), 2 * d.μ; atol = 1e-8)
end

@testset "Distribution statistics" begin
    d_test = HyperbolicSecant(2.0, 3.0)
    
    # Test mean
    @test mean(d_test) == 2.0
    
    # Test variance and standard deviation
    @test var(d_test) == 9.0
    @test std(d_test) == 3.0
    
    # Test skewness (should be 0 for symmetric distribution)
    @test skewness(d_test) == 0.0
    
    # Test kurtosis (excess kurtosis should be 2)
    @test kurtosis(d_test) == 2.0
end

@testset "Parameter scaling" begin
    # Test quantile accuracy for different parameter values
    test_cases = [
        (0.0, 0.5),   # σ < 1
        (0.0, 1.0),   # σ = 1
        (0.0, 2.0),   # σ > 1
        (2.0, 1.0),   # μ ≠ 0, σ = 1
        (-1.0, 3.0),  # μ < 0, σ > 1
    ]

    ps = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    for (μ, σ) in test_cases
        d_test = HyperbolicSecant(μ, σ)

        # Test quantile accuracy
        check_quantile_accuracy(d_test, ps)

        # Test CDF normalization (should approach 1 for large x)
        cdf_large = cdf(d_test, μ + 10 * σ)
        @test cdf_large > 0.99

        # Test PDF normalization via numerical integration
        numerical_integral = quadgk(x -> pdf(d_test, x), -Inf, Inf)[1]
        @test isapprox(numerical_integral, 1.0; atol = 1e-6) ||
              @warn "PDF normalization failed for μ = $μ, σ = $σ" numerical_integral
              
        # Test mean and variance properties
        @test mean(d_test) == μ
        @test var(d_test) == σ^2
        @test std(d_test) == σ
    end
end

@testset "Type stability" begin
    d_float64 = HyperbolicSecant(0.0, 1.0)
    d_float32 = HyperbolicSecant(0.0f0, 1.0f0)

    @test pdf(d_float64, 0.0) isa Float64
    @test pdf(d_float32, 0.0f0) isa Float32
    @test cdf(d_float64, 0.0) isa Float64
    @test cdf(d_float32, 0.0f0) isa Float32
    @test quantile(d_float64, 0.5) isa Float64
    @test quantile(d_float32, 0.5f0) isa Float32
    @test mean(d_float64) isa Float64
    @test mean(d_float32) isa Float32
    @test var(d_float64) isa Float64
    @test var(d_float32) isa Float32
end

@testset "Support interface" begin
    d_float64 = HyperbolicSecant(0.0, 1.0)
    d_float32 = HyperbolicSecant(0.0f0, 1.0f0)

    @test maximum(d_float64) == Inf
    @test maximum(d_float32) == Inf32
    @test minimum(d_float64) == -Inf
    @test minimum(d_float32) == -Inf32
    @test minimum(d_float64) == support(d_float64).lb
    @test maximum(d_float64) == support(d_float64).ub
end

@testset "Specific mathematical properties" begin
    d_std = HyperbolicSecant()  # Standard form with μ=0, σ=1
    
    # Test that PDF at origin matches expected value
    # For standard form: f(0) = (1/2) * sech(0) = (1/2) * 1 = 0.5
    @test isapprox(pdf(d_std, 0.0), 0.5; atol = 1e-10)
    
    # Test that CDF at origin is 0.5 (due to symmetry)
    @test isapprox(cdf(d_std, 0.0), 0.5; atol = 1e-10)
    
    # Test some known values for standard distribution
    # At x = ±2σ = ±2, the PDF should be smaller
    @test pdf(d_std, 2.0) < pdf(d_std, 0.0)
    @test pdf(d_std, -2.0) < pdf(d_std, 0.0)
    
    # Test that the distribution has heavy tails (hyperbolic secant property)
    # The ratio of PDF values should follow hyperbolic secant behavior
    @test isapprox(pdf(d_std, 1.0), pdf(d_std, -1.0); atol = 1e-10)
end

end