using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test

# Test distribution with M = 0.1, Γ = 1.0 (standard case)
d = RelativisticBreitWigner(0.1, 1.0)  # M, Γ

@testset "RelativisticBreitWigner Distribution" verbose = true begin
    @testset "Parameter validation" begin
        # M must be positive
        @test_throws ErrorException RelativisticBreitWigner(-1.0, 1.0)

        # Γ must be positive
        @test_throws ErrorException RelativisticBreitWigner(1.0, -1.0)
    end

    @testset "PDF properties" begin
        # PDF should be positive and finite everywhere in its support
        x_test_points = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        for x in x_test_points
            @test pdf(d, x) > 0
            @test isfinite(pdf(d, x))
        end

        # PDF should be zero for negative values
        @test pdf(d, -1.0) == 0.0
        @test pdf(d, -0.1) == 0.0

        # PDF should integrate to 1 over its support [0, Inf)
        numerical_integral = quadgk(x -> pdf(d, x), 0.0, Inf)[1]
        @test isapprox(numerical_integral, 1.0; atol=1e-6)
    end

    @testset "CDF properties" begin
        # CDF should be between 0 and 1
        x_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        for x in x_values
            cdf_val = cdf(d, x)
            @test cdf_val >= 0
            @test cdf_val <= 1
        end

        # CDF for negative x should be zero
        @test cdf(d, -1.0) == 0.0
        @test cdf(d, -0.1) == 0.0

        # CDF far above the peak should approach 1
        @test isapprox(cdf(d, d.M + 5 * d.Γ), 1; atol=1e-4)

        # Compare CDF with numerical integration of PDF
        x_test_points = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        for x in x_test_points
            cdf_val = cdf(d, x)
            quad_val = quadgk(t -> pdf(d, t), 0.0, x)[1]
            @test isapprox(cdf_val, quad_val; atol=1e-5)
        end
    end

    @testset "Type stability" begin
        d_float64 = RelativisticBreitWigner(0.1, 1.0)
        d_float32 = RelativisticBreitWigner(0.1f0, 1.0f0)
        d_int = RelativisticBreitWigner(1, 2)
        d_mix = RelativisticBreitWigner(1, 2.0f0)

        @test pdf(d_float64, 0.0) isa Float64
        @test pdf(d_float32, 0.0f0) isa Float32
        @test cdf(d_float64, 0.0) isa Float64
        @test cdf(d_float32, 0.0f0) isa Float32

        # Integer constructor should promote to Float64
        @test pdf(d_int, 1) isa Float64
        @test cdf(d_int, 1) isa Float64

        # Mixed type constructor should promote to Float32
        @test pdf(d_mix, 2) isa Float32
        @test cdf(d_mix, 2) isa Float32
    end

    @testset "Support interface" begin
        d_float64 = RelativisticBreitWigner(0.1, 1.0)
        d_float32 = RelativisticBreitWigner(0.1f0, 1.0f0)

        @test minimum(d_float64) == -Inf
        @test maximum(d_float64) == Inf
        @test minimum(d_float32) == -Inf32
        @test maximum(d_float32) == Inf32
        @test minimum(d_float64) == support(d_float64).lb
        @test maximum(d_float64) == support(d_float64).ub
    end

    @testset "Distributions.jl interface" begin
        # Test params() returns the same values as creation parameters
        test_cases = [
            (0.1, 1.0),
            (1.0, 2.0),
            (0.5, 0.8),
        ]

        for (M, Γ) in test_cases
            d_test = RelativisticBreitWigner(M, Γ)
            p = params(d_test)
            @test p == (M, Γ)
        end

        # Test location and scale
        d_test = RelativisticBreitWigner(1.5, 2.5)
        @test location(d_test) == 1.5
        @test scale(d_test) == 2.5
    end
end