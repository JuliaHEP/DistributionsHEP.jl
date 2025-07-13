using SpecialFunctions
using DistributionsHEP
using Distributions
using Polynomials
using QuadGK
using Test

# Create test objects outside of test statements
d_constant = Chebyshev([1.0, 0.0, 0.0])  # Constant polynomial
d_linear = Chebyshev([1.0, 1.0, 0.0], -1, 3)  # Linear polynomial
d_quadratic = Chebyshev([2.0, 0.0, 1.0], -2.0, 2.0)  # Quadratic polynomial

# # for visual inspection
# all polynomials positive in the range of definition
# using Plots
# theme(:boxed)
# let
#     plot(leg = :topleft)
#     map([
#         :d_constant
#         :d_linear
#         :d_quadratic]) do l
#         d = eval(l)
#         plot!(x -> pdf(d, x), minimum(d), maximum(d), label = "$l", fill = 0, fillalpha = 0.3)
#     end
#     plot!()
# end

@testset "Chebyshev Distribution" verbose = true begin
    @testset "Construction" begin
        @test minimum(d_constant) == support(d_constant).lb == -1.0
        @test maximum(d_constant) == support(d_constant).ub == 1.0

        @test minimum(d_linear) == -1.0
        @test maximum(d_linear) == 3.0

        @test minimum(d_quadratic) == -2.0
        @test maximum(d_quadratic) == 2.0
    end

    @testset "PDF properties" begin
        # Test constant polynomial (should be uniform)
        # PDF should be constant
        pdf_val = pdf(d_constant, 0.0)
        @test pdf(d_constant, 0.5) ≈ pdf_val
        @test pdf(d_constant, -0.5) ≈ pdf_val

        # Check normalization
        numerical_integral =
            quadgk(x -> pdf(d_constant, x), minimum(d_constant), maximum(d_constant))[1]
        @test isapprox(numerical_integral, 1.0; atol = 1e-6)

        # Check normalization
        numerical_integral =
            quadgk(x -> pdf(d_quadratic, x), minimum(d_quadratic), maximum(d_quadratic))[1]
        @test isapprox(numerical_integral, 1.0; atol = 1e-6)
    end

    @testset "CDF properties" begin
        # Test constant polynomial
        # CDF should be linear for uniform distribution
        @test isapprox(cdf(d_constant, -1.0), 0.0; atol = 1e-10)
        @test isapprox(cdf(d_constant, 0.0), 0.5; atol = 1e-6)
        @test isapprox(cdf(d_constant, 1.0), 1.0; atol = 1e-10)

        # CDF should be monotonic
        @test cdf(d_linear, 0.5) < cdf(d_linear, 0.51)

        # Test quadratic polynomial
        @test isapprox(cdf(d_quadratic, minimum(d_quadratic)), 0.0; atol = 1e-10)
        @test isapprox(cdf(d_quadratic, maximum(d_quadratic)), 1.0; atol = 1e-10)
        @test cdf(d_quadratic, -1.0) < cdf(d_quadratic, 1.0)
    end

    @testset "Random sampling" begin
        # Test single sample
        sample = rand(d_quadratic)
        @test -2.0 <= sample <= 2.0

        # Test multiple samples
        samples = rand(d_quadratic, 100)
        @test length(samples) == 100
        @test all(-2.0 .<= samples .<= 2.0)

        # Test that samples follow the distribution (basic check)
        @test mean(samples) > -2.0
        @test mean(samples) < 2.0
    end
end
