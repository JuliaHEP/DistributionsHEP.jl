using SpecialFunctions
using DistributionsHEP
using Distributions
using Polynomials
using QuadGK
using Test

# Create test objects outside of test statements
d_argus01 = ArgusBG(1.0, -0.005, 0.5)
d_argus05 = ArgusBG(1.0, -0.125, 0.5)
d_argus1 = ArgusBG(1.0, -0.5, 0.5)
d_argus2 = ArgusBG(1.0, -2.0, 0.5)
d_argus3 = ArgusBG(1.0, -4.5, 0.5)

d_set = [
    d_argus01
    d_argus05
    d_argus1
    d_argus2
    d_argus3
]

#= for visual inspection
using Plots
theme(:boxed)
let
    plot(leg = :topleft)
    map([
         :d_argus01 :d_argus05 :d_argus1 :d_argus2 :d_argus3 ]) do l
         d = eval(l)
         plot!(x -> pdf(d, x), minimum(d), maximum(d), label = "$l")
     end
     plot!()
end
=#

@testset "ArgusGB Distribution" verbose=true begin
    @testset "Construction" begin
        for d in d_set 
            @test minimum(d) == support(d).lb == 0.0
            @test maximum(d) == support(d).ub == 1.0
        end
    end

    @testset "PDF properties" begin

        # Check normalization
        for d in d_set
            numerical_integral = quadgk(x -> pdf(d, x), minimum(d), maximum(d))[1]
            @test isapprox(numerical_integral, 1.0; atol = 1e-6)
        end
    end

    @testset "CDF properties" begin
        for d in d_set
            # Test CDF at minimum and maximum
            @test isapprox(cdf(d, minimum(d)), 0.0; atol = 1e-10)
            @test isapprox(cdf(d, maximum(d)), 1.0; atol = 1e-10)

            # CDF should be monotonic
            @test cdf(d, 0.5) < cdf(d, 0.51)
        end
    end

    @testset "Random sampling" begin
        for d in d_set
            # Test single sample
            sample = rand(d)
            @test 0.0 <= sample <= 1.0

            # Test multiple samples
            samples = rand(d, 100)
            @test length(samples) == 100
            @test all(0.0 .<= samples .<= 1.0)

            # Test that samples follow the distribution (basic check)
            @test mean(samples) > 0.0
            @test mean(samples) < 1.0
        end
    end
    @testset "Parameter extraction" begin
        for d in d_set
            @test scale(d) == d.m₀
            @test shape(d) == d.c
            @test params(d) == (d.m₀, d.c, d.p)
            @test partype(d) == Float64
        end
    end
end


