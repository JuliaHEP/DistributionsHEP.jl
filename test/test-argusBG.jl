using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test

# Create test objects outside of test statements
d_argus01 = ArgusBG(-0.005, 0.5)
d_argus05 = ArgusBG(-0.125, 0.6, -2, 0)
d_argus1 = ArgusBG(-0.5, 0.4)
d_argus2 = ArgusBG(-2.0, 0.8, -1.0, 3.0)
d_argus3 = ArgusBG(-4.5, 0.5, 0.5, 1.5)

d_set = [
    d_argus01
    d_argus05
    d_argus1
    d_argus2
    d_argus3
]

# for visual inspection
# using Plots
# theme(:boxed)
# let
#     plot(leg = :topleft)
#     map([
#         :d_argus01
#         :d_argus05
#         :d_argus1
#         :d_argus2
#         :d_argus3
#     ]) do l
#         d = eval(l)
#         plot!(x -> pdf(d, x), minimum(d), maximum(d), label = "$l")
#     end
#     plot!()
# end

@testset "ArgusBG Distribution" verbose = true begin
    @testset "Construction" begin
        for d in d_set
            @test minimum(d) == support(d).lb
            @test maximum(d) == support(d).ub
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
            mid = (minimum(d) + maximum(d)) / 2
            @test cdf(d, mid) < cdf(d, mid + 0.01)
        end
    end

    @testset "Inverse CDF properties" begin
        for d in d_set
            for x in range(0.1, 0.9, 11) .* (maximum(d) - minimum(d)) .+ minimum(d)
                _cdf = cdf(d, x)
                _inv_cdf = quantile(d, _cdf)
                @test isapprox(x, _inv_cdf; atol = 1e-10)
            end
        end
    end
    @testset "Random sampling" begin
        for d in d_set
            # Test single sample
            sample = rand(d)
            @test minimum(d) <= sample <= maximum(d)

            # Test multiple samples
            samples = rand(d, 100)
            @test length(samples) == 100
            @test all(minimum(d) .<= samples .<= maximum(d))
        end
    end

    @testset "Parameter extraction" begin
        for d in d_set
            @test params(d.ρ) == (d.ρ.c, d.ρ.p)
            @test partype(d) == Float64
        end
    end
end


