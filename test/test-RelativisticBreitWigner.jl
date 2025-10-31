using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test

d = RelativisticBreitWigner(0.0,1.0)

@testset "RelativisticBreitWigner Distribution" verbose = true begin
@testset "Parameter validation" begin
    @test_throws ErrorException RelativisticBreitWigner()
    @test_throws ErrorException RelativisticBreitWigner()
end

@testset "PDF properties" begin
    x_test_points = [85.0,87.0,89.0,91.0,93.0,95.0]
    for x in x_test_points
        @test pdf(d,x) > 0
        @test isfinite(pdf(d,x))
        @test isfinite(pdf(d,x))
    end
end

@testset "CDF prooperties" begin
    x_values = [85.0,87.0,89.0,91.0,93.0,95.0]
    for x in x_values
        cdf_val = cdf(d,x)
        @test cdf_val >= 0
        @test cdf_val <= 1
    end
    @test cdf(d, -100) < 0.01
    @test isapprox(cdf(d, d.μ + 5 * d.σ), 1; atol = 1e-5)
end

@testset "Type stability" begin
    d_float64 = RelativisticBreitWigner(0.1,1.0)
    d_float32 = RelativisticBreitWigner(0.1f0, 1.0f0)
    d_int = RelativisticBreitWigner(1, 2)
    d_mix = RelativisticBreitWigner(1, 2.0f0)

    @test pdf(d_float64, 0.0) isa Float64
    @test pdf(d_float32, 0.0f0) isa Float32
    @test cdf(d_float64, 0.0) isa Float64
    @test cdf(d_float32, 0.0f0) isa Float32
    @test pdf(d_int, 1) isa Float64
    @test pdf(d_mix, 2) isa Float32
    @test cdf(d_int, 1) isa Float64
    @test cdf(d_mix, 2) isa Float32

end

end
