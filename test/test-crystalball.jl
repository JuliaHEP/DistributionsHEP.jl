using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test

d = CrystalBall(0.0, 1.0, 1.0, 1.6)

@testset "CrystalBall continuity on merging point" begin
    x_merge = -d.α * d.σ + d.μ
    pdf_value_left = pdf(d, x_merge + 1e-5) # just above the transition point
    pdf_value_right = pdf(d, x_merge - 1e-5) # just below the transition point
    @test isapprox(pdf_value_left, pdf_value_right; atol = 1e-5) # should be continuous
end

# for visual inspection
# using Plots
# plot(x -> pdf(d, x), -10, 10,
#     label="Crystal Ball PDF",
#     xlabel="x", ylabel="PDF",
#     title="Crystal Ball Distribution")

# check integral
numerical_integral = quadgk(x -> pdf(d, x), -Inf, Inf)[1]
@test isapprox(numerical_integral, 1.0; atol = 1e-7)

# # for visual inspection
# using Plots
# plot(x -> cdf(d, x), -10, 10,
#     label="Crystal Ball CDF",
#     xlabel="x", ylabel="CDF",
# title="Crystal Ball Distribution")

# Continuity check for CDF
@testset "CrystalBall CDF continuity on merging point" begin
    x_merge = -d.α * d.σ + d.μ
    cdf_value_left = cdf(d, x_merge + 1e-6) # just above the transition point
    cdf_value_right = cdf(d, x_merge - 1e-6) # just below the transition point
    @show cdf_value_left, cdf_value_right
    @test isapprox(cdf_value_left, cdf_value_right; atol = 1e-5) # should be continuous
end

@testset "CrystalBall CDF properties" begin
    @test cdf(d, -100) < 0.1
    @test isapprox(cdf(d, d.μ + 5 * d.σ), 1; atol = 1e-5)
end

# # for visual inspection
# using Plots
# plot(x -> cdf(d, x), -10, 10,
#     label="Crystal Ball CDF",
#     xlabel="x", ylabel="CDF",
#     title="Crystal Ball Distribution")
# 
# plot(x -> quantile(d, x), 0.2, 0.99,
#     label="Crystal Ball Quantile",
#     xlabel="x", ylabel="Quantile",
#     title="Crystal Ball Distribution")
# 

@testset "CrystalBall quantile properties" begin
    # gaussian part
    @test quantile(d, cdf(d, 0.9)) ≈ 0.9
    # in the power-law tail
    @test quantile(d, cdf(d, 0.1)) ≈ 0.1
end

d_float64 = CrystalBall(0.0, 1.0, 2.0, 3.0)
d_float32 = CrystalBall(0.0f0, 1.0f0, 2.0f0, 3.0f0)
@testset "CrystalBall maximum/minimum interfaces" begin
    @test maximum(d_float64) == Inf
    @test maximum(d_float32) == Inf32
    @test minimum(d_float64) == -Inf
    @test minimum(d_float32) == -Inf32
    @test minimum(d_float64) == support(d_float64).lb
    @test maximum(d_float64) == support(d_float64).ub
end
