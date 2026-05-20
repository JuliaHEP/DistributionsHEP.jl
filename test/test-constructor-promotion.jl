using DistributionsHEP
using Distributions
using Test

@testset "Mixed parameter constructors" begin
    distributions = [
        Chebyshev([1.0, 0.2], -1.0f0, 1.0),
        ArgusBG(-2.0f0, 0.8, -1.0f0, 3.0),
        CrystalBall(0.0f0, 1.0, 1.5, 2.5),
        DoubleCrystalBall(0.0f0, 1.0, 1.5, 2.5, 2.0, 3.0),
        HyperbolicSecant(0.0f0, 1.0),
        BifurcatedGaussian(0.0f0, 1.0, 0.2),
        DoubleSidedBifurcatedCrystalBall(0.0f0, 1.0, 0.2, 1.5, 2.5, 2.0, 3.0),
        DoubleSidedBifurcatedCrystalBallDas(0.0f0, 1.0, 0.2, 1.5, 2.5, 2.0),
    ]

    for d in distributions
        x = (minimum(d) + maximum(d)) / 2
        isfinite(x) || (x = zero(partype(d)))
        @test isfinite(pdf(d, x))
        @test zero(partype(d)) <= cdf(d, x) <= one(partype(d))
    end

    d_argus = ArgusBG(-2.0f0, 0.8f0, -1.0, 3.0)
    @test partype(d_argus.ρ) == Float32
    @test partype(d_argus) == Float64
end
