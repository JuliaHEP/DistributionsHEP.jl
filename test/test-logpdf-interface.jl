using DistributionsHEP
using Distributions
using Test

@testset "logpdf interface" begin
    test_cases = [
        (Chebyshev([1.0, 0.5]), 0.1),
        (ArgusBG(-2.0, 0.5), 0.4),
        (CrystalBall(0.0, 1.0, 1.0, 1.6), 0.2),
        (DoubleCrystalBall(0.0, 1.0, 1.5, 2.0, 2.0, 3.0), 0.2),
        (BifurcatedGaussian(0.0, 1.0, 0.25), 0.2),
        (DoubleSidedBifurcatedCrystalBall(0.0, 1.0, 0.25, 0.5, 1.25, 0.75, 1.5), 0.2),
        (DoubleSidedBifurcatedCrystalBallDas(0.0, 1.0, 0.25, 0.5, 1.25, 0.75), 0.2),
    ]

    for (d, x) in test_cases
        @test logpdf(d, x) ≈ log(pdf(d, x))

        product = product_distribution(d, Normal())
        xs = [x, 0.0]
        @test logpdf(product, xs) ≈ logpdf(d, x) + logpdf(Normal(), 0.0)
        @test pdf(product, xs) ≈ pdf(d, x) * pdf(Normal(), 0.0)
    end
end
