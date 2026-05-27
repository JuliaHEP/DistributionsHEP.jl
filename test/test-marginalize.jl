using DistributionsHEP
using Distributions
using LinearAlgebra
using Test

@testset "marginalize" begin
    @test !(:marginalize in names(DistributionsHEP))

    gaussian_product = product_distribution([Normal(-1.0, 0.5), Normal(1.0, 1.5)])
    @test DistributionsHEP.marginalize(gaussian_product, 1) == Normal(-1.0, 0.5)
    @test DistributionsHEP.marginalize(gaussian_product, 2) == Normal(1.0, 1.5)

    signal = product_distribution([Normal(-1.0, 0.5), Exponential(2.0)])
    swapped = product_distribution([Exponential(2.0), Normal(-1.0, 0.5)])
    background = product_distribution([Uniform(-2.0, 2.0), Normal(1.0, 1.5)])
    mixed = MixtureModel([signal, swapped], [0.25, 0.75])
    model = ExtendedMixtureModel([signal, mixed, background], [10.0, 5.0, 2.0])

    m1 = DistributionsHEP.marginalize(model, 1)
    @test yields(m1) == yields(model)
    @test total_yield(m1) == total_yield(model)

    x = 0.2
    expected_density =
        10.0 * pdf(DistributionsHEP.marginalize(signal, 1), x) +
        5.0 * pdf(DistributionsHEP.marginalize(mixed, 1), x) +
        2.0 * pdf(DistributionsHEP.marginalize(background, 1), x)
    @test m1(x) ≈ expected_density

    gaussian_model = ExtendedMixtureModel([gaussian_product], [10.0])
    m1_gauss = DistributionsHEP.marginalize(gaussian_model, 1)
    @test yields(m1_gauss) == yields(gaussian_model)
    @test m1_gauss(0.2) ≈ 10.0 * pdf(Normal(-1.0, 0.5), 0.2)

    correlated = MvNormal([0.0, 0.0], [1.0 0.5; 0.5 1.0])
    @test DistributionsHEP.marginalize(correlated, 1) == Normal(0.0, 1.0)
    @test DistributionsHEP.marginalize(correlated, 2) == Normal(0.0, 1.0)
end
