using DistributionsHEP
using Distributions
using LinearAlgebra
using Test

@testset "marginalize and product support" begin
    @test !(:marginalize in names(DistributionsHEP))

    gaussian_product = product_distribution([Normal(-1.0, 0.5), Normal(1.0, 1.5)])
    @test gaussian_product isa MvNormal
    @test minimum(gaussian_product) == [-Inf, -Inf]
    @test maximum(gaussian_product) == [Inf, Inf]
    @test support(gaussian_product) == Distributions.RealInterval.(minimum(gaussian_product), maximum(gaussian_product))
    @test all(minimum(gaussian_product) .== getproperty.(support(gaussian_product), :lb))
    @test all(maximum(gaussian_product) .== getproperty.(support(gaussian_product), :ub))
    @test DistributionsHEP.marginalize(gaussian_product, 1) == Normal(-1.0, 0.5)
    @test DistributionsHEP.marginalize(gaussian_product, 2) == Normal(1.0, 1.5)

    signal = product_distribution([Normal(-1.0, 0.5), Exponential(2.0)])
    swapped = product_distribution([Exponential(2.0), Normal(-1.0, 0.5)])
    background = product_distribution([Uniform(-2.0, 2.0), Normal(1.0, 1.5)])
    @test minimum(signal) == [-Inf, 0.0]
    @test maximum(signal) == [Inf, Inf]
    @test support(signal) == Distributions.RealInterval.(minimum(signal), maximum(signal))
    @test all(minimum(signal) .== getproperty.(support(signal), :lb))
    @test all(maximum(signal) .== getproperty.(support(signal), :ub))

    mixed = MixtureModel([signal, swapped], [0.25, 0.75])
    @test minimum(mixed) == [-Inf, -Inf]
    @test maximum(mixed) == [Inf, Inf]
    @test support(mixed) == Distributions.RealInterval.(minimum(mixed), maximum(mixed))
    @test all(minimum(mixed) .== getproperty.(support(mixed), :lb))
    @test all(maximum(mixed) .== getproperty.(support(mixed), :ub))

    model = ExtendedMixtureModel([signal, mixed, background], [10.0, 5.0, 2.0])
    @test minimum(model) == [-Inf, -Inf]
    @test maximum(model) == [Inf, Inf]
    @test support(model) == Distributions.RealInterval.(minimum(model), maximum(model))
    @test all(minimum(model) .== getproperty.(support(model), :lb))
    @test all(maximum(model) .== getproperty.(support(model), :ub))

    finite_product = product_distribution([Uniform(-2.0, 2.0), Uniform(0.0, 1.0)])
    finite_model = ExtendedMixtureModel([finite_product], [1.0])
    @test minimum(finite_model) == [-2.0, 0.0]
    @test maximum(finite_model) == [2.0, 1.0]
    @test support(finite_model) == Distributions.RealInterval.([-2.0, 0.0], [2.0, 1.0])
    @test all(minimum(finite_model) .== getproperty.(support(finite_model), :lb))
    @test all(maximum(finite_model) .== getproperty.(support(finite_model), :ub))

    univariate_model = ExtendedMixtureModel([Normal(-1.0, 0.5), Uniform(-2.0, 2.0)], [3.0, 1.0])
    @test minimum(univariate_model) == -Inf
    @test maximum(univariate_model) == Inf
    @test support(univariate_model) == Distributions.RealInterval(-Inf, Inf)
    @test minimum(univariate_model) == support(univariate_model).lb
    @test maximum(univariate_model) == support(univariate_model).ub

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
    @test support(gaussian_model) == support(gaussian_product)
    m1_gauss = DistributionsHEP.marginalize(gaussian_model, 1)
    @test yields(m1_gauss) == yields(gaussian_model)
    @test m1_gauss(0.2) ≈ 10.0 * pdf(Normal(-1.0, 0.5), 0.2)

    correlated = MvNormal([0.0, 0.0], [1.0 0.5; 0.5 1.0])
    @test DistributionsHEP.marginalize(correlated, 1) == Normal(0.0, 1.0)
    @test DistributionsHEP.marginalize(correlated, 2) == Normal(0.0, 1.0)
end
