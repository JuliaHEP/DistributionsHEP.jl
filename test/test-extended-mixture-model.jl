using DistributionsHEP
using Distributions
using LinearAlgebra
using Test

@testset "ExtendedMixtureModel" begin
    components = [Normal(-1.0, 0.5), Normal(1.0, 0.25)]
    ys = [20, 5]
    model = ExtendedMixtureModel(components, ys)

    @test ncomponents(model) == 2
    @test Distributions.components(model) == components
    @test component(model, 2) == components[2]
    @test Distributions.component_type(model) == Normal{Float64}
    @test yields(model) == [20.0, 5.0]
    @test total_yield(model) == 25.0

    x = 0.1
    expected_density = 20.0 * pdf(components[1], x) + 5.0 * pdf(components[2], x)
    @test model(x) ≈ expected_density
    @test !hasmethod(pdf, Tuple{typeof(model), typeof(x)})

    normalized = MixtureModel(model)
    normalized_from_pipe = model |> MixtureModel
    @test pdf(normalized, x) ≈ expected_density / total_yield(model)
    @test probs(normalized_from_pipe) ≈ yields(model) ./ total_yield(model)

    data = [-1.0, -0.5, 0.9]
    expected_nll = -sum(log(model(xi)) for xi in data) + total_yield(model)
    @test extended_negative_log_likelihood(model, data) ≈ expected_nll

    @test_throws ArgumentError ExtendedMixtureModel(components, [1.0])
    @test_throws ArgumentError ExtendedMixtureModel(Normal{Float64}[], Float64[])
    @test yields(ExtendedMixtureModel(components, [1.0, -1.0])) == [1.0, -1.0]
    @test_throws DomainError MixtureModel(ExtendedMixtureModel(components, [2.0, -1.0]))
    @test_throws ArgumentError MixtureModel(ExtendedMixtureModel(components, [0.0, 0.0]))
end

@testset "ExtendedMixtureModel multivariate" begin
    components = [
        MvNormal([0.0, 0.0], Diagonal([1.0, 1.0])),
        MvNormal([1.0, 1.0], Diagonal([0.25, 0.25])),
    ]
    model = ExtendedMixtureModel(components, [3.0, 7.0])

    @test length(model) == 2

    x = [0.2, 0.4]
    expected_density = 3.0 * pdf(components[1], x) + 7.0 * pdf(components[2], x)
    @test model(x) ≈ expected_density

    data = hcat([0.2, 0.4], [1.0, 1.1])
    expected_nll = -sum(log(model(col)) for col in eachcol(data)) + total_yield(model)
    @test extended_negative_log_likelihood(model, data) ≈ expected_nll
end

@testset "ExtendedMixtureModel support" begin
    signal = product_distribution([Normal(-1.0, 0.5), Exponential(2.0)])
    swapped = product_distribution([Exponential(2.0), Normal(-1.0, 0.5)])
    background = product_distribution([Uniform(-2.0, 2.0), Normal(1.0, 1.5)])
    mixed = MixtureModel([signal, swapped], [0.25, 0.75])

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

    gaussian_product = product_distribution([Normal(-1.0, 0.5), Normal(1.0, 1.5)])
    gaussian_model = ExtendedMixtureModel([gaussian_product], [10.0])
    @test support(gaussian_model) == Distributions.RealInterval.([-Inf, -Inf], [Inf, Inf])

    univariate_model = ExtendedMixtureModel([Normal(-1.0, 0.5), Uniform(-2.0, 2.0)], [3.0, 1.0])
    @test minimum(univariate_model) == -Inf
    @test maximum(univariate_model) == Inf
    @test support(univariate_model) == Distributions.RealInterval(-Inf, Inf)
    @test minimum(univariate_model) == support(univariate_model).lb
    @test maximum(univariate_model) == support(univariate_model).ub
end
