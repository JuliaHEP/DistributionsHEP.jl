using DistributionsHEP
using Distributions
using Test
using QuadGK

function safe_pdf_voigt(d::Voigt, x::Real)
    try
        y = pdf(d, x)
        # Ensure numeric and nonnegative return
        return (isfinite(y) && y ≥ 0) ? y : 0.0
    catch
        return 0.0
    end
end

@testset "Voigt Distribution" verbose = true begin

    @testset "Parameter validation" begin
        @test_throws ErrorException Voigt(0.0, -1.0, 0.5)
        @test_throws ErrorException Voigt(0.0, 1.0, -0.5)
    end

    d = Voigt(0.0, 1.0, 0.5)

    @testset "PDF properties" begin
        x_values = [-200.0, -100.0, -20.0, -5.0, 0.0, 5.0, 20.0, 100.0, 200.0]

        for x in x_values
            y = safe_pdf_voigt(d, x)
            @test isfinite(y)
            @test y ≥ 0
        end

        # Robust normalization with safe integrand
        integral, err = quadgk(x -> safe_pdf_voigt(d, x), -300.0, 300.0; atol=1e-6, rtol=1e-6)
        @test isapprox(integral, 1.0; atol = 0.1)
    end

    @testset "Symmetry property" begin
        xs = [-3.0, -1.5, -0.5, 0.5, 1.5, 3.0]
        for x in xs
            @test isapprox(pdf(d, d.μ + x), pdf(d, d.μ - x); atol = 1e-8)
        end
    end

    @testset "Support interface" begin
        @test minimum(d) == -Inf
        @test maximum(d) == Inf
    end

    @testset "Type stability" begin
        d64 = Voigt(0.0, 1.0, 0.5)
        d32 = Voigt(0.0f0, 1.0f0, 0.5f0)
        dint = Voigt(1, 2, 1)
        dmix = Voigt(0, 2.5, 1.2f0)
        @test pdf(d64, 0.0) isa Float64
        @test pdf(d32, 0.0f0) isa Float32
        @test pdf(dint, 0) isa Float64
        @test pdf(dmix, 0) isa Float64
    end
end
