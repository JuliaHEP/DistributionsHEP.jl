using SpecialFunctions
using DistributionsHEP
using Distributions
using QuadGK
using Test

# Create a test instance of the RelativisticBreitWigner distribution
d = RelativisticBreitWigner(0.1, 1.0) # M, Γ

###########################################################
#   MAIN TESTSET FOR THE RELATIVISTIC BREIT-WIGNER DIST   #
###########################################################

@testset "RelativisticBreitWigner Distribution" verbose = true begin

    ###################################################
    # 1. PARAMETER VALIDATION TESTS                  #
    ###################################################
    @testset "Parameter validation" begin
        # M must be positive
        @test_throws ErrorException RelativisticBreitWigner(-1.0, 1.0)

        # Γ must be positive
        @test_throws ErrorException RelativisticBreitWigner(1.0, -1.0)
    end


    ###################################################
    # 2. PDF PROPERTIES                               #
    ###################################################
    @testset "PDF properties" begin
        # Test some x-values (far from the peak)
        x_test_points = [85.0, 87.0, 89.0, 91.0, 93.0, 95.0]

        for x in x_test_points
            # PDF should always be non-negative
            @test pdf(d, x) > 0

            # PDF should always be finite (no Inf, no NaN)
            @test isfinite(pdf(d, x))
            @test isfinite(pdf(d, x))
        end
    end


    ###################################################
    # 2.5. NORMALIZATION TEST                         #
    ###################################################
    @testset "Normalization" begin
        # PDF should integrate to 1 over its support [0, Inf)
        numerical_integral = quadgk(x -> pdf(d, x), 0.0, Inf)[1]
        @test isapprox(numerical_integral, 1.0; atol=1e-6)
    end


    ###################################################
    # 3. CDF PROPERTIES                               #
    ###################################################
    @testset "CDF properties" begin

        # Check that CDF never exceeds 1
        x_values = [85.0, 87.0, 89.0, 91.0, 93.0, 95.0]
        for x in x_values
            cdf_val = cdf(d, x)

            # CDF(x) must be ≤ 1
            @test cdf_val <= 1
        end

        # CDF for very negative x should be near zero
        @test cdf(d, -100) < 0.01

        # CDF far above the peak should approach 1
        # (Allowing a relaxed tolerance because of complex arithmetic)
        @test isapprox(cdf(d, d.M + 5 * d.Γ), 1; atol=1e-4)
    end


    ###################################################
    # 3.5. CDF VS QUAD TEST                           #
    ###################################################
    @testset "CDF vs numerical integration" begin
        # Compare CDF with numerical integration of PDF
        x_test_points = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        for x in x_test_points
            cdf_val = cdf(d, x)
            quad_val = quadgk(t -> pdf(d, t), 0.0, x)[1]
            @test isapprox(cdf_val, quad_val; atol=1e-5)
        end
    end

    ###################################################
    # 4. TYPE STABILITY TESTS                         #
    ###################################################
    @testset "Type stability" begin

        # Several RBW distributions with different numeric types
        d_float64 = RelativisticBreitWigner(0.1, 1.0)            # Float64
        d_float32 = RelativisticBreitWigner(0.1f0, 1.0f0)        # Float32
        d_int = RelativisticBreitWigner(1, 2)                # integers → promoted to Float64
        d_mix = RelativisticBreitWigner(1, 2.0f0)            # mixed types → promoted properly

        # PDF return type must match parameter type
        @test pdf(d_float64, 0.0) isa Float64
        @test pdf(d_float32, 0.0f0) isa Float32

        # CDF return type must also match parameter type
        @test cdf(d_float64, 0.0) isa Float64
        @test cdf(d_float32, 0.0f0) isa Float32

        # Integer constructor should promote to Float64
        @test pdf(d_int, 1) isa Float64
        @test cdf(d_int, 1) isa Float64

        # Mixed type constructor should promote to Float32
        @test pdf(d_mix, 2) isa Float32
        @test cdf(d_mix, 2) isa Float32
    end

end