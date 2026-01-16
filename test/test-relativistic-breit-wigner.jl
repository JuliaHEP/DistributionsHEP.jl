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
        x_test_points = [0.0, 10, 85.0, 87.0, 89.0, 91.0, 93.0, 95.0]

        for x in x_test_points
            # PDF should always be non-negative
            @test pdf(d, x) >= 0

            # PDF should always be finite (no Inf, no NaN)
            @test isfinite(pdf(d, x))
        end
    end


    ###################################################
    # 3. CDF PROPERTIES                               #
    ###################################################
    @testset "CDF properties" begin

        # Check that CDF never exceeds 1 and is non-negative
        x_values = [0, 10, 85.0, 87.0, 89.0, 91.0, 93.0, 95.0]
        for x in x_values
            cdf_val = cdf(d, x)

            # CDF(x) must be in [0, 1]
            @test cdf_val >= 0
            @test cdf_val <= 1
        end

        # CDF for very negative x should be near zero
        @test cdf(d, -100) < 0.01

        # CDF far above the peak should approach 1
        @test isapprox(cdf(d, d.M + 5 * d.Γ), 1; atol=1e-4)

        # CDF at x=0 should be 1/2 by symmetry
        @test isapprox(cdf(d, 0.0), 0.5; rtol=1e-6)
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


    ###################################################
    # 5. CDF vs QUADGK NUMERICAL INTEGRATION          #
    ###################################################
    @testset "CDF vs quadgk numerical integration" begin
        # Test multiple parameter combinations
        test_cases = [
            (0.1, 1.0),
            (1.0, 0.5),
            (2.0, 1.5),
            (5.0, 2.0),
        ]

        for (M, Γ) in test_cases
            d_test = RelativisticBreitWigner(M, Γ)

            # Test points across the distribution (full real line)
            x_test_points = [
                0.25 * M,             # Near right
                0.5 * M,             # Near right
                M,                   # At peak
                1.5 * M,             # Above peak
                2.0 * M,             # Well above peak
                5.0 * M,             # Far right
                10.0 * M,            # Very far right
            ]

            for x in x_test_points
                # Compute CDF using closed-form formula
                cdf_closed = cdf(d_test, x)

                # Compute CDF using numerical integration from -∞ to x
                # Use a large negative bound for -∞
                cdf_numerical, _ = quadgk(t -> pdf(d_test, t), -1000.0 * M, x, rtol=1e-8)

                atol = 1e-6
                # Compare with reasonable tolerance
                if abs(cdf_closed) < atol && abs(cdf_numerical) < atol
                    # Both near zero, test passes
                    @test true
                elseif abs(cdf_closed - 1.0) < atol && abs(cdf_numerical - 1.0) < atol
                    # Both near one, test passes
                    @test true
                else
                    # Compare with relative tolerance
                    @test isapprox(cdf_closed, cdf_numerical; atol=atol)
                end
            end
        end

        # Test CDF at x=0 (should be 1/2 by symmetry for full distribution)
        for (M, Γ) in test_cases
            d_test = RelativisticBreitWigner(M, Γ)
            cdf_at_zero = cdf(d_test, 0.0)
            # The closed-form CDF at x=0 should be 1/2 (by symmetry)
            @test isapprox(cdf_at_zero, 0.5, rtol=1e-6)
        end

        # Test CDF limits: very negative should be near 0, very positive should be near 1
        d_test = RelativisticBreitWigner(1.0, 0.5)
        @test cdf(d_test, -100.0) < 1e-6
        @test cdf(d_test, 100.0) > 1.0 - 1e-6
    end

end