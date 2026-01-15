using Test
using DistributionsHEP
using SpecialFunctions
using QuadGK

# Access internal functions via the module
_value = DistributionsHEP._value
_integral = DistributionsHEP._integral
_integral_inversion = DistributionsHEP._integral_inversion
_gaussian_norm_const = DistributionsHEP._gaussian_norm_const

@testset "UnNormGauss internal functions" verbose = true begin

    @testset "_value function correctness" begin
        # Test with σ = 1
        g1 = DistributionsHEP.UnNormGauss(0.0, 1.0)
        
        # _value should return: exp(-((x-μ)/σ)²/2) / σ
        # For g1: μ=0, σ=1, so _value(g1, x) = exp(-x²/2) / 1 = exp(-x²/2)
        
        test_cases = [
            (0.0, exp(-0.0^2 / 2) / 1.0),  # At mean: exp(0) / 1 = 1.0
            (1.0, exp(-1.0^2 / 2) / 1.0),  # At 1σ: exp(-0.5) / 1
            (-1.0, exp(-(-1.0)^2 / 2) / 1.0),  # At -1σ: exp(-0.5) / 1
            (2.0, exp(-2.0^2 / 2) / 1.0),  # At 2σ: exp(-2) / 1
        ]
        
        for (x, expected) in test_cases
            actual = _value(g1, x)
            @test isapprox(actual, expected; rtol=1e-15)
        end
        
        # Test with σ = 2
        g2 = DistributionsHEP.UnNormGauss(0.0, 2.0)
        # _value(g2, x) = exp(-(x/2)²/2) / 2 = exp(-x²/8) / 2
        
        test_cases_σ2 = [
            (0.0, exp(-0.0^2 / 8) / 2.0),  # At mean: exp(0) / 2 = 0.5
            (2.0, exp(-2.0^2 / 8) / 2.0),  # At 1σ: exp(-4/8) / 2 = exp(-0.5) / 2
            (4.0, exp(-4.0^2 / 8) / 2.0),  # At 2σ: exp(-16/8) / 2 = exp(-2) / 2
        ]
        
        for (x, expected) in test_cases_σ2
            actual = _value(g2, x)
            @test isapprox(actual, expected; rtol=1e-15)
        end
        
        # Test with non-zero mean
        g3 = DistributionsHEP.UnNormGauss(5.0, 1.0)
        # _value(g3, x) = exp(-((x-5)/1)²/2) / 1 = exp(-(x-5)²/2)
        
        @test isapprox(_value(g3, 5.0), exp(-0.0^2 / 2) / 1.0; rtol=1e-15)  # At mean
        @test isapprox(_value(g3, 6.0), exp(-1.0^2 / 2) / 1.0; rtol=1e-15)  # At μ+σ
    end

    @testset "_integral function correctness" begin
        g = DistributionsHEP.UnNormGauss(0.0, 1.0)
        
        # _integral should return: sqrt(π/2) * (1 + erf((a-μ)/(σ*sqrt(2))))
        # For g: μ=0, σ=1, so _integral(g, a) = sqrt(π/2) * (1 + erf(a/sqrt(2)))
        
        norm_const = _gaussian_norm_const(Float64)
        
        test_cases = [
            (0.0, norm_const * (1 + erf(0.0 / sqrt(2)))),  # At mean
            (1.0, norm_const * (1 + erf(1.0 / sqrt(2)))),  # At 1σ
            (-1.0, norm_const * (1 + erf(-1.0 / sqrt(2)))),  # At -1σ
            (2.0, norm_const * (1 + erf(2.0 / sqrt(2)))),  # At 2σ
        ]
        
        for (a, expected) in test_cases
            actual = _integral(g, a)
            @test isapprox(actual, expected; rtol=1e-15)
        end
        
        # Test that _integral correctly integrates _value numerically
        for a in [-2.0, -1.0, 0.0, 1.0, 2.0]
            # Numerically integrate _value from -∞ (approximated as -1000) to a
            numerical_integral, err = QuadGK.quadgk(x -> _value(g, x), -1000.0, a, rtol=1e-10)
            analytical_integral = _integral(g, a)
            @test isapprox(analytical_integral, numerical_integral; rtol=1e-8)
        end
        
        # Test with different σ
        g2 = DistributionsHEP.UnNormGauss(0.0, 2.0)
        for a in [-2.0, 0.0, 2.0]
            numerical_integral, err = QuadGK.quadgk(x -> _value(g2, x), -1000.0, a, rtol=1e-10)
            analytical_integral = _integral(g2, a)
            @test isapprox(analytical_integral, numerical_integral; rtol=1e-8)
        end
    end

    @testset "_integral_inversion round-trip" begin
        g = DistributionsHEP.UnNormGauss(0.0, 1.0)
        
        # Test: a -> _integral -> _integral_inversion -> a
        test_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
        for a in test_points
            integral_val = _integral(g, a)
            a_recovered = _integral_inversion(g, integral_val)
            @test isapprox(a_recovered, a; rtol=1e-10)
        end
        
        # Test: integral -> _integral_inversion -> _integral -> integral
        norm_const = _gaussian_norm_const(Float64)
        test_integrals = [0.1, 0.5, 1.0, 1.5, 2.0, 2.4]
        for integral_val in test_integrals
            if integral_val < 2 * norm_const  # Only test valid integrals
                a = _integral_inversion(g, integral_val)
                if isfinite(a)
                    integral_recovered = _integral(g, a)
                    @test isapprox(integral_recovered, integral_val; rtol=1e-10)
                end
            end
        end
    end

    @testset "Derivative relationship" begin
        # The derivative of _integral should equal _value
        # d/dx ∫[-∞ to x] _value(t) dt = _value(x)
        g = DistributionsHEP.UnNormGauss(0.0, 1.0)
        
        test_points = [-1.0, 0.0, 1.0, 2.0]
        h = 1e-6  # Small step for numerical derivative
        
        for x in test_points
            integral_at_x = _integral(g, x)
            integral_at_x_plus_h = _integral(g, x + h)
            numerical_derivative = (integral_at_x_plus_h - integral_at_x) / h
            actual_value = _value(g, x)
            
            @test isapprox(numerical_derivative, actual_value; rtol=1e-5)
        end
    end

    @testset "σ scaling behavior" begin
        # Test that _value correctly includes the 1/σ factor
        # For the same scaled coordinate x̂ = (x-μ)/σ, _value should scale with 1/σ
        
        μ = 0.0
        σ1 = 1.0
        σ2 = 2.0
        
        g1 = DistributionsHEP.UnNormGauss(μ, σ1)
        g2 = DistributionsHEP.UnNormGauss(μ, σ2)
        
        # For the same x̂ = (x-μ)/σ, we expect:
        # _value(g1, μ + x̂*σ1) / _value(g2, μ + x̂*σ2) = σ2 / σ1 = 2
        
        for x̂ in [-2.0, -1.0, 0.0, 1.0, 2.0]
            x1 = μ + x̂ * σ1  # x for g1
            x2 = μ + x̂ * σ2  # x for g2 (same scaled coordinate)
            
            val1 = _value(g1, x1)
            val2 = _value(g2, x2)
            
            # Since both have the same exp(-x̂²/2) factor, but g2 has 1/σ2 = 1/2
            # and g1 has 1/σ1 = 1, we expect val1 / val2 = (1/1) / (1/2) = 2
            @test isapprox(val1 / val2, σ2 / σ1; rtol=1e-15)
        end
    end
end
