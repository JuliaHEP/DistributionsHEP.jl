using DistributionsHEP
using Distributions
using Test

include("crystal-ball-test-helpers.jl")

@testset "Crystal Ball C¹ tail matching" verbose = true begin
    # σ values deliberately include regimes where the #41 bug was visible (σ ≠ 1)
    σ_values = [0.3, 0.5, 1.0, 5.0]

    @testset "CrystalBall derivative continuity" begin
        for σ in σ_values
            @testset "σ = $σ" begin
                d = CrystalBall(0.0, σ, 1.6, 10.0)
                test_pdf_derivative_continuous(d, d.tail.x0)
                test_tail_log_derivative(d.tail, d.gauss)
            end
        end
    end

    @testset "DoubleCrystalBall derivative continuity" begin
        for σ in σ_values
            @testset "σ = $σ" begin
                d = DoubleCrystalBall(0.0, σ, 1.6, 10.0, 1.6, 10.0)
                for tail in (d.left_tail, d.right_tail)
                    test_pdf_derivative_continuous(d, tail.x0)
                    test_tail_log_derivative(tail, d.gauss)
                end
            end
        end
    end

    @testset "Tail exponent n affects deep tail when σ ≠ 1" begin
        σ, α = 0.3, 1.6
        x0L = -α * σ
        x_deep_left = x0L - 0.5
        x0R = α * σ
        x_deep_right = x0R + 0.5

        d_cb_low = CrystalBall(0.0, σ, α, 2.0)
        d_cb_high = CrystalBall(0.0, σ, α, 100.0)
        @test pdf(d_cb_high, x_deep_left) < pdf(d_cb_low, x_deep_left)

        d_dcb_low = DoubleCrystalBall(0.0, σ, α, 2.0, α, 2.0)
        d_dcb_high = DoubleCrystalBall(0.0, σ, α, 100.0, α, 100.0)
        @test pdf(d_dcb_high, x_deep_left) < pdf(d_dcb_low, x_deep_left)
        @test pdf(d_dcb_high, x_deep_right) < pdf(d_dcb_low, x_deep_right)

        # With the #41 bug, pdf at x_deep_left was nearly identical for n = 10 vs n = 100.
        d_n10 = DoubleCrystalBall(0.0, σ, α, 10.0, α, 10.0)
        d_n100 = DoubleCrystalBall(0.0, σ, α, 100.0, α, 100.0)
        rel_diff = abs(pdf(d_n10, x_deep_left) - pdf(d_n100, x_deep_left)) /
                   pdf(d_n10, x_deep_left)
        @test rel_diff > 0.15
    end
end
