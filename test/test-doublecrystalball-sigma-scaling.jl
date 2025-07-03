using Test
using DistributionsHEP

# Helper to check quantile accuracy
function check_quantile_accuracy(d, ps; atol = 1e-8)
    for p in ps
        q = quantile(d, p)
        cdf_val = cdf(d, q)
        @test isapprox(cdf_val, p; atol = atol) || @warn "Quantile test failed" p q cdf_val
    end
end

@testset "DoubleCrystalBall quantile accuracy for sigma > 1" begin
    d = DoubleCrystalBall(0.0, 5.0, 0.5, 1.5, 3.0, 5.0)
    ps = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    check_quantile_accuracy(d, ps)
end

@testset "DoubleCrystalBall quantile accuracy for sigma < 1" begin
    d = DoubleCrystalBall(0.0, 0.5, 0.5, 1.5, 3.0, 5.0)
    ps = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    check_quantile_accuracy(d, ps)
end
