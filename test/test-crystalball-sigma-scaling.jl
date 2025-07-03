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

@testset "CrystalBall quantile accuracy for sigma > 1" begin
    d = CrystalBall(0.0, 5.0, 2.0, 3.2)
    ps = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    check_quantile_accuracy(d, ps)
end

@testset "CrystalBall quantile accuracy for sigma < 1" begin
    d = CrystalBall(0.0, 0.5, 2.0, 3.2)
    ps = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    check_quantile_accuracy(d, ps)
end

@testset "CrystalBall CDF normalization" begin
    # Test that CDF approaches 1 for large x
    d1 = CrystalBall(0.0, 1.0, 2.0, 3.2)
    d5 = CrystalBall(0.0, 5.0, 2.0, 3.2)
    d05 = CrystalBall(0.0, 0.5, 2.0, 3.2)

    for d in [d1, d5, d05]
        cdf_large = cdf(d, 1000)
        @test cdf_large > 0.999
    end
end
