using Test

@testset "DistributionsHEP tests" verbose = true begin
    include("test-chebyshev.jl")
    include("test-argusBG.jl")
    include("test-crystalball.jl")
    include("test-double-sided-crystal-ball.jl")
    include("test-secant.jl")
    include("test-bifurcated-gaussian.jl")
    include("test-double-sided-bifurcated-crystal-ball.jl")
    include("test-das-function.jl")
end
