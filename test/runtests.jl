using Test

@testset "DistributionsHEP tests" verbose=true begin 
    include("test-chebyshev.jl")
    include("test-argusBG.jl")
    include("test-crystalball.jl")
    include("test-double-sided-crystal-ball.jl")

    #Add Relativistic_Breit–Wigner_distribution
    include("test-relativistic-breit-wigner.jl")
end
