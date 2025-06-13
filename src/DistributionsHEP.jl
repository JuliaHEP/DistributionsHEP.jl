module DistributionsHEP

using Random
using Distributions
using SpecialFunctions
import Distributions: @check_args

export Chebyshev
include("chebychev.jl")

export ArgusBG
include("argusBG.jl")

export CrystalBall
include("crystalball.jl")

end
