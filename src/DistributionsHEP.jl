module DistributionsHEP

using Random
using Distributions
using SpecialFunctions

import Distributions: pdf, cdf, @check_args
export pdf, cdf

export Chebyshev
include("chebychev.jl")

export ArgusBG
include("argusBG.jl")

export CrystalBall
include("crystalball.jl")

end
