module DistributionsHEP

using Random
using Distributions
using SpecialFunctions

import Distributions: @check_args
import Distributions: pdf, cdf, mean, var, skewness, kurtosis, entropy
export pdf, cdf, mean, var, skewness, kurtosis, entropy

export Chebyshev
include("chebychev.jl")

export ArgusBG
include("argusBG.jl")

export CrystalBall
include("crystalball.jl")

end
