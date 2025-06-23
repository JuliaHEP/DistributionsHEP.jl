module DistributionsHEP

using SpecialFunctions
using Distributions
using Polynomials
using Random

import Distributions: @check_args
import Distributions.Statistics: mean, std, var, quantile
import Distributions.StatsBase: kurtosis, skewness
import Base: maximum, minimum, rand

export pdf, cdf, quantile, support
export mean, std, var, skewness, kurtosis

export Chebyshev
include("chebyshev.jl")

export ArgusBG
include("argusBG.jl")

export CrystalBall
include("crystalball.jl")

end
