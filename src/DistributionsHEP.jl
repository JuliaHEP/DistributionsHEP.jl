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
include("crystal-ball-tail.jl")
include("crystalball.jl")

export DoubleCrystalBall
include("double-sided-crystal-ball.jl")

export HyperbolicSecant
include("secant.jl")

export BifurcatedGaussian
include("bifurcated-gaussian.jl")

export DoubleSidedBifurcatedCrystalBall
include("double-sided-bifurcated-crystal-ball.jl")

export DoubleSidedBifurcatedCrystalBallDas
include("exponential-tail.jl")
include("double-sided-bifurcated-crystal-ball-das.jl")

end
