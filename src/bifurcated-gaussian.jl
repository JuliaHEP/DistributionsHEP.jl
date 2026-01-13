# Common parameter validation for bifurcated gaussian
function _check_bifurcated_gaussian_params(σ::T) where {T<:Real}
    σ > zero(T) || error("σ (scale) must be positive.")
end

"""
    BifurcatedGaussian{T<:Real} <: ContinuousUnivariateDistribution

The Bifurcated Gaussian distribution is a continuous probability distribution that extends
the standard Gaussian distribution by allowing different scale parameters on the left and
right sides of the mean. This creates an asymmetric distribution that is useful for
modeling data with different tail behaviors on either side of the peak.

The probability density function is defined as:
````math
f(x; μ, σ, ψ) = \\begin{cases}
    \\frac{1}{\\sqrt{2π} σ} \\exp\\left(-\\frac{(x-μ)^2}{2σ_L^2}\\right) & \\text{for } x \\leq μ \\\\
    \\frac{1}{\\sqrt{2π} σ} \\exp\\left(-\\frac{(x-μ)^2}{2σ_R^2}\\right) & \\text{for } x > μ
\\end{cases}
````

where:
- ``σ_L = σ(1 + κ)`` is the left-side scale parameter
- ``σ_R = σ(1 - κ)`` is the right-side scale parameter  
- ``κ = \\tanh(ψ)`` is the asymmetry parameter (bounded between -1 and 1)

The distribution is continuous at ``x = μ`` and integrates to 1. When ``ψ = 0`` (so ``κ = 0``),
the distribution reduces to a standard Gaussian with scale ``σ``.

# Arguments
- `μ`: Location parameter (mean/mode of the distribution)
- `σ`: Base scale parameter. Must be positive. This is the average of the left and right scales.
- `ψ`: Asymmetry parameter. Controls the difference between left and right scales via ``κ = \\tanh(ψ)``.
  When ``ψ = 0``, the distribution is symmetric (standard Gaussian).

The struct stores precomputed constants `σL` and `σR` for efficient PDF and CDF calculations.

# Example
```julia
using DistributionsHEP
using Plots

# Symmetric case (standard Gaussian)
d_sym = BifurcatedGaussian(0.0, 1.0, 0.0)

# Asymmetric case
d_asym = BifurcatedGaussian(0.0, 1.0, 0.5)  # μ, σ, ψ

# Evaluate PDF, CDF, and generate random samples
pdf(d_asym, 0.0)
cdf(d_asym, 1.0)
quantile(d_asym, 0.5)
rand(d_asym)
```
"""
struct BifurcatedGaussian{T<:Real} <: ContinuousUnivariateDistribution
    μ::T # elementary parameters
    σ::T
    ψ::T
    # Precomputed constants for PDF calculation (from elementary parameters)
    σL::T  # Left side parameter σ_L
    σR::T    # Right side parameter σ_R

    function BifurcatedGaussian(μ::T, σ::T, ψ::T) where {T<:Real}
        _check_bifurcated_gaussian_params(σ)

        # Calculate kappa
        κ = tanh(ψ)

        # Calculate scales for left and right sides
        σL = σ * (1 + κ)
        σR = σ * (1 - κ)

        new{T}(μ, σ, ψ, σL, σR)
    end
end


function Distributions.pdf(d::BifurcatedGaussian{T}, x::Real) where {T<:Real}
    # left side
    x <= d.μ && return one(T) / (sqrt(T(2π)) * d.σ) * exp(-T(0.5) * ((x - d.μ) / d.σL)^2)
    # right side
    return one(T) / (sqrt(T(2π)) * d.σ) * exp(-T(0.5) * ((x - d.μ) / d.σR)^2)
end

function Distributions.cdf(d::BifurcatedGaussian{T}, x::Real) where {T<:Real}

    # CDF values at transition point (from mathematical derivation)
    cdf_at_mu = d.σL / (T(2) * d.σ)

    if x <= d.μ
        return cdf_at_mu * (1 + erf((x - d.μ) / (d.σL * sqrt(T(2)))))
    else
        return cdf_at_mu + d.σR / (T(2) * d.σ) * erf((x - d.μ) / (d.σR * sqrt(T(2))))
    end
end

function Distributions.quantile(d::BifurcatedGaussian{T}, p::Real) where {T<:Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    # CDF values at transition points (same as in CDF function)
    cdf_at_mu = d.σL / (T(2) * d.σ)

    if p <= cdf_at_mu
        # Quantile is in the left side
        return d.μ + d.σL * sqrt(T(2)) * erfinv(p * (T(2) * d.σ) / d.σL - 1)
    else
        return d.μ + d.σR * sqrt(T(2)) * erfinv((p - cdf_at_mu) * (T(2) * d.σ) / d.σR)
    end
end

Distributions.maximum(d::BifurcatedGaussian{T}) where {T<:Real} = T(Inf)
Distributions.minimum(d::BifurcatedGaussian{T}) where {T<:Real} = T(-Inf)
