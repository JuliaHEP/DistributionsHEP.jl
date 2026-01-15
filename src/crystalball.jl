# Common parameter validation
function _check_crystalball_params(σ::T, α::T, n::T) where {T<:Real}
    σ > zero(T) || error("σ (scale) must be positive.")
    α > zero(T) || error("α (transition point) must be positive.")
    n > one(T) || error("n (power-law exponent) must be greater than 1.")
end

"""
    CrystalBall{T<:Real} <: ContinuousUnivariateDistribution

The Crystal Ball distribution is a probability distribution commonly used in high-energy physics to model various lossy processes.
It consists of a Gaussian core and a power-law tail on one side (typically the left side, below the mean).

The probability density function is defined as:
````math
f(x; μ, σ, α, n) = \\begin{cases}
    N \\exp\\left(-\\frac{\\hat{x}^2}{2}\\right) & \\text{for } \\hat{x} > -α \\\\
    N A (B - \\hat{x})^{-n} & \\text{for } \\hat{x} \\leq -α
\\end{cases}
````
where ``\\hat{x} = (x - μ) / σ``.
The parameters A and B are derived from α and n to ensure continuity of the function and its first derivative.
N is a normalization constant.

This implementation defines the standard Crystal Ball function with the power-law tail on the left side of the Gaussian peak.

# Arguments
- `μ`: The mean of the Gaussian core.
- `σ`: The standard deviation of the Gaussian core. Must be positive.
- `α`: The transition point, defining where the power-law tail begins. It is a positive value representing the number of standard deviations (σ) from the mean (μ) to the transition point.
- `n`: The exponent of the power-law tail. Must be greater than 1 for the distribution to be normalizable.

The struct also stores precomputed constants `norm_const` (N), `A_const` (A), and `B_const` (B) for efficient PDF and CDF calculations. These are not direct user inputs but are derived from the primary parameters.

# Example
````julia
using DistributionsHEP
using Plots

d = CrystalBall(0.0, 1.0, 2.0, 3.2)  # μ, σ, α, n
plot(-2, 4, x->pdf(d, x))
````
"""
struct CrystalBall{T<:Real} <: ContinuousUnivariateDistribution
    tail::CrystalBallTail{T}  # Tail parameters
    gauss::Normal{T}          # Normal distribution helper
    # Precomputed constants for PDF calculation
    norm_const::T # Normalization constant N

    function CrystalBall(μ::T, σ::T, α::T, n::T) where {T<:Real}
        _check_crystalball_params(σ, α, n)

        gauss = Normal(μ, σ)
        x0 = μ - α * σ
        # Use pdf from Normal to get normalized G_x0 at transition point
        G_x0 = pdf(gauss, x0)
        L_x0 = α
        tail = CrystalBallTail(G_x0, n, L_x0, x0)

        tail_contribution = _integral(tail, tail.x0)
        # Integral from x0 to +∞ = integral from -∞ to +∞ - integral from -∞ to x0
        integral_full = cdf(gauss, T(Inf))
        integral_to_x0 = cdf(gauss, tail.x0)
        core_contribution = integral_full - integral_to_x0
        N = one(T) / (tail_contribution + core_contribution)

        new{T}(tail, gauss, N)
    end
end

function Distributions.pdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    x > d.tail.x0 && return d.norm_const * pdf(d.gauss, x)

    # Tail part: _value expects absolute offset (x - x0), not scaled
    offset = x - d.tail.x0
    return d.norm_const * _value(d.tail, offset)
end

function Distributions.cdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    # Left tail
    x <= d.tail.x0 && return d.norm_const * _integral(d.tail, x)
    # Gaussian core
    cdf_at_x0 = d.norm_const * _integral(d.tail, d.tail.x0)  # CDF at transition point
    # Integral from x0 to x = integral from -∞ to x - integral from -∞ to x0
    integral_gaussian_part = cdf(d.gauss, x) - cdf(d.gauss, d.tail.x0)
    return cdf_at_x0 + d.norm_const * integral_gaussian_part
end

function Distributions.quantile(d::CrystalBall{T}, p::Real) where {T<:Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    cdf_at_x0 = d.norm_const * _integral(d.tail, d.tail.x0)

    if p <= cdf_at_x0
        # tail part
        tail_integral = p / d.norm_const
        return _integral_inversion(d.tail, tail_integral)
    else
        # Gaussian part
        gaussian_integral_at_x0 = cdf(d.gauss, d.tail.x0)
        gaussian_integral = (p - cdf_at_x0) / d.norm_const + gaussian_integral_at_x0
        return quantile(d.gauss, gaussian_integral)
    end
end

Distributions.maximum(d::CrystalBall{T}) where {T<:Real} = T(Inf)
Distributions.minimum(d::CrystalBall{T}) where {T<:Real} = T(-Inf)

# Distributions.jl interface methods
Distributions.location(d::CrystalBall) = d.gauss.μ
Distributions.scale(d::CrystalBall) = d.gauss.σ


