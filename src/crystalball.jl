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
    f(x; μ, σ, α, n) = N * exp(-(x̂^2)/2)           for x̂ > -α
                    = N * A * (B - x̂)^(-n)         for x̂ ≤ -α
````
where x̂ = (x - μ) / σ.
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
```julia
using DistributionsHEP
using Plots

d = CrystalBall(μ=0.0, σ=1.0, α=2.0, n=3.2)
plot(-2, 4, x->pdf(d, x))
"""
struct CrystalBall{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    α::T
    n::T
    # Precomputed constants for PDF calculation
    norm_const::T # Normalization constant N
    A_const::T    # Tail parameter A
    B_const::T    # Tail parameter B

    function CrystalBall(μ::T, σ::T, α::T, n::T) where {T<:Real}
        _check_crystalball_params(σ, α, n)
        # absα is effectively α due to the check α > 0
        C = n / α / (n - 1) * exp(-α^2 / 2)
        D_val = sqrt(T(π) / 2) * (one(T) + erf(α / sqrt(T(2))))
        N = one(T) / (σ * (C + D_val))

        A = (n / α)^n * exp(-α^2 / 2)
        B = n / α - α
        new{T}(μ, σ, α, n, N, A, B)
    end
end

"""
    pdf(d::CrystalBall, x::Real)

Compute the probability density function (PDF) of the Crystal Ball distribution `d` at point `x`.

The function uses precomputed normalization and tail parameters stored within the `CrystalBall` struct for efficiency.
It switches between the Gaussian core and the power-law tail based on the value of `x` relative to the transition point defined by `α`.
"""
function Distributions.pdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    x̂ = (x - d.μ) / d.σ
    # Gaussian part
    x̂ > -d.α && return d.norm_const * exp(-x̂^2 / 2)
    # Power-law tail part
    return d.norm_const * d.A_const * (d.B_const - x̂)^(-d.n)
end

"""
    cdf(d::CrystalBall, x::Real)

Compute the cumulative distribution function (CDF) of the Crystal Ball distribution `d` at point `x`.

The CDF is calculated by integrating the PDF. This implementation handles the integral of the power-law tail and the Gaussian core separately, ensuring continuity at the transition point.
"""
function Distributions.cdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    x̂ = (x - d.μ) / d.σ

    # Value of the CDF at the transition point x̂ = -α
    cdf_at_minus_alpha = d.norm_const * d.A_const / (d.n - 1) * (d.B_const - (-d.α))^(1 - d.n)

    if x̂ <= -d.α
        # CDF for the power-law tail part (x̂ ≤ -α)
        # Integral of PDF from -Inf to x̂
        return d.norm_const * d.A_const / (d.n - 1) * (d.B_const - x̂)^(1 - d.n)
    else
        # CDF for the Gaussian part (x̂ > -α)
        # CDF at -α + integral of Gaussian PDF from -α to x̂
        integral_gaussian_part = sqrt(T(π) / 2) * (erf(x̂ / sqrt(T(2))) + erf(d.α / sqrt(T(2))))
        return cdf_at_minus_alpha + d.norm_const * d.σ * integral_gaussian_part
    end
end