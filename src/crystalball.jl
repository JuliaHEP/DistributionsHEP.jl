# Common parameter validation
function _check_crystalball_params(σ::T, α::T, n::T) where {T <: Real}
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
    f(x; μ, σ, α, n) = N exp(-(x̂^2)/2)      for x̂ > -α
                    = N  A (B - x̂)^{-n}     for x̂ ≤ -α
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

d = CrystalBall(0.0, 1.0, 2.0, 3.2)  # μ, σ, α, n
plot(-2, 4, x->pdf(d, x))
"""
struct CrystalBall{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    α::T
    n::T
    # Precomputed constants for PDF calculation
    norm_const::T # Normalization constant N
    A_const::T    # Tail parameter A
    B_const::T    # Tail parameter B

    function CrystalBall(μ::T, σ::T, α::T, n::T) where {T <: Real}
        _check_crystalball_params(σ, α, n)
        # Calculate normalization constant
        # The normalization should be: N = 1 / (C + D_val)
        # where C is the tail contribution and D_val is the Gaussian core contribution
        C = n / α / (n - 1) * exp(-α^2 / 2)
        D_val = sqrt(T(π) / 2) * (one(T) + erf(α / sqrt(T(2))))
        N = one(T) / (C + D_val)

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
function Distributions.pdf(d::CrystalBall{T}, x::Real) where {T <: Real}
    x̂ = (x - d.μ) / d.σ
    # Gaussian part
    x̂ > -d.α && return d.norm_const * exp(-x̂^2 / 2) / d.σ
    # Power-law tail part
    return d.norm_const * d.A_const * (d.B_const - x̂)^(-d.n) / d.σ
end

"""
    cdf(d::CrystalBall, x::Real)

Compute the cumulative distribution function (CDF) of the Crystal Ball distribution `d` at point `x`.

The CDF is calculated by integrating the PDF. This implementation handles the integral of the power-law tail and the Gaussian core separately, ensuring continuity at the transition point.
"""
function Distributions.cdf(d::CrystalBall{T}, x::Real) where {T <: Real}
    x̂ = (x - d.μ) / d.σ

    # Value of the CDF at the transition point x̂ = -α
    cdf_at_minus_alpha =
        d.norm_const * d.A_const / (d.n - 1) * (d.B_const - (-d.α))^(1 - d.n)

    if x̂ <= -d.α
        # CDF for the power-law tail part (x̂ ≤ -α)
        # Integral of PDF from -Inf to x̂
        return d.norm_const * d.A_const / (d.n - 1) * (d.B_const - x̂)^(1 - d.n)
    else
        # CDF for the Gaussian part (x̂ > -α)
        # CDF at -α + integral of Gaussian PDF from -α to x̂
        # The integral is: ∫_{-α}^{x̂} N * exp(-t^2/2) / σ dt = N * sqrt(π/2) * (erf(x̂/sqrt(2)) + erf(α/sqrt(2)))
        integral_gaussian_part =
            sqrt(T(π) / 2) * (erf(x̂ / sqrt(T(2))) + erf(d.α / sqrt(T(2))))
        return cdf_at_minus_alpha + d.norm_const * integral_gaussian_part
    end
end

"""
    quantile(d::CrystalBall, p::Real)

Compute the quantile (inverse CDF) of the CrystalBall distribution `d` for a given probability `p`.

The function determines if the probability `p` falls into the power-law tail or the Gaussian core
and then inverts the corresponding CDF segment.
Requires `SpecialFunctions.erf` and `SpecialFunctions.erfinv` to be available.
"""
function Distributions.quantile(d::CrystalBall{T}, p::Real) where {T <: Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    # CDF value at the transition point x̂ = -α. (d.B_const + d.α) simplifies to (d.n / d.α)
    cdf_at_minus_alpha = cdf(d, d.μ - d.α * d.σ)

    x̂ = zero(T) # Scaled quantile score
    if p <= cdf_at_minus_alpha
        # Quantile is in the power-law tail. Invert the tail CDF formula.
        base = (p * (d.n - 1)) / (d.norm_const * d.A_const)
        x̂ = d.B_const - base^(one(T) / (one(T) - d.n))
    else
        # Quantile is in the Gaussian core. Invert the Gaussian core CDF formula.
        # Solve: p = cdf_at_minus_alpha + N * sqrt(π/2) * (erf(x̂/sqrt(2)) + erf(α/sqrt(2)))
        # Rearranging: erf(x̂/sqrt(2)) = (p - cdf_at_minus_alpha) / (N * sqrt(π/2)) - erf(α/sqrt(2))
        term_for_erfinv_num = (p - cdf_at_minus_alpha)
        term_for_erfinv_den = d.norm_const * sqrt(T(π) / T(2))

        erf_alpha_sqrt2 = erf(d.α / sqrt(T(2)))
        arg_erfinv = (term_for_erfinv_num / term_for_erfinv_den) - erf_alpha_sqrt2

        # arg_erfinv should be within [-1, 1] for erfinv. 
        # Clamping might be needed for extreme numerical precision issues, but typically not.
        x̂ = sqrt(T(2)) * erfinv(arg_erfinv)
    end

    return d.μ + d.σ * x̂
end

Distributions.maximum(d::CrystalBall{T}) where {T <: Real} = T(Inf)
Distributions.minimum(d::CrystalBall{T}) where {T <: Real} = T(-Inf)
