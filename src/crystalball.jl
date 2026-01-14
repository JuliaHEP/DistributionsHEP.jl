# Common parameter validation
function _check_crystalball_params(σ::T, α::T, n::T) where {T<:Real}
    σ > zero(T) || error("σ (scale) must be positive.")
    α > zero(T) || error("α (transition point) must be positive.")
    n > one(T) || error("n (power-law exponent) must be greater than 1.")
end

function _compute_standard_tail_constants(μ::T, σ::T, α::T, n::T) where {T<:Real}
    x0 = μ - α * σ
    G_x0 = exp(-α^2 / 2)
    L_x0 = α
    N = n
    return CrystalBallTail(G_x0, N, L_x0, x0)
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
    μ::T
    σ::T
    tail::CrystalBallTail{T}  # Tail parameters
    # Precomputed constants for PDF calculation
    norm_const::T # Normalization constant N

    function CrystalBall(μ::T, σ::T, α::T, n::T) where {T<:Real}
        _check_crystalball_params(σ, α, n)

        tail = _compute_standard_tail_constants(μ, σ, α, n)

        tail_contribution = _tail_norm_const(tail)
        core_contribution = sqrt(T(π) / 2) * (one(T) + erf(α / sqrt(T(2))))
        N = one(T) / (tail_contribution + core_contribution)

        new{T}(μ, σ, tail, N)
    end
end

"""
    pdf(d::CrystalBall, x::Real)

Compute the probability density function (PDF) of the Crystal Ball distribution `d` at point `x`.

The function uses precomputed normalization and tail parameters stored within the `CrystalBall` struct for efficiency.
It switches between the Gaussian core and the power-law tail based on the value of `x` relative to the transition point defined by `α`.
"""
function Distributions.pdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    x > d.tail.x0 && return d.norm_const * exp(-((x - d.μ) / d.σ)^2 / 2) / d.σ

    x̂ = (x - d.μ) / d.σ
    x̂0 = (d.tail.x0 - d.μ) / d.σ  # = -α
    return d.norm_const * _tail_function_value(d.tail, x̂ - x̂0) / d.σ
end

"""
    cdf(d::CrystalBall, x::Real)

Compute the cumulative distribution function (CDF) of the Crystal Ball distribution `d` at point `x`.

The CDF is calculated by integrating the PDF. This implementation handles the integral of the power-law tail and the Gaussian core separately, ensuring continuity at the transition point.
"""
function Distributions.cdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    # Compute CDF constant using clean 4-parameter formulation
    const_tail = _tail_norm_const(d.tail)
    x̂ = (x - d.μ) / d.σ
    x̂0 = (d.tail.x0 - d.μ) / d.σ  # = -α

    if x <= d.tail.x0
        # CDF for the power-law tail part: using clean 4-parameter formulation
        # In scaled coordinates: CDF = const_tail * ((N - L_x0*(x̂-x̂0))/N)^(1-N)
        return d.norm_const * const_tail * ((d.tail.N - d.tail.L_x0 * (x̂ - x̂0)) / d.tail.N)^(1 - d.tail.N)
    else
        # CDF for the Gaussian part (x > x0)
        # CDF at x0 + integral of Gaussian PDF from x0 to x
        cdf_at_x0 = d.norm_const * const_tail
        integral_gaussian_part =
            sqrt(T(π) / 2) * (erf(x̂ / sqrt(T(2))) - erf(x̂0 / sqrt(T(2))))
        return cdf_at_x0 + d.norm_const * integral_gaussian_part
    end
end

"""
    quantile(d::CrystalBall, p::Real)

Compute the quantile (inverse CDF) of the CrystalBall distribution `d` for a given probability `p`.

The function determines if the probability `p` falls into the power-law tail or the Gaussian core
and then inverts the corresponding CDF segment.
Requires `SpecialFunctions.erf` and `SpecialFunctions.erfinv` to be available.
"""
function Distributions.quantile(d::CrystalBall{T}, p::Real) where {T<:Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    # CDF value at the transition point x0
    const_tail = _tail_norm_const(d.tail)
    cdf_at_x0 = d.norm_const * const_tail
    x̂0 = (d.tail.x0 - d.μ) / d.σ  # = -α

    if p <= cdf_at_x0
        # Quantile is in the power-law tail. Invert using clean 4-parameter formulation.
        # From: p = norm_const * const_tail * ((N - L_x0*(x̂-x̂0))/N)^(1-N)
        # Solve for x̂: x̂ = x̂0 + (N/L_x0) * (1 - (p/(norm_const*const_tail))^(1/(1-N)))
        x̂ = x̂0 + (d.tail.N / d.tail.L_x0) * (1 - (p / (d.norm_const * const_tail))^(1 / (1 - d.tail.N)))
        return d.μ + d.σ * x̂
    else
        # Quantile is in the Gaussian core. Invert the Gaussian core CDF formula.
        # Solve: p = cdf_at_x0 + N * sqrt(π/2) * (erf(x̂/sqrt(2)) - erf(x̂0/sqrt(2)))
        # Rearranging: erf(x̂/sqrt(2)) = (p - cdf_at_x0) / (N * sqrt(π/2)) + erf(x̂0/sqrt(2))
        term_for_erfinv_num = (p - cdf_at_x0)
        term_for_erfinv_den = d.norm_const * sqrt(T(π) / T(2))

        erf_x0_sqrt2 = erf(x̂0 / sqrt(T(2)))
        arg_erfinv = (term_for_erfinv_num / term_for_erfinv_den) + erf_x0_sqrt2

        x̂ = sqrt(T(2)) * erfinv(arg_erfinv)
        return d.μ + d.σ * x̂
    end
end

Distributions.maximum(d::CrystalBall{T}) where {T<:Real} = T(Inf)
Distributions.minimum(d::CrystalBall{T}) where {T<:Real} = T(-Inf)
