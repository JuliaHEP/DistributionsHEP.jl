# Common parameter validation
function _check_crystalball_params(σ::T, α::T, n::T) where {T<:Real}
    σ > zero(T) || error("σ (scale) must be positive.")
    α > zero(T) || error("α (transition point) must be positive.")
    n > one(T) || error("n (power-law exponent) must be greater than 1.")
end

"""
    UnNormGauss{T<:Real}

A struct that encapsulates the unnormalized Gaussian computation to avoid repeated
and error-prone calculations of `exp(-((x - μ) / σ)^2 / 2)`.
"""
struct UnNormGauss{T<:Real}
    μ::T  # Mean
    σ::T  # Standard deviation
end

"""
    _value(g::UnNormGauss, x::Real)

Compute the unnormalized Gaussian value at point `x`:
`exp(-((x - g.μ) / g.σ)^2 / 2)`
"""
function _value(g::UnNormGauss{T}, x::Real) where {T<:Real}
    return exp(-((x - g.μ) / g.σ)^2 / 2)
end

"""
    _scaled_coord(g::UnNormGauss, x::Real)

Convert absolute coordinate `x` to scaled coordinate: `(x - g.μ) / g.σ`
"""
function _scaled_coord(g::UnNormGauss{T}, x::Real) where {T<:Real}
    return (x - g.μ) / g.σ
end

"""
    _from_scaled_coord(g::UnNormGauss, x̂::Real)

Convert scaled coordinate `x̂` back to absolute coordinate: `g.μ + g.σ * x̂`
"""
function _from_scaled_coord(g::UnNormGauss{T}, x̂::Real) where {T<:Real}
    return g.μ + g.σ * x̂
end

"""
    _erf_scaled(g::UnNormGauss, x̂::Real)

Compute `erf(x̂ / sqrt(2))` for scaled coordinate `x̂`.
"""
function _erf_scaled(g::UnNormGauss{T}, x̂::Real) where {T<:Real}
    return erf(x̂ / sqrt(T(2)))
end

"""
    _gaussian_cdf_integral(g::UnNormGauss, x̂1::Real, x̂0::Real)

Compute the Gaussian CDF integral from scaled coordinate `x̂0` to `x̂1`:
`sqrt(π/2) * (erf(x̂1/sqrt(2)) - erf(x̂0/sqrt(2)))`
"""
function _gaussian_cdf_integral(g::UnNormGauss{T}, x̂1::Real, x̂0::Real) where {T<:Real}
    return sqrt(T(π) / T(2)) * (_erf_scaled(g, x̂1) - _erf_scaled(g, x̂0))
end

"""
    _gaussian_quantile(g::UnNormGauss, arg_erfinv::Real)

Compute the quantile from the argument to `erfinv`:
Converts `arg_erfinv` to scaled coordinate `x̂ = sqrt(2) * erfinv(arg_erfinv)`,
then returns the absolute coordinate.
"""
function _gaussian_quantile(g::UnNormGauss{T}, arg_erfinv::Real) where {T<:Real}
    x̂ = sqrt(T(2)) * erfinv(arg_erfinv)
    return _from_scaled_coord(g, x̂)
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
    tail::CrystalBallTail{T}  # Tail parameters
    gauss::UnNormGauss{T}      # Unnormalized Gaussian helper
    # Precomputed constants for PDF calculation
    norm_const::T # Normalization constant N

    function CrystalBall(μ::T, σ::T, α::T, n::T) where {T<:Real}
        _check_crystalball_params(σ, α, n)

        tail = _compute_standard_tail_constants(μ, σ, α, n)
        gauss = UnNormGauss(μ, σ)

        tail_contribution = _norm_const(tail)
        core_contribution = sqrt(T(π) / 2) * (one(T) + erf(α / sqrt(T(2))))
        N = one(T) / (tail_contribution + core_contribution)

        new{T}(tail, gauss, N)
    end
end

"""
    pdf(d::CrystalBall, x::Real)

Compute the probability density function (PDF) of the Crystal Ball distribution `d` at point `x`.

The function uses precomputed normalization and tail parameters stored within the `CrystalBall` struct for efficiency.
It switches between the Gaussian core and the power-law tail based on the value of `x` relative to the transition point defined by `α`.
"""
function Distributions.pdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    x > d.tail.x0 && return d.norm_const * _value(d.gauss, x) / d.gauss.σ

    x̂ = _scaled_coord(d.gauss, x)
    x̂0 = _scaled_coord(d.gauss, d.tail.x0)  # = -α
    return d.norm_const * _value(d.tail, x̂ - x̂0) / d.gauss.σ
end

"""
    cdf(d::CrystalBall, x::Real)

Compute the cumulative distribution function (CDF) of the Crystal Ball distribution `d` at point `x`.

The CDF is calculated by integrating the PDF. This implementation handles the integral of the power-law tail and the Gaussian core separately, ensuring continuity at the transition point.
"""
function Distributions.cdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    if x <= d.tail.x0
        # Use _integral for tail part
        return d.norm_const * _integral(d.tail, T(x))
    else
        # CDF at transition point + Gaussian part
        const_tail = _norm_const(d.tail)
        cdf_at_x0 = d.norm_const * const_tail
        x̂ = _scaled_coord(d.gauss, x)
        x̂0 = -d.tail.L_x0  # = (d.tail.x0 - μ) / σ
        integral_gaussian_part = _gaussian_cdf_integral(d.gauss, x̂, x̂0)
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
    const_tail = _norm_const(d.tail)
    cdf_at_x0 = d.norm_const * const_tail

    if p <= cdf_at_x0
        # Use _integral_inversion for tail part
        # p = d.norm_const * _integral(d.tail, x), so _integral(d.tail, x) = p / d.norm_const
        tail_integral = T(p) / d.norm_const
        return _integral_inversion(d.tail, tail_integral)
    else
        # Gaussian part
        term_for_erfinv_num = (p - cdf_at_x0)
        term_for_erfinv_den = d.norm_const * sqrt(T(π) / T(2))
        x̂0 = -d.tail.L_x0  # = (d.tail.x0 - μ) / σ
        erf_x0_sqrt2 = _erf_scaled(d.gauss, x̂0)
        arg_erfinv = (term_for_erfinv_num / term_for_erfinv_den) + erf_x0_sqrt2
        return _gaussian_quantile(d.gauss, arg_erfinv)
    end
end

Distributions.maximum(d::CrystalBall{T}) where {T<:Real} = T(Inf)
Distributions.minimum(d::CrystalBall{T}) where {T<:Real} = T(-Inf)

# Distributions.jl interface methods
Distributions.location(d::CrystalBall) = d.gauss.μ
Distributions.scale(d::CrystalBall) = d.gauss.σ
