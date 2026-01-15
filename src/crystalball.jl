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

Compute the normalized Gaussian value at point `x`:
`(1 / (σ * sqrt(2π))) * exp(-((x - g.μ) / g.σ)^2 / 2)`

This is the standard normalized Gaussian PDF.
"""
function _value(g::UnNormGauss{T}, x::Real) where {T<:Real}
    x_T = T(x)
    return (one(T) / (g.σ * sqrt(T(2) * T(π)))) * exp(-((x_T - g.μ) / g.σ)^2 / 2)
end


"""
    _gaussian_half_norm_const(::Type{T})

Return the constant `1/2` for type `T`.

This constant appears in the integral of the normalized Gaussian:
∫[-∞ to a] _value(x) dx = (1/2) * (1 + erf((a-μ)/(σ*sqrt(2))))
"""
_gaussian_half_norm_const(::Type{T}) where {T<:Real} = T(1) / T(2)


"""
    _integral(g::UnNormGauss, a)

Compute the integral of the normalized Gaussian function from -∞ to `a` (in absolute coordinates).

The integral is computed using the error function:
∫[-∞ to a] _value(x) dx = (1/2) * (1 + erf((a - g.μ) / (g.σ * sqrt(2))))

Returns the integral value.
"""
function _integral(g::UnNormGauss{T}, a::Real) where {T<:Real}
    a_T = T(a)
    x̂ = (a_T - g.μ) / g.σ
    return _gaussian_half_norm_const(T) * (one(T) + erf(x̂ / sqrt(T(2))))
end

"""
    _integral_inversion(g::UnNormGauss, integral)

Find `a` such that the integral of the normalized Gaussian function from -∞ to `a` equals the given `integral` value.

Returns the value of `a` (in absolute coordinates).
"""
function _integral_inversion(g::UnNormGauss{T}, integral::Real) where {T<:Real}
    integral_T = T(integral)
    norm_const = _gaussian_half_norm_const(T)

    # Solve: norm_const * (1 + erf((a - μ) / (σ * sqrt(2)))) = integral
    # Rearranging: erf((a - μ) / (σ * sqrt(2))) = (integral / norm_const) - 1
    ratio = integral_T / norm_const
    erf_arg = ratio - one(T)

    # Handle edge cases
    if erf_arg <= -one(T)
        return T(-Inf)
    elseif erf_arg >= one(T)
        return T(Inf)
    end

    # (a - μ) / (σ * sqrt(2)) = erfinv(erf_arg)
    # a = μ + σ * sqrt(2) * erfinv(erf_arg)
    x̂ = sqrt(T(2)) * erfinv(erf_arg)
    a = g.μ + g.σ * x̂

    return a
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

        gauss = UnNormGauss(μ, σ)
        x0 = μ - α * σ
        # Use _value from UnNormGauss to get normalized G_x0 at transition point
        G_x0 = _value(gauss, x0)
        L_x0 = α
        tail = CrystalBallTail(G_x0, n, L_x0, x0)

        tail_contribution = _integral(tail, tail.x0)
        # Integral from x0 to +∞ = integral from -∞ to +∞ - integral from -∞ to x0
        integral_full = _integral(gauss, T(Inf))
        integral_to_x0 = _integral(gauss, tail.x0)
        core_contribution = integral_full - integral_to_x0
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
    x > d.tail.x0 && return d.norm_const * _value(d.gauss, x)

    # Tail part: _value expects absolute offset (x - x0), not scaled
    offset = x - d.tail.x0
    return d.norm_const * _value(d.tail, offset)
end

"""
    cdf(d::CrystalBall, x::Real)

Compute the cumulative distribution function (CDF) of the Crystal Ball distribution `d` at point `x`.

The CDF is calculated by integrating the PDF. This implementation handles the integral of the power-law tail and the Gaussian core separately, ensuring continuity at the transition point.
"""
function Distributions.cdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    if x <= d.tail.x0
        # Use _integral for tail part
        return d.norm_const * _integral(d.tail, x)
    else
        # CDF at transition point + Gaussian part
        cdf_at_x0 = d.norm_const * _integral(d.tail, d.tail.x0)
        # Integral from x0 to x = integral from -∞ to x - integral from -∞ to x0
        integral_gaussian_part = _integral(d.gauss, x) - _integral(d.gauss, d.tail.x0)
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
    cdf_at_x0 = d.norm_const * _integral(d.tail, d.tail.x0)

    if p <= cdf_at_x0
        # Use _integral_inversion for tail part
        # p = d.norm_const * _integral(d.tail, x), so _integral(d.tail, x) = p / d.norm_const
        tail_integral = p / d.norm_const
        return _integral_inversion(d.tail, tail_integral)
    else
        # Gaussian part
        # p = cdf_at_x0 + norm_const * (integral from x0 to x)
        # So: (p - cdf_at_x0) / norm_const = integral from x0 to x
        # = integral from -∞ to x - integral from -∞ to x0
        # Therefore: integral from -∞ to x = (p - cdf_at_x0) / norm_const + integral from -∞ to x0
        gaussian_integral_at_x0 = _integral(d.gauss, d.tail.x0)
        gaussian_integral = (p - cdf_at_x0) / d.norm_const + gaussian_integral_at_x0
        return _integral_inversion(d.gauss, gaussian_integral)
    end
end

Distributions.maximum(d::CrystalBall{T}) where {T<:Real} = T(Inf)
Distributions.minimum(d::CrystalBall{T}) where {T<:Real} = T(-Inf)

# Distributions.jl interface methods
Distributions.location(d::CrystalBall) = d.gauss.μ
Distributions.scale(d::CrystalBall) = d.gauss.σ








#  to be removed once the refactoring is complete
"""
    _erf_scaled(g::UnNormGauss, x̂::Real)

Compute `erf(x̂ / sqrt(2))` for scaled coordinate `x̂`.

Note: This function is kept for compatibility with other distributions (e.g., DoubleCrystalBall).
New code should use `_integral` and `_integral_inversion` instead.
"""
function _erf_scaled(g::UnNormGauss{T}, x̂::Real) where {T<:Real}
    x̂_T = T(x̂)
    return erf(x̂_T / sqrt(T(2)))
end

"""
    _gaussian_cdf_integral(g::UnNormGauss, x̂1::Real, x̂0::Real)

Compute the Gaussian CDF integral from scaled coordinate `x̂0` to `x̂1`:
`sqrt(π/2) * (erf(x̂1/sqrt(2)) - erf(x̂0/sqrt(2)))`

Note: This function is kept for compatibility with other distributions (e.g., DoubleCrystalBall).
New code should use `_integral` instead.
"""
function _gaussian_cdf_integral(g::UnNormGauss{T}, x̂1::Real, x̂0::Real) where {T<:Real}
    x̂1_T = T(x̂1)
    x̂0_T = T(x̂0)
    return _gaussian_half_norm_const(T) * (_erf_scaled(g, x̂1_T) - _erf_scaled(g, x̂0_T))
end

"""
    _gaussian_quantile(g::UnNormGauss, arg_erfinv::Real)

Compute the quantile from the argument to `erfinv`:
Converts `arg_erfinv` to scaled coordinate `x̂ = sqrt(2) * erfinv(arg_erfinv)`,
then returns the absolute coordinate.

Note: This function is kept for compatibility with other distributions (e.g., DoubleCrystalBall).
New code should use `_integral_inversion` instead.
"""
function _gaussian_quantile(g::UnNormGauss{T}, arg_erfinv::Real) where {T<:Real}
    arg_erfinv_T = T(arg_erfinv)
    x̂ = sqrt(T(2)) * erfinv(arg_erfinv_T)
    return _from_scaled_coord(g, x̂)
end



"""
    _scaled_coord(g::UnNormGauss, x::Real)

Convert absolute coordinate `x` to scaled coordinate: `(x - g.μ) / g.σ`
"""
function _scaled_coord(g::UnNormGauss{T}, x::Real) where {T<:Real}
    x_T = T(x)
    return (x_T - g.μ) / g.σ
end

"""
    _from_scaled_coord(g::UnNormGauss, x̂::Real)

Convert scaled coordinate `x̂` back to absolute coordinate: `g.μ + g.σ * x̂`
"""
function _from_scaled_coord(g::UnNormGauss{T}, x̂::Real) where {T<:Real}
    x̂_T = T(x̂)
    return g.μ + g.σ * x̂_T
end
