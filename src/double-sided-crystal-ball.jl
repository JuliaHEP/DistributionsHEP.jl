# Note: CrystalBallTail and helper functions are defined in crystalball.jl
# They are available here since crystalball.jl is included before this file in DistributionsHEP.jl

# Common parameter validation for two-sided crystal ball
function _check_double_crystalball_params(σ::T, αL::T, nL::T, αR::T, nR::T) where {T<:Real}
    σ > zero(T) || error("σ (scale) must be positive.")
    αL > zero(T) || error("αL (left transition point) must be positive.")
    nL > one(T) || error("nL (left power-law exponent) must be greater than 1.")
    αR > zero(T) || error("αR (right transition point) must be positive.")
    nR > one(T) || error("nR (right power-law exponent) must be greater than 1.")
end

"""
    DoubleCrystalBall{T<:Real} <: ContinuousUnivariateDistribution

The Double Crystal Ball distribution is a probability distribution commonly used in high-energy physics 
to model various lossy processes with power-law tails on both sides of a Gaussian core.

The probability density function is defined as:
````math
f(x; μ, σ, α_L, n_L, α_R, n_R) = \\begin{cases}
    N A_L (B_L - \\hat{x})^{-n_L} & \\text{for } \\hat{x} < -α_L \\\\
    N \\exp\\left(-\\frac{\\hat{x}^2}{2}\\right) & \\text{for } -α_L \\leq \\hat{x} \\leq α_R \\\\
    N A_R (B_R + \\hat{x})^{-n_R} & \\text{for } \\hat{x} > α_R
\\end{cases}
````
where ``\\hat{x} = (x - μ) / σ``.
The parameters A_L, B_L, A_R, B_R are derived from α_L, n_L, α_R, n_R to ensure continuity 
of the function and its first derivative. N is a normalization constant.

# Arguments
- `μ`: The mean of the Gaussian core.
- `σ`: The standard deviation of the Gaussian core. Must be positive.
- `αL`: The left transition point, defining where the left power-law tail begins.
- `nL`: The exponent of the left power-law tail. Must be greater than 1.
- `αR`: The right transition point, defining where the right power-law tail begins.
- `nR`: The exponent of the right power-law tail. Must be greater than 1.

# Example
```julia
using DistributionsHEP
using Plots

d = DoubleCrystalBall(0.0, 1.0, 1.5, 2.0, 2.0, 3.0)  # μ, σ, αL, nL, αR, nR
plot(-5, 5, x->pdf(d, x))
```
"""
struct DoubleCrystalBall{T<:Real} <: ContinuousUnivariateDistribution
    left_tail::CrystalBallTail{T}
    right_tail::CrystalBallTail{T}
    gauss::UnNormGauss{T}      # Unnormalized Gaussian helper
    norm_const::T

    function DoubleCrystalBall(μ::T, σ::T, αL::T, nL::T, αR::T, nR::T) where {T<:Real}
        _check_double_crystalball_params(σ, αL, nL, αR, nR)

        gauss = UnNormGauss(μ, σ)

        x0L = μ - αL * σ
        # Use _value from UnNormGauss to get normalized G_x0 at transition point
        G_x0L = _value(gauss, x0L)
        L_x0L = αL # log derivative is just equal to αL
        left_tail = CrystalBallTail(G_x0L, nL, L_x0L, x0L)

        x0R = μ + αR * σ
        # Use _value from UnNormGauss to get normalized G_x0 at transition point
        G_x0R = _value(gauss, x0R)
        L_x0R = -αR  # Negative as should be for rightward tail
        right_tail = CrystalBallTail(G_x0R, nR, L_x0R, x0R)

        left_tail_contribution = _integral(left_tail, left_tail.x0)
        right_tail_contribution = -_integral(right_tail, right_tail.x0)  # Right tail has negative L_x0, so negate
        # Integral from x0L to x0R = integral from -∞ to x0R - integral from -∞ to x0L
        integral_to_x0R = _integral(gauss, right_tail.x0)
        integral_to_x0L = _integral(gauss, left_tail.x0)
        core_contribution = integral_to_x0R - integral_to_x0L
        N = one(T) / (left_tail_contribution + right_tail_contribution + core_contribution)

        new{T}(left_tail, right_tail, gauss, N)
    end
end

# Helper function to compute CDF values at transition points
function _compute_transition_cdf_values(d::DoubleCrystalBall{T}) where {T<:Real}
    # CDF values at transition points using _integral
    cdf_at_minus_alphaL = d.norm_const * _integral(d.left_tail, d.left_tail.x0)
    # Integral from x0L to x0R = integral from -∞ to x0R - integral from -∞ to x0L
    integral_to_x0R = _integral(d.gauss, d.right_tail.x0)
    integral_to_x0L = _integral(d.gauss, d.left_tail.x0)
    cdf_at_plus_alphaR = cdf_at_minus_alphaL + d.norm_const * (integral_to_x0R - integral_to_x0L)

    return cdf_at_minus_alphaL, cdf_at_plus_alphaR
end


function Distributions.pdf(d::DoubleCrystalBall{T}, x::Real) where {T<:Real}
    x̂ = _scaled_coord(d.gauss, x)
    # Left power-law tail: using clean 4-parameter formulation (in scaled coordinates)
    if x < d.left_tail.x0
        x̂0L = _scaled_coord(d.gauss, d.left_tail.x0)  # = -αL
        return d.norm_const * _value(d.left_tail, x̂ - x̂0L)
    end
    # Gaussian core
    if x <= d.right_tail.x0
        return d.norm_const * _value(d.gauss, x)
    end
    # Right power-law tail: using clean 4-parameter formulation (in scaled coordinates)
    x̂0R = _scaled_coord(d.gauss, d.right_tail.x0)  # = αR
    return d.norm_const * _value(d.right_tail, x̂ - x̂0R)
end

function Distributions.cdf(d::DoubleCrystalBall{T}, x::Real) where {T<:Real}
    # CDF values at transition points using clean 4-parameter formulation
    cdf_at_minus_alphaL, cdf_at_plus_alphaR = _compute_transition_cdf_values(d)
    if x <= d.left_tail.x0
        # CDF for the left power-law tail: use _integral
        return d.norm_const * _integral(d.left_tail, x)
    elseif x >= d.right_tail.x0
        # CDF for the right power-law tail: use _integral
        # For right tail, _integral(right_tail, x) = -∫[x, +∞] and _integral(right_tail, x0R) = -∫[x0R, +∞]
        # CDF(x) = CDF(x0R) + ∫[x0R, x] = CDF(x0R) + (∫[x0R, +∞] - ∫[x, +∞])
        # = CDF(x0R) + (_integral(right_tail, x) - _integral(right_tail, x0R))
        integral_at_x = _integral(d.right_tail, x)
        integral_at_x0R = _integral(d.right_tail, d.right_tail.x0)
        return cdf_at_plus_alphaR + d.norm_const * (integral_at_x - integral_at_x0R)
    else
        # CDF for the Gaussian core
        # Integral from x0L to x = integral from -∞ to x - integral from -∞ to x0L
        integral_to_x = _integral(d.gauss, x)
        integral_to_x0L = _integral(d.gauss, d.left_tail.x0)
        integral_gaussian_part = integral_to_x - integral_to_x0L
        return cdf_at_minus_alphaL + d.norm_const * integral_gaussian_part
    end
end


function Distributions.quantile(d::DoubleCrystalBall{T}, p::Real) where {T<:Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    # CDF values at transition points (same as in CDF function)
    cdf_at_minus_alphaL, cdf_at_plus_alphaR = _compute_transition_cdf_values(d)

    if p <= cdf_at_minus_alphaL
        # Quantile is in the left power-law tail: use _integral_inversion
        tail_integral = p / d.norm_const
        return _integral_inversion(d.left_tail, tail_integral)
    elseif p >= cdf_at_plus_alphaR
        # Quantile is in the right power-law tail: use _integral_inversion
        # For right tail: CDF(x) = CDF(x0R) + N * (_integral(right_tail, x) - _integral(right_tail, x0R))
        # So: _integral(right_tail, x) = (p - cdf_at_plus_alphaR) / N + _integral(right_tail, x0R)
        integral_at_x0R = _integral(d.right_tail, d.right_tail.x0)
        tail_integral = (p - cdf_at_plus_alphaR) / d.norm_const + integral_at_x0R
        return _integral_inversion(d.right_tail, tail_integral)
    else
        # Quantile is in the Gaussian core
        # p = cdf_at_minus_alphaL + N * (integral from x0L to x)
        # So: integral from -∞ to x = (p - cdf_at_minus_alphaL) / N + integral from -∞ to x0L
        gaussian_integral_at_x0L = _integral(d.gauss, d.left_tail.x0)
        gaussian_integral = (p - cdf_at_minus_alphaL) / d.norm_const + gaussian_integral_at_x0L
        return _integral_inversion(d.gauss, gaussian_integral)
    end
end

Distributions.maximum(d::DoubleCrystalBall{T}) where {T<:Real} = T(Inf)
Distributions.minimum(d::DoubleCrystalBall{T}) where {T<:Real} = T(-Inf)

# Distributions.jl interface methods
Distributions.location(d::DoubleCrystalBall) = d.gauss.μ
Distributions.scale(d::DoubleCrystalBall) = d.gauss.σ
# Compute αL, nL, αR, nR from tail structs when needed for params()
Distributions.params(d::DoubleCrystalBall) = (
    d.gauss.μ,
    d.gauss.σ,
    d.left_tail.L_x0,
    d.left_tail.N,
    -d.right_tail.L_x0,  # Negative as should be for rightward tail
    d.right_tail.N
)
Distributions.partype(::DoubleCrystalBall{T}) where {T} = T
