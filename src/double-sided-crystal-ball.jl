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
    μ::T
    σ::T
    left_tail::CrystalBallTail{T}
    right_tail::CrystalBallTail{T}
    norm_const::T

    function DoubleCrystalBall(μ::T, σ::T, αL::T, nL::T, αR::T, nR::T) where {T<:Real}
        _check_double_crystalball_params(σ, αL, nL, αR, nR)

        x0L = μ - αL * σ
        G_x0L = exp(-αL^2 / 2) # Unnormalized in scaled coordinates
        L_x0L = αL # log derivative is just equal to αL
        left_tail = CrystalBallTail(G_x0L, nL, L_x0L, x0L)

        x0R = μ + αR * σ
        G_x0R = exp(-αR^2 / 2) # Unnormalized in scaled coordinates
        L_x0R = -αR  # Negative as should be for rightward tail
        right_tail = CrystalBallTail(G_x0R, nR, L_x0R, x0R)

        norm_left_tail = _tail_norm_const(left_tail)
        norm_right_tail = -_tail_norm_const(right_tail)  # Right tail has negative L_x0, so negate
        core_contribution = sqrt(T(π) / 2) * (erf(αR / sqrt(T(2))) + erf(αL / sqrt(T(2))))
        N = one(T) / (norm_left_tail + norm_right_tail + core_contribution)

        new{T}(μ, σ, left_tail, right_tail, N)
    end
end

# Helper function to compute CDF values at transition points
# Using clean 4-parameter formulation
function _compute_transition_cdf_values(d::DoubleCrystalBall{T}) where {T<:Real}
    # CDF values at transition points using clean formulation
    const_left = _tail_norm_const(d.left_tail)
    cdf_at_minus_alphaL = d.norm_const * const_left
    # Compute αL and αR from x0: x0L = μ - αL*σ, x0R = μ + αR*σ
    αL = (d.μ - d.left_tail.x0) / d.σ
    αR = (d.right_tail.x0 - d.μ) / d.σ
    cdf_at_plus_alphaR =
        cdf_at_minus_alphaL +
        d.norm_const *
        sqrt(T(π) / 2) *
        (erf(αR / sqrt(T(2))) + erf(αL / sqrt(T(2))))

    return cdf_at_minus_alphaL, cdf_at_plus_alphaR
end


function Distributions.pdf(d::DoubleCrystalBall{T}, x::Real) where {T<:Real}
    x̂ = (x - d.μ) / d.σ
    # Left power-law tail: using clean 4-parameter formulation (in scaled coordinates)
    if x < d.left_tail.x0
        x̂0L = (d.left_tail.x0 - d.μ) / d.σ  # = -αL
        return d.norm_const * _tail_function_value(d.left_tail, x̂ - x̂0L) / d.σ
    end
    # Gaussian core
    if x <= d.right_tail.x0
        return d.norm_const * exp(-x̂^2 / 2) / d.σ
    end
    # Right power-law tail: using clean 4-parameter formulation (in scaled coordinates)
    x̂0R = (d.right_tail.x0 - d.μ) / d.σ  # = αR
    return d.norm_const * _tail_function_value(d.right_tail, x̂ - x̂0R) / d.σ
end

function Distributions.cdf(d::DoubleCrystalBall{T}, x::Real) where {T<:Real}
    # CDF values at transition points using clean 4-parameter formulation
    cdf_at_minus_alphaL, cdf_at_plus_alphaR = _compute_transition_cdf_values(d)
    const_left = _tail_norm_const(d.left_tail)
    const_right = -_tail_norm_const(d.right_tail)  # Right tail has negative L_x0, so negate
    x̂ = (x - d.μ) / d.σ
    x̂0L = (d.left_tail.x0 - d.μ) / d.σ  # = -αL
    x̂0R = (d.right_tail.x0 - d.μ) / d.σ  # = αR

    if x <= d.left_tail.x0
        # CDF for the left power-law tail: using clean 4-parameter formulation (in scaled coordinates)
        return d.norm_const * const_left * ((d.left_tail.N - d.left_tail.L_x0 * (x̂ - x̂0L)) / d.left_tail.N)^(1 - d.left_tail.N)
    elseif x >= d.right_tail.x0
        # CDF for the right power-law tail: using clean 4-parameter formulation (in scaled coordinates)
        return cdf_at_plus_alphaR + d.norm_const * (-const_right * ((d.right_tail.N - d.right_tail.L_x0 * (x̂ - x̂0R)) / d.right_tail.N)^(1 - d.right_tail.N) + const_right)
    else
        # CDF for the Gaussian core
        integral_gaussian_part =
            sqrt(T(π) / 2) * (erf(x̂ / sqrt(T(2))) - erf(x̂0L / sqrt(T(2))))
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

    x̂ = zero(T) # Scaled quantile score

    const_left = _tail_norm_const(d.left_tail)
    const_right = -_tail_norm_const(d.right_tail)  # Right tail has negative L_x0, so negate
    x̂0L = (d.left_tail.x0 - d.μ) / d.σ  # = -αL
    x̂0R = (d.right_tail.x0 - d.μ) / d.σ  # = αR

    if p <= cdf_at_minus_alphaL
        # Quantile is in the left power-law tail: using clean 4-parameter formulation (in scaled coordinates)
        x̂ = x̂0L + (d.left_tail.N / d.left_tail.L_x0) * (1 - (p / (d.norm_const * const_left))^(1 / (1 - d.left_tail.N)))
        return d.μ + d.σ * x̂
    elseif p >= cdf_at_plus_alphaR
        # Quantile is in the right power-law tail: using clean 4-parameter formulation (in scaled coordinates)
        x̂ = x̂0R + (d.right_tail.N / d.right_tail.L_x0) * (1 - (-1 / const_right * ((p - cdf_at_plus_alphaR) / d.norm_const - const_right))^(1 / (1 - d.right_tail.N)))
        return d.μ + d.σ * x̂
    else
        # Quantile is in the Gaussian core
        # Solve: p = cdf_at_minus_alphaL + N * sqrt(π/2) * (erf(x̂/sqrt(2)) - erf(x̂0L/sqrt(2)))
        # Rearranging: erf(x̂/sqrt(2)) = (p - cdf_at_minus_alphaL) / (N * sqrt(π/2)) + erf(x̂0L/sqrt(2))
        term_for_erfinv_num = (p - cdf_at_minus_alphaL)
        term_for_erfinv_den = d.norm_const * sqrt(T(π) / T(2))

        erf_x0L_sqrt2 = erf(x̂0L / sqrt(T(2)))
        arg_erfinv = (term_for_erfinv_num / term_for_erfinv_den) + erf_x0L_sqrt2

        x̂ = sqrt(T(2)) * erfinv(arg_erfinv)
        return d.μ + d.σ * x̂
    end
end

Distributions.maximum(d::DoubleCrystalBall{T}) where {T<:Real} = T(Inf)
Distributions.minimum(d::DoubleCrystalBall{T}) where {T<:Real} = T(-Inf)

# Distributions.jl interface methods
Distributions.location(d::DoubleCrystalBall) = d.μ
Distributions.scale(d::DoubleCrystalBall) = d.σ
# Compute αL, nL, αR, nR from tail structs when needed for params()
Distributions.params(d::DoubleCrystalBall) = (
    d.μ,
    d.σ,
    d.left_tail.L_x0,
    d.left_tail.N,
    -d.right_tail.L_x0,  # Negative as should be for rightward tail
    d.right_tail.N
)
Distributions.partype(::DoubleCrystalBall{T}) where {T} = T
