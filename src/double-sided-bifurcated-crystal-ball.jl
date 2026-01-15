# Common parameter validation for double-sided bifurcated crystal ball
function _check_double_sided_bifurcated_crystal_ball_params(σ::T, αL::T, nL::T, αR::T, nR::T) where {T<:Real}
    σ > zero(T) || error("σ (scale) must be positive.")
    αL > zero(T) || error("αL (left transition point) must be positive.")
    nL > zero(T) || error("nL (left power-law exponent) must be positive.")
    αR > zero(T) || error("αR (right transition point) must be positive.")
    nR > zero(T) || error("nR (right power-law exponent) must be positive.")
end

# Helper function logarithmic derivative
function _log_derivative_bifurcated_gaussian(d::BifurcatedGaussian, x::T) where {T<:Real}
    x <= d.μ && return -(x - d.μ) / (d.σL^2)
    return -(x - d.μ) / (d.σR^2)
end

function CrystalBallTail(
    BifGauss::BifurcatedGaussian{T},
    n::T,
    x0::T
) where {T<:Real}
    # N: Effective power (for bifurcated, N = sqrt(1 + n²))
    N = sqrt(one(T) + n^2)
    G_x0 = pdf(BifGauss, x0)
    L_x0 = _log_derivative_bifurcated_gaussian(BifGauss, x0)
    return CrystalBallTail(G_x0, N, L_x0, x0)
end

"""
    DoublesidedBifurcatedCrystalBall{T<:Real} <: ContinuousUnivariateDistribution

The Double-sided Bifurcated Crystal Ball distribution is a probability distribution commonly used in high-energy physics
to model various lossy processes. It extends the Bifurcated Gaussian distribution by adding power-law tails on both sides
of the asymmetric Gaussian core.

The distribution consists of:
- A left power-law tail for x < xL
- A bifurcated Gaussian core for xL ≤ x ≤ xR
- A right power-law tail for x > xR

The bifurcated Gaussian core uses different scale parameters on the left (σL) and right (σR) sides of the mean μ,
controlled by the asymmetry parameter ψ via κ = tanh(ψ), where σL = σ(1 + κ) and σR = σ(1 - κ).

# Arguments
- `μ`: The mean of the bifurcated Gaussian core.
- `σ`: The base scale parameter of the bifurcated Gaussian core. Must be positive.
- `ψ`: The asymmetry parameter controlling the difference between left and right scales via κ = tanh(ψ).
- `αL`: The left transition point parameter, defining where the left power-law tail begins (in units of σL). Must be positive.
- `nL`: The exponent parameter for the left power-law tail. Must be positive. The effective exponent is NL = √(1 + nL²) > 1.
- `αR`: The right transition point parameter, defining where the right power-law tail begins (in units of σR). Must be positive.
- `nR`: The exponent parameter for the right power-law tail. Must be positive. The effective exponent is NR = √(1 + nR²) > 1.

The transition points are calculated as:
- xL = μ - αL * σL (left transition)
- xR = μ + αR * σR (right transition)

The struct stores precomputed constants for efficient PDF, CDF, and quantile calculations.

# Example
```julia
using DistributionsHEP
using Plots

# Create a double-sided bifurcated crystal ball distribution
d = DoublesidedBifurcatedCrystalBall(0.0, 1.0, 0.25, 0.5, 1.25, 0.75, 1.5)  # μ, σ, ψ, αL, nL, αR, nR

# Evaluate PDF, CDF, and quantile
pdf(d, 0.0)
cdf(d, 1.0)
quantile(d, 0.5)

# Plot the distribution
plot(-5, 5, x->pdf(d, x))
```
"""
struct DoublesidedBifurcatedCrystalBall{T<:Real} <: ContinuousUnivariateDistribution
    BifGauss::BifurcatedGaussian{T}  # Bifurcated Gaussian core distribution (contains μ, σ, ψ, σL, σR)
    left_tail::CrystalBallTail{T}    # Left tail parameters (G(x0), N, L(x0), x0)
    right_tail::CrystalBallTail{T}   # Right tail parameters (G(x0), N, L(x0), x0)
    norm_const::T  # Normalization constant N
    p_xL::T        # CDF at left transition point
    p_xR::T        # CDF at right transition point
    p_mu::T        # CDF at mean μ

    function DoublesidedBifurcatedCrystalBall(μ::T, σ::T, ψ::T, αL::T, nL::T, αR::T, nR::T) where {T<:Real}
        _check_double_sided_bifurcated_crystal_ball_params(σ, αL, nL, αR, nR)

        # Calculate kappa
        κ = tanh(ψ)

        # Calculate scales for left and right sides
        σL = σ * (1 + κ)
        σR = σ * (1 - κ)

        # Calculate transition points
        xL = μ - αL * σL
        xR = μ + αR * σR

        # Pre-compute constants
        BifGauss = BifurcatedGaussian(μ, σ, ψ)

        # Create tail structures using clean 4-parameter formulation
        left_tail = CrystalBallTail(BifGauss, nL, xL)
        right_tail = CrystalBallTail(BifGauss, nR, xR)

        # Calculate normalization constant
        norm_const_left = _norm_const(left_tail)
        norm_const_right = -_norm_const(right_tail)  # Negative for right tail
        norm_const_core = cdf(BifGauss, xR) - cdf(BifGauss, xL)

        norm_const = one(T) / (norm_const_left + norm_const_right + norm_const_core)

        p_xL = norm_const * norm_const_left
        p_xR = norm_const * (norm_const_left + norm_const_core)
        p_mu = norm_const * (norm_const_left + σL / (2 * σ) - cdf(BifGauss, xL))

        new{T}(BifGauss, left_tail, right_tail, norm_const, p_xL, p_xR, p_mu)
    end
end

# Convenience constructors
DoublesidedBifurcatedCrystalBall(μ::Real, σ::Real, ψ::Real, αL::Real, nL::Real, αR::Real, nR::Real) =
    DoublesidedBifurcatedCrystalBall(promote(μ, σ, ψ, αL, nL, αR, nR)...)
DoublesidedBifurcatedCrystalBall(μ::Integer, σ::Integer, ψ::Integer, αL::Integer, nL::Integer, αR::Integer, nR::Integer) =
    DoublesidedBifurcatedCrystalBall(float(μ), float(σ), float(ψ), float(αL), float(nL), float(αR), float(nR))

"""
    pdf(d::DoublesidedBifurcatedCrystalBall, x::Real)

Compute the probability density function (PDF) of the Double-sided Bifurcated Crystal Ball distribution `d` at point `x`.

The function switches between the left power-law tail, the bifurcated Gaussian core, and the right power-law tail
based on the value of `x` relative to the transition points xL and xR.
"""
function Distributions.pdf(d::DoublesidedBifurcatedCrystalBall{T}, x::Real) where {T<:Real}
    # left tail: using clean 4-parameter formulation
    x < d.left_tail.x0 && return d.norm_const * _value(d.left_tail, x - d.left_tail.x0)
    # core
    x >= d.left_tail.x0 && x <= d.right_tail.x0 && return d.norm_const * pdf(d.BifGauss, x)
    # right tail: using clean 4-parameter formulation
    return d.norm_const * _value(d.right_tail, x - d.right_tail.x0)
end

"""
    cdf(d::DoublesidedBifurcatedCrystalBall, x::Real)

Compute the cumulative distribution function (CDF) of the Double-sided Bifurcated Crystal Ball distribution `d` at point `x`.

The CDF is calculated by integrating the PDF. This implementation handles the integral of the power-law tails
and the bifurcated Gaussian core separately, ensuring continuity at the transition points.
"""
function Distributions.cdf(d::DoublesidedBifurcatedCrystalBall{T}, x::Real) where {T<:Real}
    const_left = _norm_const(d.left_tail)
    const_right = -_norm_const(d.right_tail)

    if x < d.left_tail.x0
        # Left tail: using clean 4-parameter formulation
        return d.norm_const * const_left * ((d.left_tail.N - d.left_tail.L_x0 * (x - d.left_tail.x0)) / d.left_tail.N)^(1 - d.left_tail.N)
    elseif x >= d.left_tail.x0 && x <= d.right_tail.x0
        return d.p_xL + d.norm_const * (cdf(d.BifGauss, x) - cdf(d.BifGauss, d.left_tail.x0))
    else
        # Right tail: using clean 4-parameter formulation
        return d.p_xR + d.norm_const * (-const_right * ((d.right_tail.N - d.right_tail.L_x0 * (x - d.right_tail.x0)) / d.right_tail.N)^(1 - d.right_tail.N) + const_right)
    end
end

"""
    quantile(d::DoublesidedBifurcatedCrystalBall, p::Real)

Compute the quantile (inverse CDF) of the Double-sided Bifurcated Crystal Ball distribution `d` for a given probability `p`.

The function determines if the probability `p` falls into the left power-law tail, the left side of the core,
the right side of the core, or the right power-law tail, and then inverts the corresponding CDF segment.
Requires `SpecialFunctions.erfinv` to be available.
"""
function Distributions.quantile(d::DoublesidedBifurcatedCrystalBall{T}, p::Real) where {T<:Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    const_left = _norm_const(d.left_tail)
    const_right = -_norm_const(d.right_tail)

    if p < d.p_xL
        # Quantile is in the left power-law tail: using clean 4-parameter formulation
        return d.left_tail.x0 + (d.left_tail.N / d.left_tail.L_x0) * (1 - (1 / const_left * p / d.norm_const)^(1 / (1 - d.left_tail.N)))
    elseif p >= d.p_xL && p <= d.p_mu
        return d.BifGauss.μ + d.BifGauss.σL * sqrt(T(2)) * erfinv((2 * d.BifGauss.σ / d.BifGauss.σL) * (p / d.norm_const - const_left + cdf(d.BifGauss, d.left_tail.x0)) - 1)
    elseif p > d.p_mu && p <= d.p_xR
        return d.BifGauss.μ + d.BifGauss.σR * sqrt(T(2)) * erfinv((2 * d.BifGauss.σ / d.BifGauss.σR) * (p / d.norm_const - const_left + cdf(d.BifGauss, d.left_tail.x0) - (d.BifGauss.σL / (2 * d.BifGauss.σ))))
    else
        # Quantile is in the right power-law tail: using clean 4-parameter formulation
        return d.right_tail.x0 + (d.right_tail.N / d.right_tail.L_x0) * (1 - (-1 / const_right * ((p - d.p_xR) / d.norm_const - const_right))^(1 / (1 - d.right_tail.N)))
    end
end

Distributions.maximum(d::DoublesidedBifurcatedCrystalBall{T}) where {T<:Real} = T(Inf)
Distributions.minimum(d::DoublesidedBifurcatedCrystalBall{T}) where {T<:Real} = T(-Inf)

# Distributions.jl interface methods
Distributions.location(d::DoublesidedBifurcatedCrystalBall) = d.BifGauss.μ
Distributions.scale(d::DoublesidedBifurcatedCrystalBall) = d.BifGauss.σ
# Compute αL, nL, αR, nR from tail structs when needed for params()
# For bifurcated: x0L = μ - αL*σL, x0R = μ + αR*σR, and N = sqrt(1 + n²) → n = sqrt(N² - 1)
Distributions.params(d::DoublesidedBifurcatedCrystalBall) = (
    d.BifGauss.μ,
    d.BifGauss.σ,
    d.BifGauss.ψ,
    (d.BifGauss.μ - d.left_tail.x0) / d.BifGauss.σL,   # αL = (μ - x0L) / σL
    sqrt(d.left_tail.N^2 - 1),                         # nL = sqrt(N² - 1)
    (d.right_tail.x0 - d.BifGauss.μ) / d.BifGauss.σR,  # αR = (x0R - μ) / σR
    sqrt(d.right_tail.N^2 - 1)                         # nR = sqrt(N² - 1)
)
Distributions.partype(::DoublesidedBifurcatedCrystalBall{T}) where {T} = T