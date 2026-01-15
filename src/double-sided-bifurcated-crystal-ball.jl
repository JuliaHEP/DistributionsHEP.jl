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

        # Calculate normalization constant using _integral
        left_tail_contribution = _integral(left_tail, left_tail.x0)
        right_tail_contribution = -_integral(right_tail, right_tail.x0)  # Right tail has negative L_x0, so negate
        # Integral from xL to xR = cdf(BifGauss, xR) - cdf(BifGauss, xL)
        core_contribution = cdf(BifGauss, xR) - cdf(BifGauss, xL)
        N = one(T) / (left_tail_contribution + right_tail_contribution + core_contribution)

        new{T}(BifGauss, left_tail, right_tail, N)
    end
end

"""
    _compute_transition_cdf_values(d::DoublesidedBifurcatedCrystalBall)

Compute the CDF values at the two transition points (left and right) of the Double-sided Bifurcated Crystal Ball distribution.

Returns a tuple `(cdf_at_xL, cdf_at_xR)` where:
- `cdf_at_xL` is the CDF value at the left transition point (x = μ - αL * σL)
- `cdf_at_xR` is the CDF value at the right transition point (x = μ + αR * σR)
"""
function _compute_transition_cdf_values(d::DoublesidedBifurcatedCrystalBall{T}) where {T<:Real}
    cdf_at_xL = d.norm_const * _integral(d.left_tail, d.left_tail.x0)
    # Integral from xL to xR = cdf(BifGauss, xR) - cdf(BifGauss, xL)
    core_contribution = cdf(d.BifGauss, d.right_tail.x0) - cdf(d.BifGauss, d.left_tail.x0)
    cdf_at_xR = cdf_at_xL + d.norm_const * core_contribution

    return cdf_at_xL, cdf_at_xR
end

# Convenience constructors
DoublesidedBifurcatedCrystalBall(μ::Real, σ::Real, ψ::Real, αL::Real, nL::Real, αR::Real, nR::Real) =
    DoublesidedBifurcatedCrystalBall(promote(μ, σ, ψ, αL, nL, αR, nR)...)
DoublesidedBifurcatedCrystalBall(μ::Integer, σ::Integer, ψ::Integer, αL::Integer, nL::Integer, αR::Integer, nR::Integer) =
    DoublesidedBifurcatedCrystalBall(float(μ), float(σ), float(ψ), float(αL), float(nL), float(αR), float(nR))

function Distributions.pdf(d::DoublesidedBifurcatedCrystalBall{T}, x::Real) where {T<:Real}
    # Left power-law tail
    if x < d.left_tail.x0
        offset = x - d.left_tail.x0
        return d.norm_const * _value(d.left_tail, offset)
    end
    # Bifurcated Gaussian core
    if x <= d.right_tail.x0
        return d.norm_const * pdf(d.BifGauss, x)
    end
    # Right power-law tail
    offset = x - d.right_tail.x0
    return d.norm_const * _value(d.right_tail, offset)
end

function Distributions.cdf(d::DoublesidedBifurcatedCrystalBall{T}, x::Real) where {T<:Real}
    cdf_at_xL, cdf_at_xR = _compute_transition_cdf_values(d)

    if x <= d.left_tail.x0
        # Left power-law tail
        return d.norm_const * _integral(d.left_tail, x)
    elseif x >= d.right_tail.x0
        # Right power-law tail
        # For right tail, _integral(right_tail, x) = -∫[x, +∞] and _integral(right_tail, x0R) = -∫[x0R, +∞]
        # CDF(x) = CDF(x0R) + ∫[x0R, x] = CDF(x0R) + (∫[x0R, +∞] - ∫[x, +∞])
        # = CDF(x0R) + (_integral(right_tail, x) - _integral(right_tail, x0R))
        integral_at_x = _integral(d.right_tail, x)
        integral_at_x0R = _integral(d.right_tail, d.right_tail.x0)
        return cdf_at_xR + d.norm_const * (integral_at_x - integral_at_x0R)
    else
        # Bifurcated Gaussian core
        # Integral from xL to x = cdf(BifGauss, x) - cdf(BifGauss, xL)
        core_contribution = cdf(d.BifGauss, x) - cdf(d.BifGauss, d.left_tail.x0)
        return cdf_at_xL + d.norm_const * core_contribution
    end
end

function Distributions.quantile(d::DoublesidedBifurcatedCrystalBall{T}, p::Real) where {T<:Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    cdf_at_xL, cdf_at_xR = _compute_transition_cdf_values(d)

    if p <= cdf_at_xL
        # Left power-law tail
        tail_integral = p / d.norm_const
        return _integral_inversion(d.left_tail, tail_integral)
    elseif p >= cdf_at_xR
        # Right power-law tail
        # For right tail: CDF(x) = CDF(x0R) + N * (_integral(right_tail, x) - _integral(right_tail, x0R))
        # So: _integral(right_tail, x) = (p - cdf_at_xR) / N + _integral(right_tail, x0R)
        integral_at_x0R = _integral(d.right_tail, d.right_tail.x0)
        tail_integral = (p - cdf_at_xR) / d.norm_const + integral_at_x0R
        return _integral_inversion(d.right_tail, tail_integral)
    else
        # Bifurcated Gaussian core
        # p = cdf_at_xL + N * (cdf(BifGauss, x) - cdf(BifGauss, xL))
        # So: cdf(BifGauss, x) = (p - cdf_at_xL) / N + cdf(BifGauss, xL)
        target_cdf = (p - cdf_at_xL) / d.norm_const + cdf(d.BifGauss, d.left_tail.x0)
        return quantile(d.BifGauss, target_cdf)
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