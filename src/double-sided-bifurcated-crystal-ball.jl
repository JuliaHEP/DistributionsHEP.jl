# Common parameter validation for bifurcated gaussian
function _check_double_sided_bifurcated_crystal_ball_params(σ::T, αL::T, nL::T, αR::T, nR::T) where {T <: Real}
    σ > zero(T) || error("σ (scale) must be positive.")
    αL > zero(T) || error("αL (left transition point) must be left of μ.")
    nL > zero(T) || error("nL (left power-law exponent) must be greater than 1.")
    αR > zero(T) || error("αR (right transition point) must be right of μ.")
    nR > zero(T) || error("nR (right power-law exponent) must be greater than 1.")
end

# Helper function logarithmic derivative
function _log_derivative_bifurcated_gaussian(d::BifurcatedGaussian{T}) where {T <: Real}
    return (x::T) -> begin
        if x <= d.μ
            return -(x - d.μ) / (d.σL^2)
        else
            return -(x - d.μ) / (d.σR^2)
        end
    end
end

struct DoublesidedBifurcatedCrystalBall{T <: Real} <: ContinuousUnivariateDistribution
    μ::T # elementary parameters
    σ::T
    ψ::T
    αL::T
    αR::T
    nL::T
    nR::T
    # Precomputed constants for PDF calculation (from elementary parameters)
    σL::T    # Left side parameter σ_L
    σR::T    # Right side parameter σ_R
    xL::T    # Left transition point
    xR::T    # Right transition point
    NL::T    # Left power
    NR::T    # Right power
    G_xL::T  # Bifurcated Gaussian PDF at xL
    G_xR::T  # Bifurcated Gaussian PDF at xR
    L_xL::T  # Logarithmic derivative at xL
    L_xR::T  # Logarithmic derivative at xR
    norm_const::T  # Normalization constant N
    p_xL::T        # CDF at left transition point
    p_xR::T        # CDF at right transition point
    p_mu::T        # CDF at mean μ

    function DoublesidedBifurcatedCrystalBall(μ::T, σ::T, ψ::T, αL::T, nL::T, αR::T, nR::T) where {T <: Real}
        _check_double_sided_bifurcated_crystal_ball_params(σ, αL, nL, αR, nR)

        # Calculate kappa
        κ = tanh(ψ)

        # Calculate scales for left and right sides
        σL = σ * (1 + κ)
        σR = σ * (1 - κ)

        # Calculate transition points
        xL = μ - αL * σL
        xR = μ + αR * σR

        # Calculate powers
        NL = sqrt(one(T) + nL^2)
        NR = sqrt(one(T) + nR^2)

        # Pre-compute constants
        BifGauss = BifurcatedGaussian(μ, σ, ψ)
        G_xL = pdf(BifGauss, xL)
        G_xR = pdf(BifGauss, xR)

        L_BifGauss = _log_derivative_bifurcated_gaussian(BifGauss)
        L_xL = L_BifGauss(xL)
        L_xR = L_BifGauss(xR)

        # Calculate normalization constant
        norm_const_left = G_xL / L_xL * NL / (NL - 1)
        norm_const_right = - G_xR / L_xR * NR / (NR - 1)
        norm_const_core = cdf(BifGauss, xR) - cdf(BifGauss, xL)

        norm_const = one(T) / (norm_const_left + norm_const_right + norm_const_core)

        p_xL = norm_const * norm_const_left
        p_xR = norm_const * (norm_const_left + norm_const_core)
        p_mu = norm_const * (norm_const_left + σL / (2 * σ) - cdf(BifGauss, xL))

        new{T}(μ, σ, ψ, αL, αR, nL, nR, σL, σR, xL, xR, NL, NR, G_xL, G_xR, L_xL, L_xR, norm_const, p_xL, p_xR, p_mu)
    end
end

function Distributions.pdf(d::DoublesidedBifurcatedCrystalBall{T}, x::Real) where {T <: Real}
    BifGauss = BifurcatedGaussian(d.μ, d.σ, d.ψ)
    # left tail
    x < d.xL && return d.norm_const * d.G_xL * (d.NL / (d.NL - d.L_xL * (x - d.xL))) ^ d.NL
    # core
    x >= d.xL && x <= d.xR && return d.norm_const * pdf(BifGauss, x)
    # right tail
    return d.norm_const * d.G_xR * (d.NR / (d.NR - d.L_xR * (x - d.xR))) ^ d.NR
end

function Distributions.cdf(d::DoublesidedBifurcatedCrystalBall{T}, x::Real) where {T <: Real}
    BifGauss = BifurcatedGaussian(d.μ, d.σ, d.ψ)
    const_left = d.G_xL / d.L_xL * d.NL / (d.NL - 1)
    const_right = - d.G_xR / d.L_xR * d.NR / (d.NR - 1)

    if x < d.xL
        return d.norm_const * const_left * ((d.NL - d.L_xL * (x - d.xL)) / d.NL)^(1 - d.NL)
    elseif x >= d.xL && x <= d.xR
        return d.p_xL + d.norm_const * (cdf(BifGauss, x) - cdf(BifGauss, d.xL))
    else
        return d.p_xR + d.norm_const * (- const_right * ((d.NR - d.L_xR * (x - d.xR)) / d.NR)^(1 - d.NR) + const_right)
    end
end


function Distributions.quantile(d::DoublesidedBifurcatedCrystalBall{T}, p::Real) where {T <: Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    BifGauss = BifurcatedGaussian(d.μ, d.σ, d.ψ)
    const_left = d.G_xL / d.L_xL * d.NL / (d.NL - 1)
    const_right = - d.G_xR / d.L_xR * d.NR / (d.NR - 1)

    if p < d.p_xL
        # Quantile is in the left power-law tail
        return d.xL + (d.NL / d.L_xL) * (1 - (1 / const_left * p / d.norm_const) ^ (1 / (1 - d.NL)))
    elseif p >= d.p_xL && p <= d.p_mu
        return d.μ + d.σL * sqrt(T(2)) * erfinv((2 * d.σ / d.σL) * (p / d.norm_const - const_left + cdf(BifGauss, d.xL)) - 1)
    elseif p > d.p_mu && p <= d.p_xR
        return d.μ + d.σR * sqrt(T(2)) * erfinv((2 * d.σ / d.σR) * (p / d.norm_const - const_left + cdf(BifGauss, d.xL) - (d.σL / (2 * d.σ))))
    else
        return d.xR + (d.NR / d.L_xR) * (1 - (- 1 / const_right * ((p - d.p_xR) / d.norm_const - const_right)) ^ (1 / (1 - d.NR)))
    end
end

Distributions.maximum(d::DoublesidedBifurcatedCrystalBall{T}) where {T <: Real} = T(Inf)
Distributions.minimum(d::DoublesidedBifurcatedCrystalBall{T}) where {T <: Real} = T(-Inf)