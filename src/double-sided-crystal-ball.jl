# Common parameter validation for two-sided crystal ball
function _check_double_crystalball_params(σ::T, αL::T, nL::T, αR::T, nR::T) where {T <: Real}
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
    f(x; μ, σ, α_L, n_L, α_R, n_R) = N A_L (B_L - x̂)^{-n_L}     for x̂ < -α_L
                                   = N exp(-(x̂^2)/2)            for -α_L ≤ x̂ ≤ α_R
                                   = N A_R (B_R + x̂)^{-n_R}     for x̂ > α_R
````
where x̂ = (x - μ) / σ.
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
struct DoubleCrystalBall{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    αL::T
    nL::T
    αR::T
    nR::T
    # Precomputed constants for PDF calculation
    norm_const::T  # Normalization constant N
    AL_const::T    # Left tail parameter A_L
    BL_const::T    # Left tail parameter B_L
    AR_const::T    # Right tail parameter A_R
    BR_const::T    # Right tail parameter B_R

    function DoubleCrystalBall(μ::T, σ::T, αL::T, nL::T, αR::T, nR::T) where {T <: Real}
        _check_double_crystalball_params(σ, αL, nL, αR, nR)

        # Calculate constants for left tail
        CL = nL / αL / (nL - 1) * exp(-αL^2 / 2)
        AL = (nL / αL)^nL * exp(-αL^2 / 2)
        BL = nL / αL - αL

        # Calculate constants for right tail
        CR = nR / αR / (nR - 1) * exp(-αR^2 / 2)
        AR = (nR / αR)^nR * exp(-αR^2 / 2)
        BR = nR / αR - αR

        # Calculate normalization constant
        # The normalization should be: N = 1 / (CL + CR + D_val)
        # where CL, CR are the tail contributions and D_val is the Gaussian core contribution
        D_val = sqrt(T(π) / 2) * (erf(αR / sqrt(T(2))) + erf(αL / sqrt(T(2))))
        N = one(T) / (CL + CR + D_val)

        new{T}(μ, σ, αL, nL, αR, nR, N, AL, BL, AR, BR)
    end
end

# Helper function to compute CDF values at transition points
function _compute_transition_cdf_values(d::DoubleCrystalBall{T}) where {T <: Real}
    # CDF values at transition points (from mathematical derivation)
    cdf_at_minus_alphaL = d.norm_const * d.nL / d.αL / (d.nL - 1) * exp(-d.αL^2 / 2)
    cdf_at_plus_alphaR =
        cdf_at_minus_alphaL +
        d.norm_const *
        sqrt(T(π) / 2) *
        (erf(d.αR / sqrt(T(2))) + erf(d.αL / sqrt(T(2))))

    return cdf_at_minus_alphaL, cdf_at_plus_alphaR
end


function Distributions.pdf(d::DoubleCrystalBall{T}, x::Real) where {T <: Real}
    x̂ = (x - d.μ) / d.σ
    # Left power-law tail
    x̂ < -d.αL && return d.norm_const * d.AL_const * (d.BL_const - x̂)^(-d.nL) / d.σ
    # Gaussian core
    x̂ < d.αR && return d.norm_const * exp(-x̂^2 / 2) / d.σ
    # Right power-law tail
    return d.norm_const * d.AR_const * (d.BR_const + x̂)^(-d.nR) / d.σ
end

function Distributions.cdf(d::DoubleCrystalBall{T}, x::Real) where {T <: Real}
    x̂ = (x - d.μ) / d.σ

    # CDF values at transition points (from mathematical derivation)
    cdf_at_minus_alphaL, cdf_at_plus_alphaR = _compute_transition_cdf_values(d)

    if x̂ <= -d.αL
        # CDF for the left power-law tail (x̂ ≤ -αL)
        # The integral is: ∫_{-∞}^{x̂} N * A_L * (B_L - t)^(-n_L) / σ dt
        # = N * A_L / (n_L - 1) * (B_L - x̂)^(1 - n_L)
        return d.norm_const * d.AL_const / (d.nL - 1) * (d.BL_const - x̂)^(1 - d.nL)
    elseif x̂ >= d.αR
        # CDF for the right power-law tail (x̂ ≥ αR)
        # CDF at αR + integral of right tail from αR to x̂
        # The integral should be: ∫_{αR}^{x̂} N * A_R * (B_R + t)^(-n_R) dt
        # = N * A_R / (n_R - 1) * ((B_R + αR)^(1-n_R) - (B_R + x̂)^(1-n_R))
        # The integral is: ∫_{αR}^{x̂} N * A_R * (B_R + t)^(-n_R) / σ dt
        # = N * A_R / (n_R - 1) * ((B_R + αR)^(1 - n_R) - (B_R + x̂)^(1 - n_R))
        tail_integral =
            d.norm_const * d.AR_const / (d.nR - 1) *
            ((d.BR_const + d.αR)^(1 - d.nR) - (d.BR_const + x̂)^(1 - d.nR))
        return cdf_at_plus_alphaR + tail_integral
    else
        # CDF for the Gaussian core (-αL < x̂ < αR)
        # CDF at -αL + integral of Gaussian PDF from -αL to x̂
        # The integral is: ∫_{-αL}^{x̂} N * exp(-t^2/2) / σ dt = N * sqrt(π/2) * (erf(x̂/sqrt(2)) + erf(αL/sqrt(2)))
        integral_gaussian_part =
            sqrt(T(π) / 2) * (erf(x̂ / sqrt(T(2))) + erf(d.αL / sqrt(T(2))))
        return cdf_at_minus_alphaL + d.norm_const * integral_gaussian_part
    end
end


function Distributions.quantile(d::DoubleCrystalBall{T}, p::Real) where {T <: Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    # CDF values at transition points (same as in CDF function)
    cdf_at_minus_alphaL, cdf_at_plus_alphaR = _compute_transition_cdf_values(d)

    x̂ = zero(T) # Scaled quantile score

    if p <= cdf_at_minus_alphaL
        # Quantile is in the left power-law tail
        base = (p * (d.nL - 1)) / (d.norm_const * d.AL_const)
        x̂ = d.BL_const - base^(one(T) / (one(T) - d.nL))
    elseif p >= cdf_at_plus_alphaR
        # Quantile is in the right power-law tail
        # Solve: p = cdf_at_plus_alphaR + tail_integral
        # where tail_integral = norm_const * AR_const / (nR - 1) * ((BR_const + αR)^(1 - nR) - (BR_const + x̂)^(1 - nR))
        base =
            (d.BR_const + d.αR)^(1 - d.nR) -
            (p - cdf_at_plus_alphaR) * (d.nR - 1) / (d.norm_const * d.AR_const)
        x̂ = base^(one(T) / (one(T) - d.nR)) - d.BR_const
    else
        # Quantile is in the Gaussian core
        # Solve: p = cdf_at_minus_alphaL + N * sqrt(π/2) * (erf(x̂/sqrt(2)) + erf(αL/sqrt(2)))
        # Rearranging: erf(x̂/sqrt(2)) = (p - cdf_at_minus_alphaL) / (N * sqrt(π/2)) - erf(αL/sqrt(2))
        term_for_erfinv_num = (p - cdf_at_minus_alphaL)
        term_for_erfinv_den = d.norm_const * sqrt(T(π) / T(2))

        erf_alphaL_sqrt2 = erf(d.αL / sqrt(T(2)))
        arg_erfinv = (term_for_erfinv_num / term_for_erfinv_den) - erf_alphaL_sqrt2

        x̂ = sqrt(T(2)) * erfinv(arg_erfinv)
    end

    return d.μ + d.σ * x̂
end

Distributions.maximum(d::DoubleCrystalBall{T}) where {T <: Real} = T(Inf)
Distributions.minimum(d::DoubleCrystalBall{T}) where {T <: Real} = T(-Inf)
