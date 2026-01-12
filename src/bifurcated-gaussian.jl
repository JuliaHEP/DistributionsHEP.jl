# Common parameter validation for bifurcated gaussian
function _check_bifurcated_gaussian_params(σ::T) where {T <: Real}
    σ > zero(T) || error("σ (scale) must be positive.")
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

d = BifurcatedGaussian(0.0, 1.0, 0.5)  # μ, σ, ψ
plot(-5, 5, x->pdf(d, x))
```
"""
struct BifurcatedGaussian{T <: Real} <: ContinuousUnivariateDistribution
    μ::T # elementary parameters
    σ::T
    ψ::T
    # Precomputed constants for PDF calculation (from elementary parameters)
    σL::T  # Left side parameter σ_L
    σR::T    # Right side parameter σ_R

    function BifurcatedGaussian(μ::T, σ::T, ψ::T) where {T <: Real}
        _check_bifurcated_gaussian_params(σ)

        # Calculate kappa
        κ = tanh(ψ)

        # Calculate scales for left and right sides
        σL = σ * (1 + κ)
        σR = σ * (1 - κ)

        new{T}(μ, σ, ψ, σL, σR)
    end
end

#=# Helper function to compute CDF values at transition points
function _compute_transition_cdf_value(d::BifurcatedGaussian{T}) where {T <: Real}
    # CDF value at transition point (from mathematical derivation)
    cdf_at_mu = 

    return cdf_at_minus_alphaL, cdf_at_plus_alphaR
end=#


function Distributions.pdf(d::BifurcatedGaussian{T}, x::Real) where {T <: Real}
    # left side
    x <= d.μ && return one(T) / (sqrt(T(2π)) * d.σ) * exp(-T(0.5) * ((x - d.μ) / d.σL)^2)
    # right side
    return one(T) / (sqrt(T(2π)) * d.σ) * exp(-T(0.5) * ((x - d.μ) / d.σR)^2)
end

function Distributions.cdf(d::BifurcatedGaussian{T}, x::Real) where {T <: Real}

    # CDF values at transition point (from mathematical derivation)
    cdf_at_mu = d.σL / (T(2) * d.σ)

    if x <= d.μ
        # CDF for the left power-law tail (x̂ ≤ -αL)
        # The integral is: ∫_{-∞}^{x̂} N * A_L * (B_L - t)^(-n_L) / σ dt
        # = N * A_L / (n_L - 1) * (B_L - x̂)^(1 - n_L)
        return cdf_at_mu * (1 + erf((x - d.μ) / (d.σL * sqrt(T(2)))))
    else
        # CDF for the right power-law tail (x̂ ≥ αR)
        # CDF at αR + integral of right tail from αR to x̂
        # The integral should be: ∫_{αR}^{x̂} N * A_R * (B_R + t)^(-n_R) dt
        # = N * A_R / (n_R - 1) * ((B_R + αR)^(1-n_R) - (B_R + x̂)^(1-n_R))
        # The integral is: ∫_{αR}^{x̂} N * A_R * (B_R + t)^(-n_R) / σ dt
        # = N * A_R / (n_R - 1) * ((B_R + αR)^(1 - n_R) - (B_R + x̂)^(1 - n_R))
        return cdf_at_mu + d.σR / (T(2) * d.σ) * erf((x - d.μ) / (d.σR * sqrt(T(2))))
    end
end


function Distributions.quantile(d::BifurcatedGaussian{T}, p::Real) where {T <: Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)

    # CDF values at transition points (same as in CDF function)
    cdf_at_mu = d.σL / (T(2) * d.σ)

    if p <= cdf_at_mu
        # Quantile is in the left power-law tail
        return d.μ + d.σL * sqrt(T(2)) * erfinv(p * (T(2) * d.σ) / d.σL - 1)
    else
        # Quantile is in the Gaussian core
        # Solve: p = cdf_at_minus_alphaL + N * sqrt(π/2) * (erf(x̂/sqrt(2)) + erf(αL/sqrt(2)))
        # Rearranging: erf(x̂/sqrt(2)) = (p - cdf_at_minus_alphaL) / (N * sqrt(π/2)) - erf(αL/sqrt(2))
        return d.μ + d.σR * sqrt(T(2)) * erfinv((p - cdf_at_mu) * (T(2) * d.σ) / d.σR)
    end
end

Distributions.maximum(d::BifurcatedGaussian{T}) where {T <: Real} = T(Inf)
Distributions.minimum(d::BifurcatedGaussian{T}) where {T <: Real} = T(-Inf)