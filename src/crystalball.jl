# Common parameter validation
function _check_crystalball_params(σ::T, α::T, n::T) where {T<:Real}
    σ > zero(T) || error("σ (scale) must be positive.")
    α > zero(T) || error("α (transition point) must be positive.")
    n > one(T) || error("n (power-law exponent) must be greater than 1.")
end

struct CrystalBall{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    α::T
    n::T
    # Precomputed constants for PDF calculation
    norm_const::T # Normalization constant N
    A_const::T    # Tail parameter A
    B_const::T    # Tail parameter B

    function CrystalBall(μ::T, σ::T, α::T, n::T) where {T<:Real}
        _check_crystalball_params(σ, α, n)
        # absα is effectively α due to the check α > 0
        C = n / α / (n - 1) * exp(-α^2 / 2)
        D_val = sqrt(T(π) / 2) * (one(T) + erf(α / sqrt(T(2))))
        N = one(T) / (σ * (C + D_val))

        A = (n / α)^n * exp(-α^2 / 2)
        B = n / α - α
        new{T}(μ, σ, α, n, N, A, B)
    end
end

function Distributions.pdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    x̂ = (x - d.μ) / d.σ
    if x̂ > -d.α # Gaussian part
        return d.norm_const * exp(-x̂^2 / 2)
    else # Power-law tail part
        return d.norm_const * d.A_const * (d.B_const - x̂)^(-d.n)
    end
end

function Distributions.cdf(d::CrystalBall{T}, x::Real) where {T<:Real}
    x̂ = (x - d.μ) / d.σ

    # Integral of the tail from -Inf to -α
    # This is σ * C from the constructor's N calculation, divided by N itself to get the unnormalized integral, then multiplied by d.norm_const
    # Integral_tail_part = (d.n / d.α / (d.n - 1) * exp(-d.α^2 / 2))
    # Simplified: this is the value of the tail part of CDF at x̂ = -d.α
    cdf_at_minus_alpha = d.norm_const * d.A_const / (d.n - 1) * (d.B_const - (-d.α))^(1 - d.n)

    if x̂ <= -d.α
        # Power-law tail part: integral of d.norm_const * d.A_const * (d.B_const - t)^(-d.n) dt
        # from -Inf to x̂
        # Indefinite integral is d.norm_const * d.A_const / (d.n-1) * (d.B_const - t)^(1-d.n)
        # As t -> -Inf, (d.B_const - t)^(1-d.n) -> 0 because 1-d.n < 0.
        return d.norm_const * d.A_const / (d.n - 1) * (d.B_const - x̂)^(1 - d.n)
    else
        # Gaussian part: CDF_at_minus_alpha + integral of d.norm_const * exp(-t^2/2) dt from -α to x̂
        # Integral of exp(-t^2/2) from -α to x̂ is
        # sqrt(π/2) * (erf(x̂/√2) - erf(-α/√2))
        # = sqrt(π/2) * (erf(x̂/√2) + erf(α/√2))
        integral_gaussian_part = sqrt(T(π) / 2) * (erf(x̂ / sqrt(T(2))) + erf(d.α / sqrt(T(2))))
        return cdf_at_minus_alpha + d.norm_const * d.σ * integral_gaussian_part
    end
end