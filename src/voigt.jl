function _check_voigt_params(σ::T, γ::T) where {T <: Real}
    σ > zero(T) || error("σ (Gaussian width) must be positive.")
    γ > zero(T) || error("γ (Lorentzian width) must be positive.")
end

struct Voigt{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    γ::T
    function Voigt(μ::T, σ::T, γ::T) where {T <: Real}
        _check_voigt_params(σ, γ)
        new{T}(μ, σ, γ)
    end
end

function voigt(x::Real, μ::Real, σ::Real, γ::Real)
    z = (x - μ + im * γ) / (σ * sqrt(2))
    return real(exp(-z^2) * erfc(-im * z)) / (σ * sqrt(2π))
end

function Distributions.pdf(d::Voigt{T}, x::Real) where {T <: Real}
    return voigt(x, d.μ, d.σ, d.γ)
end

Distributions.minimum(d::Voigt{T}) where {T <: Real} = T(-Inf)
Distributions.maximum(d::Voigt{T}) where {T <: Real} = T(Inf)
