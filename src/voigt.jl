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

Voigt(μ::Real, σ::Real, γ::Real) = Voigt(promote(μ, σ, γ)...)
Voigt(μ::Integer, σ::Integer, γ::Integer) = Voigt(float(μ), float(σ), float(γ))

function Distributions.pdf(d::Voigt{T}, x::Real) where {T <: Real}
    let μ = d.μ, σ = d.σ, γ = d.γ
        z = (x - μ + im * γ) / (σ * sqrt(T(2)))
        real(exp(-z^2) * erfc(-im * z)) / (σ * sqrt(T(2π)))
    end
end

Distributions.minimum(d::Voigt{T}) where {T <: Real} = T(-Inf)
Distributions.maximum(d::Voigt{T}) where {T <: Real} = T(Inf)
