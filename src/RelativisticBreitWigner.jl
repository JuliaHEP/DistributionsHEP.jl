function check_bw_params( M::T, Γ::T) where {T <: Real}
    M > zero(T) || error("M must be positive")
    Γ > zero(T) || error("Γ must be positive")
end

struct RelativisticBreitWigner{T <: Real} <: ContinuousUnivariateDistribution 
    M::T
    Γ::T

    function RelativisticBreitWigner(M::T, Γ::T) where {T <: Real}
        check_bw_params(M, Γ)
        new{T}(M, Γ)
    end
end 

function Distributions.pdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    M,Γ = r.M, r.Γ
    γ = sqrt(M^2 * (M^2 + Γ^2))
    k = (2*sqrt(2) * M * Γ * γ)/(π * sqrt(M^2 + γ))
    dom = (x^2 - M^2)^2 + M^2 * Γ^2
    return k/dom
end

function Distributions.logpdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    return log(pdf(r,x))
end

function Distributions.cdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    let ρ = r.M/r.Γ, C =  1/π * √(2 / (1 + √ (1 + 1/ρ²) ))
        z1 = sqrt(-1 + im / ρ)
        z2 = sqrt(-ρ * (ρ + im))
        term = z1 * atan(x / z2)
        result = 2 * C * imag(term)
        return min(result, one(T))
    end
end

location(r::RelativisticBreitWigner) = r.M
scale(r::RelativisticBreitWigner) = r.Γ

params(r::RelativisticBreitWigner) = (r.M, r.Γ)
@inline partype(r::RelativisticBreitWigner{T}) where {T<:Real} = T

mean(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)
median(r::RelativisticBreitWigner) = r.M
mode(r::RelativisticBreitWigner) = r.M

var(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)
skewness(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)
kurtosis(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)

entropy(r::RelativisticBreitWigner) = log4π + log(r.Γ)

Distributions.minimum(r::RelativisticBreitWigner{T}) where {T <: Real} = T(-Inf)
Distributions.maximum(r::RelativisticBreitWigner{T}) where {T <: Real} = T(Inf)


