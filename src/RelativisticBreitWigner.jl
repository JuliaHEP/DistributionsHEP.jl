struct RelativisticBreitWigner{T <: Real} <: ContinuousUnivariateDistribution 
    M::T
    Γ::T

    function RelativisticBreitWigner(M::T, Γ::T) where {T <: Real}
        M > zero(T) || error("M must be positive")
        Γ > zero(T) || error("Γ must be positive")
        new{T}(M, Γ)
    end
end 


RelativisticBreitWigner(M::Real , Γ::Real) = RelativisticBreitWigner(promote(M, Γ)...)
RelativisticBreitWigner(M::Integer, Γ::Integer) = RelativisticBreitWigner(float(M), float(Γ))

function Distributions.pdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    if x < zero(T)
        zero(T)
    else
        M, Γ = r.M, r.Γ
        γ = sqrt(M^2 * (M^2 + Γ^2))
        k = (T(2)*sqrt(T(2)) * M * Γ * γ)/(π * sqrt(M^2 + γ))
        dom = (x^2 - M^2)^2 + (M^2 * Γ^2)
        k/dom
    end
end

function Distributions.logpdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    return log(pdf(r,x))
end

function Distributions.cdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    if x < zero(T)
        zero(T)
    else
        let ρ = r.M/r.Γ, two = T(2)
            C =  1/T(π) * √(two / (1 + √ (1 + 1/ρ^2) ))
            z1 = sqrt(-1 + im / ρ)
            z2 = sqrt(-ρ * (ρ + im))
            term = z1 * atan(x / z2)
            result = abs(two * C * imag(term))
            return min(result, one(T))
        end
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

Distributions.minimum(r::RelativisticBreitWigner{T}) where {T <: Real} = T(-Inf)
Distributions.maximum(r::RelativisticBreitWigner{T}) where {T <: Real} = T(Inf)


