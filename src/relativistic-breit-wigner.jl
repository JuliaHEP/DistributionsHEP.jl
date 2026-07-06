"""
    RelativisticBreitWigner{T<:Real} <: ContinuousUnivariateDistribution

The Relativistic Breit-Wigner distribution is a probability distribution used extensively in particle physics to describe
resonance behavior in scattering processes, particularly for unstable particles like the Z boson or the Delta baryon.

This version accounts for relativistic effects and is more accurate at higher energies than the classical (non-relativistic)
Breit-Wigner distribution.

The probability density function (PDF) is defined on ``x >= 0`` as:
````math
f(x; M, Γ) = \\frac{k}{(x^2 - M^2)^2 + M^2 Γ^2}
````
where ``k = \\frac{2 M^3 γ S}{π R}``, ``γ = Γ / M``,
``S = \\sqrt{1 + γ^2}``, and ``R = \\sqrt{(1 + S) / 2}``.

The quantile function is not implemented. Use NumericalDistributions.jl or solve
``cdf(d, x) - p = 0`` numerically for inverse-transform sampling.

#Arguments
- "M" : A real-valued location parameter that shifts the center of the distribution.
- "Γ" : A positive real-valued scale parameter that controls the spread (width) of the distribution.

#Example
```julia
using DistributionsHEP
using Plots

d = RelativisticBreitWigner(1.0, 2.5) # M, Γ
x = range(0, 100.0, length=500)
y = pdf.(d, x)
plot(x, y, xlabel="x", ylabel="PDF")
```

"""

struct RelativisticBreitWigner{T <: Real} <: ContinuousUnivariateDistribution 
    M::T # Location parameter (peak position)
    Γ::T # Scale parameter (width)

    # Constructor with input checks for positivity
    function RelativisticBreitWigner(M::T, Γ::T) where {T <: Real}
        M > zero(T) || error("M must be positive")
        Γ > zero(T) || error("Γ must be positive")
        new{T}(M, Γ)
    end
end 

# Including the type stability (Flexible to all types of inputs)
RelativisticBreitWigner(M::Real , Γ::Real) = RelativisticBreitWigner(promote(M, Γ)...)
RelativisticBreitWigner(M::Integer, Γ::Integer) = RelativisticBreitWigner(float(M), float(Γ))

# Probability Density Function (PDF) 
# The Wikipedia reference is https://en.wikipedia.org/wiki/Relativistic_Breit–Wigner_distribution
function Distributions.pdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    if x < zero(T)
        zero(T) # PDF is zero for negative values
    else
        (; M, Γ) = r
        γ = Γ / M
        S = sqrt(one(T) + γ^2)
        R = sqrt((one(T) + S) / T(2))
        normalization = T(2) * M^3 * γ * S / (T(π) * R)
        Msq_minus_xsq = (M-x)*(M+x)
        denominator = Msq_minus_xsq^2 + (M^2 * Γ^2)
        normalization/denominator
    end
end

# logarithm of the PDF
function Distributions.logpdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    return log(pdf(r,x))
end

function Distributions.cdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    x < zero(T) && return zero(T)
    isinf(x) && return one(T)

    (; M, Γ) = r
    xT = T(x)
    γ = Γ / M
    S = sqrt(one(T) + γ^2)
    R = sqrt((one(T) + S) / T(2))
    K = γ / (T(2) * R)
    y = xT / M

    atan_term = (atan((y + R) / K) + atan((y - R) / K)) / T(π)
    log_term = γ / (T(2) * T(π) * (one(T) + S)) *
        log(((y + R)^2 + K^2) / ((y - R)^2 + K^2))
    result = atan_term + log_term
    return min(max(result, zero(T)), one(T))
end

function Distributions.quantile(::RelativisticBreitWigner, p::Real)
    zero(p) <= p <= one(p) || throw(DomainError(p, "p must be in [0, 1]"))
    throw(ArgumentError(
        "quantile is not implemented for RelativisticBreitWigner; use NumericalDistributions.jl or solve cdf(d, x) - p = 0 numerically",
    ))
end

# Parameters
Distributions.location(r::RelativisticBreitWigner) = r.M
Distributions.scale(r::RelativisticBreitWigner) = r.Γ

Distributions.params(r::RelativisticBreitWigner) = (r.M, r.Γ)
@inline partype(r::RelativisticBreitWigner{T}) where {T<:Real} = T

# Statistics
mode(r::RelativisticBreitWigner) = r.M
skewness(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)

function mean(r::RelativisticBreitWigner{T}) where {T<:Real}
    γ = r.Γ / r.M
    S = sqrt(one(T) + γ^2)
    R = sqrt((one(T) + S) / T(2))
    return r.M * (T(π) - atan(γ)) * S / (T(π) * R)
end

function var(r::RelativisticBreitWigner{T}) where {T<:Real}
    γ = r.Γ / r.M
    S = sqrt(one(T) + γ^2)
    μ = mean(r)
    return r.M^2 * S - μ^2
end

std(r::RelativisticBreitWigner) = sqrt(var(r))
kurtosis(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)


Distributions.minimum(r::RelativisticBreitWigner{T}) where {T <: Real} = zero(T)
Distributions.maximum(r::RelativisticBreitWigner{T}) where {T <: Real} = T(Inf)

