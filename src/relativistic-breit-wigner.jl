"""
    RelativisticBreitWigner{T<:Real} <: ContinuousUnivariateDistribution

The Relativistic Breit-Wigner distribution is a probability distribution used extensively in particle physics to describe
resonance behavior in scattering processes, particularly for unstable particles like the Z boson or the Delta baryon.

This version accounts for relativistic effects and is more accurate at higher energies than the classical (non-relativistic)
Breit-Wigner distribution.

The probability density function (PDF) is defined as:
````math
    f(x; M, Γ) = \\frac{k / [ (x^2 - M^2)^2 + M^2 Γ^2 ]}

    where     k = \\frac{2 \\sqrt{2} M Γ γ}{π \\sqrt{M^2 + γ}} and γ = \\sqrt{M^2 (M^2 + Γ^2)}
````

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

# Including the type stability
RelativisticBreitWigner(M::Real , Γ::Real) = RelativisticBreitWigner(promote(M, Γ)...)
RelativisticBreitWigner(M::Integer, Γ::Integer) = RelativisticBreitWigner(float(M), float(Γ))

# Probability Density Function (PDF) 
# The Wikipedia reference is https://en.wikipedia.org/wiki/Relativistic_Breit–Wigner_distribution
function Distributions.pdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    if x < zero(T)
        zero(T) # PDF is zero for negative values
    else
        M, Γ = r.M, r.Γ
        γ = sqrt(M^2 * (M^2 + Γ^2))
        k = (T(2)*sqrt(T(2)) * M * Γ * γ)/(π * sqrt(M^2 + γ))
        dom = (x^2 - M^2)^2 + (M^2 * Γ^2)
        k/dom
    end
end

# logarithm of the PDF
function Distributions.logpdf(r::RelativisticBreitWigner{T}, x::Real) where {T <: Real}
    return log(pdf(r,x))
end

# Cumulative Distribution Function (CDF)
# Extracted from the continous distribution section of SciPy
# https://github.com/scipy/scipy/blob/b1296b9b4393e251511fe8fdd3e58c22a1124899/scipy/stats/_continuous_distns.py#L12407C6-L12416C40
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

# Parameters
location(r::RelativisticBreitWigner) = r.M
scale(r::RelativisticBreitWigner) = r.Γ

params(r::RelativisticBreitWigner) = (r.M, r.Γ)
@inline partype(r::RelativisticBreitWigner{T}) where {T<:Real} = T

# Statistics
mean(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)
median(r::RelativisticBreitWigner) = r.M
mode(r::RelativisticBreitWigner) = r.M

var(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)
skewness(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)
kurtosis(r::RelativisticBreitWigner{T}) where {T<:Real} = T(NaN)

Distributions.minimum(r::RelativisticBreitWigner{T}) where {T <: Real} = T(-Inf)
Distributions.maximum(r::RelativisticBreitWigner{T}) where {T <: Real} = T(Inf)


