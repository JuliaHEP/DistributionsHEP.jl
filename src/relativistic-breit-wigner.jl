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

struct RelativisticBreitWigner{T<:Real} <: ContinuousUnivariateDistribution
    M::T # Location parameter (peak position)
    Γ::T # Scale parameter (width)

    # Constructor with input checks for positivity
    function RelativisticBreitWigner(M::T, Γ::T) where {T<:Real}
        M > zero(T) || error("M must be positive")
        Γ > zero(T) || error("Γ must be positive")
        new{T}(M, Γ)
    end
end

# Including the type stability (Flexible to all types of inputs)
RelativisticBreitWigner(M::Real, Γ::Real) = RelativisticBreitWigner(promote(M, Γ)...)
RelativisticBreitWigner(M::Integer, Γ::Integer) = RelativisticBreitWigner(float(M), float(Γ))

# Probability Density Function (PDF) 
# Defined on the full real line (-∞, +∞)
# See docs/RelativisticBreitWignerMath.md for derivation
function Distributions.pdf(r::RelativisticBreitWigner{T}, x::Real) where {T<:Real}
    (; M, Γ) = r
    # Define constants: A = M², B = MΓ
    A = M^2
    B = M * Γ
    r_val = sqrt(A^2 + B^2)
    v = sqrt((r_val - A) / T(2))

    # Normalized PDF: f(x) = (2rv/π) / ((M² - x²)² + M²Γ²)
    # where the denominator is (A - x²)² + B²
    normalization = T(2) * r_val * v / π
    denominator = (A - x^2)^2 + B^2
    normalization / denominator
end

# logarithm of the PDF
function Distributions.logpdf(r::RelativisticBreitWigner{T}, x::Real) where {T<:Real}
    return log(pdf(r, x))
end

# Cumulative Distribution Function (CDF)
# Closed-form expression for constant width Γ
# See docs/RelativisticBreitWignerMath.md for derivation
function Distributions.cdf(r::RelativisticBreitWigner{T}, x::Real) where {T<:Real}
    (; M, Γ) = r
    # Define constants: A = M², B = MΓ
    A = M^2
    B = M * Γ
    r_val = sqrt(A^2 + B^2)
    u = sqrt((r_val + A) / T(2))
    v = sqrt((r_val - A) / T(2))

    # Closed-form CDF formula
    # F(x) = 1/2 + (1/(2π)) * [arctan((x+u)/v) + arctan((x-u)/v)] 
    #       + (v/(4πu)) * ln(((x+u)^2+v^2)/((x-u)^2+v^2))
    term1 = T(1) / T(2)
    term2 = (T(1) / (T(2) * π)) * (atan((x + u) / v) + atan((x - u) / v))
    term3 = (v / (T(4) * π * u)) * log(((x + u)^2 + v^2) / ((x - u)^2 + v^2))

    result = term1 + term2 + term3

    # Ensure result is in [0, 1]
    return clamp(result, zero(T), one(T))
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

Distributions.minimum(r::RelativisticBreitWigner{T}) where {T<:Real} = T(-Inf)
Distributions.maximum(r::RelativisticBreitWigner{T}) where {T<:Real} = T(Inf)


