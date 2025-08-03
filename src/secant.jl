"""
    HyperbolicSecant{T<:Real} <: ContinuousUnivariateDistribution

The Hyperbolic Secant distribution is a continuous probability distribution that forms a location-scale family.
It has interesting parallels to the normal distribution and is characterized by a bell-shaped curve that is 
more peaked at the mode and has fatter tails compared to the normal distribution.

The probability density function for the standard hyperbolic secant distribution (μ=0, σ=1) is:
````math
    f(z) = (1/2) * sech(π*z/2)    for z ∈ ℝ
````

For the general case with location parameter μ and scale parameter σ:
````math
    f(x; μ, σ) = (1/(2σ)) * sech(π*(x-μ)/(2σ))    for x ∈ ℝ
````

where sech(x) = 2/(e^x + e^(-x)) is the hyperbolic secant function.

The distribution function (CDF) is:
````math
    F(x; μ, σ) = (2/π) * arctan(exp(π*(x-μ)/(2σ)))
````

The quantile function (inverse CDF) is:
````math
    F^(-1)(p; μ, σ) = μ + σ * (2/π) * log(tan(π*p/2))
````

# Arguments
- `μ`: Location parameter (mean of the distribution)
- `σ`: Scale parameter (standard deviation of the distribution). Must be positive.

# Properties
- Mean: μ
- Variance: σ²
- Skewness: 0 (symmetric distribution)
- Kurtosis: 5 (excess kurtosis = 2)

# Example
```julia
using DistributionsHEP

# Standard hyperbolic secant distribution (μ=0, σ=1)
d1 = HyperbolicSecant()

# Custom parameters
d2 = HyperbolicSecant(2.0, 1.5)  # μ=2.0, σ=1.5

# Evaluate PDF, CDF, and generate random samples
pdf(d2, 2.0)
cdf(d2, 2.0)
rand(d2)
```

See also: The standard hyperbolic secant distribution has special mathematical properties:
- Characteristic function: χ(t) = sech(t)
- Moment generating function: M(t) = sec(t) for |t| < π/2
"""
struct HyperbolicSecant{T <: Real} <: ContinuousUnivariateDistribution
    μ::T  # location parameter (mean)
    σ::T  # scale parameter (standard deviation)

    function HyperbolicSecant{T}(μ::T, σ::T) where {T <: Real}
        new{T}(μ, σ)
    end
end

# Convenience constructors
function HyperbolicSecant(μ::T, σ::T; check_args::Bool = true) where {T <: Real}
    @check_args HyperbolicSecant (σ, σ > zero(σ))
    return HyperbolicSecant{T}(μ, σ)
end
HyperbolicSecant(μ::Real, σ::Real) = HyperbolicSecant(promote(μ, σ)...)
HyperbolicSecant(μ::Integer, σ::Integer) = HyperbolicSecant(float(μ), float(σ))
HyperbolicSecant() = HyperbolicSecant(0.0, 1.0)  # Standard hyperbolic secant
HyperbolicSecant(μ::Real) = HyperbolicSecant(μ, one(μ))  # Unit scale

# Support
Distributions.minimum(::HyperbolicSecant) = -Inf
Distributions.maximum(::HyperbolicSecant) = Inf

# Parameters
Distributions.location(d::HyperbolicSecant) = d.μ
Distributions.scale(d::HyperbolicSecant) = d.σ
Distributions.params(d::HyperbolicSecant) = (d.μ, d.σ)
Distributions.partype(::HyperbolicSecant{T}) where {T} = T

# Basic statistics
Distributions.mean(d::HyperbolicSecant) = d.μ
Distributions.var(d::HyperbolicSecant) = d.σ^2
Distributions.std(d::HyperbolicSecant) = d.σ
Distributions.skewness(::HyperbolicSecant{T}) where {T} = zero(T)
Distributions.kurtosis(::HyperbolicSecant{T}) where {T} = T(5)
Distributions.mode(d::HyperbolicSecant) = d.μ
Distributions.median(d::HyperbolicSecant) = d.μ

# Probability density function
function Distributions.pdf(d::HyperbolicSecant{T}, x::Real) where {T}
    z = (x - d.μ) / d.σ
    return T(1) / (T(2) * d.σ) * sech(T(π) * z / T(2))
end

# Cumulative distribution function
function Distributions.cdf(d::HyperbolicSecant{T}, x::Real) where {T}
    z = (x - d.μ) / d.σ
    return T(2) / T(π) * atan(exp(T(π) * z / T(2)))
end

# Quantile function (inverse CDF)
function Distributions.quantile(d::HyperbolicSecant{T}, p::Real) where {T}
    0 < p < 1 || throw(ArgumentError("p must be in (0,1)"))
    return d.μ + d.σ * T(2) / T(π) * log(tan(T(π) * p / T(2)))
end

# Log-probability density function (for numerical stability)
function Distributions.logpdf(d::HyperbolicSecant{T}, x::Real) where {T}
    z = (x - d.μ) / d.σ
    return -log(T(2) * d.σ) - log(cosh(T(π) * z / T(2)))
end

# Random number generation using quantile method
function Base.rand(rng::Random.AbstractRNG, d::HyperbolicSecant)
    u = rand(rng)
    return quantile(d, u)
end

# Moment generating function (for |t| < π/(2σ))
function mgf(d::HyperbolicSecant{T}, t::Real) where {T}
    abs(t) < T(π) / (T(2) * d.σ) || throw(DomainError(t, "MGF only defined for |t| < π/(2σ)"))
    return exp(d.μ * t) * sec(d.σ * t)
end

# Characteristic function
function cf(d::HyperbolicSecant{T}, t::Real) where {T}
    return exp(Complex{T}(0, 1) * d.μ * t) * sech(d.σ * t)
end