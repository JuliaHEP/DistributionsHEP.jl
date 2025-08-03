# Common parameter validation
function _check_hyperbolic_secant_params(σ::T) where {T <: Real}
    σ > zero(T) || error("σ (scale) must be positive.")
end

"""
    HyperbolicSecant{T<:Real} <: ContinuousUnivariateDistribution

The Hyperbolic secant distribution is a continuous probability distribution whose probability 
density function and characteristic function are hyperbolic functions.

The probability density function is defined as:
````math
    f(x; μ, σ) = \\frac{1}{2σ} \\operatorname{sech}\\left(\\frac{π(x-μ)}{2σ}\\right)
````
where sech(u) = 2/(e^u + e^(-u)) = 1/cosh(u).

The cumulative distribution function is:
````math
    F(x; μ, σ) = \\frac{2}{π} \\arctan\\left(\\exp\\left(\\frac{π(x-μ)}{2σ}\\right)\\right)
````

# Arguments
- `μ`: The location parameter (mean). Default is 0.0.
- `σ`: The scale parameter (standard deviation). Must be positive. Default is 1.0.

# Properties
- Mean: μ
- Mode: μ  
- Variance: σ²
- Standard deviation: σ
- Skewness: 0 (symmetric distribution)
- Excess kurtosis: 2
- Support: (-∞, ∞)

# Example
```julia
using DistributionsHEP
using Plots

d = HyperbolicSecant(0.0, 1.0)  # μ, σ
plot(-4, 4, x->pdf(d, x))
```
"""
struct HyperbolicSecant{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    function HyperbolicSecant(μ::T, σ::T) where {T <: Real}
        _check_hyperbolic_secant_params(σ)
        new{T}(μ, σ)
    end
end

# Convenience constructors
HyperbolicSecant(μ::Real, σ::Real) = HyperbolicSecant(promote(μ, σ)...)
HyperbolicSecant(μ::Real) = HyperbolicSecant(μ, one(μ))
HyperbolicSecant() = HyperbolicSecant(0.0, 1.0)

"""
    pdf(d::HyperbolicSecant, x::Real)

Compute the probability density function (PDF) of the Hyperbolic secant distribution `d` at point `x`.

The PDF is: f(x; μ, σ) = (1/(2σ)) * sech(π(x-μ)/(2σ))
where sech(u) = 1/cosh(u).
"""
function Distributions.pdf(d::HyperbolicSecant{T}, x::Real) where {T <: Real}
    z = T(π) * (x - d.μ) / (2 * d.σ)
    return one(T) / (2 * d.σ * cosh(z))
end

"""
    cdf(d::HyperbolicSecant, x::Real)

Compute the cumulative distribution function (CDF) of the Hyperbolic secant distribution `d` at point `x`.

The CDF is: F(x; μ, σ) = (2/π) * arctan(exp(π(x-μ)/(2σ)))
"""
function Distributions.cdf(d::HyperbolicSecant{T}, x::Real) where {T <: Real}
    z = T(π) * (x - d.μ) / (2 * d.σ)
    return 2 / T(π) * atan(exp(z))
end

"""
    quantile(d::HyperbolicSecant, p::Real)

Compute the quantile (inverse CDF) of the Hyperbolic secant distribution `d` for a given probability `p`.

The quantile function is: Q(p; μ, σ) = μ + (2σ/π) * log(tan(πp/2))
"""
function Distributions.quantile(d::HyperbolicSecant{T}, p::Real) where {T <: Real}
    if p < zero(T) || p > one(T)
        throw(DomainError(p, "Probability p must be in [0,1]."))
    end
    p == zero(T) && return T(-Inf)
    p == one(T) && return T(Inf)
    
    # For p close to 0 or 1, tan(πp/2) can become extreme
    # Use more stable computation near the boundaries
    if p ≈ 0.5
        return d.μ
    end
    
    # Q(p) = μ + (2σ/π) * log(tan(πp/2))
    # This comes from inverting F(x) = (2/π) * arctan(exp(π(x-μ)/(2σ)))
    z = tan(T(π) * p / 2)
    return d.μ + 2 * d.σ / T(π) * log(z)
end

"""
    mean(d::HyperbolicSecant)

Compute the mean of the Hyperbolic secant distribution `d`.
"""
Distributions.mean(d::HyperbolicSecant) = d.μ

"""
    var(d::HyperbolicSecant)

Compute the variance of the Hyperbolic secant distribution `d`.
"""
Distributions.var(d::HyperbolicSecant) = d.σ^2

"""
    std(d::HyperbolicSecant)

Compute the standard deviation of the Hyperbolic secant distribution `d`.
"""
Distributions.std(d::HyperbolicSecant) = d.σ

"""
    skewness(d::HyperbolicSecant)

Compute the skewness of the Hyperbolic secant distribution `d`.
The distribution is symmetric, so skewness is always 0.
"""
Distributions.skewness(d::HyperbolicSecant{T}) where {T <: Real} = zero(T)

"""
    kurtosis(d::HyperbolicSecant)

Compute the excess kurtosis of the Hyperbolic secant distribution `d`.
The excess kurtosis is always 2.
"""
Distributions.kurtosis(d::HyperbolicSecant{T}) where {T <: Real} = 2 * one(T)

Distributions.maximum(d::HyperbolicSecant{T}) where {T <: Real} = T(Inf)
Distributions.minimum(d::HyperbolicSecant{T}) where {T <: Real} = T(-Inf)