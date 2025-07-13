"""
    StandardArgusBG <: ContinuousUnivariateDistribution

A continuous univariate distribution based on the ARGUS shape defined on the standard interval `[0, 1]`.

# Constructor
```julia
StandardArgusBG(c, p)
```
where:
- `c`: Shape parameter (must be negative)
- `p`: Power parameter (must be ≥ -1)

# Examples
```julia
# Standard ARGUS distribution
d = StandardArgusBG(-2.0, 0.5)

# Different power parameter
d = StandardArgusBG(-1.5, 1.0)
```
"""
struct StandardArgusBG{T <: Real} <: ContinuousUnivariateDistribution
    c::T
    p::T
    integral::T
    function StandardArgusBG(c::T, p::T = T(0.5)) where {T <: Real}
        integral = F_argus_std(T(1), c, p) - F_argus_std(T(0), c, p)
        new{T}(c, p, integral)
    end
end

"""
    ArgusBG(c, p = 0.5, a = 0.0, b = 1.0)

Distribution describing the ARGUS shape ([wikipedia](https://en.wikipedia.org/wiki/ARGUS_distribution)) on the interval `[a, b]`.

# Arguments
- `c`: A shape parameter (must be negative)
- `p`: A power parameter (must be ≥ -1)
- `a`: The lower limit of the distribution
- `b`: The upper limit of the distribution

The distribution is normalized to 1 over the range `[a, b]`.

# Example
```julia
ArgusBG(c)           # ARGUS distribution with p = 0.5 and in the range [0, 1]
ArgusBG(c, p)        # ARGUS distribution with p and in the range [0, 1]
ArgusBG(c, p, a, b)  # ARGUS distribution with p and in the range [a, b]

params(d)            # Get the parameters, i.e. (c, p)
shape(d)             # Get the shape parameter, i.e. c
```
"""
function ArgusBG(c::T, p = T(0.5), a = 0, b = 1) where {T <: Real}
    return StandardArgusBG(c, p) * (b - a) + a
end

# Standardized ARGUS PDF on [0,1]
function f_argus_std(x, c, p)
    x * (1 - x^2)^p * exp(c * (1 - x^2))
end

function F_argus_std(x, c, p)
    w = gamma(p + 1, -c * (1 - x^2)) / (2 * (-c)^(p + 1))
    return w
end

function Distributions.quantile(d::StandardArgusBG{T}, q::Real) where {T <: Real}
    # Special cases for boundaries
    q <= 0 && return 0.0
    q >= 1 && return 1.0
    s = d.p + 1
    χ = -d.c  # Convert to positive scale
    # Compute regularized incomplete gamma P(s, χ)
    P_chi = gamma_inc(d.p + 1, χ)[1]  # [1] = regularized lower gamma P(a,x)
    # Calculate target value for inverse gamma
    P_target = (1 - q) * P_chi
    # Compute inverse incomplete gamma
    z = gamma_inc_inv(d.p + 1, P_target, 1 - P_target)
    # Solve for x and ensure numerical stability
    x_sq = max(1 - z / χ, T(0))  # Prevent negative values from floating-point errors
    return sqrt(x_sq)
end

Distributions.maximum(d::StandardArgusBG{T}) where {T <: Real} = T(1)
Distributions.minimum(d::StandardArgusBG{T}) where {T <: Real} = T(0)

#### Parameters
Distributions.shape(d::StandardArgusBG) = d.c
Distributions.params(d::StandardArgusBG) = (d.c, d.p)
Distributions.partype(::StandardArgusBG{T}) where {T <: Real} = T

# Forwarding methods for LocationScale wrapped StandardArgusBG
Distributions.shape(d::LocationScale{<:Any, <:Any, <:StandardArgusBG}) = shape(d.ρ)
# Override params for ArgusBG distributions to return shape parameters
function Distributions.params(d::LocationScale{T1, Continuous, StandardArgusBG{T2}}) where {T1 <: Real, T2 <: Real}
    return params(d.ρ)
end

function Distributions.pdf(d::StandardArgusBG, x::Real)
    (x <= 0 || x >= 1) && return 0.0
    f_argus_std(x, d.c, d.p) / d.integral
end

function Distributions.cdf(d::StandardArgusBG, x::Real)
    (x <= 0) && return 0.0
    (x >= 1) && return 1.0
    (F_argus_std(x, d.c, d.p) - F_argus_std(0, d.c, d.p)) / d.integral
end

Base.rand(rng::AbstractRNG, d::StandardArgusBG) = quantile(d, rand(rng))
Base.rand(rng::AbstractRNG, d::StandardArgusBG, n::Int) =
    quantile.(Ref(d), rand(rng, n))
