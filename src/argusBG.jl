"""
    StandardArgusBG <: ContinuousUnivariateDistribution

A continuous univariate distribution based on the ARGUS shape defined on the standard interval `[0, 1]`.

# Constructor
```julia
StandardArgusBG(c, p)
```

See also [`ArgusBG(c, p, a, b)`](@ref) for an ARGUS distribution transformed to a custom interval.

# Arguments
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
    ArgusBG(c, p, a, b)

Create an ARGUS distribution on the interval `[a, b]` by linearly transforming
the standard ARGUS distribution.

# Arguments
- `c`: Shape parameter (must be negative)
- `p`: Power parameter (must be ≥ -1)
- `a`: Lower bound of the interval
- `b`: Upper bound of the interval

# Examples
```julia
# ARGUS distribution on [0, 1]
d = ArgusBG(-2.0, 0.5, 0, 1)

# ARGUS distribution on [0, 10]
d = ArgusBG(-1.5, 1.0, 0, 10)
```
"""
function ArgusBG(c::T, p = T(0.5), a = 0.0, b = 1.0) where {T <: Real}
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

Distributions.minimum(d::StandardArgusBG{T}) where {T <: Real} = T(0)
Distributions.maximum(d::StandardArgusBG{T}) where {T <: Real} = T(1)

Distributions.pdf(d::StandardArgusBG, x::Real) =
    (x <= 0 || x >= 1) ? zero(x) : f_argus_std(x, d.c, d.p) / d.integral

Distributions.cdf(d::StandardArgusBG, x::Real) =
    x <= 0 ? zero(x) : x >= 1 ? one(x) : (F_argus_std(x, d.c, d.p) - F_argus_std(0, d.c, d.p)) / d.integral

function Distributions.quantile(d::StandardArgusBG{T}, q::Real) where {T <: Real}
    # Special cases for boundaries
    q <= zero(T) && return zero(T)
    q >= one(T) && return one(T)

    χ = -d.c  # Convert to positive scale

    # Compute regularized incomplete gamma P(s, χ)
    P_chi = gamma_inc(d.p + 1, χ)[1]  # [1] = regularized lower gamma P(a,x)

    # Calculate target value for inverse gamma
    P_target = (1 - q) * P_chi

    # Compute inverse incomplete gamma
    z = gamma_inc_inv(d.p + 1, P_target, 1 - P_target)

    # Solve for x and ensure numerical stability
    x_sq = max(1 - z / χ, zero(T))  # Prevent negative values from floating-point errors
    return sqrt(x_sq)
end

#### Parameters
Distributions.params(d::StandardArgusBG) = (d.c, d.p)
Distributions.partype(::StandardArgusBG{T}) where {T <: Real} = T

Base.rand(rng::AbstractRNG, d::StandardArgusBG) = quantile(d, rand(rng))
Base.rand(rng::AbstractRNG, d::StandardArgusBG, n::Int) =
    quantile.(Ref(d), rand(rng, n))
