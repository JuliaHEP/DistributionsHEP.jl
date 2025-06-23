"""
    StandardChebyshev <: ContinuousUnivariateDistribution

A continuous univariate distribution based on Chebyshev polynomials of the first kind
defined on the standard interval `[-1, 1]`.

# Constructor
```julia
StandardChebyshev(coeffs)
```
Normalization integral is computed analytically using `Polynomials.jl` package upon construction.

See also [`Chebyshev(coefs, a, b)`](@ref) for a Chebyshev distribution transformed to a custom interval.

# Arguments
- `coeffs`: Vector of coefficients for the Chebyshev polynomial

# Examples
```julia
# Linear polynomial
d = StandardChebyshev([1.0, 1.0])

# Quadratic polynomial
d = StandardChebyshev([2.0, 0.0, 1.0])
```
"""
struct StandardChebyshev <: ContinuousUnivariateDistribution
    polynomial::ChebyshevT{Float64, :x}
    integral::Float64
    function StandardChebyshev(coeffs)
        polynomial = ChebyshevT(coeffs)
        integral = integrate(polynomial)
        new(polynomial, (integral(1.0) - integral(-1.0)))
    end
end

"""
    Chebyshev(coeffs, a, b)

Create a Chebyshev distribution on the interval `[a, b]` by linearly transforming
the standard Chebyshev distribution.

# Arguments
- `coeffs`: Vector of coefficients for the Chebyshev polynomial
- `a`: Lower bound of the interval
- `b`: Upper bound of the interval

# Examples
```julia
# Linear polynomial on [0, 10]
d = Chebyshev([1.0, 1.0], 0, 10)

# Quadratic polynomial on [-5, 5]
d = Chebyshev([4.0, 0.0, 1.0], -5, 5)
```
"""
function Chebyshev(coeffs, a::T = -1.0, b::T = 1.0) where {T <: Real}
    return StandardChebyshev(coeffs) * (b - a) / 2 + (a + b) / 2
end

Distributions.minimum(d::StandardChebyshev) = -1.0
Distributions.maximum(d::StandardChebyshev) = 1.0

Distributions.pdf(d::StandardChebyshev, x::Real) =
    d.polynomial(x) / d.integral

Distributions.cdf(d::StandardChebyshev, x::Real) =
    integrate(d.polynomial, -1.0, x) / d.integral

function Base.rand(rng::AbstractRNG, d::StandardChebyshev)
    max = sum(abs, d.polynomial)   # estimate the maximum of the polynomial
    x = rand(rng, Uniform(-1.0, 1.0))
    while rand(rng) > d.polynomial(x) / max
        x = rand(rng, Uniform(-1.0, 1.0))
    end
    return x
end

