struct StandardChebyshev <: ContinuousUnivariateDistribution
    polynomial::ChebyshevT{Float64, :x}
    integral::Float64
    function StandardChebyshev(coeffs)
        polynomial = ChebyshevT(coeffs)
        integral = integrate(polynomial)
        new(polynomial, (integral(1.0) - integral(-1.0)))
    end
end

# Main constructor that handles transformations
function Chebyshev(coeffs, a::T, b::T) where {T <: Real}
    return transformed_chebychev(coeffs, a, b)
end

# Convenience constructor for standard interval
Chebyshev(coeffs) = StandardChebyshev(coeffs)

transformed_chebychev(coeffs, a::T, b::T) where {T <: Real} = StandardChebyshev(coeffs) * (b - a) / 2 + (a + b) / 2

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

