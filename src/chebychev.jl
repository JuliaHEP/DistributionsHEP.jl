using SpecialFunctions
using Polynomials

struct Chebyshev <: ContinuousUnivariateDistribution
    polynomial::ChebyshevT{Float64,:x}
    integral::Float64
    a::Float64
    b::Float64
    function Chebyshev(coeffs, a = -1.0, b = 1.0)
        polynomial = ChebyshevT(coeffs)
        integral = integrate(polynomial)
        new(polynomial, (integral(1.0) - integral(-1.0)), a, b)
    end
end

Distributions.maximum(d::Chebyshev) = d.b
Distributions.minimum(d::Chebyshev) = d.a

function Distributions.pdf(d::Chebyshev, x::Real)
    x′ = (2x - d.a - d.b) / (d.b - d.a)
    d.polynomial(x′) / (d.integral * (d.b - d.a) / 2)
end

function Distributions.pdf(d::Chebyshev, x::AbstractArray{<:Real})
    x′ = (2x .- d.a .- d.b) ./ (d.b - d.a)
    d.polynomial.(x′) / (d.integral * (d.b - d.a) / 2)
end

function Distributions.cdf(d::Chebyshev, x::Real)
    x′ = (2x - d.a - d.b) / (d.b - d.a)
    integrate(d.polynomial, -1.0, x′) / d.integral
end

function Base.rand(rng::AbstractRNG, d::Chebyshev)
    max = sum(abs, d.polynomial)   # estimate the maximum of the polynomial
    x = rand(rng, Uniform(-1.0, 1.0))
    while rand(rng) > d.polynomial(x) / max
        x = rand(rng, Uniform(-1.0, 1.0))
    end
    return (x * (d.b - d.a) + d.a + d.b) / 2
end

