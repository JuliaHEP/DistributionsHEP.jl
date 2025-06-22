using SpecialFunctions
using Polynomials

struct Chebyshev <: ContinuousUnivariateDistribution
    polynomial::ChebyshevT{Float64, :x}
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

"""
Compute n-th central moment of the distribution using numerical integration of the polynomial.
It's implemented for [-1,1] range. See 
https://github.com/JuliaHEP/DistributionsHEP.jl/issues/11
"""
function pure_moment(d::Chebyshev, n::Int; μ = mean(d))
    pol_x_minus_mu = ChebyshevT([-μ, 1.0])
    integrand = d.polynomial * pol_x_minus_mu^n
    _int = integrate(integrand, -1.0, 1.0) / d.integral
    return _int
end

Distributions.mean(model::Chebyshev) = pure_moment(model, 1, 0.0)

Distributions.var(model::Chebyshev) = pure_moment(model, 2)

function Distributions.skewness(model::Chebyshev)
    σ = std(model)
    μ3 = pure_moment(model, 3)
    return μ3 / σ^3
end

function Distributions.kurtosis(model::Chebyshev)
    σ = std(model)
    μ4 = pure_moment(model, 4)
    return μ4 / σ^4 - 3
end
