struct Chebyshev <: ContinuousUnivariateDistribution
    polynomial::ChebyshevT{Float64, :x}
    integral::Float64
    a::Float64
    b::Float64
    function Chebyshev(coeffs)
        polynomial = ChebyshevT(coeffs)
        integral = integrate(polynomial)
        new(polynomial, (integral(1.0) - integral(-1.0)), -1, 1)
    end
end

Chebyshev(coeffs, a::T, b::T) where {T <: Real} = Chebyshev(coeffs) * (b - a) / 2 + (a + b) / 2

Distributions.maximum(d::Chebyshev) = d.b
Distributions.minimum(d::Chebyshev) = d.a

function Distributions.pdf(d::Chebyshev, x::Real)
    d.polynomial(x) / d.integral
end

function Distributions.pdf(d::Chebyshev, x::AbstractArray{<:Real})
    d.polynomial.(x) ./ d.integral
end

function Distributions.cdf(d::Chebyshev, x::Real)
    integrate(d.polynomial, -1.0, x) / d.integral
end

function Base.rand(rng::AbstractRNG, d::Chebyshev)
    max = sum(abs, d.polynomial)   # estimate the maximum of the polynomial
    x = rand(rng, Uniform(-1.0, 1.0))
    while rand(rng) > d.polynomial(x) / max
        x = rand(rng, Uniform(-1.0, 1.0))
    end
    return x
end

