struct Chebyshev <: ContinuousUnivariateDistribution
    polynomial::ChebyshevT{Float64, :x}
    integral::Float64
    function Chebyshev(coeffs)
        polynomial = ChebyshevT(coeffs)
        integral = integrate(polynomial)
        new(polynomial, (integral(1.0) - integral(-1.0)))
    end
end

Chebyshev(coeffs, a::T, b::T) where {T <: Real} = Chebyshev(coeffs) * (b - a) / 2 + (a + b) / 2

Distributions.minimum(d::Chebyshev) = -1.0
Distributions.maximum(d::Chebyshev) = 1.0

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

