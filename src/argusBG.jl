"""
    ArgusBG(m₀, c, p = 0.5, a = 0., b = m₀)

    Distribution describing the ARGUS background shape

```math
  \\mathrm{Argus}(m; m_0, c, p) = \\mathcal{N} \\cdot m \\cdot \\left[ 1 - \\left( \\frac{m}{m_0} \\right)^2 \\right]^p
  \\cdot \\exp\\left[ c \\cdot \\left(1 - \\left(\\frac{m}{m_0}\\right)^2 \\right) \\right]
```
where `m` is the mass, `m₀` is the maximum mass, `c` is a shape parameter, and `p` is a power parameter. The distribution is normalized to 1 over the range `[a, b]`.


```julia
ArgusBG(m₀, c)           # ARGUS background distribution with p = 0.5 and in the range [0, m₀]
ArgusBG(m₀, c, p)        # ARGUS background distribution with p and in the range [0, m₀]
ArgusBG(m₀, c, p, a, b)  # ARGUS background distribution with p and in the range [a, b]

params(d)                # Get the parameters, i.e. (m₀, c, p)
shape(d)                 # Get the shape parameter, i.e. c
scale(d)                 # Get the scale parameter, i.e. m₀
```

External links

* [ARGUS distribution on Wikipedia](https://en.wikipedia.org/wiki/ARGUS_distribution)
"""
struct ArgusBG{T<:Real} <: ContinuousUnivariateDistribution
    m₀::T
    c::T
    p::T
    a::T
    b::T
    integral::T
    ArgusBG{T}(m₀::T, c::T, p::T, a::T, b::T, integral::T) where {T} =
        new{T}(m₀, c, p, a, b, integral)
end

function ArgusBG(
    m₀::T,
    c::T,
    p::T = T(0.5),
    a::T = T(0),
    b::T = m₀;
    check_args::Bool = true,
) where {T<:Real}
    @check_args ArgusBG (m₀, m₀ > zero(m₀)) (c, c > zero(c)) (p, p >= -1) (a, a < m₀) (
        b,
        b > a,
    )
    integral = F_argus(b, m₀, c, p) - F_argus(a, m₀, c, p)
    return ArgusBG{T}(m₀, c, p, a, b, integral)
end

function f_argus(m, m₀, c, p)
    m >= m₀ && return 0.0
    m * (1 - (m / m₀)^2)^p * exp(c * (1 - (m / m₀)^2))
end

function F_argus(m, m₀, c, p)
    m >= m₀ && (m = m₀ - 1e-10)
    f = (m / m₀)^2 - 1
    w = -(c * f + 0im)^(-p) * m₀^2 * (-f + 0im)^p * gamma(1 + p, c * f + 0im) / 2c
    return isreal(w) ? real(w) : zero(m)
end

#### Parameters

scale(d::ArgusBG) = d.m₀
shape(d::ArgusBG) = d.c

params(d::ArgusBG) = (d.m₀, d.c, d.p)
partype(::ArgusBG{T}) where {T<:Real} = T

#### Statistics

mean(d::ArgusBG) = d.θ
median(d::ArgusBG) = logtwo * d.θ
mode(::ArgusBG{T}) where {T<:Real} = zero(T)

var(d::ArgusBG) = d.θ^2
skewness(::ArgusBG{T}) where {T} = T(2)
kurtosis(::ArgusBG{T}) where {T} = T(6)

entropy(d::ArgusBG{T}) where {T} = one(T) + log(d.θ)




function kldivergence(p::Exponential, q::Exponential)
    λq_over_λp = scale(q) / scale(p)
    return -logmxp1(λq_over_λp)
end

#### Evaluation
function Distributions.pdf(d::ArgusBG, m::Real)
    (; m₀, c, p) = d
    f_argus(m, m₀, c, p) / d.integral
end


function Distributions.cdf(d::ArgusBG, m::Real)
    (; m₀, c, p) = d
    (F_argus(m, m₀, c, p) - F_argus(d.a, m₀, c, p)) / d.integral
end

function Base.rand(rng::AbstractRNG, d::ArgusBG, n::Int64 = 1)
    (; m₀, c, p, a, b) = d
    max = maximum(_argus.(range(a, b, 100), m₀, c, p)) # estimate the maximum
    r = Float64[]
    for i = 1:n
        m = rand(rng, Uniform(a, b))
        while rand(rng) > _argus(m, m₀, c, p) / max
            m = rand(rng, Uniform(a, b))
        end
        push!(r, m)
    end
    return n == 1 ? r[1] : r
end
