function _union_minimum(components::AbstractVector{<:Distribution})
    first_component = first(components)
    if Distributions.variate_form(typeof(first_component)) == Univariate
        return minimum(minimum(c) for c in components)
    end
    return mapreduce(minimum, (x, y) -> min.(x, y), components)
end

function _union_maximum(components::AbstractVector{<:Distribution})
    first_component = first(components)
    if Distributions.variate_form(typeof(first_component)) == Univariate
        return maximum(maximum(c) for c in components)
    end
    return mapreduce(maximum, (x, y) -> max.(x, y), components)
end

function _union_support(components::AbstractVector{<:Distribution})
    return Distributions.RealInterval.(_union_minimum(components), _union_maximum(components))
end

Distributions.support(d::Distributions.Product) =
    Distributions.RealInterval.(minimum(d), maximum(d))

Distributions.support(d::MvNormal) =
    Distributions.RealInterval.(minimum(d), maximum(d))

Distributions.minimum(d::Distributions.AbstractMixtureModel) = _union_minimum(components(d))
Distributions.maximum(d::Distributions.AbstractMixtureModel) = _union_maximum(components(d))
Distributions.support(d::Distributions.AbstractMixtureModel) = _union_support(components(d))

Distributions.minimum(d::MixtureModel) = _union_minimum(components(d))
Distributions.maximum(d::MixtureModel) = _union_maximum(components(d))

Distributions.minimum(d::ExtendedMixtureModel) = _union_minimum(components(d))
Distributions.maximum(d::ExtendedMixtureModel) = _union_maximum(components(d))
Distributions.support(d::ExtendedMixtureModel) = _union_support(components(d))

factor(d::Distributions.Product, k::Int) = d.v[k]

"""
    marginalize(model, k)

Return the exact one-dimensional marginal distribution for coordinate `k`.

For [`Product`](@ref), dimension `k` is an independent factor. For
[`MvNormal`](@ref), the coordinate marginal is
`Normal(μ[k], sqrt(Σ[k,k]))`, including when `Σ` has off-diagonal terms.
Mixtures and [`ExtendedMixtureModel`](@ref) are handled component-wise.
"""
function marginalize(d::Distributions.Product, k::Int)
    return factor(d, k)
end

function marginalize(d::MvNormal, k::Int)
    return Normal(d.μ[k], sqrt(d.Σ[k, k]))
end

function marginalize(d::Distributions.AbstractMixtureModel, k::Int)
    return MixtureModel(
        [marginalize(c, k) for c in components(d)],
        probs(d),
    )
end

function marginalize(d::ExtendedMixtureModel, k::Int)
    return ExtendedMixtureModel(
        [marginalize(c, k) for c in components(d)],
        yields(d),
    )
end
