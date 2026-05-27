"""
    ExtendedMixtureModel(components, yields)

Mixture-like distribution whose weights are expected event yields instead of
normalized probabilities. For components `f_k` and yields `y_k`, the density is
`sum(y_k * f_k(x) for k)`. This is the density term used in extended
likelihood fits where the fitted parameters are component yields.

Yields must have the same non-empty length as `components`. Use `model(x)` to
evaluate the extended density and `MixtureModel(model)` to convert to the
corresponding normalized mixture.
"""
struct ExtendedMixtureModel{
    VF<:VariateForm,
    VS<:ValueSupport,
    C<:Distribution,
    T<:Real,
}
    components::Vector{C}
    yields::Vector{T}

    function ExtendedMixtureModel{VF,VS,C,T}(components::Vector{C}, yields::Vector{T}) where {VF,VS,C,T}
        length(components) == length(yields) ||
            throw(ArgumentError("ExtendedMixtureModel: $(length(components)) components vs $(length(yields)) yields"))
        !isempty(components) ||
            throw(ArgumentError("ExtendedMixtureModel: components and yields must be non-empty"))
        return new{VF,VS,C,T}(components, yields)
    end
end

function ExtendedMixtureModel(components::AbstractVector{C}, yields::AbstractVector{<:Real}) where {C<:Distribution}
    VF = Distributions.variate_form(C)
    VS = Distributions.value_support(C)
    T = promote_type(Float64, eltype(yields))
    return ExtendedMixtureModel{VF,VS,C,T}(collect(components), T.(yields))
end

Distributions.ncomponents(d::ExtendedMixtureModel) = length(d.components)
Distributions.components(d::ExtendedMixtureModel) = d.components
Distributions.component(d::ExtendedMixtureModel, k::Int) = d.components[k]
Distributions.component_type(d::ExtendedMixtureModel{VF,VS,C}) where {VF,VS,C} = C

"""Return the expected event yields of an `ExtendedMixtureModel`."""
yields(d::ExtendedMixtureModel) = d.yields

"""Return `sum(yields(d))` for an `ExtendedMixtureModel`."""
total_yield(d::ExtendedMixtureModel) = sum(yields(d))

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

function (::Type{MixtureModel})(d::ExtendedMixtureModel)
    total = total_yield(d)
    total > zero(total) ||
        throw(ArgumentError("MixtureModel(::ExtendedMixtureModel): total yield must be positive"))
    return MixtureModel(components(d), yields(d) ./ total)
end

function (d::ExtendedMixtureModel{Univariate})(x::Real)
    return sum(y * pdf(component(d, i), x) for (i, y) in enumerate(yields(d)) if !iszero(y))
end

function (d::ExtendedMixtureModel{Multivariate})(x::AbstractVector{<:Real})
    return sum(y * pdf(component(d, i), x) for (i, y) in enumerate(yields(d)) if !iszero(y))
end

"""
    extended_negative_log_likelihood(model, data)

Evaluate `-sum(log(model(x)) for x in data) + total_yield(model)`.
For multivariate models, an `AbstractMatrix` is interpreted column-wise.
"""
function extended_negative_log_likelihood(d::ExtendedMixtureModel, data)
    nll = zero(float(total_yield(d)))
    for x in data
        density = d(x)
        density > zero(density) || return oftype(nll, Inf)
        nll -= log(density)
    end
    return nll + total_yield(d)
end

function extended_negative_log_likelihood(d::ExtendedMixtureModel{Multivariate}, data::AbstractMatrix)
    return extended_negative_log_likelihood(d, eachcol(data))
end

Base.length(d::ExtendedMixtureModel{Multivariate}) = length(d.components[1])
