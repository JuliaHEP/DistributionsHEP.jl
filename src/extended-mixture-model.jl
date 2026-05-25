"""
    ExtendedMixtureModel(components, yields)

Mixture-like distribution whose weights are expected event yields instead of
normalized probabilities. For components `f_k` and yields `y_k`, the density is
`sum(y_k * f_k(x) for k)`. This is the density term used in extended
likelihood fits where the fitted parameters are component yields.

Yields must be non-negative and have the same length as `components`.
Use `MixtureModel(model)` to convert to the corresponding normalized mixture.
"""
struct ExtendedMixtureModel{
    VF<:VariateForm,
    VS<:ValueSupport,
    C<:Distribution,
    T<:Real,
} <: Distribution{VF,VS}
    components::Vector{C}
    yields::Vector{T}

    function ExtendedMixtureModel{VF,VS,C,T}(components::Vector{C}, yields::Vector{T}) where {VF,VS,C,T}
        length(components) == length(yields) ||
            throw(ArgumentError("ExtendedMixtureModel: $(length(components)) components vs $(length(yields)) yields"))
        return new{VF,VS,C,T}(components, yields)
    end
end

function ExtendedMixtureModel(components::AbstractVector{C}, yields::AbstractVector{<:Real}) where {C<:Distribution}
    length(components) == length(yields) || throw(ArgumentError("ExtendedMixtureModel: length mismatch"))
    any(y -> y < zero(y), yields) && throw(ArgumentError("ExtendedMixtureModel: yields must be non-negative"))

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

factor(d::Distributions.Product, k::Int) = d.v[k]

"""
    marginalize(model, k)

Return the exact one-dimensional marginal distribution for dimension `k` when
the model is built from independent product components and mixtures.
"""
function marginalize(d::Distributions.Product, k::Int)
    return factor(d, k)
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

function (::Type{MixtureModel})(d::ExtendedMixtureModel)
    total = total_yield(d)
    total > zero(total) ||
        throw(ArgumentError("MixtureModel(::ExtendedMixtureModel): total yield must be positive"))
    return MixtureModel(components(d), yields(d) ./ total)
end

function Distributions.pdf(d::ExtendedMixtureModel{Univariate}, x::Real)
    return sum(y * pdf(component(d, i), x) for (i, y) in enumerate(yields(d)) if !iszero(y))
end

function Distributions.pdf(d::ExtendedMixtureModel{Multivariate}, x::AbstractVector{<:Real})
    return sum(y * pdf(component(d, i), x) for (i, y) in enumerate(yields(d)) if !iszero(y))
end

function Distributions.logpdf(d::ExtendedMixtureModel{Univariate}, x::Real)
    density = pdf(d, x)
    return density <= zero(density) ? -Inf : log(density)
end

function Distributions.logpdf(d::ExtendedMixtureModel{Multivariate}, x::AbstractVector{<:Real})
    density = pdf(d, x)
    return density <= zero(density) ? -Inf : log(density)
end

"""
    extended_negative_log_likelihood(model, data)

Evaluate `-sum(logpdf(model, x) for x in data) + total_yield(model)`.
For multivariate models, an `AbstractMatrix` is interpreted column-wise.
"""
function extended_negative_log_likelihood(d::ExtendedMixtureModel, data)
    nll = zero(float(total_yield(d)))
    for x in data
        lp = logpdf(d, x)
        isfinite(lp) || return oftype(nll, Inf)
        nll -= lp
    end
    return nll + total_yield(d)
end

function extended_negative_log_likelihood(d::ExtendedMixtureModel{Multivariate}, data::AbstractMatrix)
    return extended_negative_log_likelihood(d, eachcol(data))
end

Base.length(d::ExtendedMixtureModel{Multivariate}) = length(d.components[1])
