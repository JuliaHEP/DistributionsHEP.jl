# Tail parameters for Exponential tail distribution
#
struct ExponentialTail{T<:Real}
    G_x0::T       # G(x0): Core PDF at transition point x0
    L_x0::T       # L(x0): Logarithmic derivative of core PDF at transition point x0
    x0::T         # x0: Absolute transition point
end

function _norm_const(tail::ExponentialTail{T}) where {T<:Real}
    (; G_x0, L_x0) = tail
    return G_x0 / L_x0
end

function _value(tail::ExponentialTail{T}, offset::T) where {T<:Real}
    (; G_x0, L_x0) = tail
    return G_x0 * exp(L_x0 * offset)
end

"""
    _integral(t::ExponentialTail, a)

Compute the integral of the ExponentialTail function.

For left tail (L_x0 > 0): integral from -∞ to a
For right tail (L_x0 < 0): returns -integral from [a, +∞]

The integral is computed using the antiderivative of the tail function:
∫[-Inf*sign(L) to a] G_x0 / L_x0 * exp(L_x0 * (x - x0)) dx

For L_x0 < 0, the result is the negative of the integral from a to +∞.

Returns the integral value.
"""
function _integral(t::ExponentialTail{T}, a::Real) where {T<:Real}
    (; G_x0, L_x0, x0) = t
    # Compute offset from transition point
    a_T = T(a)
    offset_a = a_T - x0
    const_tail = _norm_const(t)
    return const_tail * exp(L_x0 * offset_a)
end

"""
    _integral_inversion(t::ExponentialTail, integral)

Find `a` such that the integral relationship holds.

For left tail (L_x0 > 0): finds `a` such that integral from -∞ to `a` equals the given `integral` value.
For right tail (L_x0 < 0): finds `a` such that `_integral(t, a) = integral` (where integral should be negative).

Returns the value of `a` (in absolute coordinates).
"""
function _integral_inversion(t::ExponentialTail{T}, integral::Real) where {T<:Real}
    (; L_x0, x0) = t
    const_tail = _norm_const(t)
    # 
    integral_T = T(integral)
    ratio = integral_T / const_tail
    offset_a = log(ratio) / L_x0
    # Convert back to absolute coordinate
    a = x0 + offset_a
    return a
end
