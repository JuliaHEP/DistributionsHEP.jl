# Tail parameters for Crystal Ball distribution
# used in CrystalBall, DoubleCrystalBall,
# also in DoubleBifurcatedCrystalBall
# 
struct CrystalBallTail{T<:Real}
    G_x0::T       # G(x0): Core PDF at transition point x0
    N::T          # Effective power
    L_x0::T       # L(x0): Logarithmic derivative of core PDF at transition point x0
    x0::T         # x0: Absolute transition point
end

function _norm_const(tail::CrystalBallTail{T}) where {T<:Real}
    (; G_x0, N, L_x0) = tail
    return G_x0 / L_x0 * N / (N - 1)
end

function _value(tail::CrystalBallTail{T}, offset::T) where {T<:Real}
    (; G_x0, N, L_x0) = tail
    return G_x0 * (N / (N - L_x0 * offset))^N
end

"""
    _integral(t::CrystalBallTail, a)

Compute the integral of the CrystalBallTail function.

For left tail (L_x0 > 0): integral from -∞ to a
For right tail (L_x0 < 0): returns -integral from [a, +∞]

The integral is computed using the antiderivative of the tail function:
∫[-Inf*sign(L) to a] G_x0 * (N / (N - L_x0 * (x - x0)))^N dx

For L_x0 < 0, the result is the negative of the integral from a to +∞.

Returns the integral value.
"""
function _integral(t::CrystalBallTail{T}, a::T) where {T<:Real}
    (; G_x0, N, L_x0, x0) = t
    # Compute offset from transition point
    offset_a = a - x0
    const_tail = _norm_const(t)
    return const_tail * ((N - L_x0 * offset_a) / N)^(one(T) - N)
end

"""
    _integral_inversion(t::CrystalBallTail, integral)

Find `a` such that the integral relationship holds.

For left tail (L_x0 > 0): finds `a` such that integral from -∞ to `a` equals the given `integral` value.
For right tail (L_x0 < 0): finds `a` such that `_integral(t, a) = integral` (where integral should be negative).

Returns the value of `a` (in absolute coordinates).
"""
function _integral_inversion(t::CrystalBallTail{T}, integral::T) where {T<:Real}
    (; N, L_x0, x0) = t
    const_tail = _norm_const(t)
    # 
    ratio = integral / const_tail
    ratio_power = ratio^(one(T) / (one(T) - N))
    offset_a = (N / L_x0) * (one(T) - ratio_power)
    # Convert back to absolute coordinate
    a = x0 + offset_a
    return a
end
