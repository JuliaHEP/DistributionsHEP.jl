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

function _tail_norm_const(tail::CrystalBallTail{T}) where {T<:Real}
    (; G_x0, N, L_x0) = tail
    return G_x0 / L_x0 * N / (N - 1)
end

function _tail_function_value(tail::CrystalBallTail{T}, offset::T) where {T<:Real}
    (; G_x0, N, L_x0) = tail
    return G_x0 * (N / (N - L_x0 * offset))^N
end

"""
    _integral(t::CrystalBallTail, a)

Compute the integral of the CrystalBallTail function from `-Inf * sign(L_x0)` to `a` (in absolute coordinates).

For left tail (L_x0 > 0): integral from -∞ to a
For right tail (L_x0 < 0): integral from +∞ to a

The integral is computed using the antiderivative of the tail function:
∫[-Inf*sign(L) to a] G_x0 * (N / (N - L_x0 * (x - x0)))^N dx

Returns the integral value.
"""
function _integral(t::CrystalBallTail{T}, a::T) where {T<:Real}
    (; G_x0, N, L_x0, x0) = t

    # Only support left tail (L_x0 > 0)
    L_x0 > zero(T) || error("_integral only supports left tail (L_x0 > 0), got L_x0 = $L_x0")

    # Compute offset from transition point
    offset_a = a - x0

    # Antiderivative formula: const_tail * ((N - L_x0 * offset) / N)^(1 - N)
    # For left tail (L_x0 > 0): integral from -∞ to a
    # At -∞, offset → -∞, so (N - L_x0 * offset) → +∞, antiderivative → 0
    # At a: const_tail * ((N - L_x0 * offset_a) / N)^(1 - N)
    const_tail = _tail_norm_const(t)

    return const_tail * ((N - L_x0 * offset_a) / N)^(one(T) - N)
end

"""
    _integral_inversion(t::CrystalBallTail, integral)

Find `a` such that the integral of the CrystalBallTail function from -∞ to `a` equals the given `integral` value.

Only supports left tail (L_x0 > 0). Throws an error if L_x0 <= 0.

Returns the value of `a` (in absolute coordinates).
"""
function _integral_inversion(t::CrystalBallTail{T}, integral::T) where {T<:Real}
    (; G_x0, N, L_x0, x0) = t

    # Only support left tail (L_x0 > 0)
    L_x0 > zero(T) || error("_integral_inversion only supports left tail (L_x0 > 0), got L_x0 = $L_x0")

    const_tail = _tail_norm_const(t)

    # Left tail: integral = const_tail * ((N - L_x0 * offset_a) / N)^(1 - N)
    # Solve for offset_a:
    # (N - L_x0 * offset_a) / N = (integral / const_tail)^(1 / (1 - N))
    # N - L_x0 * offset_a = N * (integral / const_tail)^(1 / (1 - N))
    # L_x0 * offset_a = N - N * (integral / const_tail)^(1 / (1 - N))
    # offset_a = (N / L_x0) * (1 - (integral / const_tail)^(1 / (1 - N)))

    ratio = integral / const_tail
    if ratio <= zero(T)
        # Integral is 0 or negative, return -Inf
        return T(-Inf)
    elseif ratio >= one(T)
        # Integral equals or exceeds const_tail, return x0 (transition point)
        return x0
    end

    ratio_power = ratio^(one(T) / (one(T) - N))
    offset_a = (N / L_x0) * (one(T) - ratio_power)

    # Convert back to absolute coordinate
    a = x0 + offset_a

    return a
end
