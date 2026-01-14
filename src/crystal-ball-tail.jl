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
