# Shared checks for Crystal Ball tail matching.
# Regression guard: #41 used L_x0 = ±α (standardized log-derivative) instead of ±α/σ
# (absolute coordinates). That preserves pdf continuity and ∫pdf = 1, but breaks C¹
# matching whenever σ ≠ 1.

"""One-sided numerical derivative of `pdf(d, x)` near `x0`."""
function pdf_derivative_one_sided(d, x0, side::Symbol; h=1e-7)
    side == :left && return (pdf(d, x0 - h) - pdf(d, x0 - 2h)) / h
    side == :right && return (pdf(d, x0 + 2h) - pdf(d, x0 + h)) / h
    throw(ArgumentError("side must be :left or :right"))
end

"""Expected d ln pdf / dx for a Normal core at `x`."""
gaussian_log_derivative(μ, σ, x) = -(x - μ) / σ^2

"""
    test_pdf_derivative_continuous(d, x0; rtol=1e-5)

Check that the PDF has a continuous first derivative at a Gaussian–tail transition.
"""
function test_pdf_derivative_continuous(d, x0; rtol=1e-5)
    deriv_left = pdf_derivative_one_sided(d, x0, :left)
    deriv_right = pdf_derivative_one_sided(d, x0, :right)
    @test isapprox(deriv_left, deriv_right; rtol=rtol)
end

"""
    test_tail_log_derivative(d, tail, gauss)

Check that the stored tail log-derivative matches the Gaussian core at the join.
"""
function test_tail_log_derivative(tail, gauss)
    expected = gaussian_log_derivative(gauss.μ, gauss.σ, tail.x0)
    @test isapprox(tail.L_x0, expected)
end
