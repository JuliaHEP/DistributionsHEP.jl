# ARGUS Distribution - Mathematical Summary

## Probability Density Function (PDF)
The unnormalized PDF for the standardized ARGUS distribution is:

\[
f_{\text{argus}}(x; c, p) = x (1 - x^2)^p \exp\left[c (1 - x^2)\right]
\]

**Domain**:  
- \( x \in [0, 1] \)
- Shape parameter \( c < 0 \)
- Power parameter \( p > -1 \)

## Cumulative Distribution Function (CDF)
The unnormalized CDF is expressed via the lower incomplete gamma function:

\[
F_{\text{argus}}(x; c, p) = \frac{\gamma\left(p + 1, -c (1 - x^2)\right)}{2 (-c)^{p + 1}}
\]

where \(\gamma(s, x) = \int_0^x t^{s-1} e^{-t} dt\) is the **lower incomplete gamma function**.

The **normalized CDF** (with integral constraint \(\int_0^1 f_{\text{argus}} dx = 1\)) is:

\[
\Phi(x; c, p) = \frac{F_{\text{argus}}(x) - F_{\text{argus}}(0)}{F_{\text{argus}}(1) - F_{\text{argus}}(0)}
\]

## Quantile Function (Inverse CDF)
The quantile function \( Q(q; c, p) \) solves \( \Phi(x; c, p) = q \) for \( x \):

\[
Q(q) = \sqrt{1 - \frac{1}{-c} \gamma^{-1}\left(p + 1, (1 - q) \cdot \gamma(p + 1, -c)\right)}
\]

where \(\gamma^{-1}(s, y)\) is the **inverse incomplete gamma function**, defined by \(\gamma(s, z) = y \iff z = \gamma^{-1}(s, y)\).

### Key Components:
1. **Regularization**:  
   Uses the regularized incomplete gamma \( P(s, x) = \gamma(s, x) / \Gamma(s) \).
2. **Inversion**:  
   The inverse \( P^{-1}(s, u) \) satisfies \( P(s, z) = u \iff z = P^{-1}(s, u) \).

## Boundary Conditions
- **PDF**: \( f_{\text{argus}}(x) = 0 \) for \( x \leq 0 \) or \( x \geq 1 \)
- **CDF**: 
  - \( \Phi(x) = 0 \) for \( x \leq 0 \)
  - \( \Phi(x) = 1 \) for \( x \geq 1 \)
- **Quantile**: 
  - \( Q(0) = 0 \)
  - \( Q(1) = 1 \)

## Normalization
The normalization constant \( \mathcal{N} \) ensures \( \int_0^1 f_{\text{argus}} dx = 1 \):

\[
\mathcal{N} = \frac{2 (-c)^{p + 1}}{\gamma(p + 1, -c)}
\]

This appears in the PDF as \( f_{\text{argus}}(x) / \mathcal{N} \).