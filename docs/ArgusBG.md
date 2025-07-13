# ARGUS Distribution: Mathematical Summary

This document provides detailed mathematical derivations for the ARGUS distribution, a continuous univariate distribution based on the ARGUS shape function. The distribution is defined on the standard interval `[0, 1]` and can be transformed to arbitrary intervals `[a, b]`.

---

## Distribution Overview

The ARGUS distribution is characterized by:
- **Domain**: \( x \in [0, 1] \) (standard form)
- **Shape parameter**: \( c < 0 \) (controls the peak position and width)
- **Power parameter**: \( p > -1 \) (controls the tail behavior)

The distribution is normalized to ensure \( \int_0^1 f(x) dx = 1 \).

---

## Probability Density Function (PDF)

### Unnormalized PDF
The unnormalized probability density function for the standardized ARGUS distribution is:

\[
f_{\text{argus}}(x; c, p) = x (1 - x^2)^p \exp\left[c (1 - x^2)\right]
\]

### Parameter Constraints
- **Domain**: \( x \in [0, 1] \)
- **Shape parameter**: \( c < 0 \) (must be negative)
- **Power parameter**: \( p > -1 \) (ensures integrability)

### Normalized PDF
The normalized PDF is obtained by dividing by the normalization constant:

\[
f(x; c, p) = \frac{f_{\text{argus}}(x; c, p)}{\mathcal{N}}
\]

where \( \mathcal{N} \) is the normalization constant (see below).

---

## Cumulative Distribution Function (CDF)

### Unnormalized CDF
The unnormalized cumulative distribution function is expressed via the lower incomplete gamma function:

\[
F_{\text{argus}}(x; c, p) = \frac{\gamma\left(p + 1, -c (1 - x^2)\right)}{2 (-c)^{p + 1}}
\]

where \( \gamma(s, x) = \int_0^x t^{s-1} e^{-t} dt \) is the **lower incomplete gamma function**.

### Normalized CDF
The **normalized CDF** (with integral constraint \( \int_0^1 f_{\text{argus}} dx = 1 \)) is:

\[
\Phi(x; c, p) = \frac{F_{\text{argus}}(x) - F_{\text{argus}}(0)}{F_{\text{argus}}(1) - F_{\text{argus}}(0)}
\]

---

## Quantile Function (Inverse CDF)

The quantile function \( Q(q; c, p) \) solves \( \Phi(x; c, p) = q \) for \( x \):

\[
Q(q) = \sqrt{1 - \frac{1}{-c} \gamma^{-1}\left(p + 1, (1 - q) \cdot \gamma(p + 1, -c)\right)}
\]

where \( \gamma^{-1}(s, y) \) is the **inverse incomplete gamma function**, defined by \( \gamma(s, z) = y \iff z = \gamma^{-1}(s, y) \).

### Implementation Details

#### Regularized Incomplete Gamma Function
The implementation uses the regularized incomplete gamma function:
\[
P(s, x) = \frac{\gamma(s, x)}{\Gamma(s)}
\]

#### Inverse Function
The inverse \( P^{-1}(s, u) \) satisfies:
\[
P(s, z) = u \iff z = P^{-1}(s, u)
\]

---

## Normalization Constant

The normalization constant \( \mathcal{N} \) ensures \( \int_0^1 f_{\text{argus}} dx = 1 \):

\[
\mathcal{N} = F_{\text{argus}}(1) - F_{\text{argus}}(0) = \frac{\gamma(p + 1, -c)}{2 (-c)^{p + 1}}
\]

This appears in the PDF as \( f_{\text{argus}}(x) / \mathcal{N} \).

---

## Implementation Notes

### Numerical Stability
- The quantile function includes numerical stability checks to prevent negative values from floating-point errors
- The inverse gamma function is computed using specialized algorithms for numerical accuracy

### Parameter Validation
- Shape parameter \( c \) must be negative for proper distribution behavior
- Power parameter \( p \) must be greater than -1 for integrability
- The implementation includes parameter validation to ensure these constraints are met

### Random Sampling
Random samples are generated using the inverse CDF method:
\[
X \sim \text{ARGUS}(c, p) \iff X = Q(U) \text{ where } U \sim \text{Uniform}(0, 1)
\]

This method is exact and efficient, requiring only one uniform random number per sample.