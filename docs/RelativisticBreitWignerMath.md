# Relativistic Breit-Wigner Distribution: Mathematical Derivations

This document contains detailed mathematical derivations for the Relativistic Breit-Wigner distribution, used extensively in particle physics to describe resonance behavior in scattering processes.

---

## PDF Definition

For constant width \(\Gamma\), the relativistic Breit-Wigner distribution has the form:

\[
f(x) = \frac{k}{(x^2 - M^2)^2 + M^2 \Gamma^2}
\]

where \(M > 0\) is the resonance mass and \(\Gamma > 0\) is the width parameter.

## Normalization

Let us define:

\[
A = M^2, \qquad B = M\Gamma, \qquad r = \sqrt{A^2 + B^2}
\]

\[
u = \sqrt{\frac{r + A}{2}}, \qquad v = \sqrt{\frac{r - A}{2}}
\]

Note that these definitions imply:

\[
2uv = B, \qquad u^2 - v^2 = A
\]

The denominator can be factorized as:

\[
(A - x^2)^2 + B^2 = \bigl((x - u)^2 + v^2\bigr)\bigl((x + u)^2 + v^2\bigr)
\]

The normalization integral is:

\[
\int_{-\infty}^{\infty} \frac{dx}{(A - x^2)^2 + B^2} = \frac{\pi}{2rv}
\]

Therefore, the normalized PDF is:

\[
f(x) = \frac{2rv}{\pi} \cdot \frac{1}{(A - x^2)^2 + B^2}
\]

or equivalently:

\[
f(x) = \frac{2rv}{\pi} \cdot \frac{1}{(M^2 - x^2)^2 + M^2 \Gamma^2}
\]

## Closed-Form CDF

For constant width \(\Gamma\), the CDF has an elementary antiderivative (involving logarithms and arctangents), yielding a closed-form expression:

\[
F(x) = \frac{1}{2} + \frac{1}{2\pi} \left[ \arctan\left(\frac{x + u}{v}\right) + \arctan\left(\frac{x - u}{v}\right) \right] + \frac{v}{4\pi u} \ln\left(\frac{(x + u)^2 + v^2}{(x - u)^2 + v^2}\right)
\]

**Properties:**
- \(F(-\infty) = 0\)
- \(F(\infty) = 1\)
- \(F(0) = \frac{1}{2}\) (by symmetry)

**Note:** If the width \(\Gamma\) becomes energy-dependent, \(\Gamma(x)\), then this simple closed form generally breaks down and numerical integration is required.

---

## Summary of Key Formulas

### Precomputed Constants
- \(A = M^2\)
- \(B = M\Gamma\)
- \(r = \sqrt{A^2 + B^2}\)
- \(u = \sqrt{\frac{r + A}{2}}\)
- \(v = \sqrt{\frac{r - A}{2}}\)

### PDF
\[
f(x) = \frac{2rv}{\pi} \cdot \frac{1}{(M^2 - x^2)^2 + M^2 \Gamma^2}
\]

### CDF
\[
F(x) = \frac{1}{2} + \frac{1}{2\pi} \left[ \arctan\left(\frac{x + u}{v}\right) + \arctan\left(\frac{x - u}{v}\right) \right] + \frac{v}{4\pi u} \ln\left(\frac{(x + u)^2 + v^2}{(x - u)^2 + v^2}\right)
\]
