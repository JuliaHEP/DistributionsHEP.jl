# Double-Sided Crystal Ball Distribution - Complete Mathematical Derivation

## Parameters
- \( \mu \in \mathbb{R} \): mean
- \( \sigma > 0 \): width
- \( \alpha_L > 0, n_L > 1 \): left tail parameters
- \( \alpha_R > 0, n_R > 1 \): right tail parameters

Let \( \hat{x} = \frac{x - \mu}{\sigma} \).

## PDF Definition
The probability density function is defined piecewise:
\[
f(x) =
\begin{cases}
    N A_L (B_L - \hat{x})^{-n_L} & \text{for } \hat{x} < -\alpha_L \\
    N \exp\left(-\frac{\hat{x}^2}{2}\right) & \text{for } -\alpha_L \leq \hat{x} \leq \alpha_R \\
    N A_R (B_R + \hat{x})^{-n_R} & \text{for } \hat{x} > \alpha_R
\end{cases}
\]

## Tail Parameters (Continuity and Differentiability Conditions)

### Left Tail Parameters
At the transition point \( \hat{x} = -\alpha_L \), we require:
1. **Continuity**: \( N A_L (B_L + \alpha_L)^{-n_L} = N \exp\left(-\frac{\alpha_L^2}{2}\right) \)
2. **Differentiability**: \( -N A_L n_L (B_L + \alpha_L)^{-n_L-1} = -N \alpha_L \exp\left(-\frac{\alpha_L^2}{2}\right) \)

From condition 1: \( A_L (B_L + \alpha_L)^{-n_L} = \exp\left(-\frac{\alpha_L^2}{2}\right) \)

From condition 2: \( A_L n_L (B_L + \alpha_L)^{-n_L-1} = \alpha_L \exp\left(-\frac{\alpha_L^2}{2}\right) \)

Dividing condition 2 by condition 1: \( \frac{n_L}{B_L + \alpha_L} = \alpha_L \)

Therefore: \( B_L = \frac{n_L}{\alpha_L} - \alpha_L \)

Substituting back into condition 1: \( A_L = \left(\frac{n_L}{\alpha_L}\right)^{n_L} \exp\left(-\frac{\alpha_L^2}{2}\right) \)

### Right Tail Parameters
At the transition point \( \hat{x} = \alpha_R \), we require:
1. **Continuity**: \( N A_R (B_R + \alpha_R)^{-n_R} = N \exp\left(-\frac{\alpha_R^2}{2}\right) \)
2. **Differentiability**: \( -N A_R n_R (B_R + \alpha_R)^{-n_R-1} = -N \alpha_R \exp\left(-\frac{\alpha_R^2}{2}\right) \)

Following the same logic:
\( B_R = \frac{n_R}{\alpha_R} - \alpha_R \)
\( A_R = \left(\frac{n_R}{\alpha_R}\right)^{n_R} \exp\left(-\frac{\alpha_R^2}{2}\right) \)

## Normalization Constant

The normalization constant \( N \) is determined by requiring \( \int_{-\infty}^{\infty} f(x) dx = 1 \).

Breaking the integral into three parts:

### Left Tail Integral (\( \hat{x} < -\alpha_L \))
\[
\int_{-\infty}^{-\alpha_L} N A_L (B_L - \hat{x})^{-n_L} d\hat{x} = N A_L \int_{-\infty}^{-\alpha_L} (B_L - \hat{x})^{-n_L} d\hat{x}
\]

Let \( u = B_L - \hat{x} \), then \( du = -d\hat{x} \):
\[
= N A_L \int_{B_L + \alpha_L}^{\infty} u^{-n_L} du = N A_L \frac{(B_L + \alpha_L)^{1-n_L}}{n_L - 1}
\]

Substituting \( B_L = \frac{n_L}{\alpha_L} - \alpha_L \):
\[
= N A_L \frac{(\frac{n_L}{\alpha_L})^{1-n_L}}{n_L - 1} = N \frac{n_L}{\alpha_L (n_L - 1)} \exp\left(-\frac{\alpha_L^2}{2}\right)
\]

### Gaussian Core Integral (\( -\alpha_L \leq \hat{x} \leq \alpha_R \))
\[
\int_{-\alpha_L}^{\alpha_R} N \exp\left(-\frac{\hat{x}^2}{2}\right) d\hat{x} = N \sqrt{2\pi} \frac{1}{2} \left[ \operatorname{erf}\left(\frac{\alpha_R}{\sqrt{2}}\right) + \operatorname{erf}\left(\frac{\alpha_L}{\sqrt{2}}\right) \right]
\]

### Right Tail Integral (\( \hat{x} > \alpha_R \))
\[
\int_{\alpha_R}^{\infty} N A_R (B_R + \hat{x})^{-n_R} d\hat{x} = N A_R \int_{\alpha_R}^{\infty} (B_R + \hat{x})^{-n_R} d\hat{x}
\]

Let \( u = B_R + \hat{x} \), then \( du = d\hat{x} \):
\[
= N A_R \int_{B_R + \alpha_R}^{\infty} u^{-n_R} du = N A_R \frac{(B_R + \alpha_R)^{1-n_R}}{n_R - 1}
\]

Substituting \( B_R = \frac{n_R}{\alpha_R} - \alpha_R \):
\[
= N A_R \frac{(\frac{n_R}{\alpha_R})^{1-n_R}}{n_R - 1} = N \frac{n_R}{\alpha_R (n_R - 1)} \exp\left(-\frac{\alpha_R^2}{2}\right)
\]

### Total Normalization
\[
1 = N \left[ \frac{n_L}{\alpha_L (n_L - 1)} \exp\left(-\frac{\alpha_L^2}{2}\right) + \sqrt{\frac{\pi}{2}} \left( \operatorname{erf}\left(\frac{\alpha_R}{\sqrt{2}}\right) + \operatorname{erf}\left(\frac{\alpha_L}{\sqrt{2}}\right) \right) + \frac{n_R}{\alpha_R (n_R - 1)} \exp\left(-\frac{\alpha_R^2}{2}\right) \right]
\]

Therefore:
\[
N = \frac{1}{\sigma \left[ C_L + D + C_R \right]}
\]
where:
- \( C_L = \frac{n_L}{\alpha_L (n_L - 1)} \exp\left(-\frac{\alpha_L^2}{2}\right) \)
- \( D = \sqrt{\frac{\pi}{2}} \left( \operatorname{erf}\left(\frac{\alpha_R}{\sqrt{2}}\right) + \operatorname{erf}\left(\frac{\alpha_L}{\sqrt{2}}\right) \right) \)
- \( C_R = \frac{n_R}{\alpha_R (n_R - 1)} \exp\left(-\frac{\alpha_R^2}{2}\right) \)

## CDF Derivation

The CDF is obtained by integrating the PDF: \( F(x) = \int_{-\infty}^{x} f(t) dt \)

### CDF Values at Transition Points

**CDF at left transition (\( \hat{x} = -\alpha_L \)):**
\[
F_L = \int_{-\infty}^{-\alpha_L} f(x) dx = N A_L \frac{(B_L + \alpha_L)^{1-n_L}}{n_L - 1} = N \frac{n_L}{\alpha_L (n_L - 1)} \exp\left(-\frac{\alpha_L^2}{2}\right)
\]

**CDF at right transition (\( \hat{x} = \alpha_R \)):**
\[
F_R = F_L + \int_{-\alpha_L}^{\alpha_R} f(x) dx = F_L + N \sigma \sqrt{\frac{\pi}{2}} \left( \operatorname{erf}\left(\frac{\alpha_R}{\sqrt{2}}\right) + \operatorname{erf}\left(\frac{\alpha_L}{\sqrt{2}}\right) \right)
\]

### CDF Formula

\[
F(x) =
\begin{cases}
    N A_L \frac{(B_L - \hat{x})^{1-n_L}}{n_L - 1} & \text{for } \hat{x} < -\alpha_L \\
    F_L + N \sigma \sqrt{\frac{\pi}{2}} \left[ \operatorname{erf}\left(\frac{\hat{x}}{\sqrt{2}}\right) + \operatorname{erf}\left(\frac{\alpha_L}{\sqrt{2}}\right) \right] & \text{for } -\alpha_L \leq \hat{x} \leq \alpha_R \\
    F_R + N A_R \frac{(B_R + \hat{x})^{1-n_R} - (B_R + \alpha_R)^{1-n_R}}{n_R - 1} & \text{for } \hat{x} > \alpha_R
\end{cases}
\]

**Justification for each region:**

1. **Left tail (\( \hat{x} < -\alpha_L \))**: Direct integration of the power-law PDF from \(-\infty\) to \(\hat{x}\).

2. **Gaussian core (\( -\alpha_L \leq \hat{x} \leq \alpha_R \))**: CDF at left transition plus integral of Gaussian PDF from \(-\alpha_L\) to \(\hat{x}\).

3. **Right tail (\( \hat{x} > \alpha_R \))**: CDF at right transition plus integral of power-law PDF from \(\alpha_R\) to \(\hat{x}\).

## Quantile (Inverse CDF) Derivation

The quantile function \( Q(p) \) is the inverse of the CDF: \( Q(p) = F^{-1}(p) \)

### Quantile Formula

\[
Q(p) =
\begin{cases}
    \mu + \sigma \left[ B_L - \left( \frac{p (n_L-1)}{N A_L} \right)^{1/(1-n_L)} \right] & \text{for } p \leq F_L \\
    \mu + \sigma \sqrt{2} \operatorname{erf}^{-1}\left( \frac{p - F_L}{N \sigma \sqrt{\pi/2}} - \operatorname{erf}\left(\frac{\alpha_L}{\sqrt{2}}\right) \right) & \text{for } F_L < p \leq F_R \\
    \mu + \sigma \left[ \left( (p - F_R) \frac{n_R-1}{N A_R} + (B_R + \alpha_R)^{1-n_R} \right)^{1/(1-n_R)} - B_R \right] & \text{for } p > F_R
\end{cases}
\]

**Justification for each region:**

1. **Left tail (\( p \leq F_L \))**: Invert the left tail CDF formula.
2. **Gaussian core (\( F_L < p \leq F_R \))**: Invert the Gaussian core CDF formula.
3. **Right tail (\( p > F_R \))**: Invert the right tail CDF formula.

---

## Summary of Key Formulas

### Precomputed Constants
- \( A_L = \left(\frac{n_L}{\alpha_L}\right)^{n_L} \exp\left(-\frac{\alpha_L^2}{2}\right) \)
- \( B_L = \frac{n_L}{\alpha_L} - \alpha_L \)
- \( A_R = \left(\frac{n_R}{\alpha_R}\right)^{n_R} \exp\left(-\frac{\alpha_R^2}{2}\right) \)
- \( B_R = \frac{n_R}{\alpha_R} - \alpha_R \)
- \( N = \frac{1}{\sigma (C_L + D + C_R)} \)

### Transition Point CDF Values
- \( F_L = N \frac{n_L}{\alpha_L (n_L - 1)} \exp\left(-\frac{\alpha_L^2}{2}\right) \)
- \( F_R = F_L + N \sigma \sqrt{\frac{\pi}{2}} \left( \operatorname{erf}\left(\frac{\alpha_R}{\sqrt{2}}\right) + \operatorname{erf}\left(\frac{\alpha_L}{\sqrt{2}}\right) \right) \) 