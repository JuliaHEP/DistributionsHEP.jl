# Bifurcated Gaussian Distribution: Mathematical Derivations

This document contains detailed mathematical derivations for the Bifurcated Gaussian distribution, which extends the standard Gaussian distribution by allowing different scale parameters on the left and right sides of the mean.

---

## 1. PDF Definition

Let \( x \in \mathbb{R} \), \( \mu \in \mathbb{R} \), \( \sigma > 0 \), \( \psi \in \mathbb{R} \).

Define:
- \( \kappa = \tanh(\psi) \) (asymmetry parameter, bounded between -1 and 1)
- \( \sigma_L = \sigma(1 + \kappa) \) (left-side scale parameter)
- \( \sigma_R = \sigma(1 - \kappa) \) (right-side scale parameter)

The probability density function (PDF) is:

\[
f(x; \mu, \sigma, \psi) = \begin{cases}
    \frac{1}{\sqrt{2\pi} \sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma_L^2}\right) & \text{for } x \leq \mu \\
    \frac{1}{\sqrt{2\pi} \sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma_R^2}\right) & \text{for } x > \mu
\end{cases}
\]

**Note**: The PDF uses \( \sigma \) in the normalization factor (denominator) but \( \sigma_L \) or \( \sigma_R \) in the exponential. This ensures:
1. Continuity at \( x = \mu \)
2. Proper normalization (integrates to 1)

When \( \psi = 0 \) (so \( \kappa = 0 \)), the distribution reduces to a standard Gaussian with scale \( \sigma \).

---

## 2. CDF Derivation

The cumulative distribution function (CDF) is obtained by integrating the PDF:

\[
F(x) = \int_{-\infty}^{x} f(t) dt
\]

### CDF at the Transition Point

At \( x = \mu \), the CDF value is:

\[
F(\mu) = \frac{\sigma_L}{2\sigma}
\]

This comes from integrating the left side PDF from \( -\infty \) to \( \mu \).

### CDF Formula

\[
F(x) = \begin{cases}
    F(\mu) \left[1 + \operatorname{erf}\left(\frac{x-\mu}{\sigma_L \sqrt{2}}\right)\right] & \text{for } x \leq \mu \\
    F(\mu) + \frac{\sigma_R}{2\sigma} \operatorname{erf}\left(\frac{x-\mu}{\sigma_R \sqrt{2}}\right) & \text{for } x > \mu
\end{cases}
\]

where \( \operatorname{erf} \) is the error function.

---

## 3. Quantile Function (Inverse CDF)

The quantile function \( F^{-1}(p) \) for probability \( p \in [0,1] \) is:

\[
F^{-1}(p) = \begin{cases}
    \mu + \sigma_L \sqrt{2} \cdot \operatorname{erf}^{-1}\left(\frac{2\sigma p}{\sigma_L} - 1\right) & \text{for } p \leq F(\mu) \\
    \mu + \sigma_R \sqrt{2} \cdot \operatorname{erf}^{-1}\left(\frac{2\sigma(p - F(\mu))}{\sigma_R}\right) & \text{for } p > F(\mu)
\end{cases}
\]

where \( \operatorname{erf}^{-1} \) is the inverse error function.

---

## 4. Moments and Statistics

Let \( Y = X - \mu \). The PDF of \( Y \) is:

\[
f_Y(y) = \frac{1}{\sqrt{2\pi} \sigma} \times \begin{cases}
\exp\left(-\frac{y^2}{2\sigma_L^2}\right), & y \leq 0 \\
\exp\left(-\frac{y^2}{2\sigma_R^2}\right), & y > 0
\end{cases}
\]

with \( \sigma = \frac{\sigma_L + \sigma_R}{2} \).

Using the identity \( \int_0^\infty t^n e^{-t^2/2} dt = 2^{(n-1)/2} \Gamma\left(\frac{n+1}{2}\right) \), the moments about \( \mu \) are:

\[
m_1 = \mathbb{E}[Y] = \frac{\sigma_R^2 - \sigma_L^2}{\sqrt{2\pi} \sigma}
\]

\[
m_2 = \mathbb{E}[Y^2] = \frac{\sigma_L^3 + \sigma_R^3}{2\sigma}
\]

\[
m_3 = \mathbb{E}[Y^3] = \frac{2(\sigma_R^4 - \sigma_L^4)}{\sqrt{2\pi} \sigma}
\]

### Mean

\[
\mathbb{E}[X] = \mu + m_1 = \mu - \frac{4\sigma}{\sqrt{2\pi}} \kappa
\]

where \( \kappa = \tanh(\psi) \).

**Note**: The mean is **not** equal to \( \mu \) when the distribution is asymmetric (\( \psi \neq 0 \)). The location parameter \( \mu \) is the mode (peak) of the distribution, not the mean.

### Variance

The central moments about the mean (\( \Delta = m_1 \)) are:

\[
\mu_2 = m_2 - \Delta^2, \qquad \mu_3 = m_3 - 3\Delta m_2 + 2\Delta^3
\]

The variance is:

\[
\operatorname{Var}(X) = \mu_2 = \sigma^2 \cdot \frac{\pi + (3\pi - 8)\kappa^2}{\pi}
\]

### Standard Deviation

\[
\sigma_{\text{actual}} = \sqrt{\operatorname{Var}(X)} = \sigma \sqrt{\frac{\pi + (3\pi - 8)\kappa^2}{\pi}}
\]

### Skewness

The skewness (third standardized moment) is:

\[
\gamma_1 = \frac{\mu_3}{\mu_2^{3/2}} = \frac{2\sqrt{2} \kappa \left((5\pi - 16)\kappa^2 - \pi\right)}{\left(\pi + (3\pi - 8)\kappa^2\right)^{3/2}}
\]

where \( \kappa = \tanh(\psi) \).

**Derivation:**

Starting from the moments about \( \mu \):

\[
m_1 = \frac{\sigma_R^2 - \sigma_L^2}{\sqrt{2\pi} \sigma}, \quad
m_2 = \frac{\sigma_L^3 + \sigma_R^3}{2\sigma}, \quad
m_3 = \frac{2(\sigma_R^4 - \sigma_L^4)}{\sqrt{2\pi} \sigma}
\]

With the parameterization \( \sigma_L = \sigma(1 + \kappa) \), \( \sigma_R = \sigma(1 - \kappa) \), we have:

\[
m_1 = \frac{\sigma^2((1-\kappa)^2 - (1+\kappa)^2)}{\sqrt{2\pi} \sigma} = \frac{-4\sigma\kappa}{\sqrt{2\pi}} = -\frac{4\sigma\kappa}{\sqrt{2\pi}}
\]

\[
m_2 = \frac{\sigma^3((1+\kappa)^3 + (1-\kappa)^3)}{2\sigma} = \frac{\sigma^2(2 + 6\kappa^2)}{2} = \sigma^2(1 + 3\kappa^2)
\]

\[
m_3 = \frac{2\sigma^4((1-\kappa)^4 - (1+\kappa)^4)}{\sqrt{2\pi} \sigma} = \frac{2\sigma^3(-8\kappa - 8\kappa^3)}{\sqrt{2\pi}} = -\frac{16\sigma^3\kappa(1+\kappa^2)}{\sqrt{2\pi}}
\]

The central moments about the mean are:

\[
\mu_2 = m_2 - m_1^2 = \sigma^2(1 + 3\kappa^2) - \frac{16\sigma^2\kappa^2}{2\pi} = \sigma^2\left(1 + 3\kappa^2 - \frac{8\kappa^2}{\pi}\right) = \sigma^2 \cdot \frac{\pi + (3\pi - 8)\kappa^2}{\pi}
\]

\[
\mu_3 = m_3 - 3m_1 m_2 + 2m_1^3
\]

After algebraic simplification, this yields:

\[
\mu_3 = \frac{2\sqrt{2}\sigma^3 \kappa \left((5\pi - 16)\kappa^2 - \pi\right)}{\sqrt{\pi}}
\]

Therefore:

\[
\gamma_1 = \frac{\mu_3}{\mu_2^{3/2}} = \frac{2\sqrt{2} \kappa \left((5\pi - 16)\kappa^2 - \pi\right)}{\left(\pi + (3\pi - 8)\kappa^2\right)^{3/2}}
\]

**Properties:**
- When \( \kappa = 0 \) (symmetric case, \( \psi = 0 \)), skewness = 0
- When \( \kappa > 0 \) (left side wider, \( \sigma_L > \sigma_R \)), skewness < 0 (left-skewed)
- When \( \kappa < 0 \) (right side wider, \( \sigma_L < \sigma_R \)), skewness > 0 (right-skewed)
- The skewness is independent of the overall scale \( \sigma \)

### Kurtosis

The excess kurtosis (fourth standardized moment minus 3) is:

\[
\gamma_2(\kappa) = \frac{4\kappa^2 \left[\pi(3\pi - 8) - (3\pi^2 - 40\pi + 96)\kappa^2\right]}{\left[\pi + (3\pi - 8)\kappa^2\right]^2}
\]

where \( \kappa = \tanh(\psi) \).

The (non-excess) kurtosis is:

\[
\beta_2(\kappa) = 3 + \gamma_2(\kappa)
\]

**Properties:**
- When \( \kappa = 0 \) (symmetric case, \( \psi = 0 \)), excess kurtosis = 0 and \( \beta_2 = 3 \) (same as ordinary Gaussian)
- The excess kurtosis is independent of the overall scale \( \sigma \)
- For asymmetric distributions (\( \kappa \neq 0 \)), the excess kurtosis is generally non-zero

---

## 5. Parameter Relationships

The asymmetry parameter \( \psi \) controls the difference between left and right scales:

\[
\kappa = \tanh(\psi) \in (-1, 1)
\]

\[
\sigma_L = \sigma(1 + \kappa) \in (0, 2\sigma)
\]

\[
\sigma_R = \sigma(1 - \kappa) \in (0, 2\sigma)
\]

Note that \( \sigma_L + \sigma_R = 2\sigma \), so \( \sigma \) represents the average of the two scale parameters.

---

## 6. Special Cases

### Symmetric Case (\( \psi = 0 \))

When \( \psi = 0 \):
- \( \kappa = 0 \)
- \( \sigma_L = \sigma_R = \sigma \)
- The distribution reduces to a standard Gaussian \( \mathcal{N}(\mu, \sigma^2) \)
- Skewness = 0

### Extreme Asymmetry

As \( \psi \to \infty \):
- \( \kappa \to 1 \)
- \( \sigma_L \to 2\sigma \), \( \sigma_R \to 0 \)
- The distribution becomes highly left-skewed

As \( \psi \to -\infty \):
- \( \kappa \to -1 \)
- \( \sigma_L \to 0 \), \( \sigma_R \to 2\sigma \)
- The distribution becomes highly right-skewed
