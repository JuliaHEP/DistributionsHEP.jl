# Crystal Ball Distributions: Mathematical Formulations

This document contains the essential mathematical formulas for the Crystal Ball family of distributions, widely used in high-energy physics for modeling signal and background shapes with Gaussian cores and power-law tails.

---

## 1. Crystal Ball Distribution (Left-sided)

### Parameters
- $\mu \in \mathbb{R}$: mean
- $\sigma > 0$: standard deviation
- $\alpha > 0$: transition point (in units of $\sigma$)
- $n > 1$: power-law exponent

### PDF Definition

Let $x_0 = \mu - \alpha \sigma$ be the transition point, and define $\text{offset} = x - x_0$.

$$
f(x) = N \begin{cases}
    G(x_0) \left(\frac{n}{n - \alpha \cdot \text{offset}}\right)^n & \text{for } x \leq x_0 \\
    \text{pdf}(\text{Normal}(\mu, \sigma), x) & \text{for } x > x_0
\end{cases}
$$

where $G(x_0) = \text{pdf}(\text{Normal}(\mu, \sigma), x_0)$ is the Gaussian PDF value at the transition point.

### Normalization Constant

$$
N = \frac{1}{I_{\text{tail}} + I_{\text{core}}}
$$

where:
- $I_{\text{tail}} = \frac{G(x_0)}{\alpha} \cdot \frac{n}{n-1}$
- $I_{\text{core}} = 1 - \text{cdf}(\text{Normal}(\mu, \sigma), x_0)$

### CDF

$$
F(x) = N \begin{cases}
    \frac{G(x_0)}{\alpha} \cdot \frac{n}{n-1} \cdot \left(\frac{n - \alpha \cdot \text{offset}}{n}\right)^{1-n} & \text{for } x \leq x_0 \\
    I_{\text{tail}} + \left[\text{cdf}(\text{Normal}(\mu, \sigma), x) - \text{cdf}(\text{Normal}(\mu, \sigma), x_0)\right] & \text{for } x > x_0
\end{cases}
$$

### Quantile (Inverse CDF)

For $p \in [0, 1]$:

$$
Q(p) = \begin{cases}
    x_0 + \frac{n}{\alpha} \left(1 - \left(\frac{p / N}{I_{\text{tail}} / N}\right)^{1/(1-n)}\right) & \text{for } p \leq F(x_0) \\
    \text{quantile}\left(\text{Normal}(\mu, \sigma), \frac{p - F(x_0)}{N} + \text{cdf}(\text{Normal}(\mu, \sigma), x_0)\right) & \text{for } p > F(x_0)
\end{cases}
$$

---

## 2. Double-Sided Crystal Ball Distribution

### Parameters
- $\mu \in \mathbb{R}$: mean
- $\sigma > 0$: standard deviation
- $\alpha_L > 0, n_L > 1$: left tail parameters
- $\alpha_R > 0, n_R > 1$: right tail parameters

### PDF Definition

Let $x_{0L} = \mu - \alpha_L \sigma$ and $x_{0R} = \mu + \alpha_R \sigma$ be the transition points.

$$
f(x) = N \begin{cases}
    G(x_{0L}) \left(\frac{n_L}{n_L - \alpha_L \cdot (x - x_{0L})}\right)^{n_L} & \text{for } x < x_{0L} \\
    \text{pdf}(\text{Normal}(\mu, \sigma), x) & \text{for } x_{0L} \leq x \leq x_{0R} \\
    G(x_{0R}) \left(\frac{n_R}{n_R + \alpha_R \cdot (x - x_{0R})}\right)^{n_R} & \text{for } x > x_{0R}
\end{cases}
$$

where:
- $G(x_{0L}) = \text{pdf}(\text{Normal}(\mu, \sigma), x_{0L})$
- $G(x_{0R}) = \text{pdf}(\text{Normal}(\mu, \sigma), x_{0R})$

### Normalization Constant

$$
N = \frac{1}{I_{\text{left}} + I_{\text{core}} + I_{\text{right}}}
$$

where:
- $I_{\text{left}} = \frac{G(x_{0L})}{\alpha_L} \cdot \frac{n_L}{n_L-1}$
- $I_{\text{core}} = \text{cdf}(\text{Normal}(\mu, \sigma), x_{0R}) - \text{cdf}(\text{Normal}(\mu, \sigma), x_{0L})$
- $I_{\text{right}} = \frac{G(x_{0R})}{\alpha_R} \cdot \frac{n_R}{n_R-1}$

### CDF

$$
F(x) = N \begin{cases}
    \frac{G(x_{0L})}{\alpha_L} \cdot \frac{n_L}{n_L-1} \cdot \left(\frac{n_L - \alpha_L \cdot (x - x_{0L})}{n_L}\right)^{1-n_L} & \text{for } x < x_{0L} \\
    F(x_{0L}) + \left[\text{cdf}(\text{Normal}(\mu, \sigma), x) - \text{cdf}(\text{Normal}(\mu, \sigma), x_{0L})\right] & \text{for } x_{0L} \leq x \leq x_{0R} \\
    F(x_{0R}) - \frac{G(x_{0R})}{\alpha_R} \cdot \frac{n_R}{n_R-1} \cdot \left[\left(\frac{n_R + \alpha_R \cdot (x - x_{0R})}{n_R}\right)^{1-n_R} - 1\right] & \text{for } x > x_{0R}
\end{cases}
$$

### Quantile (Inverse CDF)

For $p \in [0, 1]$:

$$
Q(p) = \begin{cases}
    x_{0L} + \frac{n_L}{\alpha_L} \left(1 - \left(\frac{p / N}{I_{\text{left}} / N}\right)^{1/(1-n_L)}\right) & \text{for } p \leq F(x_{0L}) \\
    \text{quantile}\left(\text{Normal}(\mu, \sigma), \frac{p - F(x_{0L})}{N} + \text{cdf}(\text{Normal}(\mu, \sigma), x_{0L})\right) & \text{for } F(x_{0L}) < p \leq F(x_{0R}) \\
    x_{0R} - \frac{n_R}{\alpha_R} \left(1 - \left(\frac{(p - F(x_{0R})) / N + I_{\text{right}} / N}{I_{\text{right}} / N}\right)^{1/(1-n_R)}\right) & \text{for } p > F(x_{0R})
\end{cases}
$$

---

## Implementation Notes

The implementation uses a unified `CrystalBallTail` structure parameterized by:
- `G_x0`: The core PDF value at the transition point
- `N`: The power-law exponent
- `L_x0`: The logarithmic derivative at the transition point ($\alpha$ for left tail, $-\alpha$ for right tail)
- `x0`: The absolute transition point

This approach simplifies the formulas and makes the implementation more general, allowing the same tail structure to work with different core distributions.
