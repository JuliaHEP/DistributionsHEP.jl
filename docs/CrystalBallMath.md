# Crystal Ball Distributions: Mathematical Formulations

This document contains the essential mathematical formulas for the Crystal Ball family of distributions, widely used in high-energy physics for modeling signal and background shapes with Gaussian cores and power-law tails.

---

## 1. Crystal Ball Distribution (Left-sided)

### Parameters
- $\mu \in \mathbb{R}$: mean
- $\sigma > 0$: standard deviation
- $\alpha > 0$: transition point (in units of $\sigma$)
- $n > 1$: power-law exponent

### PDF Definition (standardized coordinates)

Let $\hat{x} = (x - \mu)/\sigma$ and $x_0 = \mu - \alpha \sigma$ (so $\hat{x}_0 = -\alpha$).

$$
f(x) = N \begin{cases}
    A\,(B - \hat{x})^{-n} & \text{for } \hat{x} \leq -\alpha \\
    \exp\!\left(-\dfrac{\hat{x}^2}{2}\right) & \text{for } \hat{x} > -\alpha
\end{cases}
$$

with $A = (\tfrac{n}{\alpha})^n \exp(-\alpha^2/2)$ and $B = \tfrac{n}{\alpha} - \alpha$, chosen so that $f$ and $df/dx$ are continuous at $\hat{x} = -\alpha$. The overall constant $N$ normalizes $\int_{-\infty}^{\infty} f(x)\,dx = 1$.

### Equivalent implementation form (absolute coordinates)

Define $\text{offset} = x - x_0$ and $G(x_0) = \mathrm{pdf}(\mathcal{N}(\mu,\sigma), x_0)$. The left tail is

$$
f(x) = N\,G(x_0)\left(\frac{n}{n - L_{x_0}\,\text{offset}}\right)^n,
\qquad x \leq x_0,
$$

where $L_{x_0} = \left.\dfrac{d}{dx}\ln G(x)\right|_{x_0} = \dfrac{\alpha}{\sigma}$ for a Normal core.

### Normalization Constant

$$
N = \frac{1}{I_{\text{tail}} + I_{\text{core}}}
$$

where:
- $I_{\text{tail}} = \dfrac{G(x_0)}{L_{x_0}} \cdot \dfrac{n}{n-1} = \dfrac{G(x_0)\,\sigma}{\alpha}\cdot\dfrac{n}{n-1}$
- $I_{\text{core}} = 1 - \mathrm{cdf}(\mathcal{N}(\mu, \sigma), x_0)$

### CDF

$$
F(x) = N \begin{cases}
    \dfrac{G(x_0)}{L_{x_0}} \cdot \dfrac{n}{n-1} \cdot \left(\dfrac{n - L_{x_0}\,\text{offset}}{n}\right)^{1-n} & \text{for } x \leq x_0 \\
    I_{\text{tail}} + \left[\mathrm{cdf}(\mathcal{N}(\mu, \sigma), x) - \mathrm{cdf}(\mathcal{N}(\mu, \sigma), x_0)\right] & \text{for } x > x_0
\end{cases}
$$

### Quantile (Inverse CDF)

For $p \in [0, 1]$:

$$
Q(p) = \begin{cases}
    x_0 + \dfrac{n}{L_{x_0}} \left(1 - \left(\dfrac{p / N}{I_{\text{tail}} / N}\right)^{1/(1-n)}\right) & \text{for } p \leq F(x_0) \\
    \mathrm{quantile}\!\left(\mathcal{N}(\mu, \sigma), \dfrac{p - F(x_0)}{N} + \mathrm{cdf}(\mathcal{N}(\mu, \sigma), x_0)\right) & \text{for } p > F(x_0)
\end{cases}
$$

---

## 2. Double-Sided Crystal Ball Distribution

### Parameters
- $\mu \in \mathbb{R}$: mean
- $\sigma > 0$: standard deviation
- $\alpha_L > 0, n_L > 1$: left tail parameters
- $\alpha_R > 0, n_R > 1$: right tail parameters

### PDF Definition (standardized coordinates)

Let $\hat{x} = (x - \mu)/\sigma$, $x_{0L} = \mu - \alpha_L \sigma$, and $x_{0R} = \mu + \alpha_R \sigma$.

$$
f(x) = N \begin{cases}
    A_L\,(B_L - \hat{x})^{-n_L} & \text{for } \hat{x} < -\alpha_L \\
    \exp\!\left(-\dfrac{\hat{x}^2}{2}\right) & \text{for } -\alpha_L \leq \hat{x} \leq \alpha_R \\
    A_R\,(B_R + \hat{x})^{-n_R} & \text{for } \hat{x} > \alpha_R
\end{cases}
$$

with $A_L, B_L, A_R, B_R$ derived from $(\alpha_L, n_L, \alpha_R, n_R)$ to enforce value and first-derivative continuity at both joins (same construction as the one-sided case on each side).

### Equivalent implementation form (absolute coordinates)

$$
f(x) = N \begin{cases}
    G(x_{0L}) \left(\dfrac{n_L}{n_L - L_{L}\,(x - x_{0L})}\right)^{n_L} & \text{for } x < x_{0L} \\
    \mathrm{pdf}(\mathcal{N}(\mu, \sigma), x) & \text{for } x_{0L} \leq x \leq x_{0R} \\
    G(x_{0R}) \left(\dfrac{n_R}{n_R - L_{R}\,(x - x_{0R})}\right)^{n_R} & \text{for } x > x_{0R}
\end{cases}
$$

where $G(x_{0}) = \mathrm{pdf}(\mathcal{N}(\mu,\sigma), x_{0})$ and, for a Normal core,

$$
L_L = \left.\frac{d}{dx}\ln G\right|_{x_{0L}} = \frac{\alpha_L}{\sigma},
\qquad
L_R = \left.\frac{d}{dx}\ln G\right|_{x_{0R}} = -\frac{\alpha_R}{\sigma}.
$$

### Normalization Constant

$$
N = \frac{1}{I_{\text{left}} + I_{\text{core}} + I_{\text{right}}}
$$

where:
- $I_{\text{left}} = \dfrac{G(x_{0L})}{L_L} \cdot \dfrac{n_L}{n_L-1}$
- $I_{\text{core}} = \mathrm{cdf}(\mathcal{N}(\mu, \sigma), x_{0R}) - \mathrm{cdf}(\mathcal{N}(\mu, \sigma), x_{0L})$
- $I_{\text{right}} = \dfrac{G(x_{0R})}{-L_R} \cdot \dfrac{n_R}{n_R-1} = \dfrac{G(x_{0R})\,\sigma}{\alpha_R}\cdot\dfrac{n_R}{n_R-1}$

### CDF

$$
F(x) = N \begin{cases}
    \dfrac{G(x_{0L})}{L_L} \cdot \dfrac{n_L}{n_L-1} \cdot \left(\dfrac{n_L - L_L\,(x - x_{0L})}{n_L}\right)^{1-n_L} & \text{for } x < x_{0L} \\
    F(x_{0L}) + \left[\mathrm{cdf}(\mathcal{N}(\mu, \sigma), x) - \mathrm{cdf}(\mathcal{N}(\mu, \sigma), x_{0L})\right] & \text{for } x_{0L} \leq x \leq x_{0R} \\
    F(x_{0R}) - \dfrac{G(x_{0R})}{-L_R} \cdot \dfrac{n_R}{n_R-1} \cdot \left[\left(\dfrac{n_R - L_R\,(x - x_{0R})}{n_R}\right)^{1-n_R} - 1\right] & \text{for } x > x_{0R}
\end{cases}
$$

---

## Implementation Notes

The code uses a unified `CrystalBallTail` structure parameterized by:
- `G_x0`: core PDF value at the transition point ($G(x_0)$)
- `N`: power-law exponent ($n$)
- `L_x0`: $\left.\dfrac{d}{dx}\ln f_{\text{core}}\right|_{x_0}$ in **absolute** coordinates ($\alpha/\sigma$ on the left, $-\alpha/\sigma$ on the right for a Normal core)
- `x0`: absolute transition point

**Important:** $L_{x_0}$ is not the same as $\alpha$ from the docstring. The standardized log-slope at the left join is $\left.\dfrac{d}{d\hat{x}}\ln f\right|_{\hat{x}=-\alpha} = \alpha$, while the absolute log-slope is $L_{x_0} = \alpha/\sigma$. Confusing the two breaks $C^1$ matching whenever $\sigma \neq 1$.
