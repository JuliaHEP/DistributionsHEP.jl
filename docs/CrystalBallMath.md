# Crystal Ball Distribution (Left-sided)

## PDF Definition
Let \( x \in \mathbb{R} \), \( \mu \in \mathbb{R} \), \( \sigma > 0 \), \( \alpha > 0 \), \( n > 1 \).

Define \( \hat{x} = \frac{x - \mu}{\sigma} \).

The probability density function (PDF) is:

\[
    f(x; \mu, \sigma, \alpha, n) =
    \begin{cases}
        N \cdot \exp\left(-\frac{\hat{x}^2}{2}\right) & \text{for } \hat{x} > -\alpha \\
        N \cdot A \cdot (B - \hat{x})^{-n} & \text{for } \hat{x} \leq -\alpha
    \end{cases}
\]

where:
- \( N \) is the normalization constant
- \( A \) and \( B \) are tail parameters

## Tail Parameters
To ensure continuity and differentiability at the transition point \( \hat{x} = -\alpha \):

\[
A = \left(\frac{n}{\alpha}\right)^n \exp\left(-\frac{\alpha^2}{2}\right)
\]
\[
B = \frac{n}{\alpha} - \alpha
\]

## Normalization Constant
Let
\[
C = \frac{n}{\alpha (n-1)} \exp\left(-\frac{\alpha^2}{2}\right)
\]
\[
D = \sqrt{\frac{\pi}{2}} \left[1 + \operatorname{erf}\left(\frac{\alpha}{\sqrt{2}}\right)\right]
\]
Then
\[
N = \frac{1}{\sigma (C + D)}
\]

## CDF
Let
\[
\text{CDF at } \hat{x} = -\alpha: \quad F_{-\alpha} = N A \frac{(B + \alpha)^{1-n}}{n-1}
\]

The cumulative distribution function (CDF) is:
\[
F(x) =
\begin{cases}
    N A \frac{(B - \hat{x})^{1-n}}{n-1} & \text{for } \hat{x} \leq -\alpha \\
    F_{-\alpha} + N \sigma \sqrt{\frac{\pi}{2}} \left[ \operatorname{erf}\left(\frac{\hat{x}}{\sqrt{2}}\right) + \operatorname{erf}\left(\frac{\alpha}{\sqrt{2}}\right) \right] & \text{for } \hat{x} > -\alpha
\end{cases}
\]

## Quantile (Inverse CDF)
Let \( p \in [0, 1] \).

- If \( p \leq F_{-\alpha} \):
\[
    \hat{x} = B - \left( \frac{p (n-1)}{N A} \right)^{1/(1-n)}
\]
- If \( p > F_{-\alpha} \):
\[
    \hat{x} = \sqrt{2} \operatorname{erf}^{-1}\left( \frac{p - F_{-\alpha}}{N \sigma \sqrt{\pi/2}} - \operatorname{erf}\left(\frac{\alpha}{\sqrt{2}}\right) \right)
\]

Then \( x = \mu + \sigma \hat{x} \). 