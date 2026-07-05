# DistributionsHEP.jl

`DistributionsHEP.jl` is a package extending the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package with High Energy Physics (HEP) specific distributions.

This package specializes in distributions with a closed-form or special-algorithm CDFs,
any distributions requiring numerical integration can be wrapped ith [`NumericalDistributions.jl`](https://github.com/mmikhasenko/NumericalDistributions.jl).


## Implemented Distributions

- **Chebyshev**: Chebyshev polynomial distribution
- **ArgusBG**: ARGUS background distribution
- **CrystalBall**, **DoubleCrystalBall**: One-sided and two-sided Crystal Ball distribution with Gaussian core and power-law tail
- **HyperbolicSecant**: Hyperbolic secant distribution with location-scale family similar to normal distribution but with fatter tails
- **BifurcatedGaussian**: Asymmetric Gaussian distribution with different scale parameters on left and right sides
- **ExtendedMixtureModel**: Mixture-like model for extended likelihood fits where component weights are expected yields instead of normalized probabilities

Mathematical derivations for Crystal Ball distributions are in [`docs/CrystalBallMath.md`](docs/CrystalBallMath.md), formulas for ARGUS background distribution are in [`docs/ArgusBG.md`](docs/ArgusBG.md), and derivations for Bifurcated Gaussian (including skewness formula) are in [`docs/BifurcatedGaussianMath.md`](docs/BifurcatedGaussianMath.md).

## Installation

To install `DistributionsHEP.jl`, use Julia's built-in package manager.
In the Julia REPL, type:

```julia
julia> using Pkg
julia> Pkg.add("DistributionsHEP")
```

Or in Pkg mode (`]`):
```julia
pkg> add DistributionsHEP
```

## Usage

```julia
using DistributionsHEP

# Chebyshev distribution
c0, c1, c2 = 1.0, 0.2, 0.3
a, b = 0.0, 10.0
cheb = Chebyshev([c0, c1, c2], a, b)

# Use standard Distributions.jl API
pdf(cheb, 3.3)
cdf(cheb, 3.3)
rand(cheb)
```

The rest of the interface (`pdf`, `cdf`, `rand`, etc.) follows the standard `Distributions.jl` API.

### Extended mixture models

`ExtendedMixtureModel` is useful for extended likelihood fits where fitted
component weights are expected event yields, not normalized mixture fractions.
For components `f_k` and yields `y_k`, call the model directly to evaluate

```math
f_{\mathrm{ext}}(x) = \sum_k y_k f_k(x).
```

```julia
using Distributions
using DistributionsHEP

components = [Normal(-1.0, 0.5), Normal(1.0, 0.25)]
model = ExtendedMixtureModel(components, [20.0, 5.0])

model(0.1)
extended_negative_log_likelihood(model, [-1.0, -0.5, 0.9])
```

`ExtendedMixtureModel` deliberately does not define `pdf(model, x)` or
`logpdf(model, x)`, because the yield-weighted density is not normalized.
Use `MixtureModel(model)` when a normalized mixture is needed. The helper
accessors `yields(model)` and `total_yield(model)` are available as
`DistributionsHEP.yields` and `DistributionsHEP.total_yield`, or by importing
them explicitly.

## Contributing

We welcome contributions to improve this project! If you're interested in contributing, please:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

You can also open an issue if you encounter any problems or have feature suggestions.

## Acknowledgements

This project is part of the JuliaHEP ecosystem, which is developed by a community of scientists
and developers passionate about using Julia for high-energy physics. We are grateful to
all contributors and users who support the growth of this project.

## License

`DistributionsHEP.jl` is licensed under the MIT License.
See the [LICENSE](https://github.com/JuliaHEP/DistributionsHEP.jl/blob/main/LICENSE) file for more details.
