# DistributionsHEP.jl

`DistributionsHEP.jl` is a package extending the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package with High Energy Physics (HEP) specific distributions.

This package specializes in distributions with a closed-form or special-algorithm CDFs,
any distributions requiring numerical integration can be wrapped ith [`NumericalDistributions.jl`](https://github.com/mmikhasenko/NumericalDistributions.jl).


## Implemented Distributions

- **Chebyshev**: Chebyshev polynomial distribution
- **ArgusBG**: ARGUS background distribution
- **CrystalBall**, **DoubleCrystalBall**: One-sided and two-sided Crystal Ball distribution with Gaussian core and power-law tail

Mathematical derivations for Crystal Ball distributions are in [`docs/CrystalBallMath.md`](docs/CrystalBallMath.md), and formulas for ARGUS background distribution are in [`docs/ArgusBG.md`](docs/ArgusBG.md).

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
