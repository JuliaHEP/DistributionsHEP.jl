# DistributionsHEP.jl

`DistributionsHEP.jl` is a package extending the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package with HEP specific distributions. 
Whereas the package `Distributions.jl` already provides a large collection of common distributions out of the box, we use in HEP special distributions that make sense to be grouped in tis package. 

Generally, you don't have to implement every API method listed in the documentation. We just need to implement a small number of internal methods that will be used by `Distributions.jl` to provide all the user-end API methods.

## Implemented Distributions

- **ChebyshevDist**: Chebychev polynomial distribution
- **ArgusBGDist**: Distribution describing the ARGUS background
- **CrystalBall**: One-sided Crystal Ball distribution with power-law tail
- **DoubleCrystalBall**: Two-sided Crystal Ball distribution with power-law tails on both sides of a Gaussian core

Mathematical derivations for both the one-sided and two-sided Crystal Ball distributions are found in `docs/CrystalBallMath.md`.

## Installation

To install `DistributionsHEP.jl`, use Julia's built-in package manager. In the Julia REPL, type:

```julia
julia> using Pkg
julia> Pkg.add("DistributionsHEP")
```

Alternatively, you can enter the Pkg mode by pressing `]` in the REPL, then type:

```julia
pkg> add DistributionsHEP
```

## Usage

Once installed, you can begin using `DistributionsHEP.jl` by importing the package:

```julia
using DistributionsHEP

c0, c1, c2 = 1.0, 0.2, 0.3
a, b = 0.0, 10.0

cheb = Chebyshev([c0, c1, c2], a, b) # default values for [a,b] are [-1,1]
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
