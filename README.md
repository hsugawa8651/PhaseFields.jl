# PhaseFields.jl

[![CI](https://github.com/hsugawa8651/PhaseFields.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/hsugawa8651/PhaseFields.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/hsugawa8651/PhaseFields.jl/actions/workflows/Docs.yml/badge.svg)](https://hsugawa8651.github.io/PhaseFields.jl/dev/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18649171.svg)](https://doi.org/10.5281/zenodo.18649171)

A Julia package for phase field simulations with CALPHAD thermodynamics coupling.

## Features (v0.1)

* Multiple phase field models (Allen-Cahn, Cahn-Hilliard, KKS, WBM, Thermal)
* 1D and 2D simulations (FDM built-in, FEM via Gridap.jl extension)
* Unified problem/solve API with automatic FDM/FEM backend selection
* DifferentialEquations.jl integration (adaptive time stepping, callbacks)
* OpenCALPHAD.jl CALPHAD coupling via Package Extension
* Automatic differentiation via DifferentiationInterface.jl

## Models

| Model | Type | Use Case | Dimension |
|-------|------|----------|-----------|
| Allen-Cahn | Non-conserved | Phase transitions, interface migration | 1D, 2D |
| Cahn-Hilliard | Conserved | Spinodal decomposition, coarsening | 1D |
| KKS | Local equilibrium | Multi-component solidification | 1D |
| WBM | Dilute alloy | Binary alloy solidification | 1D |
| Thermal | Heat-coupled | Stefan problem, latent heat release | 1D, 2D |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hsugawa8651/PhaseFields.jl")
```

## Quick Example

```julia
using PhaseFields
using OrdinaryDiffEq

# Create model and grid
model = AllenCahnModel(τ=1.0, W=0.1)
grid = UniformGrid1D(N=100, L=1.0)

# Initial condition: step function
φ0 = [x < 0.5 ? 1.0 : 0.0 for x in grid.x]

# Create and solve problem
problem = PhaseFieldProblem(
    model=model,
    domain=grid,
    φ0=φ0,
    tspan=(0.0, 1.0)
)
sol = solve(problem, Tsit5())

# Access solution
φ_final = sol.u[end]
```

### Low-level API

```julia
# Direct RHS computation (for custom integrators)
model = AllenCahnModel(τ=1.0, W=1.0, m=0.1)
φ = 0.5       # Order parameter
∇²φ = -0.1    # Laplacian
ΔG = -100.0   # Driving force (negative = solidification)

dφdt = allen_cahn_rhs(model, φ, ∇²φ, ΔG)
```

## Optional Dependencies

| Package | Purpose |
|---------|---------|
| OrdinaryDiffEq | Adaptive time stepping, implicit solvers |
| Gridap | 2D/3D finite element method |
| OpenCALPHAD | CALPHAD thermodynamics coupling |

## Examples

Examples use a 3-digit numbering system:

| 1st Digit | Physical Phenomenon |
|-----------|---------------------|
| 1 | Allen-Cahn |
| 2 | Cahn-Hilliard |
| 3 | Solidification |
| 4 | Coarsening |
| 9 | Benchmarks |

| 2nd Digit | Coupling |
|-----------|----------|
| 0 | Standalone |
| 5 | DifferentialEquations.jl |
| 8 | OpenCALPHAD.jl |

```bash
# Run examples
julia --project=. examples/101_allen_cahn_1d.jl
julia --project=. examples/301_kks_solidification.jl
julia --project=. examples/381_calphad_coupling_demo.jl
```

See [examples/000_examples.md](examples/000_examples.md) for full list.

## Documentation

Build documentation locally:

```bash
cd docs
julia --project make.jl
# Open docs/build/index.html
```

## Testing

```bash
julia --project -e 'using Pkg; Pkg.test()'
# 644 tests
```

## Related Packages

* [OpenCALPHAD.jl](https://github.com/hsugawa8651/OpenCALPHAD.jl) - CALPHAD thermodynamic calculations

## Contributing

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/hsugawa8651/PhaseFields.jl/issues).
Before opening a pull request, start an issue or a discussion on the topic.
This project follows the [Julia Community Standards](https://julialang.org/community/standards/).

## References

* Allen & Cahn, Acta Metallurgica 27, 1085 (1979)
* Cahn & Hilliard, J. Chem. Phys. 28, 258 (1958)
* Wheeler, Boettinger, McFadden, Phys. Rev. A 45, 7424 (1992)
* Kim, Kim, Suzuki, Phys. Rev. E 60, 7186 (1999)
* PFHub Benchmarks: https://pages.nist.gov/pfhub/

## License

MIT License

See [LICENSE](LICENSE) for details.
