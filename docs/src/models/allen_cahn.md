# Allen-Cahn Model

## Overview

The Allen-Cahn equation describes the evolution of a non-conserved order parameter.
It is widely used for modeling solidification, grain growth, and phase transformations.

The phase field variable φ represents:
- φ = 1: Solid phase
- φ = 0: Liquid phase
- 0 < φ < 1: Diffuse interface

## Mathematical Formulation

The Allen-Cahn equation:

```math
\tau \frac{\partial \phi}{\partial t} = W^2 \nabla^2 \phi - g'(\phi) + m \cdot h'(\phi) \cdot \Delta G
```

where:
- τ: Relaxation time [s]
- W: Interface width parameter
- g(φ): Double-well potential (e.g., φ²(1-φ)²)
- h(φ): Interpolation function (e.g., 3φ² - 2φ³)
- ΔG: Thermodynamic driving force [J/mol]
- m: Scaling factor for driving force

## Quick Example

### Basic Usage

```julia
using PhaseFields

# Create model
model = AllenCahnModel(τ=1.0, W=1.0, m=0.1)

# Compute right-hand side
φ = 0.5
∇²φ = -0.1  # Laplacian
ΔG = -100.0  # Driving force (negative = solidification)

dφdt = allen_cahn_rhs(model, φ, ∇²φ, ΔG)
```

### With DifferentialEquations.jl

```julia
using PhaseFields
using OrdinaryDiffEq

# Model and grid
model = AllenCahnModel(τ=1.0, W=0.05)
grid = UniformGrid1D(N=101, L=1.0)
bc = NeumannBC()

# Initial condition: tanh interface
x0 = 0.3
φ0 = [0.5 * (1 - tanh((x - x0) / (2 * model.W))) for x in grid.x]

# Create and solve ODE problem
prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 2.0))
sol = solve(prob, Tsit5())
```

## API Reference

```@docs
AllenCahnModel
allen_cahn_rhs
AllenCahnODEParams
allen_cahn_ode!
create_allen_cahn_problem
```

## See Also

- [101_allen_cahn_1d.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/101_allen_cahn_1d.jl) - Basic 1D evolution
- [151_diffeq_allen_cahn.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/151_diffeq_allen_cahn.jl) - With DifferentialEquations.jl callbacks
