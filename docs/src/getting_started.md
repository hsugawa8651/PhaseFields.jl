# Getting Started

This guide covers the basic concepts of PhaseFields.jl.

## Phase Field Method Overview

The phase field method represents interfaces using a continuous order parameter φ:
- φ = 1: Solid phase
- φ = 0: Liquid phase
- 0 < φ < 1: Diffuse interface region

This approach avoids explicit interface tracking and naturally handles
topological changes like merging and splitting.

## Available Models

PhaseFields.jl provides several phase field models for different applications:

| Model | Application | Order Parameter |
|-------|-------------|-----------------|
| [Allen-Cahn](models/allen_cahn.md) | Solidification, grain growth | Non-conserved |
| [Cahn-Hilliard](models/cahn_hilliard.md) | Spinodal decomposition | Conserved |
| [KKS](models/kks.md) | Multi-component alloys | Local equilibrium |
| [WBM](models/wbm.md) | Dilute alloys | Mixture rule |
| [Thermal](models/thermal.md) | Solidification with heat | Coupled φ-T |

## Quick Example

```julia
using PhaseFields
using OrdinaryDiffEq

# 1. Create model
model = AllenCahnModel(τ=1.0, W=0.05)

# 2. Set up spatial grid
grid = UniformGrid1D(N=101, L=1.0)
bc = NeumannBC()

# 3. Initial condition
φ0 = [0.5 * (1 - tanh((x - 0.3) / 0.1)) for x in grid.x]

# 4. Create ODE problem
prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 2.0))

# 5. Solve
sol = solve(prob, Tsit5())
```

## Key Concepts

### Interpolation and Double-Well Functions

All models use:
- **h(φ)**: Interpolation function (0→1 transition)
- **g(φ)**: Double-well potential (energy barrier)

```julia
# Standard choices
h = h_polynomial(φ)  # 3φ² - 2φ³
g = g_standard(φ)    # φ²(1-φ)²
```

See [Common API Reference](reference/common.md) for details.

### DifferentialEquations.jl Integration

PhaseFields.jl uses Method of Lines for time integration:

```julia
using OrdinaryDiffEq

# Explicit solver (fast, non-stiff)
sol = solve(prob, Tsit5())

# Implicit solver (stiff problems)
sol = solve(prob, QNDF(autodiff=false))
```

See [DifferentialEquations.jl Integration](integration/diffeq.md) for details.

### CALPHAD Coupling (Optional)

For realistic thermodynamics:

```julia
using PhaseFields
using OpenCALPHAD  # Activates extension

db = read_tdb("AgCu.TDB")
ΔG = calphad_driving_force(db, 1000.0, 0.3, "FCC_A1", "LIQUID")
```

See [CALPHAD Coupling](integration/calphad.md) for details.

## Next Steps

- Explore specific [Models](models/allen_cahn.md)
- Learn about [DifferentialEquations.jl Integration](integration/diffeq.md)
- Set up [CALPHAD Coupling](integration/calphad.md)
