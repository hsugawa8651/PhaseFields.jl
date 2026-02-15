# PhaseFields.jl

Phase Field method simulation package for materials science.

## Features

- **Interface modeling**: Interpolation functions, double-well potentials, anisotropy
- **Phase field models**: Allen-Cahn, Cahn-Hilliard, KKS, WBM, Thermal coupling
- **CALPHAD coupling**: Optional integration with [OpenCALPHAD.jl](https://github.com/hsugawa8651/OpenCALPHAD.jl)
- **DifferentialEquations.jl integration**: Method of Lines with adaptive time stepping
- **AD-compatible design**: Using [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hsugawa8651/PhaseFields.jl")
```

For CALPHAD coupling:
```julia
Pkg.add(url="https://github.com/hsugawa8651/OpenCALPHAD.jl")
```

## Quick Start

### Basic Allen-Cahn simulation

```julia
using PhaseFields
using OrdinaryDiffEq

# Create model and grid
model = AllenCahnModel(τ=1.0, W=1.0, m=0.1)
grid = UniformGrid1D(100, 10.0)

# Initial condition: tanh interface
x0 = grid.L / 2
φ0 = [0.5 * (1 - tanh((x - x0) / 1.0)) for x in grid.x]

# Create and solve ODE problem
prob = create_allen_cahn_problem(model, grid, NeumannBC(), φ0, (0.0, 10.0))
sol = solve(prob, Tsit5())
```

### With CALPHAD thermodynamics

```julia
using PhaseFields
using OpenCALPHAD  # Activates CALPHAD extension

# Load thermodynamic database
db = read_tdb("AgCu.TDB")

# Create CALPHAD-coupled model
T = 1000.0  # Temperature [K]
x = 0.3     # Composition (Cu mole fraction)
model = create_calphad_allen_cahn(db, T, x, "FCC_A1", "LIQUID")

# Driving force from CALPHAD
ΔG = calphad_driving_force(db, T, x, "FCC_A1", "LIQUID")
```

## Package Structure

```
PhaseFields.jl/
├── src/
│   ├── PhaseFields.jl       # Main module
│   ├── types.jl             # Type definitions
│   ├── interpolation.jl     # h(φ), g(φ) functions
│   ├── anisotropy.jl        # Anisotropy functions
│   ├── models/              # Phase field models
│   │   ├── allen_cahn.jl
│   │   ├── cahn_hilliard.jl
│   │   ├── kks.jl
│   │   ├── wbm.jl
│   │   └── thermal.jl
│   └── integration/
│       └── diffeq.jl        # DifferentialEquations.jl integration
└── ext/
    └── OpenCALPHADExt.jl    # CALPHAD coupling extension
```

## Contents

```@contents
Pages = ["getting_started.md", "api.md"]
Depth = 2
```
