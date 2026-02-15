# Cahn-Hilliard Model

## Overview

The Cahn-Hilliard equation describes the evolution of a conserved order parameter.
It is used for modeling spinodal decomposition, phase separation, and coarsening.

The concentration field c represents the local composition (mole fraction).

## Mathematical Formulation

The Cahn-Hilliard equation:

```math
\frac{\partial c}{\partial t} = M \nabla^2 \mu
```

where the chemical potential μ is:

```math
\mu = \frac{df}{dc} - \kappa \nabla^2 c
```

Parameters:
- M: Mobility coefficient
- κ: Gradient energy coefficient
- f(c): Bulk free energy density (e.g., double-well)

Combining these gives the 4th-order equation:

```math
\frac{\partial c}{\partial t} = M \nabla^2 \left( \frac{df}{dc} \right) - M \kappa \nabla^4 c
```

## Quick Example

### Basic Usage

```julia
using PhaseFields

# Create model and free energy
model = CahnHilliardModel(M=1.0, κ=1.0)
f = DoubleWellFreeEnergy(A=1.0, c_eq_α=0.2, c_eq_β=0.8)

# Compute chemical potential
c = 0.5
∇²c = -0.01
μ = cahn_hilliard_chemical_potential(f, model, c, ∇²c)

# Compute concentration change rate
∇²μ = 0.001
dcdt = cahn_hilliard_rhs(model, ∇²μ)
```

### Free Energy Functions

```julia
# Double-well free energy: f(c) = A(c - c_α)²(c - c_β)²
f = DoubleWellFreeEnergy(A=1.0, c_eq_α=0.2, c_eq_β=0.8)

# Evaluate
energy = free_energy_density(f, 0.5)
μ_bulk = chemical_potential_bulk(f, 0.5)
```

### Interface Properties

```julia
# Estimate interface width
W = cahn_hilliard_interface_width(model, f)

# Stability constraint for explicit time stepping
dt_max = cahn_hilliard_stability_dt(model, f, dx)
```

## API Reference

```@docs
CahnHilliardModel
DoubleWellFreeEnergy
free_energy_density
chemical_potential_bulk
cahn_hilliard_chemical_potential
cahn_hilliard_rhs
cahn_hilliard_interface_width
cahn_hilliard_stability_dt
```

## See Also

- [201_spinodal_1d.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/201_spinodal_1d.jl) - 1D spinodal decomposition
- [401_ostwald_ripening.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/401_ostwald_ripening.jl) - Ostwald ripening (multi-particle coarsening)
