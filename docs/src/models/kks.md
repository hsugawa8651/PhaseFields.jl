# KKS Model (Kim-Kim-Suzuki)

## Overview

The KKS model handles multi-phase, multi-component systems with local thermodynamic
equilibrium at the interface. It avoids the artificial interface energy problem
by partitioning the composition between phases.

Key feature: Local equilibrium constraint ensures equal diffusion potentials
between phases at each point.

## Mathematical Formulation

### Phase Field Equation

```math
\tau \frac{\partial \phi}{\partial t} = W^2 \nabla^2 \phi - g'(\phi) + h'(\phi) \cdot \Delta\omega
```

### Concentration Equation

```math
\frac{\partial c}{\partial t} = \nabla \cdot \left( M(\phi) \nabla \mu \right)
```

### Local Equilibrium Constraints

At each point, the phase compositions (c_s, c_l) satisfy:

```math
c = h(\phi) c_s + (1 - h(\phi)) c_l \quad \text{(mass conservation)}
```

```math
\mu_s(c_s) = \mu_l(c_l) \quad \text{(equal chemical potential)}
```

### Grand Potential Difference

```math
\Delta\omega = f_s(c_s) - f_l(c_l) - \mu (c_s - c_l)
```

## Quick Example

### Basic Usage with Parabolic Free Energy

```julia
using PhaseFields

# Create model
model = KKSModel(τ=1.0, W=1.0, m=1.0, M_s=1.0, M_l=10.0)

# Parabolic free energies for testing
f_s = ParabolicFreeEnergy(A=1.0, c_eq=0.2)  # Solid
f_l = ParabolicFreeEnergy(A=1.0, c_eq=0.8)  # Liquid

# Partition composition at interface
c_avg = 0.5  # Average composition
φ = 0.5      # At interface

c_s, c_l, μ, converged = kks_partition(c_avg, φ, f_s, f_l)
println("Solid: c_s = $c_s, Liquid: c_l = $c_l")

# Grand potential difference
Δω = kks_grand_potential_diff(f_s, f_l, c_s, c_l, μ)
```

### With CALPHAD Thermodynamics

```julia
using PhaseFields
using OpenCALPHAD

# Load database
db = read_tdb("AgCu.TDB")

# Create CALPHAD-coupled KKS model
T = 1000.0  # Temperature [K]
model, f_s, f_l = create_calphad_kks_model(db, T, "FCC_A1", "LIQUID")

# Partition with real thermodynamics
c_s, c_l, μ, converged = kks_partition(c_avg, φ, f_s, f_l)
```

## API Reference

```@docs
KKSModel
ParabolicFreeEnergy
free_energy
chemical_potential
d2f_dc2
kks_partition
kks_grand_potential_diff
kks_phase_rhs
kks_concentration_rhs
kks_mobility
kks_interface_width
```

## See Also

- [301_kks_solidification.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/301_kks_solidification.jl) - KKS solidification simulation
- [381_calphad_coupling_demo.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/381_calphad_coupling_demo.jl) - KKS with CALPHAD thermodynamics
