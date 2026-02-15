# WBM Model (Wheeler-Boettinger-McFadden)

## Overview

The WBM model is designed for dilute alloy solidification. It couples phase field
evolution with solute diffusion using a mixture rule for free energy.

Reference: Wheeler, Boettinger, McFadden, Phys. Rev. A 45, 7424 (1992)

## Mathematical Formulation

### Total Free Energy

```math
F = \int \left[ f(\phi, c, T) + \frac{\kappa}{2} |\nabla\phi|^2 \right] dV
```

### Bulk Free Energy (Mixture Rule)

```math
f = h(\phi) f_S(c, T) + (1 - h(\phi)) f_L(c, T) + W g(\phi)
```

### Phase Field Equation

```math
\frac{\partial \phi}{\partial t} = M_\phi \left[ \kappa \nabla^2 \phi - h'(\phi)(f_S - f_L) - W g'(\phi) \right]
```

### Concentration Equation

```math
\frac{\partial c}{\partial t} = \nabla \cdot \left[ D(\phi) \nabla c \right]
```

where diffusivity interpolates between phases:

```math
D(\phi) = h(\phi) D_S + (1 - h(\phi)) D_L
```

## Quick Example

### Basic Usage

```julia
using PhaseFields

# Create model
model = WBMModel(
    M_φ = 1.0,      # Phase field mobility
    κ = 1.0,        # Gradient coefficient
    W = 1.0,        # Barrier height
    D_s = 1e-13,    # Solid diffusivity [m²/s]
    D_l = 1e-9      # Liquid diffusivity [m²/s]
)

# Free energies (parabolic for testing)
f_s = ParabolicFreeEnergy(A=1.0, c_eq=0.2)
f_l = ParabolicFreeEnergy(A=1.0, c_eq=0.8)

# Compute driving force
φ = 0.5
c = 0.5
Δf = wbm_driving_force(f_s, f_l, c)

# Interpolated diffusivity
D = wbm_diffusivity(model, φ)
```

### Interface Properties

```julia
# Interface width (from κ and W)
W_int = wbm_interface_width(model)

# Interface energy
σ = wbm_interface_energy(model)
```

### With CALPHAD

```julia
using PhaseFields
using OpenCALPHAD

db = read_tdb("AgCu.TDB")
T = 1000.0

model, f_s, f_l = create_calphad_wbm_model(db, T, "FCC_A1", "LIQUID")
```

## API Reference

```@docs
WBMModel
wbm_bulk_free_energy
wbm_chemical_potential
wbm_driving_force
wbm_phase_rhs
wbm_concentration_rhs
wbm_diffusivity
wbm_interface_width
wbm_interface_energy
```

## See Also

- [302_wbm_solidification.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/302_wbm_solidification.jl) - WBM solidification simulation
- [303_wbm_wheeler1992.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/303_wbm_wheeler1992.jl) - Validation vs Wheeler 1992 paper
