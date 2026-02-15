# CALPHAD Coupling

## Overview

PhaseFields.jl provides optional integration with [OpenCALPHAD.jl](https://github.com/hsugawa8651/OpenCALPHAD.jl)
for realistic thermodynamic calculations. This coupling is implemented as a
Julia package extension that loads automatically when OpenCALPHAD is imported.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hsugawa8651/OpenCALPHAD.jl")
```

## Basic Usage

```julia
using PhaseFields
using OpenCALPHAD  # Activates extension automatically

# Load thermodynamic database
db = read_tdb("AgCu.TDB")

# Calculate driving force
T = 1000.0  # Temperature [K]
x = 0.3     # Composition (Cu mole fraction)
ΔG = calphad_driving_force(db, T, x, "FCC_A1", "LIQUID")
```

## CALPHAD-Coupled Models

### Allen-Cahn with CALPHAD

```julia
using PhaseFields
using OpenCALPHAD

db = read_tdb("AgCu.TDB")
T = 1000.0
x = 0.3

# Create coupled model
model = create_calphad_allen_cahn(db, T, x, "FCC_A1", "LIQUID";
    τ = 1.0,
    W = 1.0,
    m = 1e-4  # Scaling factor for ΔG
)

# The model caches the driving force from CALPHAD
```

### KKS with CALPHAD

```julia
db = read_tdb("AgCu.TDB")
T = 1000.0

# Create model with CALPHAD free energies
model, f_s, f_l = create_calphad_kks_model(db, T, "FCC_A1", "LIQUID";
    τ = 1.0,
    W = 1.0,
    m = 1.0,
    M_s = 1.0,
    M_l = 10.0
)

# Partition with real thermodynamics
c_avg = 0.3
φ = 0.5
c_s, c_l, μ, converged = kks_partition(c_avg, φ, f_s, f_l)
```

### WBM with CALPHAD

```julia
db = read_tdb("AgCu.TDB")
T = 1000.0

model, f_s, f_l = create_calphad_wbm_model(db, T, "FCC_A1", "LIQUID";
    M_φ = 1.0,
    κ = 1.0,
    W = 1.0,
    D_s = 1e-13,
    D_l = 1e-9
)
```

## Thermodynamic Functions

### Driving Force

```julia
# Gibbs energy difference: ΔG = G_solid - G_liquid
# Negative ΔG means solid is stable
ΔG = calphad_driving_force(db, T, x, "FCC_A1", "LIQUID")
```

### Chemical Potential

```julia
# Get chemical potentials for a phase
μ = calphad_chemical_potential(db, "FCC_A1", T, x)
```

### Diffusion Potential

```julia
# Second derivative d²G/dx² (thermodynamic factor)
d2G = calphad_diffusion_potential(db, "FCC_A1", T, x)
```

## API Reference

```@docs
AbstractCALPHADCoupledModel
calphad_driving_force
calphad_chemical_potential
calphad_diffusion_potential
create_calphad_allen_cahn
create_calphad_kks_model
create_calphad_wbm_model
```

## See Also

- [381_calphad_coupling_demo.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/381_calphad_coupling_demo.jl) - Complete CALPHAD coupling demo with plots
