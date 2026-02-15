# Thermal Phase Field Model

## Overview

The thermal phase field model couples solidification with heat transfer,
capturing latent heat release and thermal gradients. It is essential for
modeling realistic solidification where temperature evolution affects
interface kinetics.

## Mathematical Formulation

### Phase Field Equation

```math
\tau \frac{\partial \phi}{\partial t} = W^2 \nabla^2 \phi - g'(\phi) + \lambda h'(\phi) u
```

### Heat Equation

```math
\frac{\partial u}{\partial t} = \alpha \nabla^2 u + \frac{1}{2} \frac{\partial \phi}{\partial t}
```

### Dimensionless Temperature

```math
u = \frac{T - T_m}{L / C_p}
```

where:
- T: Physical temperature [K]
- T_m: Melting temperature [K]
- L: Latent heat [J/m³]
- C_p: Heat capacity [J/(m³·K)]
- α: Thermal diffusivity [m²/s]

### Stefan Number

```math
St = \frac{C_p \Delta T}{L}
```

## Quick Example

### Basic Usage

```julia
using PhaseFields

# Create model (nickel-like parameters)
model = ThermalPhaseFieldModel(
    τ = 1e-6,       # Relaxation time [s]
    W = 1e-6,       # Interface width [m]
    λ = 2.0,        # Coupling strength
    α = 1e-5,       # Thermal diffusivity [m²/s]
    L = 2.35e9,     # Latent heat [J/m³]
    Cp = 5.42e6,    # Heat capacity [J/(m³·K)]
    Tm = 1728.0     # Melting point [K]
)

# Convert temperature
T = 1700.0  # Physical temperature [K]
u = dimensionless_temperature(model, T)

# Convert back
T_back = physical_temperature(model, u)
```

### With DifferentialEquations.jl

```julia
using PhaseFields
using OrdinaryDiffEq

model = ThermalPhaseFieldModel(
    τ=1e-6, W=1e-6, λ=2.0,
    α=1e-5, L=2.35e9, Cp=5.42e6, Tm=1728.0
)

grid = UniformGrid1D(N=101, L=1e-4)
bc_φ = NeumannBC()
bc_u = NeumannBC()

# Initial conditions
φ0 = [x < grid.L/2 ? 1.0 : 0.0 for x in grid.x]  # Solid seed
u0 = fill(-0.05, grid.N)  # Slight undercooling

# Create and solve
prob = create_thermal_problem(model, grid, bc_φ, bc_u, φ0, u0, (0.0, 1e-3))
sol = solve(prob, QNDF(autodiff=false))

# Extract solution
φ_hist, u_hist = extract_thermal_solution(sol, grid.N)
```

### Stefan Problem

The model can simulate the classical Stefan problem where a planar
interface advances into an undercooled melt, releasing latent heat.

```julia
# Undercooling determines interface velocity
ΔT = 10.0  # [K]
u0 = dimensionless_temperature(model, model.Tm - ΔT)
```

## API Reference

```@docs
ThermalPhaseFieldModel
dimensionless_temperature
physical_temperature
stefan_number
thermal_phase_rhs
thermal_heat_rhs
thermal_stability_dt
ThermalODEParams
thermal_ode!
create_thermal_problem
extract_thermal_solution
```

## See Also

- [304_thermal_solidification.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/304_thermal_solidification.jl) - Thermal + phase field coupling
- [305_stefan_problem_1d.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/305_stefan_problem_1d.jl) - Classical Stefan problem
- [351_diffeq_thermal_solidification.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/351_diffeq_thermal_solidification.jl) - With DifferentialEquations.jl
