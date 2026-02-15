# DifferentialEquations.jl Integration

## Overview

PhaseFields.jl integrates with [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
using the Method of Lines approach:
1. Discretize space with finite differences
2. Convert PDE to system of ODEs
3. Solve with adaptive time stepping

## Grid and Boundary Conditions

### Uniform Grid

```julia
using PhaseFields

# Create 1D uniform grid
grid = UniformGrid1D(N=101, L=1.0)  # 101 points, length 1.0

# Access properties
grid.N    # Number of points
grid.L    # Domain length
grid.dx   # Grid spacing
grid.x    # Coordinate array
```

### Boundary Conditions

```julia
# Neumann (zero-flux)
bc = NeumannBC()

# Dirichlet (fixed values)
bc = DirichletBC(left=0.0, right=1.0)

# Periodic
bc = PeriodicBC()
```

### Laplacian Operator

```julia
# In-place computation
∇²φ = zeros(grid.N)
laplacian_1d!(∇²φ, φ, grid.dx, bc)

# Allocating version
∇²φ = laplacian_1d(φ, grid.dx, bc)

# Sparse matrix form (for implicit solvers)
L = laplacian_matrix_1d(grid.N, grid.dx, bc)
```

## Callbacks

### Interface Tracking

```julia
using PhaseFields
using OrdinaryDiffEq

# Track interface position during simulation
cb, saved = create_interface_saving_callback(grid; saveat=0.1)

sol = solve(prob, Tsit5(); callback=cb)

# Access results
times = saved.t
positions = saved.saveval
```

### Solid Fraction Tracking

```julia
cb, saved = create_solid_fraction_callback(; threshold=0.5, saveat=0.1)

sol = solve(prob, Tsit5(); callback=cb)
fractions = saved.saveval
```

### Steady State Termination

```julia
# Stop when solution reaches steady state
cb = create_steady_state_callback(abstol=1e-6, reltol=1e-4, min_t=1.0)

sol = solve(prob, Tsit5(); callback=cb)
# sol.t[end] is when steady state was reached
```

### Combined Callbacks

```julia
result = create_phase_field_callbacks(grid;
    track_interface=true,
    track_solid_fraction=true,
    terminate_steady_state=true,
    saveat=0.1)

sol = solve(prob, Tsit5(); callback=result.callback)

# Access tracked data
result.interface_data.t
result.interface_data.saveval
result.solid_fraction_data.saveval
```

## Solver Selection

### Explicit Solvers (Non-stiff)

```julia
# Tsit5: Good general-purpose choice
sol = solve(prob, Tsit5())

# Vern7: Higher accuracy
sol = solve(prob, Vern7())
```

### Implicit Solvers (Stiff)

For thermal coupling or fine grids:

```julia
# QNDF: Quasi-constant step BDF (recommended)
sol = solve(prob, QNDF(autodiff=false))

# Rodas5: Rosenbrock method
sol = solve(prob, Rodas5(autodiff=false))
```

Note: Use `autodiff=false` with pre-allocated workspaces.

## API Reference

### Grid and Boundary Conditions

```@docs
UniformGrid1D
BoundaryCondition
NeumannBC
DirichletBC
PeriodicBC
laplacian_1d!
laplacian_1d
laplacian_matrix_1d
```

### Callbacks

```@docs
interface_position_1d
solid_fraction
create_interface_saving_callback
create_solid_fraction_callback
create_steady_state_callback
create_phase_field_callbacks
```

## See Also

- [151_diffeq_allen_cahn.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/151_diffeq_allen_cahn.jl) - Allen-Cahn with callbacks and analysis
- [351_diffeq_thermal_solidification.jl](https://github.com/hsugawa8651/PhaseFields.jl/blob/main/examples/351_diffeq_thermal_solidification.jl) - Thermal solidification with implicit solver
