# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Abstract problem type definitions

"""
    AbstractDomain{N}

Abstract type for N-dimensional computational domains.

Type parameter `N` indicates the spatial dimension (1, 2, or 3).

# Concrete Subtypes
- `UniformGrid1D <: AbstractDomain{1}`: 1D uniform finite difference grid

# Interface
Concrete subtypes should provide:
- Grid spacing information
- Coordinate arrays
- Dimension-specific properties

# Example
```julia
grid = UniformGrid1D(N=100, L=1.0)
grid isa AbstractDomain{1}  # true
```
"""
abstract type AbstractDomain{N} end

"""
    AbstractPhaseFieldProblem

Abstract type for phase field problem definitions.

A problem combines:
- Physical model (e.g., AllenCahnModel)
- Computational domain (e.g., UniformGrid1D)
- Initial conditions
- Time span
- Boundary conditions

# Concrete Subtypes
- `PhaseFieldProblem`: Generic phase field problem

# Interface
The `solve` function dispatches on the combination of model and domain types
to select appropriate numerical methods.
"""
abstract type AbstractPhaseFieldProblem end

# =============================================================================
# Grid Interface Functions
# =============================================================================

"""
    gridsize(domain::AbstractDomain)

Return the number of grid points as a tuple.
- 1D: `(N,)`
- 2D: `(Nx, Ny)`
- 3D: `(Nx, Ny, Nz)`

Must be implemented by concrete domain types.
"""
function gridsize(domain::AbstractDomain)
    error("gridsize not implemented for $(typeof(domain))")
end

"""
    spacing(domain::AbstractDomain)

Return the grid spacing as a tuple.
- 1D: `(dx,)`
- 2D: `(dx, dy)`
- 3D: `(dx, dy, dz)`

Must be implemented by concrete domain types.
"""
function spacing(domain::AbstractDomain)
    error("spacing not implemented for $(typeof(domain))")
end

"""
    coordinates(domain::AbstractDomain)

Return the coordinate arrays as a tuple.
- 1D: `(x,)`
- 2D: `(x, y)`
- 3D: `(x, y, z)`

Must be implemented by concrete domain types.
"""
function coordinates(domain::AbstractDomain)
    error("coordinates not implemented for $(typeof(domain))")
end

