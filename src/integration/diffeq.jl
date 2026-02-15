# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - DifferentialEquations.jl Integration

using SparseArrays

# =============================================================================
# Grid Types
# =============================================================================

"""
    UniformGrid1D

1D uniform grid for finite difference discretization.

# Fields
- `N::Int`: Number of grid points
- `L::Float64`: Domain length
- `dx::Float64`: Grid spacing (computed)
- `x::Vector{Float64}`: Grid point coordinates

# Example
```julia
grid = UniformGrid1D(N=100, L=1.0)
grid isa AbstractDomain{1}  # true
```
"""
struct UniformGrid1D <: AbstractDomain{1}
    N::Int
    L::Float64
    dx::Float64
    x::Vector{Float64}

    function UniformGrid1D(; N::Int, L::Float64)
        dx = L / (N - 1)
        x = collect(range(0, L, length=N))
        new(N, L, dx, x)
    end
end

# Grid interface implementations for UniformGrid1D
gridsize(g::UniformGrid1D) = (g.N,)
spacing(g::UniformGrid1D) = (g.dx,)
coordinates(g::UniformGrid1D) = (g.x,)

"""
    UniformGrid2D

2D uniform grid for finite difference discretization.

# Fields
- `Nx::Int`: Number of grid points in x direction
- `Ny::Int`: Number of grid points in y direction
- `Lx::Float64`: Domain length in x direction
- `Ly::Float64`: Domain length in y direction
- `dx::Float64`: Grid spacing in x (computed)
- `dy::Float64`: Grid spacing in y (computed)
- `x::Vector{Float64}`: x coordinates
- `y::Vector{Float64}`: y coordinates

# Example
```julia
grid = UniformGrid2D(Nx=100, Ny=100, Lx=1.0, Ly=1.0)
grid isa AbstractDomain{2}  # true
```
"""
struct UniformGrid2D <: AbstractDomain{2}
    Nx::Int
    Ny::Int
    Lx::Float64
    Ly::Float64
    dx::Float64
    dy::Float64
    x::Vector{Float64}
    y::Vector{Float64}

    function UniformGrid2D(; Nx::Int, Ny::Int, Lx::Float64, Ly::Float64)
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)
        x = collect(range(0, Lx, length=Nx))
        y = collect(range(0, Ly, length=Ny))
        new(Nx, Ny, Lx, Ly, dx, dy, x, y)
    end
end

# Grid interface implementations for UniformGrid2D
gridsize(g::UniformGrid2D) = (g.Nx, g.Ny)
spacing(g::UniformGrid2D) = (g.dx, g.dy)
coordinates(g::UniformGrid2D) = (g.x, g.y)

# =============================================================================
# Boundary Conditions
# =============================================================================

"""
    BoundaryCondition

Abstract type for boundary conditions.
"""
abstract type BoundaryCondition end

"""
    NeumannBC

Zero-flux (Neumann) boundary condition: ∂u/∂n = 0.
"""
struct NeumannBC <: BoundaryCondition end

"""
    DirichletBC

Fixed value (Dirichlet) boundary condition for 1D.

# Fields
- `left::Float64`: Value at left boundary
- `right::Float64`: Value at right boundary
"""
struct DirichletBC <: BoundaryCondition
    left::Float64
    right::Float64
end

"""
    DirichletBC2D

Fixed value (Dirichlet) boundary condition for 2D.

# Fields
- `left::Float64`: Value at x=0 boundary
- `right::Float64`: Value at x=Lx boundary
- `bottom::Float64`: Value at y=0 boundary
- `top::Float64`: Value at y=Ly boundary
"""
struct DirichletBC2D <: BoundaryCondition
    left::Float64
    right::Float64
    bottom::Float64
    top::Float64
end

"""
    PeriodicBC

Periodic boundary condition: u(0) = u(L).
"""
struct PeriodicBC <: BoundaryCondition end

# =============================================================================
# Generic Laplacian Interface
# =============================================================================

"""
    laplacian!(∇²u, u, domain::AbstractDomain, bc::BoundaryCondition)

Compute Laplacian on any domain type. Dispatches based on domain type.

This is the recommended API for dimension-independent code.
Works for 1D, 2D, and 3D domains.

# Arguments
- `∇²u`: Output array for Laplacian values (modified in-place)
- `u`: Input field values
- `domain`: Computational domain
- `bc`: Boundary condition

# Throws
- `ErrorException` if the domain type is not supported
"""
function laplacian!(∇²u, u, domain::AbstractDomain, bc::BoundaryCondition)
    error("laplacian! not implemented for domain type $(typeof(domain))")
end

# Generic laplacian! dispatch for UniformGrid1D
function laplacian!(∇²u, u, grid::UniformGrid1D, bc::BoundaryCondition)
    laplacian_1d!(∇²u, u, grid.dx, bc)
end

# Generic laplacian! dispatch for UniformGrid2D
function laplacian!(∇²u, u, grid::UniformGrid2D, bc::BoundaryCondition)
    laplacian_2d!(∇²u, u, grid.dx, grid.dy, bc)
end

# =============================================================================
# Laplacian Operators (1D implementations)
# =============================================================================

"""
    laplacian_1d!(∇²u, u, dx, bc::BoundaryCondition)

Compute 1D Laplacian in-place using second-order central differences.

# Arguments
- `∇²u`: Output array for Laplacian values (modified in-place)
- `u`: Input field values
- `dx`: Grid spacing
- `bc`: Boundary condition
"""
function laplacian_1d!(∇²u, u, dx, bc::NeumannBC)
    N = length(u)
    dx² = dx^2

    # Interior points: central difference
    @inbounds for i in 2:N-1
        ∇²u[i] = (u[i+1] - 2u[i] + u[i-1]) / dx²
    end

    # Neumann BC: ghost point approach (∂u/∂x = 0)
    # u[-1] = u[1], u[N+1] = u[N-1]
    @inbounds ∇²u[1] = 2(u[2] - u[1]) / dx²
    @inbounds ∇²u[N] = 2(u[N-1] - u[N]) / dx²

    return ∇²u
end

function laplacian_1d!(∇²u, u, dx, bc::DirichletBC)
    N = length(u)
    dx² = dx^2

    # Interior points: central difference
    @inbounds for i in 2:N-1
        ∇²u[i] = (u[i+1] - 2u[i] + u[i-1]) / dx²
    end

    # Dirichlet BC: use boundary values
    @inbounds ∇²u[1] = (u[2] - 2u[1] + bc.left) / dx²
    @inbounds ∇²u[N] = (bc.right - 2u[N] + u[N-1]) / dx²

    return ∇²u
end

function laplacian_1d!(∇²u, u, dx, bc::PeriodicBC)
    N = length(u)
    dx² = dx^2

    # Interior points: central difference
    @inbounds for i in 2:N-1
        ∇²u[i] = (u[i+1] - 2u[i] + u[i-1]) / dx²
    end

    # Periodic BC: wrap around
    @inbounds ∇²u[1] = (u[2] - 2u[1] + u[N]) / dx²
    @inbounds ∇²u[N] = (u[1] - 2u[N] + u[N-1]) / dx²

    return ∇²u
end

"""
    laplacian_1d(u, dx, bc::BoundaryCondition)

Compute 1D Laplacian (allocating version).
"""
function laplacian_1d(u, dx, bc::BoundaryCondition)
    ∇²u = similar(u)
    laplacian_1d!(∇²u, u, dx, bc)
    return ∇²u
end

# =============================================================================
# Laplacian Operators (2D implementations)
# =============================================================================

"""
    laplacian_2d!(∇²u, u, dx, dy, bc::BoundaryCondition)

Compute 2D Laplacian in-place using second-order central differences.

# Arguments
- `∇²u`: Output array for Laplacian values (Nx × Ny matrix, modified in-place)
- `u`: Input field values (Nx × Ny matrix)
- `dx`: Grid spacing in x direction
- `dy`: Grid spacing in y direction
- `bc`: Boundary condition
"""
function laplacian_2d!(∇²u, u, dx, dy, bc::NeumannBC)
    Nx, Ny = size(u)
    dx² = dx^2
    dy² = dy^2

    # Interior points: central difference
    @inbounds for j in 2:Ny-1
        for i in 2:Nx-1
            ∇²u[i, j] = (u[i+1, j] - 2u[i, j] + u[i-1, j]) / dx² +
                        (u[i, j+1] - 2u[i, j] + u[i, j-1]) / dy²
        end
    end

    # Neumann BC: ghost point approach (∂u/∂n = 0)
    # Interior edges (not corners)
    @inbounds for j in 2:Ny-1
        # Left edge (i=1)
        ∇²u[1, j] = 2(u[2, j] - u[1, j]) / dx² +
                    (u[1, j+1] - 2u[1, j] + u[1, j-1]) / dy²
        # Right edge (i=Nx)
        ∇²u[Nx, j] = 2(u[Nx-1, j] - u[Nx, j]) / dx² +
                     (u[Nx, j+1] - 2u[Nx, j] + u[Nx, j-1]) / dy²
    end

    @inbounds for i in 2:Nx-1
        # Bottom edge (j=1)
        ∇²u[i, 1] = (u[i+1, 1] - 2u[i, 1] + u[i-1, 1]) / dx² +
                    2(u[i, 2] - u[i, 1]) / dy²
        # Top edge (j=Ny)
        ∇²u[i, Ny] = (u[i+1, Ny] - 2u[i, Ny] + u[i-1, Ny]) / dx² +
                     2(u[i, Ny-1] - u[i, Ny]) / dy²
    end

    # Corners (Neumann on both directions)
    @inbounds ∇²u[1, 1] = 2(u[2, 1] - u[1, 1]) / dx² + 2(u[1, 2] - u[1, 1]) / dy²
    @inbounds ∇²u[Nx, 1] = 2(u[Nx-1, 1] - u[Nx, 1]) / dx² + 2(u[Nx, 2] - u[Nx, 1]) / dy²
    @inbounds ∇²u[1, Ny] = 2(u[2, Ny] - u[1, Ny]) / dx² + 2(u[1, Ny-1] - u[1, Ny]) / dy²
    @inbounds ∇²u[Nx, Ny] = 2(u[Nx-1, Ny] - u[Nx, Ny]) / dx² +
                            2(u[Nx, Ny-1] - u[Nx, Ny]) / dy²

    return ∇²u
end

function laplacian_2d!(∇²u, u, dx, dy, bc::DirichletBC2D)
    Nx, Ny = size(u)
    dx² = dx^2
    dy² = dy^2

    # Interior points: central difference
    @inbounds for j in 2:Ny-1
        for i in 2:Nx-1
            ∇²u[i, j] = (u[i+1, j] - 2u[i, j] + u[i-1, j]) / dx² +
                        (u[i, j+1] - 2u[i, j] + u[i, j-1]) / dy²
        end
    end

    # Dirichlet BC: use boundary values
    # Interior edges (not corners)
    @inbounds for j in 2:Ny-1
        # Left edge (i=1): use bc.left
        ∇²u[1, j] = (u[2, j] - 2u[1, j] + bc.left) / dx² +
                    (u[1, j+1] - 2u[1, j] + u[1, j-1]) / dy²
        # Right edge (i=Nx): use bc.right
        ∇²u[Nx, j] = (bc.right - 2u[Nx, j] + u[Nx-1, j]) / dx² +
                     (u[Nx, j+1] - 2u[Nx, j] + u[Nx, j-1]) / dy²
    end

    @inbounds for i in 2:Nx-1
        # Bottom edge (j=1): use bc.bottom
        ∇²u[i, 1] = (u[i+1, 1] - 2u[i, 1] + u[i-1, 1]) / dx² +
                    (u[i, 2] - 2u[i, 1] + bc.bottom) / dy²
        # Top edge (j=Ny): use bc.top
        ∇²u[i, Ny] = (u[i+1, Ny] - 2u[i, Ny] + u[i-1, Ny]) / dx² +
                     (bc.top - 2u[i, Ny] + u[i, Ny-1]) / dy²
    end

    # Corners (use appropriate boundary values)
    @inbounds ∇²u[1, 1] = (u[2, 1] - 2u[1, 1] + bc.left) / dx² +
                          (u[1, 2] - 2u[1, 1] + bc.bottom) / dy²
    @inbounds ∇²u[Nx, 1] = (bc.right - 2u[Nx, 1] + u[Nx-1, 1]) / dx² +
                           (u[Nx, 2] - 2u[Nx, 1] + bc.bottom) / dy²
    @inbounds ∇²u[1, Ny] = (u[2, Ny] - 2u[1, Ny] + bc.left) / dx² +
                           (bc.top - 2u[1, Ny] + u[1, Ny-1]) / dy²
    @inbounds ∇²u[Nx, Ny] = (bc.right - 2u[Nx, Ny] + u[Nx-1, Ny]) / dx² +
                            (bc.top - 2u[Nx, Ny] + u[Nx, Ny-1]) / dy²

    return ∇²u
end

function laplacian_2d!(∇²u, u, dx, dy, bc::PeriodicBC)
    Nx, Ny = size(u)
    dx² = dx^2
    dy² = dy^2

    # Interior points: central difference
    @inbounds for j in 2:Ny-1
        for i in 2:Nx-1
            ∇²u[i, j] = (u[i+1, j] - 2u[i, j] + u[i-1, j]) / dx² +
                        (u[i, j+1] - 2u[i, j] + u[i, j-1]) / dy²
        end
    end

    # Periodic BC: wrap around
    # Interior edges (not corners)
    @inbounds for j in 2:Ny-1
        # Left edge (i=1): wrap to Nx
        ∇²u[1, j] = (u[2, j] - 2u[1, j] + u[Nx, j]) / dx² +
                    (u[1, j+1] - 2u[1, j] + u[1, j-1]) / dy²
        # Right edge (i=Nx): wrap to 1
        ∇²u[Nx, j] = (u[1, j] - 2u[Nx, j] + u[Nx-1, j]) / dx² +
                     (u[Nx, j+1] - 2u[Nx, j] + u[Nx, j-1]) / dy²
    end

    @inbounds for i in 2:Nx-1
        # Bottom edge (j=1): wrap to Ny
        ∇²u[i, 1] = (u[i+1, 1] - 2u[i, 1] + u[i-1, 1]) / dx² +
                    (u[i, 2] - 2u[i, 1] + u[i, Ny]) / dy²
        # Top edge (j=Ny): wrap to 1
        ∇²u[i, Ny] = (u[i+1, Ny] - 2u[i, Ny] + u[i-1, Ny]) / dx² +
                     (u[i, 1] - 2u[i, Ny] + u[i, Ny-1]) / dy²
    end

    # Corners (wrap in both directions)
    @inbounds ∇²u[1, 1] = (u[2, 1] - 2u[1, 1] + u[Nx, 1]) / dx² +
                          (u[1, 2] - 2u[1, 1] + u[1, Ny]) / dy²
    @inbounds ∇²u[Nx, 1] = (u[1, 1] - 2u[Nx, 1] + u[Nx-1, 1]) / dx² +
                           (u[Nx, 2] - 2u[Nx, 1] + u[Nx, Ny]) / dy²
    @inbounds ∇²u[1, Ny] = (u[2, Ny] - 2u[1, Ny] + u[Nx, Ny]) / dx² +
                           (u[1, 1] - 2u[1, Ny] + u[1, Ny-1]) / dy²
    @inbounds ∇²u[Nx, Ny] = (u[1, Ny] - 2u[Nx, Ny] + u[Nx-1, Ny]) / dx² +
                            (u[Nx, 1] - 2u[Nx, Ny] + u[Nx, Ny-1]) / dy²

    return ∇²u
end

"""
    laplacian_2d(u, dx, dy, bc::BoundaryCondition)

Compute 2D Laplacian (allocating version).
"""
function laplacian_2d(u, dx, dy, bc::BoundaryCondition)
    ∇²u = similar(u)
    laplacian_2d!(∇²u, u, dx, dy, bc)
    return ∇²u
end

# =============================================================================
# Sparse Laplacian Matrix
# =============================================================================

"""
    laplacian_matrix_1d(N, dx, bc::BoundaryCondition)

Create sparse Laplacian matrix for 1D finite differences.

Returns sparse matrix L such that L * u ≈ ∇²u.
"""
function laplacian_matrix_1d(N, dx, bc::NeumannBC)
    dx² = dx^2

    # Tridiagonal: [-1, 2, -1] / dx²
    diag_main = fill(-2.0 / dx², N)
    diag_off = fill(1.0 / dx², N - 1)

    L = spdiagm(-1 => diag_off, 0 => diag_main, 1 => diag_off)

    # Neumann BC modification
    L[1, 1] = -2.0 / dx²
    L[1, 2] = 2.0 / dx²
    L[N, N-1] = 2.0 / dx²
    L[N, N] = -2.0 / dx²

    return L
end

function laplacian_matrix_1d(N, dx, bc::DirichletBC)
    dx² = dx^2

    diag_main = fill(-2.0 / dx², N)
    diag_off = fill(1.0 / dx², N - 1)

    L = spdiagm(-1 => diag_off, 0 => diag_main, 1 => diag_off)

    return L
end

function laplacian_matrix_1d(N, dx, bc::PeriodicBC)
    dx² = dx^2

    diag_main = fill(-2.0 / dx², N)
    diag_off = fill(1.0 / dx², N - 1)

    L = spdiagm(-1 => diag_off, 0 => diag_main, 1 => diag_off)

    # Periodic BC: corner elements
    L[1, N] = 1.0 / dx²
    L[N, 1] = 1.0 / dx²

    return L
end

# =============================================================================
# Allen-Cahn ODE Functions
# =============================================================================

"""
    AllenCahnODEParams{M, G, BC, A}

Parameters for Allen-Cahn ODE integration.

# Type Parameters
- `M`: Model type
- `G`: Grid type (UniformGrid1D or UniformGrid2D)
- `BC`: Boundary condition type
- `A`: Laplacian workspace array type

# Fields
- `model::M`: Phase field model parameters
- `grid::G`: Spatial discretization (1D or 2D)
- `bc::BC`: Boundary conditions
- `∇²φ::A`: Pre-allocated Laplacian workspace
"""
struct AllenCahnODEParams{M, G, BC, A}
    model::M
    grid::G
    bc::BC
    ∇²φ::A
end

# 1D constructor
function AllenCahnODEParams(model, grid::UniformGrid1D, bc::BoundaryCondition)
    ∇²φ = zeros(grid.N)
    return AllenCahnODEParams(model, grid, bc, ∇²φ)
end

# 2D constructor
function AllenCahnODEParams(model, grid::UniformGrid2D, bc::BoundaryCondition)
    ∇²φ = zeros(grid.Nx, grid.Ny)
    return AllenCahnODEParams(model, grid, bc, ∇²φ)
end

"""
    allen_cahn_ode!(dφ, φ, p::AllenCahnODEParams, t)

In-place RHS function for Allen-Cahn equation compatible with OrdinaryDiffEq.jl.

Computes: dφ/dt = (W²∇²φ - g'(φ)) / τ

Works with both 1D and 2D grids. For 2D, input/output vectors are automatically
reshaped to/from matrices for Laplacian computation.
"""
function allen_cahn_ode!(dφ, φ, p::AllenCahnODEParams{M, UniformGrid1D}, t) where {M}
    model = p.model
    grid = p.grid
    bc = p.bc
    ∇²φ = p.∇²φ

    τ = model.τ
    W = model.W

    # Compute Laplacian
    laplacian_1d!(∇²φ, φ, grid.dx, bc)

    # Compute RHS: (W²∇²φ - g'(φ)) / τ
    @inbounds for i in 1:grid.N
        g_prime = 2 * φ[i] * (1 - φ[i]) * (1 - 2φ[i])
        dφ[i] = (W^2 * ∇²φ[i] - g_prime) / τ
    end

    return nothing
end

# 2D version: handles vectorized state
function allen_cahn_ode!(dφ_vec, φ_vec, p::AllenCahnODEParams{M, UniformGrid2D}, t) where {M}
    model = p.model
    grid = p.grid
    bc = p.bc
    ∇²φ = p.∇²φ

    τ = model.τ
    W = model.W
    Nx, Ny = grid.Nx, grid.Ny

    # Reshape vector to matrix
    φ = reshape(φ_vec, Nx, Ny)

    # Compute Laplacian
    laplacian_2d!(∇²φ, φ, grid.dx, grid.dy, bc)

    # Compute RHS: (W²∇²φ - g'(φ)) / τ
    @inbounds for j in 1:Ny
        for i in 1:Nx
            idx = i + (j - 1) * Nx
            g_prime = 2 * φ[i, j] * (1 - φ[i, j]) * (1 - 2φ[i, j])
            dφ_vec[idx] = (W^2 * ∇²φ[i, j] - g_prime) / τ
        end
    end

    return nothing
end

"""
    create_allen_cahn_problem(model, grid, bc, φ0, tspan)

Create ODEProblem for Allen-Cahn equation.

# Arguments
- `model::AllenCahnModel`: Phase field model
- `grid::UniformGrid1D`: Spatial grid
- `bc::BoundaryCondition`: Boundary conditions
- `φ0`: Initial condition
- `tspan`: Time span (t_start, t_end)

# Returns
- `ODEProblem` ready for solve()

# Example
```julia
using OrdinaryDiffEq

model = AllenCahnModel(τ=1.0, W=0.1)
grid = UniformGrid1D(N=100, L=1.0)
φ0 = [0.5 + 0.1*sin(2π*x) for x in grid.x]

prob = create_allen_cahn_problem(model, grid, NeumannBC(), φ0, (0.0, 10.0))
sol = solve(prob, Tsit5())
```
"""
function create_allen_cahn_problem(model, grid::UniformGrid1D, bc::BoundaryCondition,
                                    φ0::AbstractVector, tspan::Tuple)
    params = AllenCahnODEParams(model, grid, bc)

    # Import ODEProblem from OrdinaryDiffEq
    # Note: OrdinaryDiffEq should be loaded by the user
    return OrdinaryDiffEq.ODEProblem(allen_cahn_ode!, φ0, tspan, params)
end

"""
    create_allen_cahn_problem(model, grid::UniformGrid2D, bc, φ0, tspan)

Create ODEProblem for 2D Allen-Cahn equation.

# Arguments
- `model::AllenCahnModel`: Phase field model
- `grid::UniformGrid2D`: 2D spatial grid
- `bc::BoundaryCondition`: Boundary conditions
- `φ0::AbstractMatrix`: Initial condition (Nx × Ny matrix)
- `tspan`: Time span (t_start, t_end)

# Returns
- `ODEProblem` ready for solve()

# Note
The initial condition matrix is vectorized column-major (Julia default).
Solution vectors can be reshaped back using `reshape(sol.u[i], grid.Nx, grid.Ny)`.

# Example
```julia
using OrdinaryDiffEq

model = AllenCahnModel(τ=1.0, W=0.1)
grid = UniformGrid2D(Nx=50, Ny=50, Lx=1.0, Ly=1.0)

# Circular interface
φ0 = [sqrt((x-0.5)^2 + (y-0.5)^2) < 0.3 ? 1.0 : 0.0 for x in grid.x, y in grid.y]

prob = create_allen_cahn_problem(model, grid, NeumannBC(), φ0, (0.0, 1.0))
sol = solve(prob, Tsit5())

# Reshape solution
φ_final = reshape(sol.u[end], grid.Nx, grid.Ny)
```
"""
function create_allen_cahn_problem(model, grid::UniformGrid2D, bc::BoundaryCondition,
                                    φ0::AbstractMatrix, tspan::Tuple)
    params = AllenCahnODEParams(model, grid, bc)

    # Vectorize initial condition (column-major order)
    φ0_vec = vec(φ0)

    return OrdinaryDiffEq.ODEProblem(allen_cahn_ode!, φ0_vec, tspan, params)
end

# =============================================================================
# Thermal Model ODE Functions
# =============================================================================

"""
    ThermalODEParams{M, G, BC1, BC2, A}

Parameters for coupled thermal phase field ODE integration.

# Type Parameters
- `M`: Model type
- `G`: Grid type (UniformGrid1D or UniformGrid2D)
- `BC1`: Phase field boundary condition type
- `BC2`: Temperature boundary condition type
- `A`: Laplacian workspace array type

# Fields
- `model::M`: Thermal phase field model
- `grid::G`: Spatial discretization (1D or 2D)
- `bc_φ::BC1`: Phase field boundary conditions
- `bc_u::BC2`: Temperature boundary conditions
- `∇²φ::A`: Pre-allocated Laplacian workspace for φ
- `∇²u::A`: Pre-allocated Laplacian workspace for u
- `dφdt::A`: Pre-allocated dφ/dt workspace
"""
struct ThermalODEParams{M, G, BC1, BC2, A}
    model::M
    grid::G
    bc_φ::BC1
    bc_u::BC2
    ∇²φ::A
    ∇²u::A
    dφdt::A
end

# 1D constructor
function ThermalODEParams(model, grid::UniformGrid1D, bc_φ::BoundaryCondition,
                          bc_u::BoundaryCondition)
    N = grid.N
    ∇²φ = zeros(N)
    ∇²u = zeros(N)
    dφdt = zeros(N)
    return ThermalODEParams(model, grid, bc_φ, bc_u, ∇²φ, ∇²u, dφdt)
end

# 2D constructor
function ThermalODEParams(model, grid::UniformGrid2D, bc_φ::BoundaryCondition,
                          bc_u::BoundaryCondition)
    ∇²φ = zeros(grid.Nx, grid.Ny)
    ∇²u = zeros(grid.Nx, grid.Ny)
    dφdt = zeros(grid.Nx, grid.Ny)
    return ThermalODEParams(model, grid, bc_φ, bc_u, ∇²φ, ∇²u, dφdt)
end

"""
    thermal_ode!(dy, y, p::ThermalODEParams, t)

In-place RHS function for coupled thermal phase field equations.

State vector y = [φ; u] where:
- φ: Phase field (indices 1:N for 1D, 1:Nx*Ny for 2D)
- u: Dimensionless temperature (indices N+1:2N for 1D)
"""
function thermal_ode!(dy, y, p::ThermalODEParams{M, UniformGrid1D}, t) where {M}
    model = p.model
    grid = p.grid
    N = grid.N
    dx = grid.dx

    # Extract views
    φ = @view y[1:N]
    u = @view y[N+1:2N]
    dφ = @view dy[1:N]
    du = @view dy[N+1:2N]

    # Compute Laplacians
    laplacian_1d!(p.∇²φ, φ, dx, p.bc_φ)
    laplacian_1d!(p.∇²u, u, dx, p.bc_u)

    # Phase field equation
    τ = model.τ
    W = model.W
    λ = model.λ
    α = model.α

    @inbounds for i in 1:N
        # g'(φ) = 2φ(1-φ)(1-2φ)
        g_prime = 2 * φ[i] * (1 - φ[i]) * (1 - 2φ[i])
        # h'(φ) = 6φ(1-φ)
        h_prime = 6 * φ[i] * (1 - φ[i])

        # dφ/dt = (W²∇²φ - g'(φ) - λu·h'(φ)) / τ
        dφ[i] = (W^2 * p.∇²φ[i] - g_prime - λ * u[i] * h_prime) / τ
        p.dφdt[i] = dφ[i]
    end

    # Temperature equation
    @inbounds for i in 1:N
        # du/dt = α∇²u + (1/2)dφ/dt
        du[i] = α * p.∇²u[i] + 0.5 * p.dφdt[i]
    end

    return nothing
end

# 2D version: handles vectorized state
function thermal_ode!(dy, y, p::ThermalODEParams{M, UniformGrid2D}, t) where {M}
    model = p.model
    grid = p.grid
    Nx, Ny = grid.Nx, grid.Ny
    N_total = Nx * Ny

    # Extract views and reshape
    φ_vec = @view y[1:N_total]
    u_vec = @view y[N_total+1:2N_total]
    φ = reshape(φ_vec, Nx, Ny)
    u = reshape(u_vec, Nx, Ny)

    # Compute Laplacians
    laplacian_2d!(p.∇²φ, φ, grid.dx, grid.dy, p.bc_φ)
    laplacian_2d!(p.∇²u, u, grid.dx, grid.dy, p.bc_u)

    # Phase field and temperature equations
    τ = model.τ
    W = model.W
    λ = model.λ
    α = model.α

    @inbounds for j in 1:Ny
        for i in 1:Nx
            idx = i + (j - 1) * Nx

            # g'(φ) = 2φ(1-φ)(1-2φ)
            g_prime = 2 * φ[i, j] * (1 - φ[i, j]) * (1 - 2φ[i, j])
            # h'(φ) = 6φ(1-φ)
            h_prime = 6 * φ[i, j] * (1 - φ[i, j])

            # dφ/dt = (W²∇²φ - g'(φ) - λu·h'(φ)) / τ
            dφdt_val = (W^2 * p.∇²φ[i, j] - g_prime - λ * u[i, j] * h_prime) / τ
            dy[idx] = dφdt_val
            p.dφdt[i, j] = dφdt_val

            # du/dt = α∇²u + (1/2)dφ/dt
            dy[N_total + idx] = α * p.∇²u[i, j] + 0.5 * dφdt_val
        end
    end

    return nothing
end

"""
    create_thermal_problem(model, grid, bc_φ, bc_u, φ0, u0, tspan)

Create ODEProblem for coupled thermal phase field equations.

# Arguments
- `model::ThermalPhaseFieldModel`: Thermal phase field model
- `grid::UniformGrid1D`: Spatial grid
- `bc_φ::BoundaryCondition`: Phase field boundary conditions
- `bc_u::BoundaryCondition`: Temperature boundary conditions
- `φ0`: Initial phase field
- `u0`: Initial dimensionless temperature
- `tspan`: Time span (t_start, t_end)

# Returns
- `ODEProblem` with state vector [φ; u]
"""
function create_thermal_problem(model, grid::UniformGrid1D,
                                 bc_φ::BoundaryCondition, bc_u::BoundaryCondition,
                                 φ0::AbstractVector, u0::AbstractVector, tspan::Tuple)
    params = ThermalODEParams(model, grid, bc_φ, bc_u)

    # Combine initial conditions
    y0 = vcat(φ0, u0)

    return OrdinaryDiffEq.ODEProblem(thermal_ode!, y0, tspan, params)
end

"""
    extract_thermal_solution(sol, N)

Extract φ and u from thermal ODE solution.

# Arguments
- `sol`: ODESolution from solve()
- `N`: Number of grid points

# Returns
- `(φ_history, u_history)`: Arrays of shape (N, n_times)
"""
function extract_thermal_solution(sol, N)
    n_times = length(sol.t)
    φ_history = zeros(N, n_times)
    u_history = zeros(N, n_times)

    for (i, y) in enumerate(sol.u)
        φ_history[:, i] = y[1:N]
        u_history[:, i] = y[N+1:2N]
    end

    return φ_history, u_history
end

# =============================================================================
# WBM Model ODE Functions
# =============================================================================

"""
    WBMODEParams{M, G, BC1, BC2, F1, F2, A}

Parameters for WBM phase field ODE integration.

# Type Parameters
- `M`: Model type (WBMModel)
- `G`: Grid type (UniformGrid1D or UniformGrid2D)
- `BC1`: Phase field boundary condition type
- `BC2`: Concentration boundary condition type
- `F1`: Solid free energy type
- `F2`: Liquid free energy type
- `A`: Laplacian workspace array type

# Fields
- `model::M`: WBM model parameters
- `grid::G`: Spatial discretization
- `bc_φ::BC1`: Phase field boundary conditions
- `bc_c::BC2`: Concentration boundary conditions
- `f_s::F1`: Solid phase free energy function
- `f_l::F2`: Liquid phase free energy function
- `∇²φ::A`: Pre-allocated Laplacian workspace for φ
- `∇²c::A`: Pre-allocated Laplacian workspace for c
"""
struct WBMODEParams{M, G, BC1, BC2, F1, F2, A}
    model::M
    grid::G
    bc_φ::BC1
    bc_c::BC2
    f_s::F1
    f_l::F2
    ∇²φ::A
    ∇²c::A
end

# 1D constructor
function WBMODEParams(model, grid::UniformGrid1D, bc_φ::BoundaryCondition,
                      bc_c::BoundaryCondition, f_s, f_l)
    N = grid.N
    ∇²φ = zeros(N)
    ∇²c = zeros(N)
    return WBMODEParams(model, grid, bc_φ, bc_c, f_s, f_l, ∇²φ, ∇²c)
end

# 2D constructor
function WBMODEParams(model, grid::UniformGrid2D, bc_φ::BoundaryCondition,
                      bc_c::BoundaryCondition, f_s, f_l)
    ∇²φ = zeros(grid.Nx, grid.Ny)
    ∇²c = zeros(grid.Nx, grid.Ny)
    return WBMODEParams(model, grid, bc_φ, bc_c, f_s, f_l, ∇²φ, ∇²c)
end

"""
    wbm_ode!(dy, y, p::WBMODEParams, t)

In-place RHS function for WBM phase field equations.

State vector y = [φ; c] where:
- φ: Phase field (solid=1, liquid=0)
- c: Concentration

Evolution equations:
- ∂φ/∂t = M_φ[κ∇²φ - h'(φ)(f_S - f_L) - W·g'(φ)]
- ∂c/∂t = D(φ)·∇²c
"""
function wbm_ode!(dy, y, p::WBMODEParams{M, UniformGrid1D}, t) where {M}
    model = p.model
    grid = p.grid
    N = grid.N
    dx = grid.dx

    # Extract views
    φ = @view y[1:N]
    c = @view y[N+1:2N]
    dφ = @view dy[1:N]
    dc = @view dy[N+1:2N]

    # Compute Laplacians
    laplacian_1d!(p.∇²φ, φ, dx, p.bc_φ)
    laplacian_1d!(p.∇²c, c, dx, p.bc_c)

    # Phase field and concentration equations
    M_φ = model.M_φ
    κ = model.κ
    W = model.W

    @inbounds for i in 1:N
        # Phase field RHS using wbm_phase_rhs
        dφ[i] = wbm_phase_rhs(model, φ[i], p.∇²φ[i], c[i], p.f_s, p.f_l)

        # Concentration RHS using wbm_concentration_rhs
        dc[i] = wbm_concentration_rhs(model, φ[i], p.∇²c[i])
    end

    return nothing
end

# 2D version
function wbm_ode!(dy, y, p::WBMODEParams{M, UniformGrid2D}, t) where {M}
    model = p.model
    grid = p.grid
    Nx, Ny = grid.Nx, grid.Ny
    N_total = Nx * Ny

    # Extract views and reshape
    φ_vec = @view y[1:N_total]
    c_vec = @view y[N_total+1:2N_total]
    φ = reshape(φ_vec, Nx, Ny)
    c = reshape(c_vec, Nx, Ny)

    # Compute Laplacians
    laplacian_2d!(p.∇²φ, φ, grid.dx, grid.dy, p.bc_φ)
    laplacian_2d!(p.∇²c, c, grid.dx, grid.dy, p.bc_c)

    @inbounds for j in 1:Ny
        for i in 1:Nx
            idx = i + (j - 1) * Nx

            # Phase field RHS
            dy[idx] = wbm_phase_rhs(model, φ[i, j], p.∇²φ[i, j], c[i, j], p.f_s, p.f_l)

            # Concentration RHS
            dy[N_total + idx] = wbm_concentration_rhs(model, φ[i, j], p.∇²c[i, j])
        end
    end

    return nothing
end

# =============================================================================
# Cahn-Hilliard Model ODE Functions
# =============================================================================

"""
    CahnHilliardODEParams{M, G, BC, F, A}

Parameters for Cahn-Hilliard ODE integration.

# Type Parameters
- `M`: Model type (CahnHilliardModel)
- `G`: Grid type (UniformGrid1D or UniformGrid2D)
- `BC`: Boundary condition type
- `F`: Free energy type (DoubleWellFreeEnergy)
- `A`: Workspace array type

# Fields
- `model::M`: Cahn-Hilliard model parameters
- `grid::G`: Spatial discretization
- `bc::BC`: Boundary conditions
- `f::F`: Free energy function
- `∇²c::A`: Pre-allocated Laplacian workspace for c
- `μ::A`: Pre-allocated chemical potential workspace
- `∇²μ::A`: Pre-allocated Laplacian of chemical potential workspace
"""
struct CahnHilliardODEParams{M, G, BC, F, A}
    model::M
    grid::G
    bc::BC
    f::F
    ∇²c::A
    μ::A
    ∇²μ::A
end

# 1D constructor
function CahnHilliardODEParams(model, grid::UniformGrid1D, bc::BoundaryCondition, f)
    N = grid.N
    ∇²c = zeros(N)
    μ = zeros(N)
    ∇²μ = zeros(N)
    return CahnHilliardODEParams(model, grid, bc, f, ∇²c, μ, ∇²μ)
end

# 2D constructor
function CahnHilliardODEParams(model, grid::UniformGrid2D, bc::BoundaryCondition, f)
    ∇²c = zeros(grid.Nx, grid.Ny)
    μ = zeros(grid.Nx, grid.Ny)
    ∇²μ = zeros(grid.Nx, grid.Ny)
    return CahnHilliardODEParams(model, grid, bc, f, ∇²c, μ, ∇²μ)
end

"""
    cahn_hilliard_ode!(dc, c, p::CahnHilliardODEParams, t)

In-place RHS function for Cahn-Hilliard equation.

    ∂c/∂t = M∇²μ
    μ = df/dc - κ∇²c

This is a 4th order PDE that requires computing:
1. ∇²c at each point
2. μ = df/dc - κ∇²c at each point
3. ∇²μ at each point
4. dc/dt = M∇²μ
"""
function cahn_hilliard_ode!(dc, c, p::CahnHilliardODEParams{M, UniformGrid1D}, t) where {M}
    model = p.model
    grid = p.grid
    N = grid.N
    dx = grid.dx
    κ = model.κ
    M_mob = model.M

    # Step 1: Compute ∇²c
    laplacian_1d!(p.∇²c, c, dx, p.bc)

    # Step 2: Compute chemical potential μ = df/dc - κ∇²c
    @inbounds for i in 1:N
        μ_bulk = chemical_potential_bulk(p.f, c[i])
        p.μ[i] = μ_bulk - κ * p.∇²c[i]
    end

    # Step 3: Compute ∇²μ
    laplacian_1d!(p.∇²μ, p.μ, dx, p.bc)

    # Step 4: Compute dc/dt = M∇²μ
    @inbounds for i in 1:N
        dc[i] = M_mob * p.∇²μ[i]
    end

    return nothing
end

# 2D version
function cahn_hilliard_ode!(dc_vec, c_vec, p::CahnHilliardODEParams{M, UniformGrid2D}, t) where {M}
    model = p.model
    grid = p.grid
    Nx, Ny = grid.Nx, grid.Ny
    κ = model.κ
    M_mob = model.M

    # Reshape vector to matrix
    c = reshape(c_vec, Nx, Ny)

    # Step 1: Compute ∇²c
    laplacian_2d!(p.∇²c, c, grid.dx, grid.dy, p.bc)

    # Step 2: Compute chemical potential μ = df/dc - κ∇²c
    @inbounds for j in 1:Ny
        for i in 1:Nx
            μ_bulk = chemical_potential_bulk(p.f, c[i, j])
            p.μ[i, j] = μ_bulk - κ * p.∇²c[i, j]
        end
    end

    # Step 3: Compute ∇²μ
    laplacian_2d!(p.∇²μ, p.μ, grid.dx, grid.dy, p.bc)

    # Step 4: Compute dc/dt = M∇²μ
    @inbounds for j in 1:Ny
        for i in 1:Nx
            idx = i + (j - 1) * Nx
            dc_vec[idx] = M_mob * p.∇²μ[i, j]
        end
    end

    return nothing
end

# =============================================================================
# KKS Model ODE Functions
# =============================================================================

"""
    KKSODEParams{M, G, BC1, BC2, F1, F2, A}

Parameters for KKS phase field ODE integration.

# Type Parameters
- `M`: Model type (KKSModel)
- `G`: Grid type (UniformGrid1D or UniformGrid2D)
- `BC1`: Phase field boundary condition type
- `BC2`: Concentration boundary condition type
- `F1`: Solid free energy type
- `F2`: Liquid free energy type
- `A`: Workspace array type

# Fields
- `model::M`: KKS model parameters
- `grid::G`: Spatial discretization
- `bc_φ::BC1`: Phase field boundary conditions
- `bc_c::BC2`: Concentration boundary conditions
- `f_s::F1`: Solid phase free energy function
- `f_l::F2`: Liquid phase free energy function
- `∇²φ::A`: Pre-allocated Laplacian workspace for φ
- `μ::A`: Pre-allocated chemical potential workspace
- `∇²μ::A`: Pre-allocated Laplacian of μ workspace
"""
struct KKSODEParams{M, G, BC1, BC2, F1, F2, A}
    model::M
    grid::G
    bc_φ::BC1
    bc_c::BC2
    f_s::F1
    f_l::F2
    ∇²φ::A
    μ::A
    ∇²μ::A
end

# 1D constructor
function KKSODEParams(model, grid::UniformGrid1D, bc_φ::BoundaryCondition,
                      bc_c::BoundaryCondition, f_s, f_l)
    N = grid.N
    ∇²φ = zeros(N)
    μ = zeros(N)
    ∇²μ = zeros(N)
    return KKSODEParams(model, grid, bc_φ, bc_c, f_s, f_l, ∇²φ, μ, ∇²μ)
end

# 2D constructor
function KKSODEParams(model, grid::UniformGrid2D, bc_φ::BoundaryCondition,
                      bc_c::BoundaryCondition, f_s, f_l)
    ∇²φ = zeros(grid.Nx, grid.Ny)
    μ = zeros(grid.Nx, grid.Ny)
    ∇²μ = zeros(grid.Nx, grid.Ny)
    return KKSODEParams(model, grid, bc_φ, bc_c, f_s, f_l, ∇²φ, μ, ∇²μ)
end

"""
    kks_ode!(dy, y, p::KKSODEParams, t)

In-place RHS function for KKS phase field equations.

State vector y = [φ; c] where:
- φ: Phase field (solid=1, liquid=0)
- c: Average concentration

Evolution equations:
- τ·∂φ/∂t = W²∇²φ - g'(φ) + h'(φ)·Δω
- ∂c/∂t = M(φ)·∇²μ

At each point, the KKS partition equation is solved to find (c_s, c_l, μ).
"""
function kks_ode!(dy, y, p::KKSODEParams{M, UniformGrid1D}, t) where {M}
    model = p.model
    grid = p.grid
    N = grid.N
    dx = grid.dx

    # Extract views
    φ = @view y[1:N]
    c = @view y[N+1:2N]
    dφ = @view dy[1:N]
    dc = @view dy[N+1:2N]

    # Compute Laplacian of φ
    laplacian_1d!(p.∇²φ, φ, dx, p.bc_φ)

    # First pass: compute chemical potential at each point
    @inbounds for i in 1:N
        c_s, c_l, μ_local, converged = kks_partition(c[i], φ[i], p.f_s, p.f_l)
        p.μ[i] = μ_local
    end

    # Compute Laplacian of μ
    laplacian_1d!(p.∇²μ, p.μ, dx, p.bc_c)

    # Second pass: compute RHS
    @inbounds for i in 1:N
        # Solve partition equation
        c_s, c_l, μ_local, converged = kks_partition(c[i], φ[i], p.f_s, p.f_l)

        # Grand potential difference
        Δω = kks_grand_potential_diff(p.f_s, p.f_l, c_s, c_l, μ_local)

        # Phase field RHS
        dφ[i] = kks_phase_rhs(model, φ[i], p.∇²φ[i], Δω)

        # Concentration RHS
        dc[i] = kks_concentration_rhs(model, φ[i], p.∇²μ[i])
    end

    return nothing
end

# 2D version
function kks_ode!(dy, y, p::KKSODEParams{M, UniformGrid2D}, t) where {M}
    model = p.model
    grid = p.grid
    Nx, Ny = grid.Nx, grid.Ny
    N_total = Nx * Ny

    # Extract views and reshape
    φ_vec = @view y[1:N_total]
    c_vec = @view y[N_total+1:2N_total]
    φ = reshape(φ_vec, Nx, Ny)
    c = reshape(c_vec, Nx, Ny)

    # Compute Laplacian of φ
    laplacian_2d!(p.∇²φ, φ, grid.dx, grid.dy, p.bc_φ)

    # First pass: compute chemical potential at each point
    @inbounds for j in 1:Ny
        for i in 1:Nx
            c_s, c_l, μ_local, converged = kks_partition(c[i, j], φ[i, j], p.f_s, p.f_l)
            p.μ[i, j] = μ_local
        end
    end

    # Compute Laplacian of μ
    laplacian_2d!(p.∇²μ, p.μ, grid.dx, grid.dy, p.bc_c)

    # Second pass: compute RHS
    @inbounds for j in 1:Ny
        for i in 1:Nx
            idx = i + (j - 1) * Nx

            # Solve partition equation
            c_s, c_l, μ_local, converged = kks_partition(c[i, j], φ[i, j], p.f_s, p.f_l)

            # Grand potential difference
            Δω = kks_grand_potential_diff(p.f_s, p.f_l, c_s, c_l, μ_local)

            # Phase field RHS
            dy[idx] = kks_phase_rhs(model, φ[i, j], p.∇²φ[i, j], Δω)

            # Concentration RHS
            dy[N_total + idx] = kks_concentration_rhs(model, φ[i, j], p.∇²μ[i, j])
        end
    end

    return nothing
end
