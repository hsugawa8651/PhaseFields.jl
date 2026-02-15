# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - FDM solver dispatches

# =============================================================================
# FDM Solver for UniformGrid1D
# =============================================================================

"""
    solve(problem::PhaseFieldProblem{<:AllenCahnModel, UniformGrid1D}, solver; kwargs...)

Solve an Allen-Cahn problem on a 1D uniform grid using finite differences.

Uses OrdinaryDiffEq.jl for time integration.

# Arguments
- `problem`: PhaseFieldProblem with AllenCahnModel and UniformGrid1D
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Keyword Arguments
Passed to OrdinaryDiffEq.solve()

# Returns
- ODESolution from OrdinaryDiffEq.jl

# Example
```julia
using OrdinaryDiffEq

model = AllenCahnModel(τ=1.0, W=0.1)
grid = UniformGrid1D(N=100, L=1.0)
φ0 = [x < 0.5 ? 1.0 : 0.0 for x in grid.x]

problem = PhaseFieldProblem(
    model=model,
    domain=grid,
    φ0=φ0,
    tspan=(0.0, 1.0)
)

sol = solve(problem, Tsit5())
```
"""
function solve(
    problem::PhaseFieldProblem{<:AllenCahnModel, UniformGrid1D},
    solver;
    kwargs...
)
    # Create ODE parameters
    params = AllenCahnODEParams(problem.model, problem.domain, problem.bc)

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        allen_cahn_ode!,
        problem.φ0,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end

# =============================================================================
# FDM Solver for UniformGrid2D
# =============================================================================

"""
    solve(problem::PhaseFieldProblem{<:AllenCahnModel, UniformGrid2D}, solver; kwargs...)

Solve an Allen-Cahn problem on a 2D uniform grid using finite differences.

Uses OrdinaryDiffEq.jl for time integration. The 2D field is vectorized internally
for compatibility with ODE solvers.

# Arguments
- `problem`: PhaseFieldProblem with AllenCahnModel and UniformGrid2D
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Keyword Arguments
Passed to OrdinaryDiffEq.solve()

# Returns
- ODESolution from OrdinaryDiffEq.jl

# Note
Solution vectors can be reshaped back to 2D using:
`reshape(sol.u[i], grid.Nx, grid.Ny)`

# Example
```julia
using OrdinaryDiffEq

model = AllenCahnModel(τ=1.0, W=0.1)
grid = UniformGrid2D(Nx=50, Ny=50, Lx=1.0, Ly=1.0)

# Circular interface
φ0 = [sqrt((x-0.5)^2 + (y-0.5)^2) < 0.3 ? 1.0 : 0.0 for x in grid.x, y in grid.y]

problem = PhaseFieldProblem(
    model=model,
    domain=grid,
    φ0=φ0,
    tspan=(0.0, 1.0)
)

sol = solve(problem, Tsit5())

# Reshape solution
φ_final = reshape(sol.u[end], grid.Nx, grid.Ny)
```
"""
function solve(
    problem::PhaseFieldProblem{<:AllenCahnModel, UniformGrid2D},
    solver;
    kwargs...
)
    # Create ODE parameters
    params = AllenCahnODEParams(problem.model, problem.domain, problem.bc)

    # Vectorize initial condition (column-major order)
    φ0_vec = vec(problem.φ0)

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        allen_cahn_ode!,
        φ0_vec,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end

# =============================================================================
# FDM Solver for ThermalPhaseFieldModel
# =============================================================================

"""
    solve(problem::PhaseFieldProblem{<:ThermalPhaseFieldModel, UniformGrid1D}, solver; kwargs...)

Solve a thermal phase field problem on a 1D uniform grid using finite differences.

# Arguments
- `problem`: PhaseFieldProblem with ThermalPhaseFieldModel and UniformGrid1D
  - `problem.φ0` should be a tuple `(φ0, u0)` of initial conditions
  - `problem.bc` should be a tuple `(bc_φ, bc_u)` of boundary conditions
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Returns
- ODESolution with state vector [φ; u]
"""
function solve(
    problem::PhaseFieldProblem{<:ThermalPhaseFieldModel, UniformGrid1D},
    solver;
    kwargs...
)
    # Extract initial conditions and boundary conditions
    φ0, u0 = problem.φ0
    bc_φ, bc_u = problem.bc

    # Create ODE parameters
    params = ThermalODEParams(problem.model, problem.domain, bc_φ, bc_u)

    # Combine initial conditions
    y0 = vcat(φ0, u0)

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        thermal_ode!,
        y0,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end

"""
    solve(problem::PhaseFieldProblem{<:ThermalPhaseFieldModel, UniformGrid2D}, solver; kwargs...)

Solve a thermal phase field problem on a 2D uniform grid using finite differences.

# Arguments
- `problem`: PhaseFieldProblem with ThermalPhaseFieldModel and UniformGrid2D
  - `problem.φ0` should be a tuple `(φ0, u0)` of initial conditions (matrices)
  - `problem.bc` should be a tuple `(bc_φ, bc_u)` of boundary conditions
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Returns
- ODESolution with state vector [φ_vec; u_vec]

# Note
Solution vectors can be reshaped back to 2D:
```julia
N_total = grid.Nx * grid.Ny
φ = reshape(sol.u[i][1:N_total], grid.Nx, grid.Ny)
u = reshape(sol.u[i][N_total+1:end], grid.Nx, grid.Ny)
```
"""
function solve(
    problem::PhaseFieldProblem{<:ThermalPhaseFieldModel, UniformGrid2D},
    solver;
    kwargs...
)
    # Extract initial conditions and boundary conditions
    φ0, u0 = problem.φ0
    bc_φ, bc_u = problem.bc
    grid = problem.domain

    # Create ODE parameters
    params = ThermalODEParams(problem.model, grid, bc_φ, bc_u)

    # Vectorize and combine initial conditions (column-major order)
    y0 = vcat(vec(φ0), vec(u0))

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        thermal_ode!,
        y0,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end

# =============================================================================
# FDM Solver for WBMModel
# =============================================================================

"""
    solve(problem::PhaseFieldProblem{<:WBMProblemData, UniformGrid1D}, solver; kwargs...)

Solve a WBM problem on a 1D uniform grid using finite differences.

# Arguments
- `problem`: PhaseFieldProblem created with WBMProblem()
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Returns
- ODESolution with state vector [φ; c]
"""
function solve(
    problem::PhaseFieldProblem{<:WBMProblemData, UniformGrid1D},
    solver;
    kwargs...
)
    # Extract model data
    model_data = problem.model
    model = model_data.model
    f_s = model_data.f_s
    f_l = model_data.f_l

    # Extract initial conditions and boundary conditions
    φ0, c0 = problem.φ0
    bc_φ, bc_c = problem.bc
    grid = problem.domain

    # Create ODE parameters
    params = WBMODEParams(model, grid, bc_φ, bc_c, f_s, f_l)

    # Combine initial conditions
    y0 = vcat(φ0, c0)

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        wbm_ode!,
        y0,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end

"""
    solve(problem::PhaseFieldProblem{<:WBMProblemData, UniformGrid2D}, solver; kwargs...)

Solve a WBM problem on a 2D uniform grid using finite differences.

# Arguments
- `problem`: PhaseFieldProblem created with WBMProblem()
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Returns
- ODESolution with state vector [φ_vec; c_vec]
"""
function solve(
    problem::PhaseFieldProblem{<:WBMProblemData, UniformGrid2D},
    solver;
    kwargs...
)
    # Extract model data
    model_data = problem.model
    model = model_data.model
    f_s = model_data.f_s
    f_l = model_data.f_l

    # Extract initial conditions and boundary conditions
    φ0, c0 = problem.φ0
    bc_φ, bc_c = problem.bc
    grid = problem.domain

    # Create ODE parameters
    params = WBMODEParams(model, grid, bc_φ, bc_c, f_s, f_l)

    # Vectorize and combine initial conditions
    y0 = vcat(vec(φ0), vec(c0))

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        wbm_ode!,
        y0,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end

# =============================================================================
# FDM Solver for CahnHilliardModel
# =============================================================================

"""
    solve(problem::PhaseFieldProblem{<:CahnHilliardProblemData, UniformGrid1D}, solver; kwargs...)

Solve a Cahn-Hilliard problem on a 1D uniform grid using finite differences.

# Arguments
- `problem`: PhaseFieldProblem created with CahnHilliardProblem()
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Returns
- ODESolution with concentration field
"""
function solve(
    problem::PhaseFieldProblem{<:CahnHilliardProblemData, UniformGrid1D},
    solver;
    kwargs...
)
    # Extract model data
    model_data = problem.model
    model = model_data.model
    f = model_data.f

    # Create ODE parameters
    params = CahnHilliardODEParams(model, problem.domain, problem.bc, f)

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        cahn_hilliard_ode!,
        problem.φ0,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end

"""
    solve(problem::PhaseFieldProblem{<:CahnHilliardProblemData, UniformGrid2D}, solver; kwargs...)

Solve a Cahn-Hilliard problem on a 2D uniform grid using finite differences.

# Arguments
- `problem`: PhaseFieldProblem created with CahnHilliardProblem()
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Returns
- ODESolution with concentration field (vectorized)
"""
function solve(
    problem::PhaseFieldProblem{<:CahnHilliardProblemData, UniformGrid2D},
    solver;
    kwargs...
)
    # Extract model data
    model_data = problem.model
    model = model_data.model
    f = model_data.f

    # Vectorize initial condition
    c0_vec = vec(problem.φ0)

    # Create ODE parameters
    params = CahnHilliardODEParams(model, problem.domain, problem.bc, f)

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        cahn_hilliard_ode!,
        c0_vec,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end

# =============================================================================
# FDM Solver for KKSModel
# =============================================================================

"""
    solve(problem::PhaseFieldProblem{<:KKSProblemData, UniformGrid1D}, solver; kwargs...)

Solve a KKS problem on a 1D uniform grid using finite differences.

# Arguments
- `problem`: PhaseFieldProblem created with KKSProblem()
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Returns
- ODESolution with state vector [φ; c]
"""
function solve(
    problem::PhaseFieldProblem{<:KKSProblemData, UniformGrid1D},
    solver;
    kwargs...
)
    # Extract model data
    model_data = problem.model
    model = model_data.model
    f_s = model_data.f_s
    f_l = model_data.f_l

    # Extract initial conditions and boundary conditions
    φ0, c0 = problem.φ0
    bc_φ, bc_c = problem.bc
    grid = problem.domain

    # Create ODE parameters
    params = KKSODEParams(model, grid, bc_φ, bc_c, f_s, f_l)

    # Combine initial conditions
    y0 = vcat(φ0, c0)

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        kks_ode!,
        y0,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end

"""
    solve(problem::PhaseFieldProblem{<:KKSProblemData, UniformGrid2D}, solver; kwargs...)

Solve a KKS problem on a 2D uniform grid using finite differences.

# Arguments
- `problem`: PhaseFieldProblem created with KKSProblem()
- `solver`: ODE solver (e.g., Tsit5(), ROCK4())

# Returns
- ODESolution with state vector [φ_vec; c_vec]
"""
function solve(
    problem::PhaseFieldProblem{<:KKSProblemData, UniformGrid2D},
    solver;
    kwargs...
)
    # Extract model data
    model_data = problem.model
    model = model_data.model
    f_s = model_data.f_s
    f_l = model_data.f_l

    # Extract initial conditions and boundary conditions
    φ0, c0 = problem.φ0
    bc_φ, bc_c = problem.bc
    grid = problem.domain

    # Create ODE parameters
    params = KKSODEParams(model, grid, bc_φ, bc_c, f_s, f_l)

    # Vectorize and combine initial conditions
    y0 = vcat(vec(φ0), vec(c0))

    # Create ODE problem
    ode_prob = OrdinaryDiffEq.ODEProblem(
        kks_ode!,
        y0,
        problem.tspan,
        params
    )

    # Solve and return
    return OrdinaryDiffEq.solve(ode_prob, solver; kwargs...)
end
