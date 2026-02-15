# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Gridap Extension for FEM

module GridapExt

using PhaseFields
using Gridap
using Gridap.FESpaces
using Gridap.CellData

# =============================================================================
# GridapDomain Type
# =============================================================================

"""
    GridapDomain{N, M<:Gridap.DiscreteModel} <: PhaseFields.AbstractDomain{N}

FEM domain using Gridap.jl backend.

Type parameters:
- `N`: Spatial dimension (2 or 3)
- `M`: DiscreteModel subtype (CartesianDiscreteModel or GmshDiscreteModel)

The parametric design allows future extension to non-rectangular domains (GMSH)
without breaking changes.

# Fields
- `model::M`: Gridap DiscreteModel
- `Ω::Triangulation`: Domain triangulation
- `dΩ::Measure`: Integration measure
- `order::Int`: Polynomial order for FE spaces

# Example
```julia
using Gridap
domain = GridapDomain((0, 1, 0, 1), (50, 50))
```
"""
struct GridapDomain{N, M<:Gridap.DiscreteModel} <: PhaseFields.AbstractDomain{N}
    model::M
    Ω::Gridap.Triangulation
    dΩ::Gridap.CellData.Measure
    order::Int
end

"""
    GridapDomain(domain::NTuple{4,Real}, partition::NTuple{2,Int}; order=1)

Create a 2D FEM domain on a rectangular Cartesian grid.

# Arguments
- `domain`: (x_min, x_max, y_min, y_max)
- `partition`: (Nx, Ny) number of elements
- `order`: Polynomial order (default: 1)

# Example
```julia
domain = GridapDomain((0, 1, 0, 1), (50, 50))
```
"""
function GridapDomain(
    domain::NTuple{4, Real},
    partition::NTuple{2, Int};
    order::Int = 1
)
    model = Gridap.CartesianDiscreteModel(domain, partition)
    Ω = Gridap.Triangulation(model)
    degree = 2 * order
    dΩ = Gridap.CellData.Measure(Ω, degree)
    return GridapDomain{2, typeof(model)}(model, Ω, dΩ, order)
end

# Export GridapDomain through PhaseFields
PhaseFields.GridapDomain = GridapDomain

# =============================================================================
# FEMSolution Type
# =============================================================================

"""
    FEMSolution{N}

Solution from Gridap FEM solver.

# Fields
- `times::Vector{Float64}`: Time points
- `φ_history::Vector{Vector{Float64}}`: Phase field values at each time
- `domain::GridapDomain{N}`: FEM domain
"""
struct FEMSolution{N}
    times::Vector{Float64}
    φ_history::Vector{Vector{Float64}}
    domain::GridapDomain{N}
end

# =============================================================================
# Allen-Cahn FEM Solver
# =============================================================================

# Double-well potential derivative: g'(φ) = 2φ(1-φ)(1-2φ)
_g_prime(φ) = 2 * φ * (1 - φ) * (1 - 2φ)

"""
    solve(problem::PhaseFieldProblem{<:AllenCahnModel, <:GridapDomain{2}}, solver; kwargs...)

Solve an Allen-Cahn problem on a 2D FEM domain using Gridap.

Uses semi-implicit time stepping with Newton iteration.

# Arguments
- `problem`: PhaseFieldProblem with AllenCahnModel and GridapDomain{2}
- `solver`: Solver specification (e.g., `:newton`)

# Keyword Arguments
- `Δt::Float64`: Time step (default: 0.01)
- `save_every::Int`: Save every N steps (default: 10)

# Returns
- `FEMSolution{2}` containing time evolution data
"""
function PhaseFields.solve(
    problem::PhaseFields.PhaseFieldProblem{<:PhaseFields.AllenCahnModel, <:GridapDomain{2}},
    solver::Symbol;
    Δt::Float64 = 0.01,
    save_every::Int = 10
)
    model = problem.model
    domain = problem.domain

    τ = model.τ
    W = model.W

    # FE Spaces
    reffe = Gridap.ReferenceFE(Gridap.lagrangian, Float64, domain.order)
    V = Gridap.TestFESpace(domain.model, reffe; conformity=:H1)
    U = Gridap.TrialFESpace(V)

    # Integration measure
    dΩ = domain.dΩ

    # Initial condition
    φ_h = Gridap.interpolate_everywhere(problem.φ0, U)

    # Time stepping setup
    t = problem.tspan[1]
    T_end = problem.tspan[2]
    times = [t]
    φ_history = [copy(Gridap.get_free_dof_values(φ_h))]

    step = 0

    # Weak form functions
    function create_residual(φ_old)
        function res(u, v)
            # Time derivative term (implicit)
            term1 = (τ / Δt) * u * v
            # Diffusion term (implicit)
            term2 = W^2 * (∇(u) ⋅ ∇(v))
            # Double-well term (explicit, evaluated at φ_old)
            term3 = _g_prime ∘ φ_old * v
            # RHS: φⁿ contribution
            rhs = (τ / Δt) * φ_old * v

            return ∫(term1 + term2 + term3 - rhs) * dΩ
        end
        return res
    end

    function create_jacobian(φ_old)
        function jac(u, du, v)
            # Linearization of residual w.r.t. u
            term1 = (τ / Δt) * du * v
            term2 = W^2 * (∇(du) ⋅ ∇(v))
            return ∫(term1 + term2) * dΩ
        end
        return jac
    end

    # Time stepping loop
    while t < T_end
        step += 1
        t += Δt

        # Create operators for current time step
        res = create_residual(φ_h)
        jac = create_jacobian(φ_h)

        # Solve nonlinear problem
        op = Gridap.FEOperator(res, jac, U, V)

        # Newton solver
        nls = Gridap.NLSolver(show_trace=false, method=:newton, iterations=10)
        fe_solver = Gridap.FESolver(nls)

        # Solve
        φ_new, cache = Gridap.solve!(φ_h, fe_solver, op)

        # Update for next time step
        φ_h = φ_new

        # Save snapshot
        if step % save_every == 0 || t >= T_end
            push!(φ_history, copy(Gridap.get_free_dof_values(φ_h)))
            push!(times, t)
        end
    end

    return FEMSolution{2}(times, φ_history, domain)
end

end # module GridapExt
