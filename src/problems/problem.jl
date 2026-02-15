# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Phase field problem definitions

"""
    PhaseFieldProblem

Generic phase field problem combining model, domain, and conditions.

# Fields
- `model`: Phase field model (e.g., AllenCahnModel, CahnHilliardModel)
- `domain`: Computational domain (e.g., UniformGrid1D)
- `φ0`: Initial condition for phase field
- `tspan`: Time span as tuple (t_start, t_end)
- `bc`: Boundary condition (default: NeumannBC())

# Example
```julia
model = AllenCahnModel()
grid = UniformGrid1D(N=100, L=1.0)
φ0 = zeros(100)
tspan = (0.0, 1.0)

problem = PhaseFieldProblem(
    model=model,
    domain=grid,
    φ0=φ0,
    tspan=tspan
)
```
"""
@kwdef struct PhaseFieldProblem{M, D, T, BC} <: AbstractPhaseFieldProblem
    model::M
    domain::D
    φ0::T
    tspan::Tuple{Float64, Float64}
    bc::BC = NeumannBC()
end

"""
    AllenCahnProblem(model, domain, φ0, tspan; bc=NeumannBC())

Convenience constructor for Allen-Cahn problems.

# Arguments
- `model`: AllenCahnModel instance
- `domain`: Computational domain
- `φ0`: Initial phase field
- `tspan`: Time span (t_start, t_end)

# Keyword Arguments
- `bc`: Boundary condition (default: NeumannBC())

# Returns
- `PhaseFieldProblem` configured for Allen-Cahn simulation
"""
function AllenCahnProblem(model, domain, φ0, tspan; bc=NeumannBC())
    return PhaseFieldProblem(
        model=model,
        domain=domain,
        φ0=φ0,
        tspan=tspan,
        bc=bc
    )
end

"""
    ThermalProblem(model, domain, φ0, u0, tspan; bc_φ=NeumannBC(), bc_u=NeumannBC())

Convenience constructor for thermal phase field problems.

# Arguments
- `model`: ThermalPhaseFieldModel instance
- `domain`: Computational domain (UniformGrid1D or UniformGrid2D)
- `φ0`: Initial phase field
- `u0`: Initial dimensionless temperature field
- `tspan`: Time span (t_start, t_end)

# Keyword Arguments
- `bc_φ`: Phase field boundary condition (default: NeumannBC())
- `bc_u`: Temperature boundary condition (default: NeumannBC())

# Returns
- `PhaseFieldProblem` configured for thermal phase field simulation

# Note
The initial condition is stored as a tuple (φ0, u0) in problem.φ0.
Boundary conditions are stored as a tuple (bc_φ, bc_u) in problem.bc.
"""
function ThermalProblem(model, domain, φ0, u0, tspan; bc_φ=NeumannBC(), bc_u=NeumannBC())
    return PhaseFieldProblem(
        model=model,
        domain=domain,
        φ0=(φ0, u0),
        tspan=tspan,
        bc=(bc_φ, bc_u)
    )
end

"""
    WBMProblem(model, domain, φ0, c0, tspan, f_s, f_l; bc_φ=NeumannBC(), bc_c=NeumannBC())

Convenience constructor for WBM (Wheeler-Boettinger-McFadden) problems.

# Arguments
- `model`: WBMModel instance
- `domain`: Computational domain (UniformGrid1D or UniformGrid2D)
- `φ0`: Initial phase field
- `c0`: Initial concentration field
- `tspan`: Time span (t_start, t_end)
- `f_s`: Solid phase free energy (ParabolicFreeEnergy)
- `f_l`: Liquid phase free energy (ParabolicFreeEnergy)

# Keyword Arguments
- `bc_φ`: Phase field boundary condition (default: NeumannBC())
- `bc_c`: Concentration boundary condition (default: NeumannBC())

# Returns
- `PhaseFieldProblem` configured for WBM simulation

# Note
The initial condition is stored as a tuple (φ0, c0) in problem.φ0.
Boundary conditions are stored as a tuple (bc_φ, bc_c) in problem.bc.
Free energies are stored as a tuple (f_s, f_l) appended to model.
"""
struct WBMProblemData{M, F1, F2}
    model::M
    f_s::F1
    f_l::F2
end

function WBMProblem(model, domain, φ0, c0, tspan, f_s, f_l;
                    bc_φ=NeumannBC(), bc_c=NeumannBC())
    model_data = WBMProblemData(model, f_s, f_l)
    return PhaseFieldProblem(
        model=model_data,
        domain=domain,
        φ0=(φ0, c0),
        tspan=tspan,
        bc=(bc_φ, bc_c)
    )
end

"""
    CahnHilliardProblem(model, domain, c0, tspan, f; bc=NeumannBC())

Convenience constructor for Cahn-Hilliard problems.

# Arguments
- `model`: CahnHilliardModel instance
- `domain`: Computational domain (UniformGrid1D or UniformGrid2D)
- `c0`: Initial concentration field
- `tspan`: Time span (t_start, t_end)
- `f`: Free energy (DoubleWellFreeEnergy)

# Keyword Arguments
- `bc`: Boundary condition (default: NeumannBC())

# Returns
- `PhaseFieldProblem` configured for Cahn-Hilliard simulation

# Note
The free energy is stored with the model as CahnHilliardProblemData.
"""
struct CahnHilliardProblemData{M, F}
    model::M
    f::F
end

function CahnHilliardProblem(model, domain, c0, tspan, f; bc=NeumannBC())
    model_data = CahnHilliardProblemData(model, f)
    return PhaseFieldProblem(
        model=model_data,
        domain=domain,
        φ0=c0,
        tspan=tspan,
        bc=bc
    )
end

"""
    KKSProblem(model, domain, φ0, c0, tspan, f_s, f_l; bc_φ=NeumannBC(), bc_c=NeumannBC())

Convenience constructor for KKS (Kim-Kim-Suzuki) problems.

# Arguments
- `model`: KKSModel instance
- `domain`: Computational domain (UniformGrid1D or UniformGrid2D)
- `φ0`: Initial phase field
- `c0`: Initial concentration field
- `tspan`: Time span (t_start, t_end)
- `f_s`: Solid phase free energy (ParabolicFreeEnergy)
- `f_l`: Liquid phase free energy (ParabolicFreeEnergy)

# Keyword Arguments
- `bc_φ`: Phase field boundary condition (default: NeumannBC())
- `bc_c`: Concentration boundary condition (default: NeumannBC())

# Returns
- `PhaseFieldProblem` configured for KKS simulation

# Note
The initial condition is stored as a tuple (φ0, c0) in problem.φ0.
Boundary conditions are stored as a tuple (bc_φ, bc_c) in problem.bc.
Free energies are stored with the model as KKSProblemData.
"""
struct KKSProblemData{M, F1, F2}
    model::M
    f_s::F1
    f_l::F2
end

function KKSProblem(model, domain, φ0, c0, tspan, f_s, f_l;
                    bc_φ=NeumannBC(), bc_c=NeumannBC())
    model_data = KKSProblemData(model, f_s, f_l)
    return PhaseFieldProblem(
        model=model_data,
        domain=domain,
        φ0=(φ0, c0),
        tspan=tspan,
        bc=(bc_φ, bc_c)
    )
end

"""
    solve(problem::AbstractPhaseFieldProblem, solver)

Solve a phase field problem with the specified solver.

This is a fallback method that throws an error for unsupported domain types.
Specialized methods dispatch on the domain type to select appropriate
numerical methods (FDM, FEM, etc.).

# Throws
- `ErrorException` if no solver is implemented for the domain type
"""
function solve(problem::AbstractPhaseFieldProblem, solver)
    domain_type = typeof(problem.domain)
    error("No solver implemented for domain type $(domain_type). " *
          "Ensure the domain type is supported or implement a specialized solve method.")
end

