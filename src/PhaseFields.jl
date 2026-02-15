# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl

"""
    PhaseFields.jl

Phase Field method simulation package for materials science.
Provides domain-specific functionality for solidification modeling
with CALPHAD thermodynamic coupling.

# Features
- Interface modeling (interpolation, double-well, anisotropy)
- Phase field models (Allen-Cahn, Cahn-Hilliard, KKS, WBM)
- CALPHAD coupling via OpenCALPHAD.jl (optional extension)
- AD-compatible design using DifferentiationInterface.jl
"""
module PhaseFields

using LinearAlgebra
using SparseArrays
using DifferentiationInterface
import DifferentiationInterface as DI
using ForwardDiff
using OrdinaryDiffEq

# AD backend configuration
include("differentiation.jl")

# Type definitions
include("types.jl")

# Interface modeling functions
include("interpolation.jl")
include("anisotropy.jl")

# Phase field models
include("models/abstract.jl")
include("models/allen_cahn.jl")
include("models/cahn_hilliard.jl")
include("models/kks.jl")
include("models/wbm.jl")
include("models/thermal.jl")

# Problem types (must be before diffeq.jl for AbstractDomain)
include("problems/abstract.jl")
include("problems/problem.jl")

# DifferentialEquations.jl integration
include("integration/diffeq.jl")
include("integration/callbacks.jl")

# Problem solvers (must be after diffeq.jl for UniformGrid1D)
include("problems/solvers.jl")

# =============================================================================
# Gridap Extension - Placeholder for extension
# These are implemented by ext/GridapExt.jl when Gridap is loaded
# =============================================================================

"""
    GridapDomain

FEM domain using Gridap.jl backend.

**Requires Gridap.jl**: Load with `using Gridap` to enable this type.

See GridapExt documentation for details.
"""
function GridapDomain end

# =============================================================================
# CALPHAD Coupling - Zero-method definitions for extension
# These are implemented by ext/OpenCALPHADExt.jl when OpenCALPHAD is loaded
# =============================================================================

"""
    AbstractCALPHADCoupledModel

Abstract type for phase field models with CALPHAD thermodynamic coupling.
Concrete implementations are provided by the OpenCALPHAD extension.
"""
abstract type AbstractCALPHADCoupledModel end

"""
    calphad_driving_force(db, T, x, solid_phase, liquid_phase)

Calculate the thermodynamic driving force for solidification using CALPHAD data.

**Requires OpenCALPHAD.jl**: Load with `using OpenCALPHAD` to enable this function.

# Arguments
- `db`: OpenCALPHAD Database object
- `T`: Temperature [K]
- `x`: Composition (mole fraction of solute)
- `solid_phase`: Name of solid phase (e.g., "FCC_A1")
- `liquid_phase`: Name of liquid phase (e.g., "LIQUID")

# Returns
- Driving force ΔG [J/mol] (negative = solid stable)
"""
function calphad_driving_force end

"""
    calphad_chemical_potential(db, phase, T, x)

Get chemical potentials from CALPHAD database.

**Requires OpenCALPHAD.jl**: Load with `using OpenCALPHAD` to enable this function.

# Returns
- Tuple of chemical potentials (μ₁, μ₂, ...) [J/mol]
"""
function calphad_chemical_potential end

"""
    calphad_diffusion_potential(db, phase, T, x)

Get second derivative of Gibbs energy d²G/dx² from CALPHAD.

**Requires OpenCALPHAD.jl**: Load with `using OpenCALPHAD` to enable this function.

# Returns
- d²G/dx² [J/mol]
"""
function calphad_diffusion_potential end

"""
    create_calphad_allen_cahn(db, T, x, solid_phase, liquid_phase; kwargs...)

Create an Allen-Cahn model coupled with CALPHAD thermodynamics.

**Requires OpenCALPHAD.jl**: Load with `using OpenCALPHAD` to enable this function.

# Arguments
- `db`: OpenCALPHAD Database object
- `T`: Temperature [K]
- `x`: Composition (mole fraction)
- `solid_phase`: Name of solid phase
- `liquid_phase`: Name of liquid phase

# Keyword Arguments
- `τ`: Relaxation time [s] (default: 1.0)
- `W`: Interface width parameter (default: 1.0)
- `m`: Scaling factor for ΔG (default: 1e-4)

# Returns
- CALPHADAllenCahnModel with cached driving force
"""
function create_calphad_allen_cahn end

"""
    create_calphad_kks_model(db, T, solid_phase, liquid_phase; kwargs...)

Create a KKS model setup with CALPHAD free energies.

**Requires OpenCALPHAD.jl**: Load with `using OpenCALPHAD` to enable this function.

# Arguments
- `db`: OpenCALPHAD Database object
- `T`: Temperature [K]
- `solid_phase`: Name of solid phase
- `liquid_phase`: Name of liquid phase

# Keyword Arguments
- `τ`: Relaxation time [s] (default: 1.0)
- `W`: Interface width parameter (default: 1.0)
- `m`: Driving force scale (default: 1.0)
- `M_s`: Solid mobility (default: 1.0)
- `M_l`: Liquid mobility (default: 10.0)

# Returns
- Tuple of (KKSModel, CALPHADFreeEnergy_solid, CALPHADFreeEnergy_liquid)
"""
function create_calphad_kks_model end

"""
    create_calphad_wbm_model(db, T, solid_phase, liquid_phase; kwargs...)

Create a WBM model setup with CALPHAD free energies.

**Requires OpenCALPHAD.jl**: Load with `using OpenCALPHAD` to enable this function.

# Arguments
- `db`: OpenCALPHAD Database object
- `T`: Temperature [K]
- `solid_phase`: Name of solid phase
- `liquid_phase`: Name of liquid phase

# Keyword Arguments
- `M_φ`: Phase field mobility (default: 1.0)
- `κ`: Gradient energy coefficient (default: 1.0)
- `W`: Barrier height (default: 1.0)
- `D_s`: Solid diffusivity (default: 1e-13)
- `D_l`: Liquid diffusivity (default: 1e-9)

# Returns
- Tuple of (WBMModel, CALPHADFreeEnergy_solid, CALPHADFreeEnergy_liquid)
"""
function create_calphad_wbm_model end

# =============================================================================
# Exports - Types
export InterfaceParams, DiffusionParams, MaterialParams

# Exports - Interpolation functions
export h_polynomial, h_sin, g_standard, g_obstacle
export h_prime, g_prime, g_double_prime

# Exports - Anisotropy functions
export anisotropy_cubic, anisotropy_hcp, anisotropy_custom

# Exports - AD utilities
export DEFAULT_AD_BACKEND, set_ad_backend!

# Exports - Abstract types
export AbstractPhaseFieldModel
export AbstractDomain, AbstractPhaseFieldProblem

# Exports - Domain interface functions
export gridsize, spacing, coordinates
export laplacian!

# Exports - Problem types
export PhaseFieldProblem, AllenCahnProblem, ThermalProblem
export WBMProblem, CahnHilliardProblem, KKSProblem

# Exports - Models (Allen-Cahn)
export AllenCahnModel, allen_cahn_rhs

# Exports - Models (Cahn-Hilliard)
export CahnHilliardModel, DoubleWellFreeEnergy
export free_energy_density, chemical_potential_bulk
export cahn_hilliard_chemical_potential, cahn_hilliard_rhs
export cahn_hilliard_interface_width, cahn_hilliard_stability_dt

# Exports - Models (KKS)
export KKSModel, ParabolicFreeEnergy
export free_energy, chemical_potential, d2f_dc2
export kks_partition, kks_grand_potential_diff
export kks_phase_rhs, kks_concentration_rhs, kks_mobility
export kks_interface_width

# Exports - Models (WBM)
export WBMModel
export wbm_bulk_free_energy, wbm_chemical_potential, wbm_driving_force
export wbm_phase_rhs, wbm_concentration_rhs, wbm_diffusivity
export wbm_interface_width, wbm_interface_energy

# Exports - Models (Thermal)
export ThermalPhaseFieldModel
export dimensionless_temperature, physical_temperature, stefan_number
export thermal_phase_rhs, thermal_heat_rhs, thermal_stability_dt

# Exports - DifferentialEquations.jl Integration
export UniformGrid1D, UniformGrid2D
export BoundaryCondition, NeumannBC, DirichletBC, DirichletBC2D, PeriodicBC
export laplacian_1d!, laplacian_1d, laplacian_matrix_1d
export laplacian_2d!, laplacian_2d
export AllenCahnODEParams, allen_cahn_ode!, create_allen_cahn_problem
export ThermalODEParams, thermal_ode!, create_thermal_problem
export extract_thermal_solution
export WBMODEParams, wbm_ode!
export CahnHilliardODEParams, cahn_hilliard_ode!
export KKSODEParams, kks_ode!

# Exports - Callbacks
export interface_position_1d, solid_fraction
export create_interface_saving_callback, create_solid_fraction_callback
export create_steady_state_callback, create_phase_field_callbacks

# Exports - Gridap Extension (implemented by GridapExt extension)
export GridapDomain

# Exports - CALPHAD Coupling (implemented by OpenCALPHADExt extension)
export AbstractCALPHADCoupledModel
export calphad_driving_force, calphad_chemical_potential, calphad_diffusion_potential
export create_calphad_allen_cahn, create_calphad_kks_model, create_calphad_wbm_model

end # module PhaseFields
