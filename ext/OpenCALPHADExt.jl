# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - OpenCALPHAD.jl extension

"""
    OpenCALPHADExt

Extension module providing CALPHAD thermodynamic coupling for PhaseFields.jl.
Automatically loaded when OpenCALPHAD.jl is imported.

# Usage
```julia
using PhaseFields
using OpenCALPHAD  # Triggers extension loading

db = read_tdb("agcu.TDB")
ΔG = calphad_driving_force(db, 1000.0, 0.3, "FCC_A1", "LIQUID")
```
"""
module OpenCALPHADExt

using PhaseFields
using OpenCALPHAD

# Import functions that we will extend with new methods
import PhaseFields: free_energy, chemical_potential, d2f_dc2
import PhaseFields: allen_cahn_rhs, create_calphad_kks_model, create_calphad_wbm_model

# Debug: confirm extension loading
function __init__()
    @debug "PhaseFields: OpenCALPHAD extension loaded"
end

# =============================================================================
# CALPHAD-coupled Allen-Cahn model
# =============================================================================

"""
    CALPHADAllenCahnModel <: AbstractCALPHADCoupledModel

Allen-Cahn model with automatic CALPHAD thermodynamic coupling.

# Fields
- `base_model::AllenCahnModel`: Underlying Allen-Cahn model
- `db::OpenCALPHAD.Database`: Thermodynamic database
- `T::Float64`: Temperature [K]
- `x::Float64`: Composition (mole fraction)
- `solid_phase::String`: Name of solid phase
- `liquid_phase::String`: Name of liquid phase
- `ΔG::Float64`: Cached driving force [J/mol]

# Example
```julia
db = read_tdb("agcu.TDB")
model = create_calphad_allen_cahn(db, 1000.0, 0.3, "FCC_A1", "LIQUID")
dφdt = allen_cahn_rhs(model, φ, ∇²φ)  # Uses cached ΔG
```
"""
struct CALPHADAllenCahnModel <: PhaseFields.AbstractCALPHADCoupledModel
    base_model::PhaseFields.AllenCahnModel
    db::OpenCALPHAD.Database
    T::Float64
    x::Float64
    solid_phase::String
    liquid_phase::String
    ΔG::Float64
end

# =============================================================================
# Implementation of extension functions
# =============================================================================

"""
    PhaseFields.calphad_driving_force(db, T, x, solid_phase, liquid_phase)

Calculate driving force using OpenCALPHAD thermodynamic data.
Returns ΔG = G_solid - G_liquid [J/mol]. Negative means solid is stable.
"""
function PhaseFields.calphad_driving_force(
    db::OpenCALPHAD.Database,
    T::Real,
    x::Real,
    solid_phase::AbstractString,
    liquid_phase::AbstractString
)
    return OpenCALPHAD.driving_force(db, T, x, solid_phase, liquid_phase)
end

"""
    PhaseFields.calphad_chemical_potential(db, phase, T, x)

Get chemical potentials from CALPHAD database.
"""
function PhaseFields.calphad_chemical_potential(
    db::OpenCALPHAD.Database,
    phase_name::AbstractString,
    T::Real,
    x::Real
)
    phase = OpenCALPHAD.get_phase(db, phase_name)
    return OpenCALPHAD.chemical_potential(phase, T, x, db)
end

"""
    PhaseFields.calphad_diffusion_potential(db, phase, T, x)

Get second derivative of Gibbs energy d²G/dx² from CALPHAD.
"""
function PhaseFields.calphad_diffusion_potential(
    db::OpenCALPHAD.Database,
    phase_name::AbstractString,
    T::Real,
    x::Real
)
    phase = OpenCALPHAD.get_phase(db, phase_name)
    return OpenCALPHAD.diffusion_potential(phase, T, x, db)
end

"""
    PhaseFields.create_calphad_allen_cahn(db, T, x, solid_phase, liquid_phase; kwargs...)

Create an Allen-Cahn model coupled with CALPHAD thermodynamics.
"""
function PhaseFields.create_calphad_allen_cahn(
    db::OpenCALPHAD.Database,
    T::Real,
    x::Real,
    solid_phase::AbstractString,
    liquid_phase::AbstractString;
    τ::Real = 1.0,
    W::Real = 1.0,
    m::Real = 1e-4
)
    # Get driving force from CALPHAD
    ΔG = OpenCALPHAD.driving_force(db, T, x, solid_phase, liquid_phase)

    # Create base model
    base_model = PhaseFields.AllenCahnModel(τ=Float64(τ), W=Float64(W), m=Float64(m))

    return CALPHADAllenCahnModel(
        base_model, db, Float64(T), Float64(x),
        String(solid_phase), String(liquid_phase), ΔG
    )
end

# =============================================================================
# Extend Allen-Cahn RHS for CALPHAD-coupled model
# =============================================================================

"""
    allen_cahn_rhs(model::CALPHADAllenCahnModel, φ, ∇²φ)

Compute Allen-Cahn RHS using cached CALPHAD driving force.
"""
function PhaseFields.allen_cahn_rhs(
    model::CALPHADAllenCahnModel,
    φ::Real,
    ∇²φ::Real
)
    return PhaseFields.allen_cahn_rhs(model.base_model, φ, ∇²φ, model.ΔG)
end

# =============================================================================
# Utility functions
# =============================================================================

"""
    update_conditions(model::CALPHADAllenCahnModel; T=nothing, x=nothing)

Create a new model with updated temperature and/or composition.
Recalculates the driving force from CALPHAD.

# Example
```julia
model_new = update_conditions(model; T=1100.0)  # Change temperature
model_new = update_conditions(model; x=0.4)     # Change composition
```
"""
function update_conditions(
    model::CALPHADAllenCahnModel;
    T::Union{Nothing,Real} = nothing,
    x::Union{Nothing,Real} = nothing
)
    new_T = isnothing(T) ? model.T : Float64(T)
    new_x = isnothing(x) ? model.x : Float64(x)

    new_ΔG = OpenCALPHAD.driving_force(
        model.db, new_T, new_x, model.solid_phase, model.liquid_phase
    )

    return CALPHADAllenCahnModel(
        model.base_model, model.db, new_T, new_x,
        model.solid_phase, model.liquid_phase, new_ΔG
    )
end

"""
    get_driving_force(model::CALPHADAllenCahnModel)

Get the cached driving force from the model.
"""
get_driving_force(model::CALPHADAllenCahnModel) = model.ΔG

"""
    get_temperature(model::CALPHADAllenCahnModel)

Get the temperature from the model.
"""
get_temperature(model::CALPHADAllenCahnModel) = model.T

"""
    get_composition(model::CALPHADAllenCahnModel)

Get the composition from the model.
"""
get_composition(model::CALPHADAllenCahnModel) = model.x

# =============================================================================
# KKS Model with CALPHAD Free Energy
# =============================================================================

"""
    CALPHADFreeEnergy

Wrapper for CALPHAD thermodynamic data compatible with KKS model.
Provides the same interface as ParabolicFreeEnergy.

# Fields
- `db::OpenCALPHAD.Database`: Thermodynamic database
- `phase::OpenCALPHAD.Phase`: Phase object
- `T::Float64`: Temperature [K]

# Example
```julia
db = read_tdb("agcu.TDB")
f_s = CALPHADFreeEnergy(db, "FCC_A1", 1000.0)
f_l = CALPHADFreeEnergy(db, "LIQUID", 1000.0)
c_s, c_l, μ, converged = kks_partition(0.5, 0.5, f_s, f_l)
```
"""
struct CALPHADFreeEnergy
    db::OpenCALPHAD.Database
    phase::OpenCALPHAD.Phase
    T::Float64
end

"""
    CALPHADFreeEnergy(db, phase_name, T)

Create a CALPHAD free energy wrapper from database and phase name.
"""
function CALPHADFreeEnergy(
    db::OpenCALPHAD.Database,
    phase_name::AbstractString,
    T::Real
)
    phase = OpenCALPHAD.get_phase(db, phase_name)
    return CALPHADFreeEnergy(db, phase, Float64(T))
end

"""
    free_energy(f::CALPHADFreeEnergy, c)

Get Gibbs energy from CALPHAD at composition c.
"""
function free_energy(f::CALPHADFreeEnergy, c::Real)
    return OpenCALPHAD.gibbs_energy(f.phase, f.T, c, f.db)
end

"""
    chemical_potential(f::CALPHADFreeEnergy, c)

Get chemical potential (dG/dc) from CALPHAD at composition c.
"""
function chemical_potential(f::CALPHADFreeEnergy, c::Real)
    μ = OpenCALPHAD.chemical_potential(f.phase, f.T, c, f.db)
    # Return the diffusion potential (μ₂ - μ₁) for binary system
    return μ[2] - μ[1]
end

"""
    d2f_dc2(f::CALPHADFreeEnergy, c)

Get second derivative of Gibbs energy (d²G/dc²) from CALPHAD.
"""
function d2f_dc2(f::CALPHADFreeEnergy, c::Real)
    return OpenCALPHAD.diffusion_potential(f.phase, f.T, c, f.db)
end

"""
    create_calphad_kks_model(db, T, solid_phase, liquid_phase; τ=1.0, W=1.0, m=1.0, M_s=1.0, M_l=10.0)

Create a KKS model setup with CALPHAD free energies.

# Arguments
- `db`: OpenCALPHAD Database object
- `T`: Temperature [K]
- `solid_phase`: Name of solid phase
- `liquid_phase`: Name of liquid phase

# Returns
- `(model, f_s, f_l)`: KKS model and CALPHAD free energy wrappers

# Example
```julia
db = read_tdb("agcu.TDB")
model, f_s, f_l = create_calphad_kks_model(db, 1000.0, "FCC_A1", "LIQUID")
c_s, c_l, μ, converged = kks_partition(0.5, 0.5, f_s, f_l)
```
"""
function create_calphad_kks_model(
    db::OpenCALPHAD.Database,
    T::Real,
    solid_phase::AbstractString,
    liquid_phase::AbstractString;
    τ::Real = 1.0,
    W::Real = 1.0,
    m::Real = 1.0,
    M_s::Real = 1.0,
    M_l::Real = 10.0
)
    model = PhaseFields.KKSModel(
        τ=Float64(τ), W=Float64(W), m=Float64(m),
        M_s=Float64(M_s), M_l=Float64(M_l)
    )
    f_s = CALPHADFreeEnergy(db, solid_phase, Float64(T))
    f_l = CALPHADFreeEnergy(db, liquid_phase, Float64(T))
    return (model, f_s, f_l)
end

# =============================================================================
# WBM Model with CALPHAD Free Energy
# =============================================================================

"""
    create_calphad_wbm_model(db, T, solid_phase, liquid_phase; M_φ=1.0, κ=1.0, W=1.0, D_s=1e-13, D_l=1e-9)

Create a WBM model setup with CALPHAD free energies.

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
- `(model, f_s, f_l)`: WBM model and CALPHAD free energy wrappers

# Example
```julia
db = read_tdb("agcu.TDB")
model, f_s, f_l = create_calphad_wbm_model(db, 1000.0, "FCC_A1", "LIQUID")
f_bulk = wbm_bulk_free_energy(f_s, f_l, φ, c, model.W)
```

# Notes
The CALPHADFreeEnergy wrapper provides the same interface as ParabolicFreeEnergy:
- `free_energy(f, c)`: Gibbs energy at composition c
- `chemical_potential(f, c)`: dG/dc (diffusion potential for binary)
- `d2f_dc2(f, c)`: d²G/dc²
"""
function create_calphad_wbm_model(
    db::OpenCALPHAD.Database,
    T::Real,
    solid_phase::AbstractString,
    liquid_phase::AbstractString;
    M_φ::Real = 1.0,
    κ::Real = 1.0,
    W::Real = 1.0,
    D_s::Real = 1e-13,
    D_l::Real = 1e-9
)
    model = PhaseFields.WBMModel(
        M_φ=Float64(M_φ), κ=Float64(κ), W=Float64(W),
        D_s=Float64(D_s), D_l=Float64(D_l)
    )
    f_s = CALPHADFreeEnergy(db, solid_phase, Float64(T))
    f_l = CALPHADFreeEnergy(db, liquid_phase, Float64(T))
    return (model, f_s, f_l)
end

# Export extension-specific types and functions
export CALPHADAllenCahnModel, CALPHADFreeEnergy
export update_conditions, get_driving_force, get_temperature, get_composition
export create_calphad_kks_model, create_calphad_wbm_model

end # module OpenCALPHADExt
