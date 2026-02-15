# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Cahn-Hilliard equation model

"""
    CahnHilliardModel

Cahn-Hilliard model for conserved order parameter (concentration) evolution.

The Cahn-Hilliard equation:
    ∂c/∂t = ∇·(M∇μ)
    μ = df/dc - κ∇²c

where:
- c: Concentration field (conserved quantity)
- M: Mobility [m²/(J·s)]
- κ: Gradient energy coefficient [J·m²/mol] or [J/m] depending on normalization
- μ: Chemical potential [J/mol]
- f(c): Bulk free energy density [J/mol]

# Fields
- `M::Float64`: Mobility
- `κ::Float64`: Gradient energy coefficient

# Example
```julia
model = CahnHilliardModel(M=5.0, κ=2.0)
```
"""
@kwdef struct CahnHilliardModel <: AbstractPhaseFieldModel
    M::Float64 = 1.0    # Mobility
    κ::Float64 = 1.0    # Gradient energy coefficient
end

"""
    DoubleWellFreeEnergy

Symmetric double-well free energy density for binary phase separation.

    f(c) = ρs · (c - cα)² · (cβ - c)²

Minima at c = cα (α-phase) and c = cβ (β-phase).

# Fields
- `ρs::Float64`: Barrier height [J/mol]
- `cα::Float64`: α-phase equilibrium concentration
- `cβ::Float64`: β-phase equilibrium concentration

# Example
```julia
# PFHub BM1 parameters
f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)
```
"""
struct DoubleWellFreeEnergy
    ρs::Float64     # Barrier height
    cα::Float64     # α-phase equilibrium concentration
    cβ::Float64     # β-phase equilibrium concentration
end

# Constructor with keyword arguments
function DoubleWellFreeEnergy(; ρs::Float64, cα::Float64, cβ::Float64)
    return DoubleWellFreeEnergy(ρs, cα, cβ)
end

"""
    free_energy_density(f::DoubleWellFreeEnergy, c::T) where T<:Real

Compute bulk free energy density f(c).

    f(c) = ρs · (c - cα)² · (cβ - c)²

This function is AD-compatible.

# Arguments
- `f`: DoubleWellFreeEnergy parameters
- `c`: Concentration

# Returns
- Free energy density [J/mol]

# Example
```julia
f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)
energy = free_energy_density(f, 0.5)  # At spinodal point
```
"""
function free_energy_density(f::DoubleWellFreeEnergy, c::T) where T<:Real
    return f.ρs * (c - f.cα)^2 * (f.cβ - c)^2
end

"""
    chemical_potential_bulk(f::DoubleWellFreeEnergy, c::T) where T<:Real

Compute bulk chemical potential df/dc.

    df/dc = 2ρs · (c - cα) · (cβ - c) · (cα + cβ - 2c)

This function is AD-compatible.

# Arguments
- `f`: DoubleWellFreeEnergy parameters
- `c`: Concentration

# Returns
- Bulk chemical potential [J/mol]

# Example
```julia
f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)
μ_bulk = chemical_potential_bulk(f, 0.5)  # = 0 at c = (cα+cβ)/2
```
"""
function chemical_potential_bulk(f::DoubleWellFreeEnergy, c::T) where T<:Real
    # df/dc = 2ρs(c - cα)(cβ - c)(cα + cβ - 2c)
    return 2 * f.ρs * (c - f.cα) * (f.cβ - c) * (f.cα + f.cβ - 2 * c)
end

"""
    chemical_potential_bulk_deriv(f::DoubleWellFreeEnergy, c::T) where T<:Real

Compute second derivative of free energy d²f/dc².

Used for stability analysis and numerical schemes.

# Arguments
- `f`: DoubleWellFreeEnergy parameters
- `c`: Concentration

# Returns
- d²f/dc² [J/mol]
"""
function chemical_potential_bulk_deriv(f::DoubleWellFreeEnergy, c::T) where T<:Real
    # d²f/dc² = 2ρs[(cβ-c)(cα+cβ-2c) - (c-cα)(cα+cβ-2c) - 2(c-cα)(cβ-c)]
    # Simplified: d²f/dc² = 2ρs[12c² - 12c(cα+cβ)/2 + 2cαcβ + (cα+cβ)² - 2(cα²+cβ²)]
    # More directly from expanding:
    cα, cβ, ρs = f.cα, f.cβ, f.ρs
    return 2 * ρs * (12 * c^2 - 6 * (cα + cβ) * c + cα^2 + 4 * cα * cβ + cβ^2)
end

"""
    cahn_hilliard_chemical_potential(model::CahnHilliardModel, f::DoubleWellFreeEnergy,
                                     c::T, ∇²c::T) where T<:Real

Compute total chemical potential including gradient term.

    μ = df/dc - κ∇²c

# Arguments
- `model`: CahnHilliardModel parameters
- `f`: DoubleWellFreeEnergy for bulk contribution
- `c`: Concentration at current point
- `∇²c`: Laplacian of concentration

# Returns
- Total chemical potential [J/mol]

# Example
```julia
model = CahnHilliardModel(M=5.0, κ=2.0)
f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)
μ = cahn_hilliard_chemical_potential(model, f, 0.5, -0.01)
```
"""
function cahn_hilliard_chemical_potential(model::CahnHilliardModel,
                                          f::DoubleWellFreeEnergy,
                                          c::T, ∇²c::T) where T<:Real
    μ_bulk = chemical_potential_bulk(f, c)
    μ_grad = -model.κ * ∇²c
    return μ_bulk + μ_grad
end

"""
    cahn_hilliard_rhs(model::CahnHilliardModel, ∇²μ::T) where T<:Real

Compute the right-hand side of the Cahn-Hilliard equation.

    ∂c/∂t = M∇²μ

Note: This assumes ∇·(M∇μ) ≈ M∇²μ for constant mobility.

# Arguments
- `model`: CahnHilliardModel parameters
- `∇²μ`: Laplacian of chemical potential

# Returns
- Time derivative ∂c/∂t

# Example
```julia
model = CahnHilliardModel(M=5.0, κ=2.0)
dcdt = cahn_hilliard_rhs(model, 100.0)
```
"""
function cahn_hilliard_rhs(model::CahnHilliardModel, ∇²μ::T) where T<:Real
    return model.M * ∇²μ
end

"""
    cahn_hilliard_interface_width(model::CahnHilliardModel, f::DoubleWellFreeEnergy)

Estimate equilibrium interface width.

    W ≈ √(κ / (ρs · (cβ - cα)²))

# Arguments
- `model`: CahnHilliardModel parameters
- `f`: DoubleWellFreeEnergy parameters

# Returns
- Approximate interface width
"""
function cahn_hilliard_interface_width(model::CahnHilliardModel, f::DoubleWellFreeEnergy)
    Δc = f.cβ - f.cα
    return sqrt(model.κ / (f.ρs * Δc^2))
end

"""
    cahn_hilliard_stability_dt(model::CahnHilliardModel, dx::Float64)

Estimate maximum stable time step for explicit Euler scheme.

For the 4th-order Cahn-Hilliard equation, stability requires:
    dt < dx⁴ / (16 · M · κ)

# Arguments
- `model`: CahnHilliardModel parameters
- `dx`: Grid spacing

# Returns
- Maximum stable time step

# Example
```julia
model = CahnHilliardModel(M=5.0, κ=2.0)
dt_max = cahn_hilliard_stability_dt(model, 1.0)  # ≈ 0.00625
```
"""
function cahn_hilliard_stability_dt(model::CahnHilliardModel, dx::Float64)
    return dx^4 / (16 * model.M * model.κ)
end
