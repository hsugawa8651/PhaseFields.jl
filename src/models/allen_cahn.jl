# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Allen-Cahn equation model

"""
    AllenCahnModel(; τ=1.0, W=1.0, m=1.0)

Allen-Cahn model for non-conserved order parameter evolution.

The Allen-Cahn equation:
    τ ∂φ/∂t = W²∇²φ - g'(φ) + m·ΔG·h'(φ)

where:
- φ: Order parameter (0 = solid, 1 = liquid)
- τ: Relaxation time [s]
- W: Interface width parameter [m]
- g(φ): Double-well potential
- h(φ): Interpolation function
- ΔG: Driving force [J/mol]
- m: Driving force coupling constant

# Fields
- `τ::Float64`: Relaxation time [s] (default: 1.0)
- `W::Float64`: Interface width parameter [m] (default: 1.0)
- `m::Float64`: Driving force scale [-] (default: 1.0)

# Example
```julia
model = AllenCahnModel()                    # all defaults
model = AllenCahnModel(τ=3e-8, W=1e-6)      # partial
model = AllenCahnModel(τ=3e-8, W=1e-6, m=1.0)  # full
```
"""
@kwdef struct AllenCahnModel <: AbstractPhaseFieldModel
    τ::Float64 = 1.0    # Relaxation time [s]
    W::Float64 = 1.0    # Interface width parameter [m]
    m::Float64 = 1.0    # Driving force scale
end

# Constructor from InterfaceParams
function AllenCahnModel(params::InterfaceParams; m::Float64=1.0)
    return AllenCahnModel(params.τ, params.W₀, m)
end

"""
    allen_cahn_rhs(model::AllenCahnModel, φ::T, ∇²φ::T, ΔG::T) where T<:Real

Compute the right-hand side of the Allen-Cahn equation.

∂φ/∂t = (W²∇²φ - g'(φ) + m·ΔG·h'(φ)) / τ

This function is AD-compatible for sensitivity analysis.

# Arguments
- `model`: AllenCahnModel parameters
- `φ`: Order parameter at current point
- `∇²φ`: Laplacian of order parameter
- `ΔG`: Driving force [J/mol]

# Returns
- Time derivative ∂φ/∂t [1/s]

# Example
```julia
model = AllenCahnModel(τ=3e-8, W=1e-6, m=1.0)
dφdt = allen_cahn_rhs(model, 0.5, -100.0, -5000.0)
```
"""
function allen_cahn_rhs(model::AllenCahnModel, φ::T, ∇²φ::T, ΔG::T) where T<:Real
    # Derivatives of interpolation and double-well
    h_p = h_prime(φ)
    g_p = g_prime(φ)

    # Allen-Cahn equation RHS
    diffusion = model.W^2 * ∇²φ
    bulk = -g_p
    driving = model.m * ΔG * h_p

    return (diffusion + bulk + driving) / model.τ
end

"""
    allen_cahn_rhs(model::AllenCahnModel, φ, ∇²φ, ΔG)

Non-typed version for convenience.
"""
function allen_cahn_rhs(model::AllenCahnModel, φ, ∇²φ, ΔG)
    T = promote_type(typeof(φ), typeof(∇²φ), typeof(ΔG))
    return allen_cahn_rhs(model, convert(T, φ), convert(T, ∇²φ), convert(T, ΔG))
end

"""
    allen_cahn_residual(model::AllenCahnModel, φ::T, ∇²φ::T, ΔG::T) where T<:Real

Compute the residual for steady-state analysis.

R(φ) = W²∇²φ - g'(φ) + m·ΔG·h'(φ)

At steady state, R(φ) = 0.

# Arguments
- `model`: AllenCahnModel parameters
- `φ`: Order parameter
- `∇²φ`: Laplacian of order parameter
- `ΔG`: Driving force [J/mol]

# Returns
- Residual value (should be 0 at steady state)
"""
function allen_cahn_residual(model::AllenCahnModel, φ::T, ∇²φ::T, ΔG::T) where T<:Real
    h_p = h_prime(φ)
    g_p = g_prime(φ)

    return model.W^2 * ∇²φ - g_p + model.m * ΔG * h_p
end

"""
    allen_cahn_interface_width(model::AllenCahnModel)

Calculate the equilibrium interface width.

For the double-well potential g(φ) = φ²(1-φ)², the interface width
is approximately 2√2 · W.

# Returns
- Interface width [m]
"""
function allen_cahn_interface_width(model::AllenCahnModel)
    return 2 * sqrt(2) * model.W
end

"""
    allen_cahn_interface_energy(model::AllenCahnModel)

Calculate the interface energy for the equilibrium profile.

For the standard double-well, σ = W/3.

Note: This is for the dimensionless form. For physical units,
multiply by the appropriate scaling factor.

# Returns
- Dimensionless interface energy
"""
function allen_cahn_interface_energy(model::AllenCahnModel)
    return model.W / 3
end
