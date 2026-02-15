# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Wheeler-Boettinger-McFadden (WBM) model

"""
    WBMModel

Wheeler-Boettinger-McFadden model for binary alloy solidification.

The WBM model uses a single concentration field with phase-dependent free energy:
    f(φ,c) = h(φ)·f_S(c) + (1-h(φ))·f_L(c) + W·g(φ)

# Evolution equations:
    Phase field:   ∂φ/∂t = M_φ[κ∇²φ - h'(φ)(f_S - f_L) - W·g'(φ)]
    Concentration: ∂c/∂t = ∇·[D(φ)·∇c]

# Fields
- `M_φ::Float64`: Phase field mobility [m³/(J·s)]
- `κ::Float64`: Gradient energy coefficient [J/m]
- `W::Float64`: Barrier height [J/m³]
- `D_s::Float64`: Solid diffusivity [m²/s]
- `D_l::Float64`: Liquid diffusivity [m²/s]

# Example
```julia
model = WBMModel(M_φ=1.0, κ=1.0, W=1.0, D_s=1e-13, D_l=1e-9)
```

# Comparison with KKS
- WBM: Single concentration field, c = c_S = c_L at interface (simpler but interface-width dependent)
- KKS: Separate phase concentrations with equal chemical potential (thermodynamically consistent)

# Reference
Wheeler, Boettinger, McFadden, Phys. Rev. A 45, 7424 (1992)
"""
@kwdef struct WBMModel <: AbstractPhaseFieldModel
    M_φ::Float64 = 1.0  # Phase field mobility
    κ::Float64 = 1.0    # Gradient energy coefficient
    W::Float64 = 1.0    # Barrier height
    D_s::Float64 = 1.0  # Solid diffusivity
    D_l::Float64 = 1.0  # Liquid diffusivity
end

"""
    wbm_bulk_free_energy(f_s, f_l, φ, c, W)

Compute the bulk free energy density for WBM model.

    f(φ,c) = h(φ)·f_S(c) + (1-h(φ))·f_L(c) + W·g(φ)

# Arguments
- `f_s`: Solid phase free energy function
- `f_l`: Liquid phase free energy function
- `φ`: Phase field value (0=liquid, 1=solid)
- `c`: Concentration
- `W`: Barrier height [J/m³]

# Returns
- Bulk free energy density [J/m³]
"""
function wbm_bulk_free_energy(f_s, f_l, φ::Real, c::Real, W::Real)
    h = h_polynomial(φ)
    g = g_standard(φ)
    f_S = free_energy(f_s, c)
    f_L = free_energy(f_l, c)
    return h * f_S + (1 - h) * f_L + W * g
end

"""
    wbm_chemical_potential(f_s, f_l, φ, c)

Compute the chemical potential for WBM model.

    μ = ∂f/∂c = h(φ)·μ_S(c) + (1-h(φ))·μ_L(c)

# Arguments
- `f_s`: Solid phase free energy function
- `f_l`: Liquid phase free energy function
- `φ`: Phase field value
- `c`: Concentration

# Returns
- Chemical potential [J/mol]
"""
function wbm_chemical_potential(f_s, f_l, φ::Real, c::Real)
    h = h_polynomial(φ)
    μ_s = chemical_potential(f_s, c)
    μ_l = chemical_potential(f_l, c)
    return h * μ_s + (1 - h) * μ_l
end

"""
    wbm_driving_force(f_s, f_l, φ, c, W)

Compute the phase field driving force for WBM model.

    ∂f/∂φ = h'(φ)·(f_S(c) - f_L(c)) + W·g'(φ)

# Arguments
- `f_s`: Solid phase free energy function
- `f_l`: Liquid phase free energy function
- `φ`: Phase field value
- `c`: Concentration
- `W`: Barrier height [J/m³]

# Returns
- Driving force ∂f/∂φ [J/m³]
"""
function wbm_driving_force(f_s, f_l, φ::Real, c::Real, W::Real)
    h_p = h_prime(φ)
    g_p = g_prime(φ)
    f_S = free_energy(f_s, c)
    f_L = free_energy(f_l, c)
    return h_p * (f_S - f_L) + W * g_p
end

"""
    wbm_phase_rhs(model, φ, ∇²φ, c, f_s, f_l)

Compute the right-hand side of the WBM phase field equation.

    ∂φ/∂t = M_φ[κ∇²φ - h'(φ)(f_S - f_L) - W·g'(φ)]
          = M_φ[κ∇²φ - ∂f/∂φ]

# Arguments
- `model`: WBMModel parameters
- `φ`: Phase field value
- `∇²φ`: Laplacian of phase field
- `c`: Concentration
- `f_s`: Solid phase free energy function
- `f_l`: Liquid phase free energy function

# Returns
- Time derivative ∂φ/∂t
"""
function wbm_phase_rhs(model::WBMModel, φ::Real, ∇²φ::Real, c::Real, f_s, f_l)
    # Gradient term
    diffusion = model.κ * ∇²φ

    # Driving force: ∂f/∂φ
    driving = wbm_driving_force(f_s, f_l, φ, c, model.W)

    return model.M_φ * (diffusion - driving)
end

"""
    wbm_diffusivity(model, φ)

Compute effective diffusivity using interpolation.

    D(φ) = h(φ)·D_s + (1-h(φ))·D_l

# Returns
- Effective diffusivity at this phase field value [m²/s]
"""
function wbm_diffusivity(model::WBMModel, φ::Real)
    h = h_polynomial(φ)
    return h * model.D_s + (1 - h) * model.D_l
end

"""
    wbm_concentration_rhs(model, φ, ∇²c)

Compute the right-hand side of the WBM concentration equation (simplified form).

    ∂c/∂t = ∇·[D(φ)·∇c] ≈ D(φ)·∇²c  (for slowly varying D)

# Arguments
- `model`: WBMModel parameters
- `φ`: Phase field value (for diffusivity interpolation)
- `∇²c`: Laplacian of concentration

# Returns
- Time derivative ∂c/∂t
"""
function wbm_concentration_rhs(model::WBMModel, φ::Real, ∇²c::Real)
    D_eff = wbm_diffusivity(model, φ)
    return D_eff * ∇²c
end

"""
    wbm_interface_width(model)

Estimate the equilibrium interface width from model parameters.

    δ ≈ √(κ/W)

# Reference
Wheeler 1992, Eq. 32: δ = ε√(2/W), where κ = ε²/2
"""
function wbm_interface_width(model::WBMModel)
    if model.W > 0
        return sqrt(model.κ / model.W)
    else
        return Inf
    end
end

"""
    wbm_interface_energy(model)

Estimate the interface energy from model parameters.

    σ ≈ √(κ·W) / (6√2)

# Reference
Wheeler 1992, Eq. 28: σ = ε√W / (6√2), where κ = ε²/2
"""
function wbm_interface_energy(model::WBMModel)
    if model.W > 0 && model.κ > 0
        return sqrt(model.κ * model.W) / (6 * sqrt(2))
    else
        return 0.0
    end
end
