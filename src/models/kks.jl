# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Kim-Kim-Suzuki (KKS) model

"""
    KKSModel

Kim-Kim-Suzuki model for solidification with local equilibrium.

The KKS model introduces phase concentrations (c_s, c_l) that satisfy:
1. Mass conservation: h(φ)·c_s + (1-h(φ))·c_l = c
2. Equal diffusion potential: μ_s(c_s, T) = μ_l(c_l, T)

# Evolution equations:
    Phase field:  τ·∂φ/∂t = W²∇²φ - g'(φ) + h'(φ)·Δω
    Concentration: ∂c/∂t = ∇·(M(φ)·∇μ)

where Δω = f_s(c_s) - f_l(c_l) - μ·(c_s - c_l) is the grand potential difference.

# Fields
- `τ::Float64`: Relaxation time [s]
- `W::Float64`: Interface width parameter [m]
- `m::Float64`: Driving force scale (typically 1.0 for KKS)
- `M_s::Float64`: Mobility in solid phase [m²/(J·s)]
- `M_l::Float64`: Mobility in liquid phase [m²/(J·s)]

# Example
```julia
model = KKSModel(τ=1.0, W=1.0, m=1.0, M_s=1.0, M_l=10.0)
```

# Reference
Kim, Kim, Suzuki, Phys. Rev. E 60 (1999) 7186
"""
@kwdef struct KKSModel <: AbstractPhaseFieldModel
    τ::Float64 = 1.0    # Relaxation time
    W::Float64 = 1.0    # Interface width parameter
    m::Float64 = 1.0    # Driving force scale
    M_s::Float64 = 1.0  # Solid mobility
    M_l::Float64 = 1.0  # Liquid mobility
end

"""
    ParabolicFreeEnergy

Simple parabolic free energy for testing KKS model.

    f(c) = A·(c - c_eq)²

# Fields
- `A::Float64`: Curvature (must be positive for stability)
- `c_eq::Float64`: Equilibrium concentration
"""
struct ParabolicFreeEnergy
    A::Float64      # Curvature
    c_eq::Float64   # Equilibrium concentration
end

function ParabolicFreeEnergy(; A::Real, c_eq::Real)
    return ParabolicFreeEnergy(Float64(A), Float64(c_eq))
end

"""
    free_energy(f::ParabolicFreeEnergy, c)

Parabolic free energy density: f(c) = A·(c - c_eq)²
"""
function free_energy(f::ParabolicFreeEnergy, c::Real)
    return f.A * (c - f.c_eq)^2
end

"""
    chemical_potential(f::ParabolicFreeEnergy, c)

Chemical potential (first derivative): df/dc = 2A·(c - c_eq)
"""
function chemical_potential(f::ParabolicFreeEnergy, c::Real)
    return 2 * f.A * (c - f.c_eq)
end

"""
    d2f_dc2(f::ParabolicFreeEnergy, c)

Second derivative of free energy: d²f/dc² = 2A
"""
function d2f_dc2(f::ParabolicFreeEnergy, c::Real)
    return 2 * f.A
end

"""
    _invert_chemical_potential(f, μ_target, c_init; maxiter=20, tol=1e-12)

Find c such that chemical_potential(f, c) = μ_target using Newton iteration.
Internal helper function for edge cases in kks_partition.
"""
function _invert_chemical_potential(
    f, μ_target::Real, c_init::Real;
    maxiter::Int=20, tol::Real=1e-12
)
    c = c_init
    for _ in 1:maxiter
        μ = chemical_potential(f, c)
        residual = μ - μ_target
        if abs(residual) < tol
            return c
        end
        d2f = d2f_dc2(f, c)
        if abs(d2f) < 1e-15
            break
        end
        c -= residual / d2f
    end
    return c
end

"""
    kks_partition(c_avg, φ, f_s, f_l; maxiter=50, tol=1e-12, φ_min=1e-8)

Solve the KKS local equilibrium problem using Newton iteration.

Given average concentration c and phase fraction φ, find phase concentrations
(c_s, c_l) that satisfy:
1. h(φ)·c_s + (1-h(φ))·c_l = c  (mass conservation)
2. μ_s(c_s) = μ_l(c_l)         (equal chemical potential)

# Arguments
- `c_avg`: Average concentration at this point
- `φ`: Phase field value (0=liquid, 1=solid)
- `f_s`: Solid phase free energy (ParabolicFreeEnergy or similar)
- `f_l`: Liquid phase free energy

# Keyword Arguments
- `maxiter`: Maximum Newton iterations (default: 50)
- `tol`: Convergence tolerance (default: 1e-12)
- `φ_min`: Minimum φ for Newton iteration (default: 1e-8)

# Returns
- `(c_s, c_l, μ, converged)`: Phase concentrations, chemical potential, convergence flag

# Example
```julia
f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)  # Solid equilibrium at c=0.1
f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)  # Liquid equilibrium at c=0.9

c_s, c_l, μ, converged = kks_partition(0.5, 0.5, f_s, f_l)
```
"""
function kks_partition(
    c_avg::Real, φ::Real,
    f_s, f_l;
    maxiter::Int=50, tol::Real=1e-12, φ_min::Real=1e-8
)
    # Clamp φ to avoid singularity
    φ_eff = clamp(φ, φ_min, 1 - φ_min)
    h = h_polynomial(φ_eff)

    # Handle edge cases: pure phase regions
    if φ < φ_min
        # Pure liquid
        c_l = c_avg
        μ = chemical_potential(f_l, c_l)
        # Find c_s such that μ_s(c_s) = μ using Newton iteration
        c_s = _invert_chemical_potential(f_s, μ, c_avg)
        return (c_s, c_l, μ, true)
    elseif φ > 1 - φ_min
        # Pure solid
        c_s = c_avg
        μ = chemical_potential(f_s, c_s)
        # Find c_l such that μ_l(c_l) = μ using Newton iteration
        c_l = _invert_chemical_potential(f_l, μ, c_avg)
        return (c_s, c_l, μ, true)
    end

    # Initial guess: both concentrations equal to average
    c_s = c_avg
    c_l = c_avg

    for iter in 1:maxiter
        # Compute residuals
        # F1: h·c_s + (1-h)·c_l - c = 0
        F1 = h * c_s + (1 - h) * c_l - c_avg

        # F2: μ_s(c_s) - μ_l(c_l) = 0
        μ_s = chemical_potential(f_s, c_s)
        μ_l = chemical_potential(f_l, c_l)
        F2 = μ_s - μ_l

        # Check convergence
        if abs(F1) < tol && abs(F2) < tol
            μ = μ_s  # = μ_l at convergence
            return (c_s, c_l, μ, true)
        end

        # Jacobian
        # J = | h        1-h      |
        #     | d²f_s    -d²f_l   |
        d2f_s = d2f_dc2(f_s, c_s)
        d2f_l = d2f_dc2(f_l, c_l)

        # det(J) = -h·d²f_l - (1-h)·d²f_s
        det_J = -h * d2f_l - (1 - h) * d2f_s

        # Check for singular Jacobian
        if abs(det_J) < 1e-15
            return (c_s, c_l, μ_s, false)
        end

        # Newton update: [c_s; c_l] -= J⁻¹ · [F1; F2]
        # J⁻¹ = 1/det(J) * | -d²f_l   -(1-h) |
        #                   | -d²f_s    h     |
        Δc_s = (-d2f_l * F1 - (1 - h) * F2) / det_J
        Δc_l = (-d2f_s * F1 + h * F2) / det_J

        c_s -= Δc_s
        c_l -= Δc_l
    end

    # Did not converge
    μ = chemical_potential(f_s, c_s)
    return (c_s, c_l, μ, false)
end

"""
    kks_grand_potential_diff(f_s, f_l, c_s, c_l, μ)

Compute the grand potential difference (driving force for phase field).

    Δω = f_s(c_s) - f_l(c_l) - μ·(c_s - c_l)

# Arguments
- `f_s, f_l`: Solid and liquid free energy functions
- `c_s, c_l`: Phase concentrations
- `μ`: Chemical potential (equal in both phases)

# Returns
- Grand potential difference Δω [J/mol]
"""
function kks_grand_potential_diff(
    f_s, f_l,
    c_s::Real, c_l::Real, μ::Real
)
    ω_s = free_energy(f_s, c_s) - μ * c_s
    ω_l = free_energy(f_l, c_l) - μ * c_l
    return ω_s - ω_l
end

"""
    kks_phase_rhs(model, φ, ∇²φ, Δω)

Compute the right-hand side of the KKS phase field equation.

    τ·∂φ/∂t = W²∇²φ - g'(φ) + m·h'(φ)·Δω
    ⟹ ∂φ/∂t = [W²∇²φ - g'(φ) + m·h'(φ)·Δω] / τ

# Arguments
- `model`: KKSModel parameters
- `φ`: Phase field value
- `∇²φ`: Laplacian of phase field
- `Δω`: Grand potential difference from kks_grand_potential_diff

# Returns
- Time derivative ∂φ/∂t
"""
function kks_phase_rhs(model::KKSModel, φ::Real, ∇²φ::Real, Δω::Real)
    # Double-well derivative: g'(φ) = 2φ(1-φ)(1-2φ)
    g_p = g_prime(φ)

    # Interpolation derivative: h'(φ) = 6φ(1-φ)
    h_p = h_prime(φ)

    # Phase field equation
    diffusion = model.W^2 * ∇²φ
    bulk = -g_p
    driving = model.m * h_p * Δω

    return (diffusion + bulk + driving) / model.τ
end

"""
    kks_mobility(model, φ)

Compute effective mobility using interpolation.

    M(φ) = h(φ)·M_s + (1-h(φ))·M_l

# Returns
- Effective mobility at this phase field value
"""
function kks_mobility(model::KKSModel, φ::Real)
    h = h_polynomial(φ)
    return h * model.M_s + (1 - h) * model.M_l
end

"""
    kks_concentration_rhs(model, φ, ∇²μ)

Compute the right-hand side of the KKS concentration equation.

    ∂c/∂t = ∇·(M(φ)·∇μ) ≈ M(φ)·∇²μ  (for constant mobility in each phase)

# Arguments
- `model`: KKSModel parameters
- `φ`: Phase field value (for mobility interpolation)
- `∇²μ`: Laplacian of chemical potential

# Returns
- Time derivative ∂c/∂t
"""
function kks_concentration_rhs(model::KKSModel, φ::Real, ∇²μ::Real)
    M_eff = kks_mobility(model, φ)
    return M_eff * ∇²μ
end

"""
    kks_interface_width(model)

Get the interface width parameter W.
"""
kks_interface_width(model::KKSModel) = model.W
