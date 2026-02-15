# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Type definitions

"""
    InterfaceParams

Parameters for solid-liquid interface modeling.

# Fields
- `W₀::Float64`: Interface width parameter [m]
- `σ₀::Float64`: Interface energy [J/m²]
- `τ::Float64`: Relaxation time [s]
- `δ::Float64`: Anisotropy strength [-]
- `n_fold::Int`: Anisotropy symmetry (4=FCC, 6=HCP)

# Example
```julia
# FCC metal with 4-fold anisotropy
params = InterfaceParams(
    W₀ = 1e-6,    # 1 μm interface width
    σ₀ = 0.3,     # 0.3 J/m² interface energy
    τ = 3e-8,     # 30 ns relaxation time
    δ = 0.04,     # 4% anisotropy
    n_fold = 4    # 4-fold cubic symmetry
)
```
"""
struct InterfaceParams
    W₀::Float64      # Interface width [m]
    σ₀::Float64      # Interface energy [J/m²]
    τ::Float64       # Relaxation time [s]
    δ::Float64       # Anisotropy strength [-]
    n_fold::Int      # Anisotropy symmetry (4=FCC, 6=HCP)
end

# Default constructor with keyword arguments
function InterfaceParams(;
    W₀::Float64,
    σ₀::Float64,
    τ::Float64,
    δ::Float64 = 0.04,
    n_fold::Int = 4
)
    return InterfaceParams(W₀, σ₀, τ, δ, n_fold)
end

"""
    DiffusionParams

Temperature-dependent diffusion parameters.

# Fields
- `D_liquid::Float64`: Pre-exponential factor for liquid [m²/s]
- `D_solid::Float64`: Pre-exponential factor for solid [m²/s]
- `Q_liquid::Float64`: Activation energy for liquid [J/mol]
- `Q_solid::Float64`: Activation energy for solid [J/mol]

# Example
```julia
params = DiffusionParams(
    D_liquid = 1e-9,  # m²/s
    D_solid = 1e-13,  # m²/s
    Q_liquid = 40e3,  # 40 kJ/mol
    Q_solid = 80e3    # 80 kJ/mol
)
```
"""
struct DiffusionParams
    D_liquid::Float64   # Diffusion in liquid [m²/s]
    D_solid::Float64    # Diffusion in solid [m²/s]
    Q_liquid::Float64   # Activation energy [J/mol]
    Q_solid::Float64    # Activation energy [J/mol]
end

# Default constructor with keyword arguments
function DiffusionParams(;
    D_liquid::Float64,
    D_solid::Float64,
    Q_liquid::Float64 = 0.0,
    Q_solid::Float64 = 0.0
)
    return DiffusionParams(D_liquid, D_solid, Q_liquid, Q_solid)
end

"""
    MaterialParams

General material properties.

# Fields
- `Vm::Float64`: Molar volume [m³/mol]
- `L::Float64`: Latent heat [J/mol]
- `Cp::Float64`: Heat capacity [J/(mol·K)]
- `k::Float64`: Thermal conductivity [W/(m·K)]
"""
struct MaterialParams
    Vm::Float64      # Molar volume [m³/mol]
    L::Float64       # Latent heat [J/mol]
    Cp::Float64      # Heat capacity [J/(mol·K)]
    k::Float64       # Thermal conductivity [W/(m·K)]
end

# Default constructor with keyword arguments
function MaterialParams(;
    Vm::Float64,
    L::Float64 = 0.0,
    Cp::Float64 = 0.0,
    k::Float64 = 0.0
)
    return MaterialParams(Vm, L, Cp, k)
end

# Gas constant for convenience
const R_GAS = 8.314462618  # J/(mol·K)

"""
    diffusion_coefficient(params::DiffusionParams, T, φ)

Calculate temperature and phase-dependent diffusion coefficient.

D(T, φ) = D_liquid * h(φ) + D_solid * (1 - h(φ))

where h(φ) is the interpolation function and D values are
Arrhenius temperature-dependent:

D_i(T) = D_i * exp(-Q_i / (R * T))

# Arguments
- `params`: DiffusionParams struct
- `T`: Temperature [K]
- `φ`: Order parameter (0 = solid, 1 = liquid)

# Returns
- Effective diffusion coefficient [m²/s]
"""
function diffusion_coefficient(params::DiffusionParams, T::Real, φ::Real)
    # Arrhenius temperature dependence
    D_l = params.D_liquid * exp(-params.Q_liquid / (R_GAS * T))
    D_s = params.D_solid * exp(-params.Q_solid / (R_GAS * T))

    # Phase interpolation using h_polynomial
    h = 3φ^2 - 2φ^3
    return D_l * h + D_s * (1 - h)
end
