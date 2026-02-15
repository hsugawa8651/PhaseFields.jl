# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Thermal Phase Field Model
# Governing equations:
#   Phase field: τ ∂φ/∂t = W²∇²φ - g'(φ) + λu h'(φ)
#   Temperature: ∂u/∂t = α∇²u + (1/2) ∂φ/∂t
#
# where:
#   φ: Phase field (0=liquid, 1=solid)
#   u: Dimensionless temperature u = Cp(T - Tm)/L
#   τ: Relaxation time
#   W: Interface width parameter
#   λ: Coupling strength
#   α: Thermal diffusivity
#   g(φ): Double-well potential
#   h(φ): Interpolation function

"""
    ThermalPhaseFieldModel

Parameters for thermal phase field model coupling phase evolution with heat transfer.

# Fields
- `τ::Float64`: Relaxation time [s]
- `W::Float64`: Interface width parameter [m]
- `λ::Float64`: Thermal coupling strength (dimensionless)
- `α::Float64`: Thermal diffusivity [m²/s]
- `L::Float64`: Latent heat [J/m³]
- `Cp::Float64`: Volumetric heat capacity [J/(m³·K)]
- `Tm::Float64`: Melting temperature [K]
- `K::Float64`: Latent heat coefficient L/Cp [K] (computed)

# Example
```julia
model = ThermalPhaseFieldModel(
    τ = 1.0,
    W = 1.0,
    λ = 1.0,
    α = 1e-5,
    L = 2.35e9,
    Cp = 5.42e6,
    Tm = 1728.0
)
```
"""
struct ThermalPhaseFieldModel <: AbstractPhaseFieldModel
    τ::Float64
    W::Float64
    λ::Float64
    α::Float64
    L::Float64
    Cp::Float64
    Tm::Float64
    K::Float64

    function ThermalPhaseFieldModel(; τ, W, λ, α, L, Cp, Tm)
        K = L / Cp
        new(τ, W, λ, α, L, Cp, Tm, K)
    end
end

"""
    dimensionless_temperature(T, Tm, L, Cp)

Convert physical temperature to dimensionless temperature.

# Arguments
- `T`: Physical temperature [K]
- `Tm`: Melting temperature [K]
- `L`: Latent heat [J/m³]
- `Cp`: Volumetric heat capacity [J/(m³·K)]

# Returns
- `u`: Dimensionless temperature u = Cp(T - Tm)/L

# Notes
- u = 0 at melting point
- u < 0 for undercooling (T < Tm)
- u > 0 for superheating (T > Tm)
"""
function dimensionless_temperature(T, Tm, L, Cp)
    return Cp * (T - Tm) / L
end

"""
    physical_temperature(u, Tm, L, Cp)

Convert dimensionless temperature back to physical temperature.

# Arguments
- `u`: Dimensionless temperature
- `Tm`: Melting temperature [K]
- `L`: Latent heat [J/m³]
- `Cp`: Volumetric heat capacity [J/(m³·K)]

# Returns
- `T`: Physical temperature [K]
"""
function physical_temperature(u, Tm, L, Cp)
    return Tm + u * L / Cp
end

"""
    stefan_number(ΔT, L, Cp)

Calculate Stefan number.

# Arguments
- `ΔT`: Temperature difference (undercooling or superheating) [K]
- `L`: Latent heat [J/m³]
- `Cp`: Volumetric heat capacity [J/(m³·K)]

# Returns
- `St`: Stefan number St = Cp·ΔT/L (sensible heat / latent heat ratio)

# Notes
- St << 1: Latent heat dominates, slow interface motion
- St ~ 1: Comparable sensible and latent heat
- St >> 1: Sensible heat dominates, fast interface motion
"""
function stefan_number(ΔT, L, Cp)
    return Cp * ΔT / L
end

"""
    thermal_phase_rhs(model, φ, ∇²φ, u)

Compute right-hand side of phase field evolution equation with thermal driving.

# Arguments
- `model::ThermalPhaseFieldModel`: Model parameters
- `φ`: Phase field value (0=liquid, 1=solid)
- `∇²φ`: Laplacian of phase field
- `u`: Dimensionless temperature

# Returns
- `∂φ/∂t`: Time derivative of phase field (divide by τ is already done)

# Equation
```
τ ∂φ/∂t = W²∇²φ - g'(φ) + λu h'(φ)
```

Returns the RHS divided by τ:
```
∂φ/∂t = (W²∇²φ - g'(φ) + λu h'(φ)) / τ
```

# Notes
- g(φ) = φ²(1-φ)² double-well potential, g'(φ) = 2φ(1-φ)(1-2φ)
- h(φ) = φ²(3-2φ) interpolation function, h'(φ) = 6φ(1-φ)
- Undercooling (u < 0) drives solidification (increases φ)
- Superheating (u > 0) drives melting (decreases φ)
"""
function thermal_phase_rhs(model::ThermalPhaseFieldModel, φ, ∇²φ, u)
    τ = model.τ
    W = model.W
    λ = model.λ

    # Double-well derivative: g'(φ) = 2φ(1-φ)(1-2φ)
    g_prime = 2 * φ * (1 - φ) * (1 - 2φ)

    # Interpolation derivative: h'(φ) = 6φ(1-φ)
    h_prime = 6 * φ * (1 - φ)

    # Phase field RHS: (W²∇²φ - g'(φ) + λu h'(φ)) / τ
    # Note: thermal driving term has sign convention where u < 0 (undercooling)
    # drives φ to increase (solidification)
    return (W^2 * ∇²φ - g_prime - λ * u * h_prime) / τ
end

"""
    thermal_heat_rhs(model, u, ∇²u, dφdt)

Compute right-hand side of heat equation with latent heat source.

# Arguments
- `model::ThermalPhaseFieldModel`: Model parameters
- `u`: Dimensionless temperature
- `∇²u`: Laplacian of dimensionless temperature
- `dφdt`: Time derivative of phase field

# Returns
- `∂u/∂t`: Time derivative of dimensionless temperature

# Equation
```
∂u/∂t = α∇²u + (1/2) ∂φ/∂t
```

# Notes
- Solidification (dφdt > 0) releases latent heat (u increases)
- Melting (dφdt < 0) absorbs latent heat (u decreases)
- Factor 1/2 comes from h(φ) normalization where solid fraction = h(φ)
"""
function thermal_heat_rhs(model::ThermalPhaseFieldModel, u, ∇²u, dφdt)
    α = model.α

    # Heat equation RHS: α∇²u + (1/2) dφdt
    # Latent heat release during solidification increases temperature
    return α * ∇²u + 0.5 * dφdt
end

"""
    thermal_stability_dt(model, dx)

Compute stable time step for coupled thermal phase field evolution.

# Arguments
- `model::ThermalPhaseFieldModel`: Model parameters
- `dx`: Grid spacing [m]

# Returns
- `dt`: Stable time step [s]

# Notes
Combines stability conditions:
- Thermal diffusion: dt < dx²/(2α) in 1D, dx²/(4α) in 2D
- Phase field: dt < τ·dx²/(2W²)

Returns conservative (1D) limit with safety factor.
"""
function thermal_stability_dt(model::ThermalPhaseFieldModel, dx)
    τ = model.τ
    W = model.W
    α = model.α

    # Thermal diffusion limit (1D)
    dt_thermal = dx^2 / (2 * α)

    # Phase field stability limit
    dt_phase = τ * dx^2 / (2 * W^2)

    # Safety factor
    safety = 0.9

    return safety * min(dt_thermal, dt_phase)
end
