# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Anisotropic interface energy functions

"""
    anisotropy_cubic(θ::T; δ::Real=0.04) where T<:Real

4-fold anisotropic interface energy for FCC/BCC crystals.

σ(θ) = 1 + δ·cos(4θ)

# Arguments
- `θ`: Interface normal angle [rad]
- `δ`: Anisotropy strength (default: 0.04 = 4%)

# Returns
- Normalized interface energy (base value = 1)

# Example
```julia
# Interface energy at θ = 0 (⟨100⟩ direction)
σ = anisotropy_cubic(0.0, δ=0.04)  # = 1.04

# Interface energy at θ = π/4 (⟨110⟩ direction)
σ = anisotropy_cubic(π/4, δ=0.04)  # = 0.96
```
"""
function anisotropy_cubic(θ::T; δ::Real=0.04) where T<:Real
    δ_T = convert(T, δ)
    return one(T) + δ_T * cos(4 * θ)
end

"""
    anisotropy_hcp(θ::T; δ::Real=0.02) where T<:Real

6-fold anisotropic interface energy for HCP crystals.

σ(θ) = 1 + δ·cos(6θ)

# Arguments
- `θ`: Interface normal angle [rad]
- `δ`: Anisotropy strength (default: 0.02 = 2%)

# Returns
- Normalized interface energy (base value = 1)
"""
function anisotropy_hcp(θ::T; δ::Real=0.02) where T<:Real
    δ_T = convert(T, δ)
    return one(T) + δ_T * cos(6 * θ)
end

"""
    anisotropy_custom(θ::T; n::Int, δ::Real) where T<:Real

n-fold anisotropic interface energy.

σ(θ) = 1 + δ·cos(n·θ)

# Arguments
- `θ`: Interface normal angle [rad]
- `n`: Symmetry order (4 for cubic, 6 for hexagonal)
- `δ`: Anisotropy strength

# Returns
- Normalized interface energy (base value = 1)
"""
function anisotropy_custom(θ::T; n::Int, δ::Real) where T<:Real
    n_T = convert(T, n)
    δ_T = convert(T, δ)
    return one(T) + δ_T * cos(n_T * θ)
end

# ============================================================================
# Anisotropy derivatives (for interface kinetics)
# ============================================================================

"""
    anisotropy_cubic_prime(θ::T; δ::Real=0.04) where T<:Real

Derivative of 4-fold anisotropy: σ'(θ) = -4δ·sin(4θ)

# Arguments
- `θ`: Interface normal angle [rad]
- `δ`: Anisotropy strength

# Returns
- Derivative dσ/dθ
"""
function anisotropy_cubic_prime(θ::T; δ::Real=0.04) where T<:Real
    δ_T = convert(T, δ)
    return -4 * δ_T * sin(4 * θ)
end

"""
    anisotropy_hcp_prime(θ::T; δ::Real=0.02) where T<:Real

Derivative of 6-fold anisotropy: σ'(θ) = -6δ·sin(6θ)

# Arguments
- `θ`: Interface normal angle [rad]
- `δ`: Anisotropy strength

# Returns
- Derivative dσ/dθ
"""
function anisotropy_hcp_prime(θ::T; δ::Real=0.02) where T<:Real
    δ_T = convert(T, δ)
    return -6 * δ_T * sin(6 * θ)
end

"""
    anisotropy_stiffness(θ::T; δ::Real=0.04, n::Int=4) where T<:Real

Interface stiffness σ(θ) + σ''(θ), important for dendrite stability.

For n-fold anisotropy:
σ + σ'' = 1 + δ·cos(nθ) - n²δ·cos(nθ) = 1 + δ(1-n²)·cos(nθ)

# Arguments
- `θ`: Interface normal angle [rad]
- `δ`: Anisotropy strength
- `n`: Symmetry order

# Returns
- Interface stiffness value
"""
function anisotropy_stiffness(θ::T; δ::Real=0.04, n::Int=4) where T<:Real
    n_T = convert(T, n)
    δ_T = convert(T, δ)
    return one(T) + δ_T * (one(T) - n_T^2) * cos(n_T * θ)
end
