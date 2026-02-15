# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Interpolation and double-well potential functions

"""
    h_polynomial(φ::T) where T<:Real

Polynomial interpolation function h(φ) = 3φ² - 2φ³.

Most commonly used interpolation for phase field models.
Properties: h(0) = 0, h(1) = 1, h'(0) = h'(1) = 0.

# Arguments
- `φ`: Order parameter ∈ [0, 1]

# Returns
- Interpolated value ∈ [0, 1]
"""
function h_polynomial(φ::T) where T<:Real
    return 3 * φ^2 - 2 * φ^3
end

"""
    h_sin(φ::T) where T<:Real

Sinusoidal interpolation function h(φ) = (1 - cos(πφ))/2.

Alternative interpolation with similar properties to polynomial.
Properties: h(0) = 0, h(1) = 1, h'(0) = h'(1) = 0.

# Arguments
- `φ`: Order parameter ∈ [0, 1]

# Returns
- Interpolated value ∈ [0, 1]
"""
function h_sin(φ::T) where T<:Real
    return (one(T) - cos(convert(T, π) * φ)) / 2
end

"""
    g_standard(φ::T) where T<:Real

Standard double-well potential g(φ) = φ²(1-φ)².

Properties: g(0) = g(1) = 0, minimum at φ = 0 and φ = 1.
Maximum at φ = 0.5 with g(0.5) = 1/16.

# Arguments
- `φ`: Order parameter ∈ [0, 1]

# Returns
- Potential value ≥ 0
"""
function g_standard(φ::T) where T<:Real
    return φ^2 * (one(T) - φ)^2
end

"""
    g_obstacle(φ::T) where T<:Real

Obstacle double-well potential g(φ) = φ(1-φ).

Simpler form than standard, but with same zeros.
Used in some thin-interface models.

# Arguments
- `φ`: Order parameter ∈ [0, 1]

# Returns
- Potential value ≥ 0
"""
function g_obstacle(φ::T) where T<:Real
    return φ * (one(T) - φ)
end

# ============================================================================
# Analytical derivatives (for performance)
# ============================================================================

"""
    h_prime(φ::T) where T<:Real

Derivative of polynomial interpolation: h'(φ) = 6φ(1-φ).

Analytical form for better performance than AD.

# Arguments
- `φ`: Order parameter ∈ [0, 1]

# Returns
- Derivative value
"""
function h_prime(φ::T) where T<:Real
    return 6 * φ * (one(T) - φ)
end

"""
    g_prime(φ::T) where T<:Real

Derivative of standard double-well: g'(φ) = 2φ(1-φ)(1-2φ).

Analytical form for better performance than AD.

# Arguments
- `φ`: Order parameter ∈ [0, 1]

# Returns
- Derivative value
"""
function g_prime(φ::T) where T<:Real
    return 2 * φ * (one(T) - φ) * (one(T) - 2 * φ)
end

"""
    g_double_prime(φ::T) where T<:Real

Second derivative of standard double-well: g''(φ) = 2(1 - 6φ + 6φ²).

# Arguments
- `φ`: Order parameter ∈ [0, 1]

# Returns
- Second derivative value
"""
function g_double_prime(φ::T) where T<:Real
    return 2 * (one(T) - 6 * φ + 6 * φ^2)
end

# ============================================================================
# AD-based derivatives (for verification and complex functions)
# ============================================================================

"""
    h_prime_ad(φ)

Derivative of h_polynomial using automatic differentiation.

Use this for verification or when analytical form is unavailable.
"""
function h_prime_ad(φ)
    return DI.derivative(h_polynomial, get_ad_backend(), φ)
end

"""
    g_prime_ad(φ)

Derivative of g_standard using automatic differentiation.

Use this for verification or when analytical form is unavailable.
"""
function g_prime_ad(φ)
    return DI.derivative(g_standard, get_ad_backend(), φ)
end

"""
    g_double_prime_ad(φ)

Second derivative of g_standard using automatic differentiation.
"""
function g_double_prime_ad(φ)
    return DI.second_derivative(g_standard, get_ad_backend(), φ)
end
