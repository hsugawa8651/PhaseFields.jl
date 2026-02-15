# Common API Reference

This page documents the common types and functions shared across all phase field models.

## Types

### Interface Parameters

```@docs
InterfaceParams
```

### Diffusion Parameters

```@docs
DiffusionParams
```

### Material Parameters

```@docs
MaterialParams
```

## Interpolation Functions

Phase field models use interpolation functions h(φ) and double-well
potentials g(φ) to smoothly transition between phases.

### Interpolation h(φ)

Properties:
- h(0) = 0, h(1) = 1
- Smooth transition in interface region

```julia
using PhaseFields

φ = 0.5

# Polynomial: h = 3φ² - 2φ³
h = h_polynomial(φ)

# Sinusoidal: h = (1 - cos(πφ))/2
h = h_sin(φ)

# Derivative
dh = h_prime(φ)
```

```@docs
h_polynomial
h_sin
h_prime
```

### Double-Well g(φ)

Properties:
- g(0) = g(1) = 0
- Maximum at φ = 0.5
- Provides energy barrier

```julia
# Standard: g = φ²(1-φ)²
g = g_standard(φ)

# Obstacle: g = |φ(1-φ)|
g = g_obstacle(φ)

# Derivatives
dg = g_prime(φ)
d2g = g_double_prime(φ)
```

```@docs
g_standard
g_obstacle
g_prime
g_double_prime
```

## Anisotropy Functions

For dendritic growth, crystallographic anisotropy modifies the
interface energy and kinetics.

### Cubic Anisotropy (4-fold)

For FCC and BCC crystals:

```julia
θ = π/4  # Angle

# 4-fold anisotropy: a(θ) = 1 + ε cos(4θ)
a = anisotropy_cubic(θ; ε=0.05)
```

```@docs
anisotropy_cubic
```

### HCP Anisotropy (6-fold)

For hexagonal crystals:

```julia
# 6-fold anisotropy: a(θ) = 1 + ε cos(6θ)
a = anisotropy_hcp(θ; ε=0.05)
```

```@docs
anisotropy_hcp
```

### Custom Anisotropy

```julia
# n-fold anisotropy
a = anisotropy_custom(θ; ε=0.05, n=8)
```

```@docs
anisotropy_custom
```

## AD Configuration

PhaseFields.jl uses [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)
for automatic differentiation.

```@docs
DEFAULT_AD_BACKEND
set_ad_backend!
```

## Module

```@docs
PhaseFields
```
