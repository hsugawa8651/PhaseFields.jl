# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Abstract model type definitions

"""
    AbstractPhaseFieldModel

Abstract type for all phase field models.

Concrete subtypes include:
- `AllenCahnModel`: Non-conserved order parameter
- `CahnHilliardModel`: Conserved order parameter (spinodal decomposition)
- `KKSModel`: Kim-Kim-Suzuki multi-phase model
- `WBMModel`: Wheeler-Boettinger-McFadden dilute alloy model
- `ThermalPhaseFieldModel`: Coupled thermal-phase field

# Interface
Concrete subtypes should implement model-specific RHS functions.
The generic `model_rhs` function provides an error fallback for
unimplemented models.
"""
abstract type AbstractPhaseFieldModel end

"""
    model_rhs(model::AbstractPhaseFieldModel, φ, ∇²φ, args...)

Compute the right-hand side of the phase field evolution equation.

This is a fallback method that throws an error for unimplemented models.
Concrete model types should implement their own RHS functions
(e.g., `allen_cahn_rhs`, `cahn_hilliard_rhs`).

# Throws
- `ErrorException` if not implemented for the given model type
"""
function model_rhs(model::AbstractPhaseFieldModel, φ, ∇²φ, args...)
    error("model_rhs not implemented for $(typeof(model)). " *
          "Use the model-specific RHS function (e.g., allen_cahn_rhs).")
end
