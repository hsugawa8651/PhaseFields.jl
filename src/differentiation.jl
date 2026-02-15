# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Automatic Differentiation configuration

"""
Default AD backend (ForwardDiff).
Can be changed via `set_ad_backend!`.
"""
const DEFAULT_AD_BACKEND = Ref{Any}(DI.AutoForwardDiff())

"""
    set_ad_backend!(backend)

Set the default AD backend for PhaseFields.jl.

# Arguments
- `backend`: A DifferentiationInterface backend (e.g., `DI.AutoForwardDiff()`)

# Example
```julia
using DifferentiationInterface
set_ad_backend!(DI.AutoForwardDiff())  # Default
# set_ad_backend!(DI.AutoEnzyme())     # For high performance (future)
```
"""
function set_ad_backend!(backend)
    DEFAULT_AD_BACKEND[] = backend
    return backend
end

"""
    get_ad_backend()

Get the current default AD backend.
"""
get_ad_backend() = DEFAULT_AD_BACKEND[]
