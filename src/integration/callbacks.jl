# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Callback utilities for DiffEq integration
# - Steady state detection

using DiffEqCallbacks: SavedValues, SavingCallback, TerminateSteadyState, CallbackSet

# =============================================================================
# Interface Position Tracking
# =============================================================================

"""
    interface_position_1d(φ, x; contour=0.5)

Find the position where φ crosses the contour value using linear interpolation.

# Arguments
- `φ`: Phase field array
- `x`: Spatial coordinates array

# Keyword Arguments
- `contour`: Contour value to track (default: 0.5)

# Returns
- Interface position, or `NaN` if no crossing found
"""
function interface_position_1d(φ::AbstractVector, x::AbstractVector; contour::Real=0.5)
    # Check for exact match first
    for i in 1:length(φ)
        if φ[i] == contour
            return x[i]
        end
    end

    # Check for crossing
    for i in 1:length(φ)-1
        if (φ[i] - contour) * (φ[i+1] - contour) < 0
            # Linear interpolation
            t = (contour - φ[i]) / (φ[i+1] - φ[i])
            return x[i] + t * (x[i+1] - x[i])
        end
    end
    return NaN
end

"""
    create_interface_saving_callback(grid; contour=0.5, saveat=nothing, save_everystep=true)

Create a SavingCallback that tracks interface position during integration.

# Arguments
- `grid`: UniformGrid1D object

# Keyword Arguments
- `contour`: Contour value to track (default: 0.5)
- `saveat`: Specific times to save (default: nothing, uses save_everystep)
- `save_everystep`: Save at every solver step (default: true if saveat is empty)

# Returns
- Tuple of (callback, saved_values)

# Example
```julia
cb, saved = create_interface_saving_callback(grid; saveat=0.1)
sol = solve(prob, Tsit5(); callback=cb)
# Access results:
times = saved.t
positions = saved.saveval
```
"""
function create_interface_saving_callback(grid::UniformGrid1D;
                                          contour::Real=0.5,
                                          saveat=nothing,
                                          save_everystep::Bool=isnothing(saveat))
    saved_values = SavedValues(Float64, Float64)

    save_func = (u, t, integrator) -> interface_position_1d(u, grid.x; contour=contour)

    if isnothing(saveat)
        cb = SavingCallback(save_func, saved_values; save_everystep=save_everystep)
    else
        cb = SavingCallback(save_func, saved_values; saveat=saveat)
    end

    return cb, saved_values
end

# =============================================================================
# Solid Fraction Tracking
# =============================================================================

"""
    solid_fraction(φ; threshold=0.5)

Calculate the fraction of the domain in the solid phase (φ > threshold).

# Arguments
- `φ`: Phase field array

# Keyword Arguments
- `threshold`: Value above which a point is considered solid (default: 0.5)

# Returns
- Solid fraction (0 to 1)
"""
function solid_fraction(φ::AbstractVector; threshold::Real=0.5)
    return count(φ .> threshold) / length(φ)
end

"""
    create_solid_fraction_callback(; threshold=0.5, saveat=nothing, save_everystep=true)

Create a SavingCallback that tracks solid fraction during integration.

# Keyword Arguments
- `threshold`: Value above which a point is considered solid (default: 0.5)
- `saveat`: Specific times to save (default: nothing)
- `save_everystep`: Save at every solver step (default: true if saveat is empty)

# Returns
- Tuple of (callback, saved_values)

# Example
```julia
cb, saved = create_solid_fraction_callback(; saveat=0.1)
sol = solve(prob, Tsit5(); callback=cb)
# Access results:
times = saved.t
fractions = saved.saveval
```
"""
function create_solid_fraction_callback(; threshold::Real=0.5,
                                         saveat=nothing,
                                         save_everystep::Bool=isnothing(saveat))
    saved_values = SavedValues(Float64, Float64)

    save_func = (u, t, integrator) -> solid_fraction(u; threshold=threshold)

    if isnothing(saveat)
        cb = SavingCallback(save_func, saved_values; save_everystep=save_everystep)
    else
        cb = SavingCallback(save_func, saved_values; saveat=saveat)
    end

    return cb, saved_values
end

# =============================================================================
# Steady State Detection
# =============================================================================

"""
    create_steady_state_callback(; abstol=1e-8, reltol=1e-6, min_t=nothing)

Create a TerminateSteadyState callback that stops integration when the solution
reaches steady state.

# Keyword Arguments
- `abstol`: Absolute tolerance for derivatives (default: 1e-8)
- `reltol`: Relative tolerance for derivatives (default: 1e-6)
- `min_t`: Minimum time before termination is allowed (default: nothing)

# Returns
- TerminateSteadyState callback

# Example
```julia
cb = create_steady_state_callback(abstol=1e-6, reltol=1e-4, min_t=1.0)
sol = solve(prob, Tsit5(); callback=cb)
# sol.t[end] will be the time when steady state was reached
```
"""
function create_steady_state_callback(; abstol::Real=1e-8,
                                       reltol::Real=1e-6,
                                       min_t=nothing)
    if isnothing(min_t)
        return TerminateSteadyState(abstol, reltol)
    else
        return TerminateSteadyState(abstol, reltol; min_t=min_t)
    end
end

# =============================================================================
# Combined Callbacks
# =============================================================================

"""
    create_phase_field_callbacks(grid;
        track_interface=true, track_solid_fraction=false,
        terminate_steady_state=false, kwargs...)

Create a combined set of callbacks for phase field simulations.

# Arguments
- `grid`: UniformGrid1D object (required if track_interface=true)

# Keyword Arguments
- `track_interface`: Track interface position (default: true)
- `track_solid_fraction`: Track solid fraction (default: false)
- `terminate_steady_state`: Enable steady state termination (default: false)
- `contour`: Contour value for interface tracking (default: 0.5)
- `threshold`: Threshold for solid fraction (default: 0.5)
- `saveat`: Times to save tracked values (default: nothing)
- `abstol`: Steady state absolute tolerance (default: 1e-8)
- `reltol`: Steady state relative tolerance (default: 1e-6)
- `min_t`: Minimum time before steady state termination (default: nothing)

# Returns
- Named tuple with:
  - `callback`: Combined CallbackSet
  - `interface_data`: SavedValues for interface (or nothing)
  - `solid_fraction_data`: SavedValues for solid fraction (or nothing)

# Example
```julia
result = create_phase_field_callbacks(grid;
    track_interface=true,
    terminate_steady_state=true,
    saveat=0.1)

sol = solve(prob, Tsit5(); callback=result.callback)

# Access tracked data
interface_times = result.interface_data.t
interface_positions = result.interface_data.saveval
```
"""
function create_phase_field_callbacks(grid::UniformGrid1D=UniformGrid1D(1, 1.0);
                                      track_interface::Bool=true,
                                      track_solid_fraction::Bool=false,
                                      terminate_steady_state::Bool=false,
                                      contour::Real=0.5,
                                      threshold::Real=0.5,
                                      saveat=nothing,
                                      abstol::Real=1e-8,
                                      reltol::Real=1e-6,
                                      min_t=nothing)
    callbacks = []
    interface_data = nothing
    solid_fraction_data = nothing

    if track_interface
        cb, saved = create_interface_saving_callback(grid; contour=contour, saveat=saveat)
        push!(callbacks, cb)
        interface_data = saved
    end

    if track_solid_fraction
        cb, saved = create_solid_fraction_callback(; threshold=threshold, saveat=saveat)
        push!(callbacks, cb)
        solid_fraction_data = saved
    end

    if terminate_steady_state
        cb = create_steady_state_callback(; abstol=abstol, reltol=reltol, min_t=min_t)
        push!(callbacks, cb)
    end

    if isempty(callbacks)
        callback = nothing
    elseif length(callbacks) == 1
        callback = callbacks[1]
    else
        callback = CallbackSet(callbacks...)
    end

    return (callback=callback, interface_data=interface_data,
            solid_fraction_data=solid_fraction_data)
end
