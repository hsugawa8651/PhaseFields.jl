# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Snapshot types for plotting

"""
    FieldSnapshot1D

1D field snapshot at a specific time. Holds one or more named fields.

# Fields
- `x::Vector{Float64}` — spatial coordinates
- `fields::Vector{Pair{Symbol,Vector{Float64}}}` — field data (insertion order preserved)
- `t::Float64` — time
- `xlabel::String` — x-axis label
- `title::String` — plot title
"""
struct FieldSnapshot1D
    x::Vector{Float64}
    fields::Vector{Pair{Symbol,Vector{Float64}}}
    t::Float64
    xlabel::String
    title::String
end

# Single-field convenience constructor
function FieldSnapshot1D(grid::UniformGrid1D, φ::AbstractVector, t::Real;
                         field_name::Symbol=:φ, xlabel::String="x", title::String="")
    length(φ) == grid.N || throw(ArgumentError(
        "Field length $(length(φ)) does not match grid size $(grid.N)"))
    FieldSnapshot1D(grid.x, [field_name => collect(Float64, φ)], Float64(t), xlabel, title)
end

# Multi-field constructor
function FieldSnapshot1D(grid::UniformGrid1D, fields::Dict{Symbol,<:AbstractVector}, t::Real;
                         xlabel::String="x", title::String="")
    for (k, v) in fields
        length(v) == grid.N || throw(ArgumentError(
            "Field :$k length $(length(v)) does not match grid size $(grid.N)"))
    end
    pairs = [k => collect(Float64, v) for (k, v) in fields]
    FieldSnapshot1D(grid.x, pairs, Float64(t), xlabel, title)
end

"""
    SpaceTimeSnapshot1D

1D field evolution over time, suitable for heatmap display.

# Fields
- `x::Vector{Float64}` — spatial coordinates (length N)
- `t::Vector{Float64}` — time values (length M)
- `data::Matrix{Float64}` — field values (N × M)
- `field_name::Symbol` — field name for display
- `xlabel::String` — x-axis label
- `ylabel::String` — y-axis label (default: "Time")
- `title::String` — plot title
- `clims::Union{Nothing,Tuple{Float64,Float64}}` — color range
- `colormap::Symbol` — colormap (default: :RdBu)
"""
struct SpaceTimeSnapshot1D
    x::Vector{Float64}
    t::Vector{Float64}
    data::Matrix{Float64}
    field_name::Symbol
    xlabel::String
    ylabel::String
    title::String
    clims::Union{Nothing,Tuple{Float64,Float64}}
    colormap::Symbol

    function SpaceTimeSnapshot1D(x, t, data, field_name;
                                 xlabel="x", ylabel="Time", title="",
                                 clims=nothing, colormap=:RdBu)
        size(data) == (length(x), length(t)) || throw(ArgumentError(
            "Data size $(size(data)) does not match ($(length(x)), $(length(t)))"))
        new(collect(Float64, x), collect(Float64, t), collect(Float64, data),
            field_name, xlabel, ylabel, title, clims, colormap)
    end
end

"""
    FieldSnapshot2D

2D field snapshot at a specific time (single field, suitable for heatmap).

# Fields
- `x::Vector{Float64}` — x coordinates (length Nx)
- `y::Vector{Float64}` — y coordinates (length Ny)
- `field::Matrix{Float64}` — field values (Nx × Ny)
- `t::Float64` — time
- `field_name::Symbol` — field name for display
- `xlabel::String`, `ylabel::String` — axis labels
- `title::String` — plot title
- `clims::Union{Nothing,Tuple{Float64,Float64}}` — color range
- `colormap::Symbol` — colormap (default: :viridis)
"""
struct FieldSnapshot2D
    x::Vector{Float64}
    y::Vector{Float64}
    field::Matrix{Float64}
    t::Float64
    field_name::Symbol
    xlabel::String
    ylabel::String
    title::String
    clims::Union{Nothing,Tuple{Float64,Float64}}
    colormap::Symbol
end

function FieldSnapshot2D(grid::UniformGrid2D, field::AbstractMatrix, t::Real, field_name::Symbol;
                         xlabel::String="x", ylabel::String="y", title::String="",
                         clims=nothing, colormap::Symbol=:viridis)
    size(field) == (grid.Nx, grid.Ny) || throw(ArgumentError(
        "Field size $(size(field)) does not match grid size ($(grid.Nx), $(grid.Ny))"))
    FieldSnapshot2D(grid.x, grid.y, collect(Float64, field), Float64(t),
                    field_name, xlabel, ylabel, title, clims, colormap)
end
