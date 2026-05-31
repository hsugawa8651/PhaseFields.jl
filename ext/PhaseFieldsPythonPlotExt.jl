# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - PythonPlot extension (3-layer plotting API)
#
# Convention: do NOT use `PythonPlot.subplots()` — it returns a raw
# `PythonCall.Py` whose `PythonPlot.close(fig)` triggers a MethodError.
# Always create figures via `PythonPlot.figure(figsize=...)` and add axes
# with `fig.add_axes([l, b, w, h])`.

module PhaseFieldsPythonPlotExt

using PhaseFields
import PhaseFields: FieldSnapshot1D, SpaceTimeSnapshot1D, FieldSnapshot2D,
                    plot_on_axis!, figure_publication, savefig_publication
import PythonPlot

const MM_PER_INCH = 25.4

# ── ETB layout helper (shared pattern with the OC PythonPlot ext) ──
function _layout_axes(axis_width_mm, axis_height_mm, n;
        margin_left_mm = 20.0, margin_right_mm = 3.0,
        margin_bottom_mm = 15.0, margin_top_mm = 8.0,
        hgap_mm = 18.0, vgap_mm = 15.0, nrows = 1, ncols = 1)
    widths  = axis_width_mm  isa AbstractVector ? axis_width_mm  : fill(axis_width_mm,  ncols)
    heights = axis_height_mm isa AbstractVector ? axis_height_mm : fill(axis_height_mm, nrows)
    fig_w_mm = margin_left_mm   + sum(widths)  + hgap_mm * (ncols - 1) + margin_right_mm
    fig_h_mm = margin_bottom_mm + sum(heights) + vgap_mm * (nrows - 1) + margin_top_mm
    fig_w = fig_w_mm / MM_PER_INCH
    fig_h = fig_h_mm / MM_PER_INCH
    positions = Vector{NTuple{4,Float64}}()
    for row in 1:nrows, col in 1:ncols
        left   = (margin_left_mm   + sum(widths[1:(col - 1)])    + hgap_mm * (col - 1))   / fig_w_mm
        bottom = (margin_bottom_mm + sum(heights[(row + 1):end]) + vgap_mm * (nrows - row)) / fig_h_mm
        push!(positions, (left, bottom, widths[col] / fig_w_mm, heights[row] / fig_h_mm))
    end
    return fig_w, fig_h, positions
end

# single-field view of a multi-field FieldSnapshot1D (for vertical-stack savefig)
_single_field_view(snap::FieldSnapshot1D, sym::Symbol, vals::Vector{Float64}) =
    FieldSnapshot1D(snap.x, Pair{Symbol,Vector{Float64}}[sym => vals],
                    snap.t, snap.xlabel, snap.title)

# 3-stage label resolution: kwarg overrides; else snap value; else recipe default
_resolve(kwarg, snapval, default) =
    isempty(kwarg) ? (isempty(snapval) ? default : snapval) : kwarg

# clims -> vmin/vmax NamedTuple (empty when nothing)
_clims_kw(cl) = isnothing(cl) ? (;) : (vmin = cl[1], vmax = cl[2])

# ══════════════════════════════════════════════════════════════════════
# L1 — plot_on_axis!(ax, snap; ...) : mutate a user-supplied axis, return ax
# ══════════════════════════════════════════════════════════════════════

"""
    plot_on_axis!(ax, snap::FieldSnapshot1D; kwargs...) -> ax

Draw a [`FieldSnapshot1D`](@ref) onto the matplotlib axis you supply and return
`ax`, so panels can be composed. A single field is drawn as a line; multiple
fields are overlaid with a legend. Requires `using PythonPlot`; not exported
(call as `PhaseFields.plot_on_axis!`).

# Arguments

- `ax`: a matplotlib axis (PythonCall `Py`) to draw onto
- `snap::FieldSnapshot1D`: the 1D field snapshot to plot

# Keywords

- `color=nothing`: line color for a single field (matplotlib default when
    `nothing`); multiple fields are colored automatically
- `linewidth=2.0`: line width
- `linestyle="-"`: line style for a single field
- `xlabel::AbstractString=""`: x axis label; empty uses `snap.xlabel`
- `ylabel::AbstractString=""`: y axis label; empty uses the field name
- `title::AbstractString=""`: title; empty uses `snap.title` or an automatic title
"""
function PhaseFields.plot_on_axis!(ax, snap::FieldSnapshot1D;
        xlabel = "", ylabel = "", title = "",
        color = nothing, linewidth = 2.0, linestyle = "-")
    if length(snap.fields) == 1
        sym, vals = snap.fields[1]
        if color === nothing
            ax.plot(snap.x, vals; linewidth = linewidth, linestyle = linestyle)
        else
            ax.plot(snap.x, vals; color = color, linewidth = linewidth, linestyle = linestyle)
        end
        ax.set_ylabel(_resolve(ylabel, "", string(sym)))
        ax.set_title(_resolve(title, snap.title, "$(sym) (t=$(snap.t))"))
    else
        for (sym, vals) in snap.fields
            ax.plot(snap.x, vals; label = string(sym), linewidth = linewidth)
        end
        ax.legend()
        isempty(ylabel) || ax.set_ylabel(ylabel)
        ax.set_title(_resolve(title, snap.title, ""))
    end
    ax.set_xlabel(_resolve(xlabel, snap.xlabel, "x"))
    return ax
end

"""
    plot_on_axis!(ax, snap::SpaceTimeSnapshot1D; kwargs...) -> ax

Draw a [`SpaceTimeSnapshot1D`](@ref) as a pcolormesh (with a colorbar) onto `ax`
and return `ax`. Requires `using PythonPlot`; not exported.

# Arguments

- `ax`: a matplotlib axis (PythonCall `Py`) to draw onto
- `snap::SpaceTimeSnapshot1D`: the space time field evolution to plot

# Keywords

- `colormap=nothing`: colormap; empty uses `snap.colormap`
- `clims=nothing`: color range `(vmin, vmax)`; empty uses `snap.clims`
- `xlabel::AbstractString=""`: x axis label; empty uses `snap.xlabel`
- `ylabel::AbstractString=""`: y axis label; empty uses `snap.ylabel`
- `title::AbstractString=""`: title; empty uses `snap.title` or the field name
"""
function PhaseFields.plot_on_axis!(ax, snap::SpaceTimeSnapshot1D;
        xlabel = "", ylabel = "", title = "", colormap = nothing, clims = nothing)
    cmap = String(isnothing(colormap) ? snap.colormap : colormap)
    cl = isnothing(clims) ? snap.clims : clims
    im = ax.pcolormesh(snap.t, snap.x, snap.data; cmap = cmap, _clims_kw(cl)...)
    ax.set_xlabel(_resolve(xlabel, snap.xlabel, "x"))
    ax.set_ylabel(_resolve(ylabel, snap.ylabel, "Time"))
    ax.set_title(_resolve(title, snap.title, string(snap.field_name)))
    ax.get_figure().colorbar(im; ax = ax)
    return ax
end

"""
    plot_on_axis!(ax, snap::FieldSnapshot2D; kwargs...) -> ax

Draw a [`FieldSnapshot2D`](@ref) as a pcolormesh (with a colorbar and equal
aspect) onto `ax` and return `ax`. Requires `using PythonPlot`; not exported.

# Arguments

- `ax`: a matplotlib axis (PythonCall `Py`) to draw onto
- `snap::FieldSnapshot2D`: the 2D field snapshot to plot

# Keywords

- `colormap=nothing`: colormap; empty uses `snap.colormap`
- `clims=nothing`: color range `(vmin, vmax)`; empty uses `snap.clims`
- `xlabel::AbstractString=""`: x axis label; empty uses `snap.xlabel`
- `ylabel::AbstractString=""`: y axis label; empty uses `snap.ylabel`
- `title::AbstractString=""`: title; empty uses `snap.title` or an automatic title
"""
function PhaseFields.plot_on_axis!(ax, snap::FieldSnapshot2D;
        xlabel = "", ylabel = "", title = "", colormap = nothing, clims = nothing)
    cmap = String(isnothing(colormap) ? snap.colormap : colormap)
    cl = isnothing(clims) ? snap.clims : clims
    im = ax.pcolormesh(snap.x, snap.y, permutedims(snap.field); cmap = cmap, _clims_kw(cl)...)
    ax.set_aspect("equal")
    ax.set_xlabel(_resolve(xlabel, snap.xlabel, "x"))
    ax.set_ylabel(_resolve(ylabel, snap.ylabel, "y"))
    ax.set_title(_resolve(title, snap.title, "$(snap.field_name) (t=$(snap.t))"))
    ax.get_figure().colorbar(im; ax = ax, shrink = 0.9, pad = 0.02)
    return ax
end

# ══════════════════════════════════════════════════════════════════════
# L2 — figure_publication(snap; ...) : create (fig, ax), single axis
# ══════════════════════════════════════════════════════════════════════

"""
    figure_publication(snap; axis_width_mm=80.0, axis_height_mm=60.0, ylims=nothing, kwargs...) -> (fig, ax)

Create a publication matplotlib figure and a single axis for `snap`, draw it via
[`plot_on_axis!`](@ref), and return `(fig, ax)` so you can tweak it before
saving. The caller owns the figure and must call `PythonPlot.close(fig)`.
Requires `using PythonPlot`; not exported (call as `PhaseFields.figure_publication`).

# Arguments

- `snap`: a `FieldSnapshot1D`, `SpaceTimeSnapshot1D`, or `FieldSnapshot2D`

# Keywords

- `axis_width_mm=80.0`: plotting area width in mm
- `axis_height_mm=60.0`: plotting area height in mm
- `ylims=nothing`: pass a tuple to override the y axis limits
- `kwargs...`: forwarded to [`plot_on_axis!`](@ref)
"""
function PhaseFields.figure_publication(
        snap::Union{FieldSnapshot1D,SpaceTimeSnapshot1D,FieldSnapshot2D};
        axis_width_mm = 80.0, axis_height_mm = 60.0,
        ylims = nothing, kwargs...)
    # margin_right widened to leave room for the colorbar on heatmap snapshots
    fig_w, fig_h, positions = _layout_axes(axis_width_mm, axis_height_mm, 1;
                                           margin_right_mm = 15.0)
    fig = PythonPlot.figure(figsize = (fig_w, fig_h))
    try
        ax = fig.add_axes(collect(positions[1]))
        plot_on_axis!(ax, snap; kwargs...)
        isnothing(ylims) || ax.set_ylim(ylims...)
        return fig, ax
    catch
        PythonPlot.close(fig)
        rethrow()
    end
end

# ══════════════════════════════════════════════════════════════════════
# L3 — savefig_publication(snap, path; ...) : L2 + savefig + close
# (type-specific methods; no untyped catch-all to avoid shadowing the
#  src fallback. The public docstring is attached to the FieldSnapshot1D
#  method and describes all variants.)
# ══════════════════════════════════════════════════════════════════════

function PhaseFields.savefig_publication(snap::SpaceTimeSnapshot1D, path; kwargs...)
    fig, _ = figure_publication(snap; kwargs...)
    try
        fig.savefig(path)
    finally
        PythonPlot.close(fig)
    end
    return path
end

function PhaseFields.savefig_publication(snap::FieldSnapshot2D, path; kwargs...)
    fig, _ = figure_publication(snap; kwargs...)
    try
        fig.savefig(path)
    finally
        PythonPlot.close(fig)
    end
    return path
end

"""
    savefig_publication(snap, path; kwargs...) -> path

Render `snap` to `path` and return `path`; the output format follows the file
extension (`.pdf` or `.png`). This is the convenience entry point (the only
exported layer): it delegates to [`figure_publication`](@ref) and closes the
figure for you. A multi field `FieldSnapshot1D` is drawn as a vertical stack of
panels, and an `AbstractVector{<:FieldSnapshot2D}` as a `layout=(rows, cols)`
grid. Requires `using PythonPlot`.

# Arguments

- `snap`: a `FieldSnapshot1D`, `SpaceTimeSnapshot1D`, `FieldSnapshot2D`, or
    `AbstractVector{<:FieldSnapshot2D}`
- `path::AbstractString`: output file; the extension selects PDF or PNG

# Keywords

- `axis_width_mm=80.0`, `axis_height_mm=60.0`: panel size in mm
- `layout=(1, n)`: grid arrangement for a vector of 2D snapshots
- `kwargs...`: forwarded through [`figure_publication`](@ref) to
    [`plot_on_axis!`](@ref) (including any plot styling)
"""
function PhaseFields.savefig_publication(snap::FieldSnapshot1D, path;
        axis_width_mm = 80.0, axis_height_mm = 60.0, kwargs...)
    if length(snap.fields) == 1
        fig, _ = figure_publication(snap;
                    axis_width_mm = axis_width_mm, axis_height_mm = axis_height_mm, kwargs...)
        try
            fig.savefig(path)
        finally
            PythonPlot.close(fig)
        end
    else
        n = length(snap.fields)
        fig_w, fig_h, positions = _layout_axes(axis_width_mm, axis_height_mm, n;
                                               nrows = n, ncols = 1)
        fig = PythonPlot.figure(figsize = (fig_w, fig_h))
        try
            for (i, (sym, vals)) in enumerate(snap.fields)
                ax = fig.add_axes(collect(positions[i]))
                sub = _single_field_view(snap, sym, vals)
                plot_on_axis!(ax, sub;
                              xlabel = (i == n ? snap.xlabel : ""),
                              title  = (i == 1 ? snap.title : ""),
                              kwargs...)
            end
            fig.savefig(path)
        finally
            PythonPlot.close(fig)
        end
    end
    return path
end

function PhaseFields.savefig_publication(snaps::AbstractVector{<:FieldSnapshot2D}, path;
        layout = (1, length(snaps)),
        axis_width_mm = 80.0, axis_height_mm = 60.0, kwargs...)
    nrows, ncols = layout
    fig_w, fig_h, positions = _layout_axes(axis_width_mm, axis_height_mm, length(snaps);
                                           nrows = nrows, ncols = ncols)
    fig = PythonPlot.figure(figsize = (fig_w, fig_h))
    try
        for (i, snap) in enumerate(snaps)
            ax = fig.add_axes(collect(positions[i]))
            plot_on_axis!(ax, snap; kwargs...)
        end
        fig.savefig(path)
    finally
        PythonPlot.close(fig)
    end
    return path
end

end # module
