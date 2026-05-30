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

const CM_PER_INCH = 2.54

# ── ETB layout helper (shared pattern with OC/BT/PX PythonPlot ext) ──
function _layout_axes(axis_width_cm, axis_height_cm, n;
        margin_left_cm = 2.0, margin_right_cm = 0.3,
        margin_bottom_cm = 1.5, margin_top_cm = 0.8,
        hgap_cm = 1.8, vgap_cm = 1.5, nrows = 1, ncols = 1)
    widths  = axis_width_cm  isa AbstractVector ? axis_width_cm  : fill(axis_width_cm,  ncols)
    heights = axis_height_cm isa AbstractVector ? axis_height_cm : fill(axis_height_cm, nrows)
    fig_w_cm = margin_left_cm   + sum(widths)  + hgap_cm * (ncols - 1) + margin_right_cm
    fig_h_cm = margin_bottom_cm + sum(heights) + vgap_cm * (nrows - 1) + margin_top_cm
    fig_w = fig_w_cm / CM_PER_INCH
    fig_h = fig_h_cm / CM_PER_INCH
    positions = Vector{NTuple{4,Float64}}()
    for row in 1:nrows, col in 1:ncols
        left   = (margin_left_cm   + sum(widths[1:(col - 1)])    + hgap_cm * (col - 1))   / fig_w_cm
        bottom = (margin_bottom_cm + sum(heights[(row + 1):end]) + vgap_cm * (nrows - row)) / fig_h_cm
        push!(positions, (left, bottom, widths[col] / fig_w_cm, heights[row] / fig_h_cm))
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

function PhaseFields.figure_publication(
        snap::Union{FieldSnapshot1D,SpaceTimeSnapshot1D,FieldSnapshot2D};
        axis_width_cm = 8.0, axis_height_cm = 6.0, margin_right_cm = 1.5,
        ylims = nothing, kwargs...)
    fig_w, fig_h, positions = _layout_axes(axis_width_cm, axis_height_cm, 1;
                                           margin_right_cm = margin_right_cm)
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
# (all methods type-specific; no untyped catch-all to avoid shadowing the
#  src fallback)
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

function PhaseFields.savefig_publication(snap::FieldSnapshot1D, path;
        axis_width_cm = 8.0, axis_height_cm = 6.0, kwargs...)
    if length(snap.fields) == 1
        fig, _ = figure_publication(snap;
                    axis_width_cm = axis_width_cm, axis_height_cm = axis_height_cm, kwargs...)
        try
            fig.savefig(path)
        finally
            PythonPlot.close(fig)
        end
    else
        n = length(snap.fields)
        fig_w, fig_h, positions = _layout_axes(axis_width_cm, axis_height_cm, n;
                                               nrows = n, ncols = 1)
        fig = PythonPlot.figure(figsize = (fig_w, fig_h))
        try
            for (i, (sym, vals)) in enumerate(snap.fields)
                ax = fig.add_axes(collect(positions[i]))
                sub = _single_field_view(snap, sym, vals)
                plot_on_axis!(ax, sub;
                              xlabel = (i == n ? snap.xlabel : ""),
                              title  = (i == 1 ? snap.title : ""))
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
        axis_width_cm = 8.0, axis_height_cm = 6.0, kwargs...)
    nrows, ncols = layout
    fig_w, fig_h, positions = _layout_axes(axis_width_cm, axis_height_cm, length(snaps);
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
