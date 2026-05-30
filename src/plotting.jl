# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Plotting stubs
#
# Implementations in extensions:
#   ext/PhaseFieldsRecipesBaseExt.jl (RecipesBase recipes)
#   ext/PhaseFieldsPlotsExt.jl       (plot_field, animate_field)
#   ext/PhaseFieldsPythonPlotExt.jl  (plot_on_axis!, figure_publication, savefig_publication)

"""
    plot_field(snap; kwargs...)

Plot a field snapshot. Requires `using Plots`.
"""
function plot_field end

"""
    animate_field(snaps; fps=10, kwargs...)

Create animation from snapshots. Requires `using Plots`.
"""
function animate_field end

"""
    savefig_publication(snap, filepath; kwargs...)

Save publication-quality figure. Requires `using PythonPlot`.
"""
function savefig_publication end

# Fallback
function savefig_publication(args...; kwargs...)
    throw(ArgumentError(
        "savefig_publication requires PythonPlot.jl. Run `using PythonPlot` first."
    ))
end

"""
    plot_on_axis!(ax, snap; kwargs...)

Draw a snapshot onto a user-supplied matplotlib axis and return `ax`.
Requires `using PythonPlot`. Not exported; call as `PhaseFields.plot_on_axis!`.
"""
function plot_on_axis! end

# Fallback
function plot_on_axis!(args...; kwargs...)
    throw(ArgumentError(
        "plot_on_axis! requires PythonPlot.jl. Run `using PythonPlot` first."
    ))
end

"""
    figure_publication(snap; kwargs...)

Create a publication figure and axis for a snapshot, returning `(fig, ax)`.
Requires `using PythonPlot`. Not exported; call as `PhaseFields.figure_publication`.
"""
function figure_publication end

# Fallback
function figure_publication(args...; kwargs...)
    throw(ArgumentError(
        "figure_publication requires PythonPlot.jl. Run `using PythonPlot` first."
    ))
end
