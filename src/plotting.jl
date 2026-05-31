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

# PythonPlot extension API (3-layer). Implemented — with their authoritative,
# per-method docstrings — in ext/PhaseFieldsPythonPlotExt.jl. The extension
# module is added to Documenter's `modules` so `@docs` renders those docstrings.
function savefig_publication end
function plot_on_axis! end
function figure_publication end

# Fallbacks (when PythonPlot is not loaded)
function savefig_publication(args...; kwargs...)
    throw(ArgumentError(
        "savefig_publication requires PythonPlot.jl. Run `using PythonPlot` first."
    ))
end

function plot_on_axis!(args...; kwargs...)
    throw(ArgumentError(
        "plot_on_axis! requires PythonPlot.jl. Run `using PythonPlot` first."
    ))
end

function figure_publication(args...; kwargs...)
    throw(ArgumentError(
        "figure_publication requires PythonPlot.jl. Run `using PythonPlot` first."
    ))
end
