# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Plotting stubs
#
# Implementations in extensions:
#   ext/PhaseFieldsRecipesBaseExt.jl (RecipesBase recipes)
#   ext/PhaseFieldsPlotsExt.jl       (plot_field, animate_field)
#   ext/PhaseFieldsPythonPlotExt.jl  (savefig_publication)

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
