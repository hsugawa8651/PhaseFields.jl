# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - RecipesBase extension

module PhaseFieldsRecipesBaseExt

using RecipesBase
using PhaseFields: FieldSnapshot1D, SpaceTimeSnapshot1D, FieldSnapshot2D

@recipe function f(snap::FieldSnapshot1D)
    n = length(snap.fields)
    if n == 1
        sym, vals = snap.fields[1]
        xlabel --> snap.xlabel
        ylabel --> string(sym)
        title  --> (snap.title == "" ? "$(sym) (t=$(snap.t))" : snap.title)
        linewidth --> 2
        snap.x, vals
    else
        layout := (n, 1)
        for (i, (sym, vals)) in enumerate(snap.fields)
            @series begin
                subplot := i
                xlabel --> (i == n ? snap.xlabel : "")
                ylabel --> string(sym)
                title  --> (i == 1 && snap.title != "" ? snap.title : "")
                linewidth --> 2
                snap.x, vals
            end
        end
    end
end

@recipe function f(snap::SpaceTimeSnapshot1D)
    seriestype := :heatmap
    xlabel --> snap.xlabel
    ylabel --> snap.ylabel
    title  --> (snap.title == "" ? string(snap.field_name) : snap.title)
    seriescolor --> snap.colormap
    if snap.clims !== nothing
        clims --> snap.clims
    end
    snap.t, snap.x, snap.data
end

@recipe function f(snap::FieldSnapshot2D)
    seriestype := :heatmap
    xlabel --> snap.xlabel
    ylabel --> snap.ylabel
    title  --> (snap.title == "" ? "$(snap.field_name) (t=$(snap.t))" : snap.title)
    aspect_ratio --> :equal
    seriescolor --> snap.colormap
    if snap.clims !== nothing
        clims --> snap.clims
    end
    snap.x, snap.y, snap.field'
end

end # module
