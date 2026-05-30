# Plotting

PhaseFields.jl provides snapshot types and [RecipesBase.jl](https://github.com/JuliaPlots/RecipesBase.jl) plot recipes for visualizing simulation results.

| Extension | Trigger | Functionality |
|-----------|---------|---------------|
| RecipesBaseExt | `using RecipesBase` (or any backend) | `plot(snapshot)` recipes for all snapshot types |
| PlotsExt | `using Plots` | `plot_field`, `animate_field` (planned) |
| PythonPlotExt | `using PythonPlot` | `plot_on_axis!`, `figure_publication`, `savefig_publication` |

```julia
using PhaseFields
using Plots   # or CairoMakie, etc.
```

---

## RecipesBase Recipes

### 1D Field Snapshot — `FieldSnapshot1D`

Plot one or more fields at a specific time step.

**Single field:**

```julia
grid = UniformGrid1D(0.0, 1.0, 101)
snap = FieldSnapshot1D(grid, φ, t; field_name=:φ)
plot(snap)
```

**Multiple fields** (stacked subplots):

```julia
snap = FieldSnapshot1D(grid, Dict(:φ => φ, :c => c), t)
plot(snap)
```

| Attribute | Default |
|-----------|---------|
| xlabel | value of `snap.xlabel` (default `"x"`) |
| ylabel | field name (e.g. `"φ"`) |
| title | `"φ (t=0.5)"` (auto-generated from field name and time) |
| linewidth | 2 |
| layout | `(n, 1)` for n fields |

### Space-Time Heatmap — `SpaceTimeSnapshot1D`

Plot the time evolution of a 1D field as a heatmap (x vs t).

```julia
snap = SpaceTimeSnapshot1D(x, t_values, data, :φ;
    colormap=:RdBu, clims=(-1.0, 1.0))
plot(snap)
```

| Attribute | Default |
|-----------|---------|
| seriestype | `:heatmap` |
| xlabel | value of `snap.xlabel` (default `"x"`) |
| ylabel | value of `snap.ylabel` (default `"Time"`) |
| title | field name |
| seriescolor | `:RdBu` |
| clims | `nothing` (auto) |

### 2D Field Snapshot — `FieldSnapshot2D`

Plot a 2D field at a specific time step as a heatmap.

```julia
grid = UniformGrid2D(0.0, 1.0, 64, 0.0, 1.0, 64)
snap = FieldSnapshot2D(grid, field, t, :φ;
    colormap=:viridis, clims=(0.0, 1.0))
plot(snap)
```

| Attribute | Default |
|-----------|---------|
| seriestype | `:heatmap` |
| xlabel | value of `snap.xlabel` (default `"x"`) |
| ylabel | value of `snap.ylabel` (default `"y"`) |
| title | `"φ (t=0.5)"` |
| aspect_ratio | `:equal` |
| seriescolor | `:viridis` |
| clims | `nothing` (auto) |

---

## Customization

Override any recipe attribute with standard keyword arguments:

```julia
plot(snap, title="Custom Title", seriescolor=:plasma, size=(800, 600))
```

---

## Publication-quality output (PythonPlot)

`using PythonPlot` enables a three-layer [matplotlib](https://matplotlib.org/)-based
API for publication figures. Install with `pkg> add PythonPlot` (matplotlib is
provided automatically via Conda). These functions are not exported; call them as
`PhaseFields.plot_on_axis!` / `PhaseFields.figure_publication`.

| Layer | Function | Returns | Use |
|-------|----------|---------|-----|
| L1 | `PhaseFields.plot_on_axis!(ax, snap; ...)` | `ax` | draw onto your own matplotlib axis (compose subplots) |
| L2 | `PhaseFields.figure_publication(snap; ...)` | `(fig, ax)` | get a sized figure + axis to tweak before saving |
| L3 | `savefig_publication(snap, path; ...)` | `path` | one-shot save to PDF or PNG (by extension) |

```julia
using PhaseFields, PythonPlot

grid = UniformGrid2D(Nx=64, Ny=64, Lx=1.0, Ly=1.0)
snap = FieldSnapshot2D(grid, field, 0.5, :φ; colormap=:viridis)

# L3 — one-shot save (format from the file extension)
savefig_publication(snap, "phi.pdf")
savefig_publication(snap, "phi.png")

# L2 — adjust before saving
fig, ax = PhaseFields.figure_publication(snap; axis_width_cm=8.0, axis_height_cm=6.0)
ax.set_title("My field")
fig.savefig("phi_custom.pdf")
PythonPlot.close(fig)

# L1 — compose into your own subplot grid
fig = PythonPlot.figure()
ax = fig.add_subplot(1, 2, 1)
PhaseFields.plot_on_axis!(ax, snap)
```

Supported snapshots: `FieldSnapshot1D` (single field, or multiple fields stacked
vertically by L3), `SpaceTimeSnapshot1D`, `FieldSnapshot2D`, and
`AbstractVector{<:FieldSnapshot2D}` (panel grid via `layout=(rows, cols)`).

Common keyword arguments:

| Keyword | Applies to | Meaning |
|---------|-----------|---------|
| `axis_width_cm`, `axis_height_cm` | L2, L3 | physical axis size (default `8.0` × `6.0`) |
| `clims` | L1, L2, L3 | color range `(vmin, vmax)` for heatmaps |
| `colormap` | L1, L2, L3 | override `snap.colormap` |
| `ylims` | L2, L3 single-axis | y-axis limits (not supported for vertical-stack / panel paths) |
| `layout` | L3 vector | `(rows, cols)` panel arrangement |

Labels and titles follow the recipe conventions: an explicit keyword wins, then
`snap.xlabel` / `snap.ylabel` / `snap.title`, then an auto-generated default.
