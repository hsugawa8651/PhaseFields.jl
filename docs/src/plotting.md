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
grid = UniformGrid1D(N=101, L=1.0)
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
| title | `"φ (t=0.5)"` (generated from the field name and time) |
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
grid = UniformGrid2D(Nx=64, Ny=64, Lx=1.0, Ly=1.0)
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

For publication-quality static figures (PDF / PNG via matplotlib), load the
PythonPlot extension. The API has **three layers** (L1 / L2 are unexported —
call them as `PhaseFields.…`; L3 `savefig_publication` is exported). Install
with `pkg> add PythonPlot` (matplotlib is provided automatically via Conda).

| Layer | Function | Returns | Use |
|-------|----------|---------|-----|
| L3 | `savefig_publication(snap, path; ...)` | `path` | save to PDF or PNG in one call (format from the file extension; delegates to L2) |
| L2 | `PhaseFields.figure_publication(snap; ...)` | `(fig, ax)` | a sized figure + axis to tweak before saving |
| L1 | `PhaseFields.plot_on_axis!(ax, snap; ...)` | `ax` | draw onto your own matplotlib axis (compose a subplot grid) |

```julia
using PhaseFields, PythonPlot
```

On a headless machine or in CI (no display), select a non-interactive backend
before plotting: `PythonPlot.matplotlib.use("Agg")`.

In every layer the `snap` argument may be a `FieldSnapshot1D`,
`SpaceTimeSnapshot1D`, or `FieldSnapshot2D`, plus an
`AbstractVector{<:FieldSnapshot2D}` for an L3 panel grid.

```julia
grid = UniformGrid2D(Nx=64, Ny=64, Lx=1.0, Ly=1.0)
snap = FieldSnapshot2D(grid, field, 0.5, :φ; colormap=:viridis)

# L3 — save in one call (format from the file extension)
savefig_publication(snap, "phi.pdf")
savefig_publication(snap, "phi.png")

# L2 — adjust before saving
fig, ax = PhaseFields.figure_publication(snap; axis_width_mm=80.0, axis_height_mm=60.0)
ax.set_title("My field")
fig.savefig("phi_custom.pdf")
PythonPlot.close(fig)

# L1 — compose into your own subplot grid
fig = PythonPlot.figure()
ax = fig.add_subplot(1, 2, 1)
PhaseFields.plot_on_axis!(ax, snap)
fig.savefig("composite.pdf")
PythonPlot.close(fig)
```

With L1 and L2 you create the figure (via `PythonPlot.figure` / `add_subplot`),
so you own it and must call `PythonPlot.close(fig)`; `savefig_publication` (L3)
closes its figure for you.

### Keyword reference

Keywords are documented on the functions below. The plot keywords for each
snapshot type (and the label and title precedence) live on `plot_on_axis!`;
`figure_publication` adds the figure size keywords; `savefig_publication`
forwards everything:

```@docs
PhaseFields.savefig_publication
PhaseFields.figure_publication
PhaseFields.plot_on_axis!
```
