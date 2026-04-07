# Plotting

PhaseFields.jl provides snapshot types and [RecipesBase.jl](https://github.com/JuliaPlots/RecipesBase.jl) plot recipes for visualizing simulation results.

| Extension | Trigger | Functionality |
|-----------|---------|---------------|
| RecipesBaseExt | `using RecipesBase` (or any backend) | `plot(snapshot)` recipes for all snapshot types |
| PlotsExt | `using Plots` | `plot_field`, `animate_field` (planned) |
| PythonPlotExt | `using PythonPlot` | `savefig_publication` (planned) |

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
