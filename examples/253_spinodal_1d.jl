# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - 1D Cahn-Hilliard spinodal decomposition (DiffEq)

# 1D Spinodal Decomposition with Unified Solve API
#
# Demonstrates 1D spinodal decomposition using CahnHilliardProblem
# and OrdinaryDiffEq.jl. Phase separation from an unstable
# homogeneous mixture.
#
# Compare with:
#   201_spinodal_1d.jl  — standalone explicit Euler
#   251_spinodal_2d.jl  — 2D version with unified solve API
#
# Reference: Jokisaari et al. (2017) Comp. Mater. Sci. 126, 139-151
#
# Run: julia --project=. examples/253_spinodal_1d.jl

using PhaseFields
using OrdinaryDiffEq
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14,
    legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

println("=== 1D Spinodal Decomposition (Cahn-Hilliard, DiffEq) ===\n")

# ---------------------------------------------------------------------
# Model parameters (PFHub Benchmark 1 inspired)
# ---------------------------------------------------------------------
model = CahnHilliardModel(
    M = 5.0,    # Mobility
    κ = 2.0     # Gradient energy coefficient
)

f = DoubleWellFreeEnergy(
    ρs = 5.0,   # Barrier height
    cα = 0.3,   # α-phase equilibrium
    cβ = 0.7    # β-phase equilibrium
)

# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------
Nx = 200
Lx = 200.0
grid = UniformGrid1D(N=Nx, L=Lx)

# Time parameters
tspan = (0.0, 100.0)

println("Parameters:")
println("  Grid: Nx=$Nx, Lx=$Lx, dx=$(grid.dx)")
println("  Model: M=$(model.M), κ=$(model.κ)")
println("  Free energy: ρs=$(f.ρs), cα=$(f.cα), cβ=$(f.cβ)")
println("  tspan: $tspan")
println()

# ---------------------------------------------------------------------
# Initial condition: deterministic perturbation around c = 0.5
# ---------------------------------------------------------------------
c0 = [0.5 + 0.02 * sin(4π * x / Lx) + 0.01 * sin(6π * x / Lx)
      for x in grid.x]

println("Initial condition:")
println("  Mean: 0.5, perturbation: sin modes")
println("  Min c: $(round(minimum(c0), digits=3)), " *
        "Max c: $(round(maximum(c0), digits=3))")
println()

# ---------------------------------------------------------------------
# Create problem and solve
# ---------------------------------------------------------------------
problem = CahnHilliardProblem(
    model, grid, c0, tspan, f, bc=PeriodicBC()
)

println("Solving with ROCK2 (stiff solver)...")
sol = PhaseFields.solve(
    problem, ROCK2(), saveat=10.0, maxiters=100000
)
println("  Time points saved: $(length(sol.t))")
println("  Final time reached: $(sol.t[end])")

# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------
println("\nAnalysis:")
mean(x) = sum(x) / length(x)

for idx in [1, length(sol.t)]
    c = sol.u[idx]
    c_mean = mean(c)
    c_var = sum((c .- c_mean) .^ 2) / Nx
    println("  t=$(round(sol.t[idx], digits=1)):")
    println("    Mean: $(round(c_mean, digits=4)), " *
            "Variance: $(round(c_var, digits=6))")
    println("    Min: $(round(minimum(c), digits=3)), " *
            "Max: $(round(maximum(c), digits=3))")
end

# Mass conservation check
mass_initial = sum(sol.u[1])
mass_final = sum(sol.u[end])
mass_error = abs(mass_final - mass_initial) / mass_initial
println("\nMass conservation error: " *
        "$(round(mass_error * 100, digits=6))%")

# Phase separation check
var_initial = sum((sol.u[1] .- mean(sol.u[1])) .^ 2) / Nx
var_final = sum((sol.u[end] .- mean(sol.u[end])) .^ 2) / Nx
if var_final > var_initial
    println("  Variance increased — phase separation observed")
end

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
println("\nGenerating plots...")

# Plot 1: Time evolution
p1 = plot(
    title="1D Spinodal Decomposition (DiffEq)",
    xlabel="Position x",
    ylabel="Concentration c",
    ylims=(0.2, 0.8),
    legend=:topright,
    size=(800, 500)
)
hline!([f.cα, f.cβ], color=:gray, linestyle=:dash, label="")

n_curves = min(6, length(sol.t))
indices = round.(Int, range(1, length(sol.t), length=n_curves))
colors = cgrad(:viridis, n_curves, categorical=true)
for (i, idx) in enumerate(indices)
    plot!(grid.x, sol.u[idx],
          label="t=$(round(sol.t[idx], digits=1))",
          color=colors[i],
          linewidth=1.5)
end
savefig(p1, "examples/253_spinodal_1d_evolution.png")
println("  Saved: examples/253_spinodal_1d_evolution.png")

# Plot 2: Heatmap (space-time)
c_matrix = hcat(sol.u...)
p2 = heatmap(
    sol.t, grid.x, c_matrix,
    title="Concentration Field (space-time)",
    xlabel="Time",
    ylabel="Position x",
    colorbar_title="c",
    color=:RdBu,
    clims=(f.cα, f.cβ),
    size=(800, 500)
)
savefig(p2, "examples/253_spinodal_1d_heatmap.png")
println("  Saved: examples/253_spinodal_1d_heatmap.png")

# Animation
println("Generating animation...")
anim = @animate for idx in 1:length(sol.t)
    plot(grid.x, sol.u[idx],
         color=:purple, linewidth=2,
         fill=(f.cα, 0.3, :purple),
         label="",
         xlabel="Position x",
         ylabel="Concentration c",
         ylims=(0.2, 0.8),
         title="Spinodal Decomposition " *
               "(t=$(round(sol.t[idx], digits=1)))",
         size=(800, 400))
    hline!([f.cα, f.cβ],
           color=:gray, linestyle=:dash, label="")
end
gif(anim, "examples/253_spinodal_1d.gif", fps=3)
println("  Saved: examples/253_spinodal_1d.gif")

println("\n=== Simulation completed! ===")
