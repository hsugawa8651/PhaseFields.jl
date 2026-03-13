# Cahn-Hilliard 2D Spinodal Decomposition — Long-time Evolution
#
# Based on 251_spinodal_2d.jl with extended time to show clear phase separation
# and coarsening. Produces a visually striking GIF for demonstration.
#
# Run: julia --project=. examples/252_spinodal_2d_long.jl

using PhaseFields
using OrdinaryDiffEq
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=5Plots.mm, right_margin=5Plots.mm,
    top_margin=5Plots.mm, bottom_margin=5Plots.mm
)

println("=== Cahn-Hilliard 2D Spinodal Decomposition (Long-time) ===\n")

# -----------------------------------------------------------------------------
# Model parameters (PFHub Benchmark 1 inspired)
# -----------------------------------------------------------------------------
model = CahnHilliardModel(
    M = 5.0,    # Mobility
    κ = 2.0     # Gradient energy coefficient
)

# Double-well free energy with minima at cα and cβ
f = DoubleWellFreeEnergy(
    ρs = 5.0,   # Barrier height
    cα = 0.3,   # α-phase equilibrium
    cβ = 0.7    # β-phase equilibrium
)

# Grid parameters
Nx, Ny = 128, 128
Lx, Ly = 200.0, 200.0
grid = UniformGrid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)

# Time parameters — long enough for clear coarsening
tspan = (0.0, 100.0)

println("Parameters:")
println("  Grid: $(Nx)×$(Ny), domain: $(Lx)×$(Ly)")
println("  Model: M=$(model.M), κ=$(model.κ)")
println("  Free energy: ρs=$(f.ρs), cα=$(f.cα), cβ=$(f.cβ)")
println("  Time: $(tspan[1]) to $(tspan[2])")
println()

# -----------------------------------------------------------------------------
# Initial conditions: larger fluctuations for visible initial pattern
# -----------------------------------------------------------------------------
c0_mean = 0.5
fluctuation = 0.05

# Deterministic fluctuation pattern (reproducible)
c0 = [c0_mean + fluctuation * sin(4π * x / Lx) * cos(4π * y / Ly) +
      fluctuation * 0.5 * sin(6π * x / Lx) * sin(6π * y / Ly)
      for x in grid.x, y in grid.y]

println("Initial conditions:")
println("  Mean concentration: $(c0_mean)")
println("  Fluctuation amplitude: $(fluctuation)")
println("  Min c: $(round(minimum(c0), digits=3)), Max c: $(round(maximum(c0), digits=3))")
println()

# -----------------------------------------------------------------------------
# Create problem and solve
# -----------------------------------------------------------------------------
problem = CahnHilliardProblem(model, grid, c0, tspan, f, bc=PeriodicBC())

println("Solving with OrdinaryDiffEq.jl...")
sol = PhaseFields.solve(problem, ROCK2(), saveat=1.0, maxiters=500000)
println("  Time points saved: $(length(sol.t))")
println("  Final time reached: $(sol.t[end])")

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
mean(x) = sum(x) / length(x)

println("\nAnalysis:")
for idx in [1, length(sol.t)÷2, length(sol.t)]
    c = reshape(sol.u[idx], Nx, Ny)
    c_mean = mean(c)
    c_var = sum((c .- c_mean).^2) / (Nx * Ny)
    println("  t=$(round(sol.t[idx], digits=1)):")
    println("    Mean: $(round(c_mean, digits=4)), Variance: $(round(c_var, digits=6))")
    println("    Min: $(round(minimum(c), digits=3)), Max: $(round(maximum(c), digits=3))")
end

# Mass conservation check
c_initial = reshape(sol.u[1], Nx, Ny)
c_final = reshape(sol.u[end], Nx, Ny)
mass_error = abs(sum(c_final) - sum(c_initial)) / sum(c_initial)
println("\nMass conservation error: $(round(mass_error * 100, digits=6))%")

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
println("\nGenerating plots...")

# Snapshots at selected times
n_snapshots = 4
indices = round.(Int, range(1, length(sol.t), length=n_snapshots))

p1 = plot(layout=(1, n_snapshots), size=(500*n_snapshots, 500))
for (i, idx) in enumerate(indices)
    c = reshape(sol.u[idx], Nx, Ny)
    heatmap!(p1[i], grid.x, grid.y, c',
             c=:RdBu, clim=(0.2, 0.8),
             xlabel="x", ylabel="y",
             title="c (t=$(round(sol.t[idx], digits=1)))",
             aspect_ratio=:equal)
end
savefig(p1, "examples/252_spinodal_2d_long.png")
println("  Saved: examples/252_spinodal_2d_long.png")

# Animation
println("Generating animation...")
anim = @animate for idx in 1:length(sol.t)
    c = reshape(sol.u[idx], Nx, Ny)
    heatmap(grid.x, grid.y, c',
            c=:RdBu, clim=(0.2, 0.8),
            xlabel="x", ylabel="y",
            title="Spinodal Decomposition (t=$(round(sol.t[idx], digits=1)))",
            aspect_ratio=:equal,
            size=(500, 500))
end
gif(anim, "examples/252_spinodal_2d_long.gif", fps=10)
println("  Saved: examples/252_spinodal_2d_long.gif")

println("\n✅ Simulation completed!")
