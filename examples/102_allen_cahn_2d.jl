# Allen-Cahn 2D Simulation Example
#
# Demonstrates 2D phase field simulation using the unified solve API.
# Circular solid phase shrinks due to curvature-driven motion.
#
# Run: julia --project=. examples/102_allen_cahn_2d.jl

using PhaseFields
using OrdinaryDiffEq
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=5Plots.mm, right_margin=5Plots.mm,
    top_margin=5Plots.mm, bottom_margin=5Plots.mm
)

println("=== Allen-Cahn 2D Simulation ===\n")

# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------
model = AllenCahnModel(
    τ = 1.0,    # Relaxation time
    W = 0.05,   # Interface width parameter
    m = 0.0     # No external driving force (curvature-driven only)
)

# Grid parameters
Nx, Ny = 100, 100
Lx, Ly = 1.0, 1.0
grid = UniformGrid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)

# Time parameters
tspan = (0.0, 0.5)

println("Parameters:")
println("  Grid: $(Nx)×$(Ny), domain: $(Lx)×$(Ly)")
println("  Model: τ=$(model.τ), W=$(model.W)")
println("  Time: $(tspan[1]) to $(tspan[2])")
println()

# -----------------------------------------------------------------------------
# Initial condition: circular solid phase at center
# -----------------------------------------------------------------------------
R0 = 0.3  # Initial radius
cx, cy = 0.5, 0.5  # Center position

φ0 = [sqrt((x - cx)^2 + (y - cy)^2) < R0 ? 1.0 : 0.0
      for x in grid.x, y in grid.y]

initial_solid_fraction = sum(φ0) / (Nx * Ny)
println("Initial condition:")
println("  Circular solid at ($(cx), $(cy)) with radius $(R0)")
println("  Initial solid fraction: $(round(initial_solid_fraction * 100, digits=1))%")
println()

# -----------------------------------------------------------------------------
# Create problem and solve using unified API
# -----------------------------------------------------------------------------
problem = PhaseFieldProblem(
    model = model,
    domain = grid,
    φ0 = φ0,
    tspan = tspan,
    bc = NeumannBC()
)

println("Solving with OrdinaryDiffEq.jl...")
sol = PhaseFields.solve(problem, Tsit5(), saveat=0.05)
println("  Time points saved: $(length(sol.t))")

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
println("\nAnalysis:")

# Extract solutions at different times
times_to_analyze = [1, length(sol.t) ÷ 2, length(sol.t)]
for idx in times_to_analyze
    φ = reshape(sol.u[idx], Nx, Ny)
    solid_frac = sum(φ) / (Nx * Ny)
    println("  t=$(round(sol.t[idx], digits=3)): solid fraction = $(round(solid_frac * 100, digits=1))%")
end

# Verify curvature-driven shrinkage
φ_initial = reshape(sol.u[1], Nx, Ny)
φ_final = reshape(sol.u[end], Nx, Ny)
println("\nPhysical behavior:")
println("  Initial solid fraction: $(round(sum(φ_initial)/(Nx*Ny)*100, digits=1))%")
println("  Final solid fraction: $(round(sum(φ_final)/(Nx*Ny)*100, digits=1))%")
if sum(φ_final) < sum(φ_initial)
    println("  ✓ Curvature-driven shrinkage observed (correct)")
else
    println("  ✗ Warning: solid did not shrink as expected")
end

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
println("\nGenerating plots...")

# Plot: snapshots at different times
n_snapshots = min(4, length(sol.t))
indices = round.(Int, range(1, length(sol.t), length=n_snapshots))

p = plot(layout=(1, n_snapshots), size=(500*n_snapshots, 500))
for (i, idx) in enumerate(indices)
    φ = reshape(sol.u[idx], Nx, Ny)
    heatmap!(p[i], grid.x, grid.y, φ',
             c=:viridis, clim=(0, 1),
             xlabel="x", ylabel="y",
             title="t=$(round(sol.t[idx], digits=2))",
             aspect_ratio=:equal)
end
savefig(p, "examples/102_allen_cahn_2d.png")
println("  Saved: examples/102_allen_cahn_2d.png")

# Animation
println("Generating animation...")
anim = @animate for idx in 1:length(sol.t)
    φ = reshape(sol.u[idx], Nx, Ny)
    heatmap(grid.x, grid.y, φ',
            c=:viridis, clim=(0, 1),
            xlabel="x", ylabel="y",
            title="Allen-Cahn 2D (t=$(round(sol.t[idx], digits=3)))",
            aspect_ratio=:equal,
            size=(500, 500))
end
gif(anim, "examples/102_allen_cahn_2d.gif", fps=10)
println("  Saved: examples/102_allen_cahn_2d.gif")

println("\n✅ Simulation completed!")
