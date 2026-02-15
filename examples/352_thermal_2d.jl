# Thermal Phase Field 2D Simulation Example
#
# Demonstrates coupled thermal phase field simulation using the unified solve API.
# Solidification driven by undercooling with latent heat release.
#
# Run: julia --project=. examples/352_thermal_2d.jl

using PhaseFields
using OrdinaryDiffEq
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=5Plots.mm, right_margin=5Plots.mm,
    top_margin=5Plots.mm, bottom_margin=5Plots.mm
)

println("=== Thermal Phase Field 2D Simulation ===\n")

# -----------------------------------------------------------------------------
# Model parameters (dimensionless)
# -----------------------------------------------------------------------------
model = ThermalPhaseFieldModel(
    τ = 1.0,      # Relaxation time
    W = 0.04,     # Interface width parameter
    λ = 2.0,      # Coupling strength
    α = 1.0,      # Thermal diffusivity
    L = 1.0,      # Latent heat (dimensionless = 1)
    Cp = 1.0,     # Heat capacity (dimensionless = 1)
    Tm = 0.0      # Melting point (dimensionless = 0)
)

# Grid parameters
Nx, Ny = 80, 80
Lx, Ly = 1.0, 1.0
grid = UniformGrid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)

# Time parameters
tspan = (0.0, 0.3)

println("Parameters:")
println("  Grid: $(Nx)×$(Ny), domain: $(Lx)×$(Ly)")
println("  Model: τ=$(model.τ), W=$(model.W), λ=$(model.λ), α=$(model.α)")
println("  Time: $(tspan[1]) to $(tspan[2])")
println()

# -----------------------------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------------------------
# Phase field: small solid seed at center
R0 = 0.1  # Seed radius
cx, cy = 0.5, 0.5  # Center position

φ0 = [sqrt((x - cx)^2 + (y - cy)^2) < R0 ? 1.0 : 0.0
      for x in grid.x, y in grid.y]

# Temperature: uniform undercooling (u < 0 drives solidification)
undercooling = -0.3
u0 = undercooling * ones(Nx, Ny)

println("Initial conditions:")
println("  Solid seed at ($(cx), $(cy)) with radius $(R0)")
println("  Undercooling: u = $(undercooling)")
println("  Initial solid fraction: $(round(sum(φ0)/(Nx*Ny)*100, digits=1))%")
println()

# -----------------------------------------------------------------------------
# Create problem and solve using unified API
# -----------------------------------------------------------------------------
problem = ThermalProblem(model, grid, φ0, u0, tspan)

println("Solving with OrdinaryDiffEq.jl...")
sol = PhaseFields.solve(problem, Tsit5(), saveat=0.03)
println("  Time points saved: $(length(sol.t))")

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
N_total = Nx * Ny

println("\nAnalysis:")
for idx in [1, length(sol.t) ÷ 2, length(sol.t)]
    φ = reshape(sol.u[idx][1:N_total], Nx, Ny)
    u = reshape(sol.u[idx][N_total+1:end], Nx, Ny)
    solid_frac = sum(φ) / N_total
    u_mean = sum(u) / N_total
    u_max = maximum(u)
    println("  t=$(round(sol.t[idx], digits=3)):")
    println("    Solid fraction: $(round(solid_frac*100, digits=1))%")
    println("    Temperature: mean=$(round(u_mean, digits=3)), max=$(round(u_max, digits=3))")
end

# Verify physical behavior
φ_initial = reshape(sol.u[1][1:N_total], Nx, Ny)
φ_final = reshape(sol.u[end][1:N_total], Nx, Ny)
u_initial = reshape(sol.u[1][N_total+1:end], Nx, Ny)
u_final = reshape(sol.u[end][N_total+1:end], Nx, Ny)

println("\nPhysical behavior:")
if sum(φ_final) > sum(φ_initial)
    println("  ✓ Solidification observed (solid fraction increased)")
else
    println("  ✗ Warning: unexpected behavior")
end
if maximum(u_final) > maximum(u_initial)
    println("  ✓ Latent heat release observed (temperature increased)")
else
    println("  ✗ Warning: no latent heat release detected")
end

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
println("\nGenerating plots...")

# Plot: phase field and temperature at different times
n_snapshots = min(4, length(sol.t))
indices = round.(Int, range(1, length(sol.t), length=n_snapshots))

# Phase field evolution
p1 = plot(layout=(1, n_snapshots), size=(500*n_snapshots, 500))
for (i, idx) in enumerate(indices)
    φ = reshape(sol.u[idx][1:N_total], Nx, Ny)
    heatmap!(p1[i], grid.x, grid.y, φ',
             c=:viridis, clim=(0, 1),
             xlabel="x", ylabel="y",
             title="φ (t=$(round(sol.t[idx], digits=2)))",
             aspect_ratio=:equal)
end
savefig(p1, "examples/352_thermal_2d_phase.png")
println("  Saved: examples/352_thermal_2d_phase.png")

# Temperature evolution
p2 = plot(layout=(1, n_snapshots), size=(500*n_snapshots, 500))
for (i, idx) in enumerate(indices)
    u = reshape(sol.u[idx][N_total+1:end], Nx, Ny)
    heatmap!(p2[i], grid.x, grid.y, u',
             c=:coolwarm, clim=(-0.5, 0.5),
             xlabel="x", ylabel="y",
             title="u (t=$(round(sol.t[idx], digits=2)))",
             aspect_ratio=:equal)
end
savefig(p2, "examples/352_thermal_2d_temp.png")
println("  Saved: examples/352_thermal_2d_temp.png")

# Combined animation
println("Generating animation...")
anim = @animate for idx in 1:length(sol.t)
    φ = reshape(sol.u[idx][1:N_total], Nx, Ny)
    u = reshape(sol.u[idx][N_total+1:end], Nx, Ny)

    p_φ = heatmap(grid.x, grid.y, φ',
                  c=:viridis, clim=(0, 1),
                  xlabel="x", ylabel="y",
                  title="Phase field φ",
                  aspect_ratio=:equal)

    p_u = heatmap(grid.x, grid.y, u',
                  c=:coolwarm, clim=(-0.5, 0.5),
                  xlabel="x", ylabel="y",
                  title="Temperature u",
                  aspect_ratio=:equal)

    plot(p_φ, p_u, layout=(1, 2), size=(1000, 500),
         plot_title="Thermal Phase Field 2D (t=$(round(sol.t[idx], digits=3)))")
end
gif(anim, "examples/352_thermal_2d.gif", fps=8)
println("  Saved: examples/352_thermal_2d.gif")

println("\n✅ Simulation completed!")
