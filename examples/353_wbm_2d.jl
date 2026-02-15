# WBM 2D Solidification Example
#
# Demonstrates 2D binary alloy solidification using the WBM model
# with the unified solve API. Coupled phase field and concentration evolution.
#
# Run: julia --project=. examples/353_wbm_2d.jl

using PhaseFields
using OrdinaryDiffEq
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=5Plots.mm, right_margin=5Plots.mm,
    top_margin=5Plots.mm, bottom_margin=5Plots.mm
)

println("=== WBM 2D Binary Alloy Solidification ===\n")

# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------
# Free energy functions (parabolic approximation)
f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)  # Solid equilibrium at c=0.1
f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)  # Liquid equilibrium at c=0.9

# WBM model parameters
model = WBMModel(
    M_φ = 1.0,      # Phase field mobility
    κ = 1.0,        # Gradient energy coefficient
    W = 1.0,        # Barrier height
    D_s = 0.01,     # Solid diffusivity (slow)
    D_l = 1.0       # Liquid diffusivity (fast)
)

# Grid parameters
Nx, Ny = 60, 60
Lx, Ly = 1.0, 1.0
grid = UniformGrid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)

# Time parameters
tspan = (0.0, 0.1)

println("Parameters:")
println("  Grid: $(Nx)×$(Ny), domain: $(Lx)×$(Ly)")
println("  Model: M_φ=$(model.M_φ), κ=$(model.κ), W=$(model.W)")
println("  Diffusivity: D_s=$(model.D_s), D_l=$(model.D_l)")
println("  Free energy: solid c_eq=$(f_s.c_eq), liquid c_eq=$(f_l.c_eq)")
println("  Time: $(tspan[1]) to $(tspan[2])")
println()

# -----------------------------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------------------------
# Phase field: solid seed at center
R0 = 0.15  # Seed radius
cx, cy = 0.5, 0.5

φ0 = [sqrt((x - cx)^2 + (y - cy)^2) < R0 ? 1.0 : 0.0
      for x in grid.x, y in grid.y]

# Concentration: intermediate value (drives solidification)
c0_value = 0.3  # Between solid (0.1) and liquid (0.9) equilibria
c0 = c0_value * ones(Nx, Ny)

println("Initial conditions:")
println("  Solid seed at ($(cx), $(cy)) with radius $(R0)")
println("  Initial concentration: $(c0_value)")
println("  Initial solid fraction: $(round(sum(φ0)/(Nx*Ny)*100, digits=1))%")
println()

# -----------------------------------------------------------------------------
# Create problem and solve using unified API
# -----------------------------------------------------------------------------
problem = WBMProblem(model, grid, φ0, c0, tspan, f_s, f_l)

println("Solving with OrdinaryDiffEq.jl...")
sol = PhaseFields.solve(problem, Tsit5(), saveat=0.01)
println("  Time points saved: $(length(sol.t))")

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
N_total = Nx * Ny

println("\nAnalysis:")
for idx in [1, length(sol.t) ÷ 2, length(sol.t)]
    φ = reshape(sol.u[idx][1:N_total], Nx, Ny)
    c = reshape(sol.u[idx][N_total+1:end], Nx, Ny)
    solid_frac = sum(φ) / N_total
    c_mean = sum(c) / N_total
    c_solid = sum(c .* φ) / max(sum(φ), 1e-10)
    c_liquid = sum(c .* (1 .- φ)) / max(sum(1 .- φ), 1e-10)
    println("  t=$(round(sol.t[idx], digits=3)):")
    println("    Solid fraction: $(round(solid_frac*100, digits=1))%")
    println("    Mean concentration: $(round(c_mean, digits=3))")
    println("    Solid c: $(round(c_solid, digits=3)), Liquid c: $(round(c_liquid, digits=3))")
end

# Verify physical behavior
φ_initial = reshape(sol.u[1][1:N_total], Nx, Ny)
φ_final = reshape(sol.u[end][1:N_total], Nx, Ny)

println("\nPhysical behavior:")
if sum(φ_final) > sum(φ_initial)
    println("  ✓ Solidification observed (solid fraction increased)")
elseif sum(φ_final) < sum(φ_initial)
    println("  ✓ Melting observed (solid fraction decreased)")
else
    println("  ○ No significant phase change")
end

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
println("\nGenerating plots...")

# Plot: phase field and concentration at different times
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
savefig(p1, "examples/353_wbm_2d_phase.png")
println("  Saved: examples/353_wbm_2d_phase.png")

# Concentration evolution
p2 = plot(layout=(1, n_snapshots), size=(500*n_snapshots, 500))
for (i, idx) in enumerate(indices)
    c = reshape(sol.u[idx][N_total+1:end], Nx, Ny)
    heatmap!(p2[i], grid.x, grid.y, c',
             c=:plasma, clim=(0, 1),
             xlabel="x", ylabel="y",
             title="c (t=$(round(sol.t[idx], digits=2)))",
             aspect_ratio=:equal)
end
savefig(p2, "examples/353_wbm_2d_conc.png")
println("  Saved: examples/353_wbm_2d_conc.png")

# Combined animation
println("Generating animation...")
anim = @animate for idx in 1:length(sol.t)
    φ = reshape(sol.u[idx][1:N_total], Nx, Ny)
    c = reshape(sol.u[idx][N_total+1:end], Nx, Ny)

    p_φ = heatmap(grid.x, grid.y, φ',
                  c=:viridis, clim=(0, 1),
                  xlabel="x", ylabel="y",
                  title="Phase field φ",
                  aspect_ratio=:equal)

    p_c = heatmap(grid.x, grid.y, c',
                  c=:plasma, clim=(0, 1),
                  xlabel="x", ylabel="y",
                  title="Concentration c",
                  aspect_ratio=:equal)

    plot(p_φ, p_c, layout=(1, 2), size=(1000, 500),
         plot_title="WBM 2D (t=$(round(sol.t[idx], digits=3)))")
end
gif(anim, "examples/353_wbm_2d.gif", fps=8)
println("  Saved: examples/353_wbm_2d.gif")

println("\n✅ Simulation completed!")
