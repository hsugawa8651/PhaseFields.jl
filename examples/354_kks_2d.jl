# KKS 2D Solidification Example
#
# Demonstrates 2D binary alloy solidification using the KKS model
# with the unified solve API. Uses local equilibrium partitioning.
#
# Run: julia --project=. examples/354_kks_2d.jl

using PhaseFields
using OrdinaryDiffEq
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=5Plots.mm, right_margin=5Plots.mm,
    top_margin=5Plots.mm, bottom_margin=5Plots.mm
)

println("=== KKS 2D Binary Alloy Solidification ===\n")

# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------
# Free energy functions (parabolic approximation)
f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)  # Solid equilibrium at c=0.1
f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)  # Liquid equilibrium at c=0.9

# KKS model parameters
model = KKSModel(
    τ = 1.0,        # Relaxation time
    W = 0.05,       # Interface width parameter
    m = 1.0,        # Driving force scale
    M_s = 0.1,      # Solid mobility
    M_l = 1.0       # Liquid mobility
)

# Grid parameters
Nx, Ny = 50, 50
Lx, Ly = 1.0, 1.0
grid = UniformGrid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)

# Time parameters
tspan = (0.0, 0.05)

println("Parameters:")
println("  Grid: $(Nx)×$(Ny), domain: $(Lx)×$(Ly)")
println("  Model: τ=$(model.τ), W=$(model.W), m=$(model.m)")
println("  Mobility: M_s=$(model.M_s), M_l=$(model.M_l)")
println("  Free energy: solid c_eq=$(f_s.c_eq), liquid c_eq=$(f_l.c_eq)")
println("  Time: $(tspan[1]) to $(tspan[2])")
println()

# -----------------------------------------------------------------------------
# Initial conditions
# -----------------------------------------------------------------------------
# Phase field: solid seed at center with smooth interface
R0 = 0.15  # Seed radius
cx, cy = 0.5, 0.5
W_init = 0.03  # Interface width for initial profile

φ0 = [0.5 * (1 - tanh((sqrt((x - cx)^2 + (y - cy)^2) - R0) / W_init))
      for x in grid.x, y in grid.y]

# Concentration: slightly off-equilibrium to drive solidification
c0_value = 0.4
c0 = c0_value * ones(Nx, Ny)

println("Initial conditions:")
println("  Solid seed at ($(cx), $(cy)) with radius $(R0)")
println("  Smooth interface width: $(W_init)")
println("  Initial concentration: $(c0_value)")
solid_frac_init = sum(φ0 .> 0.5) / (Nx * Ny)
println("  Initial solid fraction (φ>0.5): $(round(solid_frac_init*100, digits=1))%")
println()

# -----------------------------------------------------------------------------
# Create problem and solve using unified API
# -----------------------------------------------------------------------------
problem = KKSProblem(model, grid, φ0, c0, tspan, f_s, f_l)

println("Solving with OrdinaryDiffEq.jl...")
println("  Note: KKS solves partition equation at each grid point")
sol = PhaseFields.solve(problem, Tsit5(), saveat=0.005)
println("  Time points saved: $(length(sol.t))")

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
N_total = Nx * Ny

println("\nAnalysis:")
for idx in [1, length(sol.t) ÷ 2, length(sol.t)]
    φ = reshape(sol.u[idx][1:N_total], Nx, Ny)
    c = reshape(sol.u[idx][N_total+1:end], Nx, Ny)

    # Solid fraction (φ > 0.5)
    solid_frac = sum(φ .> 0.5) / N_total

    # Mean concentration
    c_mean = sum(c) / N_total

    # Phase concentrations from partition (sample at center)
    i_center, j_center = Nx ÷ 2, Ny ÷ 2
    c_s, c_l, μ, converged = kks_partition(c[i_center, j_center], φ[i_center, j_center], f_s, f_l)

    println("  t=$(round(sol.t[idx], digits=3)):")
    println("    Solid fraction: $(round(solid_frac*100, digits=1))%")
    println("    Mean c: $(round(c_mean, digits=3))")
    println("    Center: φ=$(round(φ[i_center, j_center], digits=2)), c_s=$(round(c_s, digits=3)), c_l=$(round(c_l, digits=3))")
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

# Interface diffusion check
φ_interface_init = sum((φ_initial .> 0.1) .& (φ_initial .< 0.9))
φ_interface_final = sum((φ_final .> 0.1) .& (φ_final .< 0.9))
println("  Interface region (0.1<φ<0.9): $(φ_interface_init) → $(φ_interface_final) grid points")

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
             title="φ (t=$(round(sol.t[idx], digits=3)))",
             aspect_ratio=:equal)
end
savefig(p1, "examples/354_kks_2d_phase.png")
println("  Saved: examples/354_kks_2d_phase.png")

# Concentration evolution
p2 = plot(layout=(1, n_snapshots), size=(500*n_snapshots, 500))
for (i, idx) in enumerate(indices)
    c = reshape(sol.u[idx][N_total+1:end], Nx, Ny)
    heatmap!(p2[i], grid.x, grid.y, c',
             c=:plasma, clim=(0, 1),
             xlabel="x", ylabel="y",
             title="c (t=$(round(sol.t[idx], digits=3)))",
             aspect_ratio=:equal)
end
savefig(p2, "examples/354_kks_2d_conc.png")
println("  Saved: examples/354_kks_2d_conc.png")

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
         plot_title="KKS 2D (t=$(round(sol.t[idx], digits=3)))")
end
gif(anim, "examples/354_kks_2d.gif", fps=8)
println("  Saved: examples/354_kks_2d.gif")

println("\n✅ Simulation completed!")
