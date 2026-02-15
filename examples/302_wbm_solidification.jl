# WBM 1D Solidification Example
#
# Demonstrates the Wheeler-Boettinger-McFadden (WBM) model for binary alloy
# solidification. Uses parabolic free energies for solid and liquid phases.
#
# The WBM model uses a single concentration field with phase-dependent free
# energy, making it simpler than KKS but with interface-width dependent results.
#
# Run: julia --project=. examples/302_wbm_solidification.jl

using PhaseFields
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

println("=== WBM 1D Solidification Simulation ===\n")

# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------
# Solid phase: equilibrium at c_s = 0.2 (solute-poor)
# Liquid phase: equilibrium at c_l = 0.8 (solute-rich)
f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.2)
f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.8)

model = WBMModel(
    M_φ = 1.0,        # Phase field mobility
    κ = 4.0,          # Gradient energy coefficient
    W = 100.0,        # Barrier height
    D_s = 1e-4,       # Solid diffusivity (scaled for demo)
    D_l = 1e-2        # Liquid diffusivity (higher in liquid)
)

# Simulation parameters
Nx = 100        # Grid points
dx = 1.0        # Grid spacing
dt = 0.005      # Time step (smaller for stability due to diffusion)
Nt = 4000       # Number of time steps

# Interface properties
δ = wbm_interface_width(model)
σ = wbm_interface_energy(model)

println("Model parameters:")
println("  Solid equilibrium: c_eq = $(f_s.c_eq)")
println("  Liquid equilibrium: c_eq = $(f_l.c_eq)")
println("  Interface width: δ ≈ $(round(δ, digits=2))")
println("  Interface energy: σ ≈ $(round(σ, digits=4))")
println()
println("Simulation:")
println("  Grid: Nx=$Nx, dx=$dx")
println("  Time: dt=$dt, Nt=$Nt")
println()

# -----------------------------------------------------------------------------
# Initial condition
# -----------------------------------------------------------------------------
# Initial interface at x=30
# Solid on left (φ=1), liquid on right (φ=0)
# Initial composition: uniform at c=0.5

φ = [0.5 * (1 - tanh((i - 30) / 3)) for i in 1:Nx]
c = fill(0.5, Nx)  # Uniform concentration

# -----------------------------------------------------------------------------
# Finite difference Laplacian (Neumann BC)
# -----------------------------------------------------------------------------
function compute_laplacian!(∇²f, f, dx)
    Nx = length(f)
    for i in 2:Nx-1
        ∇²f[i] = (f[i+1] - 2f[i] + f[i-1]) / dx^2
    end
    # Neumann BC: ∂f/∂n = 0
    ∇²f[1] = (f[2] - f[1]) / dx^2
    ∇²f[Nx] = (f[Nx-1] - f[Nx]) / dx^2
    return ∇²f
end

# -----------------------------------------------------------------------------
# Time integration
# -----------------------------------------------------------------------------
∇²φ = similar(φ)
∇²c = similar(c)

# Storage for snapshots
x_grid = range(0, Nx*dx, length=Nx)
snapshots = [(t=0, φ=copy(φ), c=copy(c))]

# Animation snapshots (more frequent)
animation_snapshots = [(t=0, φ=copy(φ), c=copy(c))]
animation_interval = max(1, Nt ÷ 80)  # ~80 frames

println("Running simulation...")

for step in 1:Nt
    # Step 1: Compute Laplacians
    compute_laplacian!(∇²φ, φ, dx)
    compute_laplacian!(∇²c, c, dx)

    # Step 2: Update fields
    for i in 1:Nx
        # Phase field evolution
        dφdt = wbm_phase_rhs(model, φ[i], ∇²φ[i], c[i], f_s, f_l)
        φ[i] = clamp(φ[i] + dt * dφdt, 0.0, 1.0)

        # Concentration evolution (simple diffusion with phase-dependent D)
        dcdt = wbm_concentration_rhs(model, φ[i], ∇²c[i])
        c[i] = clamp(c[i] + dt * dcdt, 0.0, 1.0)
    end

    # Animation snapshots
    if step % animation_interval == 0
        push!(animation_snapshots, (t=step, φ=copy(φ), c=copy(c)))
    end

    # Store snapshots
    if step % 1000 == 0
        push!(snapshots, (t=step, φ=copy(φ), c=copy(c)))
        println("  t = $step")
    end
end

# Final snapshot
push!(snapshots, (t=Nt, φ=copy(φ), c=copy(c)))

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
println()
println("Results:")

# Find interface position (φ = 0.5)
interface_idx = findfirst(i -> φ[i] > 0.5 && φ[i+1] <= 0.5, 1:Nx-1)
if interface_idx !== nothing
    x_interface = interface_idx + (0.5 - φ[interface_idx]) / (φ[interface_idx+1] - φ[interface_idx])
    println("  Interface position: x ≈ $(round(x_interface, digits=1))")
end

# Solid fraction
solid_frac = sum(φ) / Nx
println("  Solid fraction: $(round(solid_frac * 100, digits=1))%")

# Average concentrations in each phase
solid_mask = φ .> 0.8
liquid_mask = φ .< 0.2
if any(solid_mask)
    avg_c_solid = sum(c[solid_mask]) / sum(solid_mask)
    println("  Avg concentration in solid: $(round(avg_c_solid, digits=3))")
end
if any(liquid_mask)
    avg_c_liquid = sum(c[liquid_mask]) / sum(liquid_mask)
    println("  Avg concentration in liquid: $(round(avg_c_liquid, digits=3))")
end

# Mass conservation check
total_mass = sum(c)
println("  Total mass (should be constant): $(round(total_mass, digits=2))")

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
println("\nGenerating plots...")

# Plot 1: Phase field evolution
p1 = plot(
    title="WBM Solidification: Phase Field",
    xlabel="Position x",
    ylabel="φ (order parameter)",
    ylims=(-0.1, 1.1),
    legend=:topright,
    size=(800, 400)
)
hline!([0, 1], color=:gray, linestyle=:dash, label="")

colors_φ = cgrad(:viridis, length(snapshots), categorical=true)
for (i, snap) in enumerate(snapshots)
    plot!(x_grid, snap.φ,
          label="t=$(snap.t)",
          color=colors_φ[i],
          linewidth=2)
end

# Plot 2: Concentration evolution
p2 = plot(
    title="WBM Solidification: Concentration",
    xlabel="Position x",
    ylabel="c (concentration)",
    ylims=(0.0, 1.0),
    legend=:topright,
    size=(800, 400)
)
hline!([f_s.c_eq, f_l.c_eq], color=[:blue, :red], linestyle=:dash,
       label=["c_eq (solid)" "c_eq (liquid)"])

for (i, snap) in enumerate(snapshots)
    plot!(x_grid, snap.c,
          label="t=$(snap.t)",
          color=colors_φ[i],
          linewidth=2)
end

# Combined plot
p_combined = plot(p1, p2, layout=(2, 1), size=(800, 700))
savefig(p_combined, "examples/302_wbm_solidification.png")
println("  Saved: examples/302_wbm_solidification.png")

# Plot 3: Final state with driving force
p3 = plot(
    title="WBM Final State: Phase and Driving Force",
    xlabel="Position x",
    ylabel="Value",
    legend=:right,
    size=(800, 400)
)

# Compute driving force at final state
driving_force = [wbm_driving_force(f_s, f_l, φ[i], c[i], model.W) for i in 1:Nx]
# Normalize for plotting
df_max = maximum(abs.(driving_force))
if df_max > 0
    driving_force_norm = driving_force ./ df_max
else
    driving_force_norm = driving_force
end

plot!(x_grid, φ, label="φ (phase)", color=:black, linewidth=2)
plot!(x_grid, c, label="c (concentration)", color=:green, linewidth=2)
plot!(x_grid, driving_force_norm, label="∂f/∂φ (normalized)", color=:red, linewidth=1.5, linestyle=:dash)

# Effective diffusivity profile
D_eff = [wbm_diffusivity(model, φ[i]) for i in 1:Nx]
D_norm = (D_eff .- minimum(D_eff)) ./ (maximum(D_eff) - minimum(D_eff))
plot!(x_grid, D_norm, label="D(φ) (normalized)", color=:orange, linewidth=1.5, linestyle=:dot)

savefig(p3, "examples/302_wbm_solidification_analysis.png")
println("  Saved: examples/302_wbm_solidification_analysis.png")

# Plot 4: Comparison of WBM vs KKS characteristics
p4 = plot(
    title="WBM Model Characteristics",
    xlabel="Position x",
    size=(800, 400),
    layout=(1, 2)
)

# Left: Free energy density profile
f_density = [wbm_bulk_free_energy(f_s, f_l, φ[i], c[i], model.W) for i in 1:Nx]
plot!(p4[1], x_grid, f_density,
      label="f(φ,c)",
      color=:purple,
      linewidth=2,
      ylabel="Free energy density",
      title="Bulk Free Energy")

# Right: Chemical potential profile
μ_field = [wbm_chemical_potential(f_s, f_l, φ[i], c[i]) for i in 1:Nx]
plot!(p4[2], x_grid, μ_field,
      label="μ(φ,c)",
      color=:teal,
      linewidth=2,
      ylabel="Chemical potential",
      title="Chemical Potential")

savefig(p4, "examples/302_wbm_solidification_characteristics.png")
println("  Saved: examples/302_wbm_solidification_characteristics.png")

# Animation
println("Generating animation...")
anim = @animate for snap in animation_snapshots
    p = plot(layout=(2,1), size=(800, 500))

    # Phase field
    plot!(p[1], x_grid, snap.φ,
          color=:black, linewidth=2,
          fill=(0, 0.3, :blue),
          label="",
          ylabel="φ",
          ylims=(-0.1, 1.1),
          title="WBM Solidification (t=$(snap.t))")
    hline!(p[1], [0, 1], color=:gray, linestyle=:dash, label="")

    # Concentration
    plot!(p[2], x_grid, snap.c,
          color=:green, linewidth=2,
          label="",
          xlabel="Position x",
          ylabel="c",
          ylims=(0.0, 1.0))
    hline!(p[2], [f_s.c_eq, f_l.c_eq], color=[:blue, :red], linestyle=:dash, label="")
end
gif(anim, "examples/302_wbm_solidification.gif", fps=15)
println("  Saved: examples/302_wbm_solidification.gif")

println("\n✓ Simulation completed!")
println()
println("WBM vs KKS comparison:")
println("  - WBM uses single concentration field (c = c_S = c_L at interface)")
println("  - KKS uses separate phase concentrations with local equilibrium")
println("  - WBM is simpler but results depend on interface width")
println("  - KKS is thermodynamically consistent but requires partition solver")
