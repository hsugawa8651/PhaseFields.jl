# KKS 1D Solidification Example
#
# Demonstrates the Kim-Kim-Suzuki (KKS) model for binary alloy solidification.
# Uses parabolic free energies for solid and liquid phases.
#
# The KKS model couples phase field evolution with concentration partitioning
# at the interface, maintaining local thermodynamic equilibrium.
#
# Run: julia --project=. examples/301_kks_solidification.jl

using PhaseFields
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

println("=== KKS 1D Solidification Simulation ===\n")

# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------
# Solid phase: equilibrium at c_s = 0.2 (solute-poor)
# Liquid phase: equilibrium at c_l = 0.8 (solute-rich)
f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.2)
f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.8)

model = KKSModel(
    τ = 1.0,      # Relaxation time
    W = 2.0,      # Interface width parameter
    m = 1.0,      # Driving force coupling
    M_s = 1.0,    # Solid mobility
    M_l = 5.0     # Liquid mobility (higher diffusivity in liquid)
)

# Simulation parameters
Nx = 100        # Grid points
dx = 1.0        # Grid spacing
dt = 0.01       # Time step (smaller for stability)
Nt = 2000       # Number of time steps

println("Model parameters:")
println("  Solid equilibrium: c_eq = $(f_s.c_eq)")
println("  Liquid equilibrium: c_eq = $(f_l.c_eq)")
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
c = fill(0.5, Nx)  # Uniform average concentration

# -----------------------------------------------------------------------------
# Finite difference Laplacian (Neumann BC)
# -----------------------------------------------------------------------------
function compute_laplacian!(∇²f, f, dx)
    Nx = length(f)
    for i in 2:Nx-1
        ∇²f[i] = (f[i+1] - 2f[i] + f[i-1]) / dx^2
    end
    ∇²f[1] = (f[2] - f[1]) / dx^2
    ∇²f[Nx] = (f[Nx-1] - f[Nx]) / dx^2
    return ∇²f
end

# -----------------------------------------------------------------------------
# Time integration
# -----------------------------------------------------------------------------
∇²φ = similar(φ)
∇²μ = similar(c)
μ_field = similar(c)  # Chemical potential field
c_s_field = similar(c)
c_l_field = similar(c)

# Storage for snapshots
x_grid = range(0, Nx*dx, length=Nx)
snapshots = [(t=0, φ=copy(φ), c=copy(c))]

# Animation snapshots (more frequent)
animation_snapshots = [(t=0, φ=copy(φ), c=copy(c))]
animation_interval = max(1, Nt ÷ 80)  # ~80 frames

println("Running simulation...")

for step in 1:Nt
    # Step 1: Solve KKS partition at each point
    for i in 1:Nx
        c_s, c_l, μ, converged = kks_partition(c[i], φ[i], f_s, f_l)
        c_s_field[i] = c_s
        c_l_field[i] = c_l
        μ_field[i] = μ

        if !converged && step == 1 && i <= 5
            println("Warning: partition did not converge at i=$i, step=$step")
        end
    end

    # Step 2: Compute Laplacians
    compute_laplacian!(∇²φ, φ, dx)
    compute_laplacian!(∇²μ, μ_field, dx)

    # Step 3: Update fields
    for i in 1:Nx
        # Grand potential difference (driving force)
        Δω = kks_grand_potential_diff(f_s, f_l, c_s_field[i], c_l_field[i], μ_field[i])

        # Phase field evolution
        dφdt = kks_phase_rhs(model, φ[i], ∇²φ[i], Δω)
        φ[i] = clamp(φ[i] + dt * dφdt, 0.0, 1.0)

        # Concentration evolution
        dcdt = kks_concentration_rhs(model, φ[i], ∇²μ[i])
        c[i] = clamp(c[i] + dt * dcdt, 0.0, 1.0)
    end

    # Animation snapshots
    if step % animation_interval == 0
        push!(animation_snapshots, (t=step, φ=copy(φ), c=copy(c)))
    end

    # Store snapshots
    if step % 500 == 0
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

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
println("\nGenerating plots...")

# Plot 1: Phase field evolution
p1 = plot(
    title="KKS Solidification: Phase Field",
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
    title="KKS Solidification: Concentration",
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
savefig(p_combined, "examples/301_kks_solidification.png")
println("  Saved: examples/301_kks_solidification.png")

# Plot 3: Final state with phase concentrations
p3 = plot(
    title="KKS Final State: Phase Concentrations",
    xlabel="Position x",
    ylabel="Concentration / Phase",
    ylims=(0.0, 1.0),
    legend=:right,
    size=(800, 400)
)

# Final partition
for i in 1:Nx
    c_s, c_l, μ, _ = kks_partition(c[i], φ[i], f_s, f_l)
    c_s_field[i] = c_s
    c_l_field[i] = c_l
end

plot!(x_grid, φ, label="φ (phase)", color=:black, linewidth=2, linestyle=:solid)
plot!(x_grid, c, label="c (average)", color=:green, linewidth=2)
plot!(x_grid, c_s_field, label="c_s (solid)", color=:blue, linewidth=1.5, linestyle=:dash)
plot!(x_grid, c_l_field, label="c_l (liquid)", color=:red, linewidth=1.5, linestyle=:dash)

savefig(p3, "examples/301_kks_solidification_partitioning.png")
println("  Saved: examples/301_kks_solidification_partitioning.png")

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
          title="KKS Solidification (t=$(snap.t))")
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
gif(anim, "examples/301_kks_solidification.gif", fps=15)
println("  Saved: examples/301_kks_solidification.gif")

println("\n✓ Simulation completed!")
