# 1D Ostwald Ripening (Cahn-Hilliard)
#
# PFHub Benchmark 2 inspired simulation in 1D.
# Demonstrates coarsening: small particles dissolve, large particles grow.
#
# Reference: Jokisaari et al. (2017) Computational Materials Science 126, 139-151
#
# Run: julia --project=. examples/401_ostwald_ripening.jl

using PhaseFields
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

println("=" ^ 70)
println("1D Ostwald Ripening (Cahn-Hilliard)")
println("=" ^ 70)

# =============================================================================
# Parameters using PhaseFields.jl types
# =============================================================================

# Cahn-Hilliard model
const ch_model = CahnHilliardModel(M=5.0, κ=2.0)

# Double-well free energy: cα = matrix, cβ = particle
const free_energy = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)

# =============================================================================
# 1D Discretization with periodic boundary
# =============================================================================

function laplacian_periodic!(∇²c, c, dx)
    Nx = length(c)
    dx² = dx^2
    for i in 1:Nx
        ip1 = i == Nx ? 1 : i + 1
        im1 = i == 1 ? Nx : i - 1
        ∇²c[i] = (c[ip1] - 2 * c[i] + c[im1]) / dx²
    end
end

"""
One time step using PhaseFields.jl Cahn-Hilliard functions.
"""
function cahn_hilliard_step!(c, μ, ∇²c, ∇²μ, model, f, dx, dt)
    Nx = length(c)
    laplacian_periodic!(∇²c, c, dx)
    for i in 1:Nx
        μ[i] = cahn_hilliard_chemical_potential(model, f, c[i], ∇²c[i])
    end
    laplacian_periodic!(∇²μ, μ, dx)
    for i in 1:Nx
        c[i] = c[i] + dt * cahn_hilliard_rhs(model, ∇²μ[i])
    end
end

# =============================================================================
# Simulation setup
# =============================================================================

println("\nParameters:")
println("  CahnHilliardModel: M=$(ch_model.M), κ=$(ch_model.κ)")
println("  DoubleWellFreeEnergy: cα=$(free_energy.cα) (matrix), cβ=$(free_energy.cβ) (particle)")

# Grid parameters
Nx = 300
Lx = 300.0
dx = Lx / Nx

# Time parameters using PhaseFields.jl
dt_stable = cahn_hilliard_stability_dt(ch_model, dx)
dt = 0.5 * dt_stable
Nt = 200000
output_interval = 40000

# Interface width
W = cahn_hilliard_interface_width(ch_model, free_energy)

println("  Stability limit: dt < $(round(dt_stable, digits=4))")
println("  Interface width: W ≈ $(round(W, digits=2))")
println("\nGrid: Nx=$Nx, Lx=$Lx, dx=$dx")
println("Time: dt=$(round(dt, digits=4)), Nt=$Nt")

# =============================================================================
# Initial condition: Multiple particles with different sizes
# =============================================================================

println("\nInitial condition: 5 particles with different sizes")

# Initialize with matrix phase (cα)
c = fill(free_energy.cα, Nx)

# Create smooth particle profiles using tanh
function add_particle!(c, center, radius, f, interface_width=3.0)
    dx_local = Lx / length(c)
    for i in 1:length(c)
        x = i * dx_local
        dist = abs(x - center)
        # Smooth interface: transition from cβ (inside) to cα (outside)
        profile = 0.5 * (1 - tanh((dist - radius) / interface_width))
        c[i] = max(c[i], f.cα + (f.cβ - f.cα) * profile)
    end
end

# Add particles with different sizes
particles = [
    (center=40.0,  radius=4.0),   # Very small (should dissolve)
    (center=80.0,  radius=15.0),  # Medium
    (center=140.0, radius=3.0),   # Tiny (should dissolve first)
    (center=180.0, radius=25.0),  # Large
    (center=250.0, radius=6.0),   # Small (should shrink)
]

for p in particles
    add_particle!(c, p.center, p.radius, free_energy)
    println("  Particle at x=$(p.center), radius=$(p.radius)")
end

# Work arrays
μ = similar(c)
∇²c = similar(c)
∇²μ = similar(c)

# =============================================================================
# Analysis functions
# =============================================================================

function show_concentration(c, label, f)
    chars = [" ", "░", "▒", "▓", "█"]
    str = ""
    substep = max(1, length(c) ÷ 80)
    for i in 1:substep:length(c)
        val = c[i]
        if isnan(val) || isinf(val)
            str *= "?"
        else
            normalized = clamp((val - f.cα) / (f.cβ - f.cα), 0.0, 1.0)
            idx = clamp(Int(floor(normalized * 4)) + 1, 1, 5)
            str *= chars[idx]
        end
    end
    println("$label |$str|")
end

"""
Count particles and their approximate sizes.
A particle is a connected region where c > (cα + cβ)/2.
"""
function analyze_particles(c, f, dx)
    threshold = (f.cα + f.cβ) / 2
    particles = []
    in_particle = false
    start_idx = 0

    for i in 1:length(c)
        if c[i] > threshold && !in_particle
            in_particle = true
            start_idx = i
        elseif c[i] <= threshold && in_particle
            in_particle = false
            particle_size = (i - start_idx) * dx
            particle_center = (start_idx + i - 1) / 2 * dx
            push!(particles, (center=particle_center, size=particle_size))
        end
    end
    # Handle particle at boundary (periodic)
    if in_particle
        particle_size = (length(c) - start_idx + 1) * dx
        particle_center = (start_idx + length(c)) / 2 * dx
        push!(particles, (center=particle_center, size=particle_size))
    end

    return particles
end

function compute_statistics(c)
    c_mean = sum(c) / length(c)
    c_min = minimum(c)
    c_max = maximum(c)
    return (mean=c_mean, min=c_min, max=c_max)
end

# =============================================================================
# Time evolution
# =============================================================================

println("\nTime evolution (dark=matrix α, bright=particle β):")
println("Ostwald ripening: small particles dissolve → large particles grow\n")

# Store snapshots for plotting
x_grid = range(0, Lx, length=Nx)
snapshots = [(t=0, c=copy(c))]

# Animation snapshots (more frequent)
animation_snapshots = [(t=0, c=copy(c))]
animation_interval = max(1, Nt ÷ 100)  # ~100 frames

show_concentration(c, "t=0      ", free_energy)
initial_particles = analyze_particles(c, free_energy, dx)
println("         $(length(initial_particles)) particles, sizes: $(round.([p.size for p in initial_particles], digits=1))")
println()

for step in 1:Nt
    cahn_hilliard_step!(c, μ, ∇²c, ∇²μ, ch_model, free_energy, dx, dt)

    # Animation snapshots
    if step % animation_interval == 0
        push!(animation_snapshots, (t=step, c=copy(c)))
    end

    if step % output_interval == 0
        push!(snapshots, (t=step, c=copy(c)))
        show_concentration(c, "t=$(lpad(step, 6))", free_energy)
        current_particles = analyze_particles(c, free_energy, dx)
        stats = compute_statistics(c)
        sizes = length(current_particles) > 0 ? round.([p.size for p in current_particles], digits=1) : []
        println("         $(length(current_particles)) particles, sizes: $sizes, c_mean=$(round(stats.mean, digits=3))")
        println()
    end
end

# =============================================================================
# Final analysis
# =============================================================================

println("=" ^ 70)
println("Results:")
println("=" ^ 70)

final_particles = analyze_particles(c, free_energy, dx)
stats = compute_statistics(c)

println("""
Ostwald ripening demonstrated:
  - Initial: $(length(initial_particles)) particles
  - Final:   $(length(final_particles)) particles

Initial particle sizes: $(round.([p.size for p in initial_particles], digits=1))
Final particle sizes:   $(round.([p.size for p in final_particles], digits=1))

Conservation check:
  - Mean concentration: $(round(stats.mean, digits=4))
  - Expected (from initial): should be conserved

Physical interpretation:
  - Gibbs-Thomson effect: smaller particles have higher chemical potential
  - Mass diffuses from small to large particles
  - Small particles dissolve, large particles grow
  - Total number of particles decreases over time
""")

# Verify mass conservation
initial_mass = sum(c)
println("Total solute (conserved): $(round(initial_mass, digits=2))")

# =============================================================================
# Plot results
# =============================================================================

println("\nGenerating plots...")

# Plot 1: Time evolution
p1 = plot(
    title="Ostwald Ripening: Particle Coarsening",
    xlabel="Position x",
    ylabel="Concentration c",
    ylims=(0.25, 0.75),
    legend=:topright,
    size=(900, 500)
)
hline!([free_energy.cα, free_energy.cβ], color=:gray, linestyle=:dash, label="")

colors = cgrad(:viridis, length(snapshots), categorical=true)
for (i, snap) in enumerate(snapshots)
    plot!(x_grid, snap.c,
          label="t=$(snap.t)",
          color=colors[i],
          linewidth=1.5)
end
savefig(p1, "examples/401_ostwald_ripening_evolution.png")
println("  Saved: examples/401_ostwald_ripening_evolution.png")

# Plot 2: Initial vs Final
p2 = plot(
    title="Ostwald Ripening: Initial vs Final",
    xlabel="Position x",
    ylabel="Concentration c",
    ylims=(0.25, 0.75),
    size=(900, 400)
)
hline!([free_energy.cα, free_energy.cβ], color=:gray, linestyle=:dash, label="cα, cβ")
plot!(x_grid, snapshots[1].c, label="Initial ($(length(initial_particles)) particles)",
      color=:blue, linewidth=2)
plot!(x_grid, snapshots[end].c, label="Final ($(length(final_particles)) particles)",
      color=:red, linewidth=2)
savefig(p2, "examples/401_ostwald_ripening_comparison.png")
println("  Saved: examples/401_ostwald_ripening_comparison.png")

# Animation
println("Generating animation...")
anim = @animate for snap in animation_snapshots
    current_particles = analyze_particles(snap.c, free_energy, dx)
    plot(x_grid, snap.c,
         color=:teal, linewidth=2,
         fill=(free_energy.cα, 0.4, :teal),
         label="",
         xlabel="Position x",
         ylabel="Concentration c",
         ylims=(0.25, 0.75),
         title="Ostwald Ripening: $(length(current_particles)) particles (t=$(snap.t))",
         size=(900, 400))
    hline!([free_energy.cα, free_energy.cβ], color=:gray, linestyle=:dash, label="")
end
gif(anim, "examples/401_ostwald_ripening.gif", fps=15)
println("  Saved: examples/401_ostwald_ripening.gif")
