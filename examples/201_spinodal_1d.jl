# 1D Spinodal Decomposition (Cahn-Hilliard)
#
# PFHub Benchmark 1 inspired simulation in 1D.
# Demonstrates phase separation via spinodal decomposition.
#
# Reference: Jokisaari et al. (2017) Computational Materials Science 126, 139-151
#
# Run: julia --project=. examples/201_spinodal_1d.jl

using PhaseFields
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

println("=" ^ 70)
println("1D Spinodal Decomposition (Cahn-Hilliard)")
println("=" ^ 70)

# =============================================================================
# PFHub BM1 Parameters using PhaseFields.jl types
# =============================================================================

# Cahn-Hilliard model: ∂c/∂t = M * ∇²μ,  μ = df/dc - κ∇²c
const ch_model = CahnHilliardModel(M=5.0, κ=2.0)

# Double-well free energy: f(c) = ρs * (c - cα)² * (cβ - c)²
const free_energy = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)

# =============================================================================
# 1D Discretization with periodic boundary
# =============================================================================

"""
Compute Laplacian with periodic boundary conditions.
"""
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
One time step of Cahn-Hilliard equation using explicit Euler.

Uses PhaseFields.jl functions:
- cahn_hilliard_chemical_potential(model, f, c, ∇²c)
- cahn_hilliard_rhs(model, ∇²μ)
"""
function cahn_hilliard_step!(c, μ, ∇²c, ∇²μ, model, f, dx, dt)
    Nx = length(c)

    # Step 1: Compute Laplacian of c
    laplacian_periodic!(∇²c, c, dx)

    # Step 2: Compute chemical potential using PhaseFields.jl
    for i in 1:Nx
        μ[i] = cahn_hilliard_chemical_potential(model, f, c[i], ∇²c[i])
    end

    # Step 3: Compute Laplacian of μ
    laplacian_periodic!(∇²μ, μ, dx)

    # Step 4: Update concentration using PhaseFields.jl
    for i in 1:Nx
        c[i] = c[i] + dt * cahn_hilliard_rhs(model, ∇²μ[i])
    end
end

# =============================================================================
# Simulation setup
# =============================================================================

println("\nParameters (PFHub BM1 style):")
println("  CahnHilliardModel: M=$(ch_model.M), κ=$(ch_model.κ)")
println("  DoubleWellFreeEnergy: ρs=$(free_energy.ρs), cα=$(free_energy.cα), cβ=$(free_energy.cβ)")

# Grid parameters
Nx = 200
Lx = 200.0
dx = Lx / Nx

# Time parameters using PhaseFields.jl stability function
dt_stable = cahn_hilliard_stability_dt(ch_model, dx)
dt = 0.5 * dt_stable  # Safety factor
Nt = 50000
output_interval = 10000

println("  Stability limit: dt < $(round(dt_stable, digits=4))")
println("  Using dt = $(round(dt, digits=4))")

# Interface width estimate
W = cahn_hilliard_interface_width(ch_model, free_energy)
println("  Interface width estimate: W ≈ $(round(W, digits=2))")

println("\nGrid: Nx=$Nx, Lx=$Lx, dx=$dx")
println("Time: dt=$dt, Nt=$Nt")

# Initial condition: c₀ = 0.5 + small random perturbation
c0 = 0.5
noise_amplitude = 0.05
c = c0 .+ noise_amplitude .* (rand(Nx) .- 0.5)

# Work arrays
μ = similar(c)
∇²c = similar(c)
∇²μ = similar(c)

# =============================================================================
# ASCII visualization
# =============================================================================

function show_concentration(c, label, f)
    # Map concentration to characters
    chars = [" ", "░", "▒", "▓", "█"]
    str = ""
    # Subsample for display
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

function compute_statistics(c, f)
    c_mean = sum(c) / length(c)
    c_min = minimum(c)
    c_max = maximum(c)
    # Count phases
    n_alpha = count(x -> x < f.cα + 0.1, c)
    n_beta = count(x -> x > f.cβ - 0.1, c)
    n_interface = length(c) - n_alpha - n_beta
    return c_mean, c_min, c_max, n_alpha, n_beta
end

# =============================================================================
# Time evolution
# =============================================================================

println("\nTime evolution (dark=α-phase c≈$(free_energy.cα), bright=β-phase c≈$(free_energy.cβ)):")
println("Initial condition: c₀ = $c0 ± $noise_amplitude (supersaturated)")
println()

# Store snapshots for plotting
x_grid = range(0, Lx, length=Nx)
snapshots = [(t=0, c=copy(c))]

# Animation snapshots (more frequent)
animation_snapshots = [(t=0, c=copy(c))]
animation_interval = max(1, Nt ÷ 100)  # ~100 frames

show_concentration(c, "t=0     ", free_energy)
c_mean, c_min, c_max, _, _ = compute_statistics(c, free_energy)
println("         c: mean=$(round(c_mean, digits=3)), min=$(round(c_min, digits=3)), max=$(round(c_max, digits=3))")
println()

for step in 1:Nt
    cahn_hilliard_step!(c, μ, ∇²c, ∇²μ, ch_model, free_energy, dx, dt)

    # Animation snapshots
    if step % animation_interval == 0
        push!(animation_snapshots, (t=step, c=copy(c)))
    end

    if step % output_interval == 0
        push!(snapshots, (t=step, c=copy(c)))
        show_concentration(c, "t=$(lpad(step, 5))", free_energy)
        local stats = compute_statistics(c, free_energy)
        println("         c: mean=$(round(stats[1], digits=3)), min=$(round(stats[2], digits=3)), max=$(round(stats[3], digits=3))")
        println()
    end
end

# =============================================================================
# Plot results
# =============================================================================

println("Generating plots...")

# Plot 1: Time evolution (multiple curves)
p1 = plot(
    title="1D Spinodal Decomposition",
    xlabel="Position x",
    ylabel="Concentration c",
    ylims=(0.2, 0.8),
    legend=:topright,
    size=(800, 500)
)
hline!([free_energy.cα, free_energy.cβ], color=:gray, linestyle=:dash, label="")

colors = cgrad(:viridis, length(snapshots), categorical=true)
for (i, snap) in enumerate(snapshots)
    plot!(x_grid, snap.c,
          label="t=$(snap.t)",
          color=colors[i],
          linewidth=1.5)
end
savefig(p1, "examples/201_spinodal_1d_evolution.png")
println("  Saved: examples/201_spinodal_1d_evolution.png")

# Plot 2: Initial vs Final comparison
p2 = plot(
    title="Spinodal Decomposition: Initial vs Final",
    xlabel="Position x",
    ylabel="Concentration c",
    ylims=(0.2, 0.8),
    size=(800, 400)
)
hline!([free_energy.cα, free_energy.cβ], color=:gray, linestyle=:dash, label="cα, cβ")
plot!(x_grid, snapshots[1].c, label="Initial (t=0)", color=:blue, linewidth=2)
plot!(x_grid, snapshots[end].c, label="Final (t=$(snapshots[end].t))", color=:red, linewidth=2)
savefig(p2, "examples/201_spinodal_1d_comparison.png")
println("  Saved: examples/201_spinodal_1d_comparison.png")

# Plot 3: Heatmap of time evolution
c_matrix = hcat([snap.c for snap in snapshots]...)
t_values = [snap.t for snap in snapshots]
p3 = heatmap(
    t_values, collect(x_grid), c_matrix,
    title="Spinodal Decomposition: Concentration Field",
    xlabel="Time step",
    ylabel="Position x",
    colorbar_title="c",
    color=:RdBu,
    clims=(free_energy.cα, free_energy.cβ),
    size=(800, 500)
)
savefig(p3, "examples/201_spinodal_1d_heatmap.png")
println("  Saved: examples/201_spinodal_1d_heatmap.png")

# Animation
println("Generating animation...")
anim = @animate for snap in animation_snapshots
    plot(x_grid, snap.c,
         color=:purple, linewidth=2,
         fill=(free_energy.cα, 0.3, :purple),
         label="",
         xlabel="Position x",
         ylabel="Concentration c",
         ylims=(0.2, 0.8),
         title="Spinodal Decomposition (t=$(snap.t))",
         size=(800, 400))
    hline!([free_energy.cα, free_energy.cβ], color=:gray, linestyle=:dash, label="")
    annotate!([(20, free_energy.cα - 0.03, text("cα", 8)),
               (20, free_energy.cβ + 0.03, text("cβ", 8))])
end
gif(anim, "examples/201_spinodal_1d.gif", fps=15)
println("  Saved: examples/201_spinodal_1d.gif")

# =============================================================================
# Final analysis
# =============================================================================

println("=" ^ 70)
println("Results:")
println("=" ^ 70)

c_mean, c_min, c_max, n_alpha, n_beta = compute_statistics(c, free_energy)

println("""
Spinodal decomposition observed:
  - Initial: homogeneous c = $c0 (supersaturated)
  - Final: phase separated into α (c≈$(free_energy.cα)) and β (c≈$(free_energy.cβ)) phases

Statistics:
  - Mean concentration: $(round(c_mean, digits=4)) (conserved ≈ $c0)
  - Min concentration: $(round(c_min, digits=4)) (approaching cα=$(free_energy.cα))
  - Max concentration: $(round(c_max, digits=4)) (approaching cβ=$(free_energy.cβ))

Phase fractions (lever rule predicts α:β = 1:1 for c₀=0.5):
  - α-phase (c < $(free_energy.cα + 0.1)): $(round(100*n_alpha/Nx, digits=1))%
  - β-phase (c > $(free_energy.cβ - 0.1)): $(round(100*n_beta/Nx, digits=1))%
  - Interface region:  $(round(100*(Nx-n_alpha-n_beta)/Nx, digits=1))%
""")
