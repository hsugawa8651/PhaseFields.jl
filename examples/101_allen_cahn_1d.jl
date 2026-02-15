# Allen-Cahn 1D Simulation Example
#
# Demonstrates basic usage of PhaseFields.jl for interface evolution
# without CALPHAD coupling (constant driving force).
#
# Run: julia --project=. examples/101_allen_cahn_1d.jl

using PhaseFields
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

println("=== Allen-Cahn 1D Simulation ===\n")

# -----------------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------------
model = AllenCahnModel(
    τ = 1.0,    # Relaxation time
    W = 2.0,    # Interface width parameter
    m = 0.5     # Driving force coupling
)

# Simulation parameters
Nx = 80         # Grid points
dx = 1.0        # Grid spacing
dt = 0.05       # Time step
Nt = 600        # Number of time steps
ΔG = -0.3       # Driving force (negative = solid grows)

println("Parameters:")
println("  Grid: Nx=$Nx, dx=$dx")
println("  Time: dt=$dt, Nt=$Nt")
println("  Model: τ=$(model.τ), W=$(model.W), m=$(model.m)")
println("  Driving force: ΔG=$ΔG")
println()

# -----------------------------------------------------------------------------
# Initial condition: tanh profile (interface at x=20)
# -----------------------------------------------------------------------------
φ = [0.5 * (1 + tanh((i - 20) / 3)) for i in 1:Nx]

# -----------------------------------------------------------------------------
# Laplacian (finite difference, Neumann BC)
# -----------------------------------------------------------------------------
function compute_laplacian!(∇²φ, φ, dx)
    Nx = length(φ)
    for i in 2:Nx-1
        ∇²φ[i] = (φ[i+1] - 2φ[i] + φ[i-1]) / dx^2
    end
    # Neumann BC: ∂φ/∂x = 0 at boundaries
    ∇²φ[1] = (φ[2] - φ[1]) / dx^2
    ∇²φ[Nx] = (φ[Nx-1] - φ[Nx]) / dx^2
    return ∇²φ
end

# -----------------------------------------------------------------------------
# ASCII visualization
# -----------------------------------------------------------------------------
function show_profile(φ, label)
    chars = [" ", "░", "▒", "▓", "█"]
    str = ""
    for v in φ
        idx = clamp(Int(floor(v * 4)) + 1, 1, 5)
        str *= chars[idx]
    end
    println("$label |$str|")
end

# -----------------------------------------------------------------------------
# Time integration (Forward Euler)
# -----------------------------------------------------------------------------
∇²φ = similar(φ)
x_grid = range(0, Nx*dx, length=Nx)
snapshots = [(t=0, φ=copy(φ))]

# Animation snapshots (more frequent)
animation_snapshots = [(t=0, φ=copy(φ))]
animation_interval = max(1, Nt ÷ 60)  # ~60 frames

println("Time evolution (φ=0: liquid, φ=1: solid):")
println("         " * "0" * " "^38 * "x" * " "^38 * "80")
show_profile(φ, "t=0   ")

for step in 1:Nt
    compute_laplacian!(∇²φ, φ, dx)

    for i in 1:Nx
        dφdt = allen_cahn_rhs(model, φ[i], ∇²φ[i], ΔG)
        φ[i] = clamp(φ[i] + dt * dφdt, 0.0, 1.0)
    end

    # Animation snapshots
    if step % animation_interval == 0
        push!(animation_snapshots, (t=step, φ=copy(φ)))
    end

    if step % 100 == 0
        push!(snapshots, (t=step, φ=copy(φ)))
        show_profile(φ, "t=$(lpad(step, 3))")
    end
end

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
println()

# Find interface position (φ = 0.5)
interface_idx = findfirst(i -> φ[i] < 0.5 && φ[i+1] >= 0.5, 1:Nx-1)
if interface_idx !== nothing
    # Linear interpolation for more accurate position
    x_interface = interface_idx + (0.5 - φ[interface_idx]) / (φ[interface_idx+1] - φ[interface_idx])
    println("Final interface position: x ≈ $(round(x_interface, digits=1))")
else
    # Check if completely solidified or liquid
    if all(φ .> 0.9)
        println("System completely solidified (φ ≈ 1)")
    elseif all(φ .< 0.1)
        println("System completely liquid (φ ≈ 0)")
    end
end

# Solid fraction
solid_frac = sum(φ) / Nx
println("Solid fraction: $(round(solid_frac * 100, digits=1))%")

# -----------------------------------------------------------------------------
# Plot results
# -----------------------------------------------------------------------------
println("\nGenerating plots...")

# Plot: Time evolution
p = plot(
    title="Allen-Cahn 1D: Interface Migration",
    xlabel="Position x",
    ylabel="Order parameter φ",
    ylims=(-0.1, 1.1),
    legend=:topleft,
    size=(800, 500)
)
hline!([0, 1], color=:gray, linestyle=:dash, label="")

colors = cgrad(:viridis, length(snapshots), categorical=true)
for (i, snap) in enumerate(snapshots)
    if i == 1 || i == length(snapshots) || snap.t % 200 == 0
        plot!(x_grid, snap.φ,
              label="t=$(snap.t)",
              color=colors[i],
              linewidth=2)
    end
end
savefig(p, "examples/101_allen_cahn_1d.png")
println("  Saved: examples/101_allen_cahn_1d.png")

# Animation
println("Generating animation...")
anim = @animate for snap in animation_snapshots
    plot(x_grid, snap.φ,
         color=:blue, linewidth=2,
         fill=(0, 0.3, :blue),
         label="",
         xlabel="Position x",
         ylabel="φ (order parameter)",
         ylims=(-0.1, 1.1),
         title="Allen-Cahn 1D: Interface Migration (t=$(snap.t))",
         size=(700, 400))
    hline!([0, 1], color=:gray, linestyle=:dash, label="")
    # Mark interface position
    idx = findfirst(i -> snap.φ[i] < 0.5 && snap.φ[i+1] >= 0.5, 1:Nx-1)
    if idx !== nothing
        vline!([idx * dx], color=:red, linestyle=:dot, label="interface")
    end
end
gif(anim, "examples/101_allen_cahn_1d.gif", fps=15)
println("  Saved: examples/101_allen_cahn_1d.gif")

println("\n✅ Simulation completed!")
