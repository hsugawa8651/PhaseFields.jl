# WBM Model: Wheeler 1992 Ni-Cu Solidification
#
# Reference: Wheeler, Boettinger, McFadden, Phys. Rev. A 45, 7424 (1992)
#
# This example reproduces the qualitative behavior of binary alloy solidification
# using dimensionless parameters derived from Ni-Cu system properties.
#
# Key features:
# - Solidification driven by undercooling (T < T_liquidus)
# - Concentration redistribution at the interface
# - Phase-dependent diffusivity (slower in solid)
#
# Run: julia --project=. examples/303_wbm_wheeler1992.jl

using PhaseFields
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

println("=== WBM Wheeler 1992 Ni-Cu Solidification ===\n")

# -----------------------------------------------------------------------------
# Physical parameters from Wheeler 1992 Table I (Ni-Cu system)
# -----------------------------------------------------------------------------
# T_M_Ni = 1728 K, T_M_Cu = 1358 K
# L_Ni = 2350 J/cm³, L_Cu = 1725 J/cm³
# σ = 3.7e-5 J/cm² (Ni), 2.8e-5 J/cm² (Cu)
# D = 1e-5 cm²/s (liquid diffusivity)
# δ = 5e-7 cm (interface width)

# -----------------------------------------------------------------------------
# Dimensionless parameters (scaled for numerical simulation)
# -----------------------------------------------------------------------------
# Length scale: interface width δ
# Time scale: δ²/D (diffusion time)
# Energy scale: RT (thermal energy)

# Free energy parameters (parabolic approximation)
# f_s(c) = A_s * (c - c_eq_s)²  for solid
# f_l(c) = A_l * (c - c_eq_l)²  for liquid
#
# Ni-Cu phase diagram (simplified):
# - Solidus: c_s ≈ 0.4 (Cu mole fraction in solid at T)
# - Liquidus: c_l ≈ 0.6 (Cu mole fraction in liquid at T)
# - At undercooling, liquid with c > c_l will solidify

# Solid phase: equilibrium at lower Cu content
f_s = ParabolicFreeEnergy(A=500.0, c_eq=0.40)

# Liquid phase: equilibrium at higher Cu content
f_l = ParabolicFreeEnergy(A=500.0, c_eq=0.60)

# Add driving force for solidification (undercooling effect)
# This shifts the liquid free energy up, making solid more stable
ΔT_undercooling = 5.0  # Small undercooling for controlled solidification

# Effective free energy with undercooling: f_l_eff = f_l + ΔG_undercool
# We model this by shifting the curvature
struct UndercooledLiquidFreeEnergy
    base::ParabolicFreeEnergy
    ΔG::Float64  # Driving force from undercooling
end

function PhaseFields.free_energy(f::UndercooledLiquidFreeEnergy, c::Real)
    return free_energy(f.base, c) + f.ΔG
end

function PhaseFields.chemical_potential(f::UndercooledLiquidFreeEnergy, c::Real)
    return chemical_potential(f.base, c)
end

function PhaseFields.d2f_dc2(f::UndercooledLiquidFreeEnergy, c::Real)
    return d2f_dc2(f.base, c)
end

# Liquid with undercooling (ΔG > 0 makes liquid less stable)
f_l_undercooled = UndercooledLiquidFreeEnergy(f_l, ΔT_undercooling)

# WBM Model parameters (Wheeler 1992 scaling)
# κ = 6σδ, W = 12σ/δ (Eq. 35-36)
# For dimensionless: κ ~ 1, W ~ 1
model = WBMModel(
    M_φ = 0.1,        # Phase field mobility (reduced for slower interface)
    κ = 1.0,          # Gradient energy (controls interface width)
    W = 1.0,          # Barrier height (controls interface energy)
    D_s = 0.1,        # Solid diffusivity (10x slower than liquid)
    D_l = 1.0         # Liquid diffusivity (reference)
)

# Simulation parameters
Nx = 200        # Grid points
dx = 0.5        # Grid spacing (in units of interface width)
dt = 0.001      # Time step
Nt = 10000      # Number of time steps

# Interface properties
δ = wbm_interface_width(model)
σ = wbm_interface_energy(model)

println("Wheeler 1992 Ni-Cu Parameters (dimensionless):")
println("  Solid equilibrium: c_eq = $(f_s.c_eq)")
println("  Liquid equilibrium: c_eq = $(f_l.c_eq)")
println("  Undercooling ΔG = $(ΔT_undercooling)")
println("  D_s/D_l = $(model.D_s/model.D_l) (solid diffusion 100x slower)")
println()
println("Model properties:")
println("  Interface width: δ ≈ $(round(δ, digits=2))")
println("  Interface energy: σ ≈ $(round(σ, digits=4))")
println()
println("Simulation:")
println("  Grid: Nx=$Nx, dx=$dx, L=$(Nx*dx)")
println("  Time: dt=$dt, Nt=$Nt, T_final=$(Nt*dt)")
println()

# -----------------------------------------------------------------------------
# Initial condition
# -----------------------------------------------------------------------------
# Initial interface at x=40
# Solid on left (φ=1), liquid on right (φ=0)
# Initial composition: c = 0.50 (between solidus and liquidus)

x_interface_init = 40.0
φ = [0.5 * (1 - tanh((i*dx - x_interface_init) / (2*δ))) for i in 1:Nx]
c = fill(0.50, Nx)  # Uniform initial concentration

# -----------------------------------------------------------------------------
# Finite difference operators
# -----------------------------------------------------------------------------
function compute_laplacian!(∇²f, f, dx)
    Nx = length(f)
    for i in 2:Nx-1
        ∇²f[i] = (f[i+1] - 2f[i] + f[i-1]) / dx^2
    end
    # Neumann BC
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
snapshot_interval = Nt ÷ 5

# Storage for animation (more frequent)
animation_snapshots = [(t=0, φ=copy(φ), c=copy(c))]
animation_interval = Nt ÷ 100  # 100 frames for smooth animation

println("Running simulation...")
println("  (Solid grows into undercooled liquid, rejecting solute)")
println()

for step in 1:Nt
    # Compute Laplacians
    compute_laplacian!(∇²φ, φ, dx)
    compute_laplacian!(∇²c, c, dx)

    # Update fields
    for i in 1:Nx
        # Phase field evolution (using undercooled liquid free energy)
        dφdt = wbm_phase_rhs(model, φ[i], ∇²φ[i], c[i], f_s, f_l_undercooled)
        φ[i] = clamp(φ[i] + dt * dφdt, 0.0, 1.0)

        # Concentration evolution
        dcdt = wbm_concentration_rhs(model, φ[i], ∇²c[i])
        c[i] = clamp(c[i] + dt * dcdt, 0.001, 0.999)
    end

    # Store animation snapshots (frequent)
    if step % animation_interval == 0
        push!(animation_snapshots, (t=step, φ=copy(φ), c=copy(c)))
    end

    # Store snapshots for static plots (less frequent)
    if step % snapshot_interval == 0
        push!(snapshots, (t=step, φ=copy(φ), c=copy(c)))

        # Find interface position
        idx = findfirst(i -> φ[i] > 0.5 && φ[i+1] <= 0.5, 1:Nx-1)
        x_int = idx !== nothing ? idx * dx : NaN

        # Solid fraction
        solid_frac = sum(φ) / Nx * 100

        println("  t=$(step): interface x≈$(round(x_int, digits=1)), solid=$(round(solid_frac, digits=1))%")
    end
end

# Final snapshot
push!(snapshots, (t=Nt, φ=copy(φ), c=copy(c)))

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
println()
println("=== Results ===")

# Interface position
interface_idx = findfirst(i -> φ[i] > 0.5 && φ[i+1] <= 0.5, 1:Nx-1)
if interface_idx !== nothing
    x_interface_final = interface_idx * dx
    interface_velocity = (x_interface_final - x_interface_init) / (Nt * dt)
    println("  Initial interface: x = $(x_interface_init)")
    println("  Final interface: x ≈ $(round(x_interface_final, digits=1))")
    println("  Interface velocity: v ≈ $(round(interface_velocity, digits=2)) (dimensionless)")
end

# Solid fraction
solid_frac = sum(φ) / Nx
println("  Solid fraction: $(round(solid_frac * 100, digits=1))%")

# Concentration in phases
solid_mask = φ .> 0.9
liquid_mask = φ .< 0.1
if any(solid_mask)
    avg_c_solid = sum(c[solid_mask]) / sum(solid_mask)
    println("  Avg concentration in solid: $(round(avg_c_solid, digits=3)) (eq: $(f_s.c_eq))")
end
if any(liquid_mask)
    avg_c_liquid = sum(c[liquid_mask]) / sum(liquid_mask)
    println("  Avg concentration in liquid: $(round(avg_c_liquid, digits=3)) (eq: $(f_l.c_eq))")
end

# Mass conservation
total_mass_init = 0.50 * Nx
total_mass_final = sum(c)
println("  Mass conservation: $(round(total_mass_final/total_mass_init * 100, digits=2))%")

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
println("\nGenerating plots...")

# Plot 1: Phase field and concentration evolution
p1 = plot(layout=(2,1), size=(900, 600))

# Phase field
colors = cgrad(:viridis, length(snapshots), categorical=true)
for (i, snap) in enumerate(snapshots)
    plot!(p1[1], x_grid, snap.φ,
          label="t=$(snap.t)",
          color=colors[i],
          linewidth=1.5)
end
plot!(p1[1], title="Wheeler 1992: Phase Field Evolution",
      xlabel="Position x", ylabel="φ",
      ylims=(-0.1, 1.1))
hline!(p1[1], [0, 1], color=:gray, linestyle=:dash, label="")

# Concentration
for (i, snap) in enumerate(snapshots)
    plot!(p1[2], x_grid, snap.c,
          label="t=$(snap.t)",
          color=colors[i],
          linewidth=1.5)
end
hline!(p1[2], [f_s.c_eq, f_l.c_eq],
       color=[:blue, :red], linestyle=:dash,
       label=["c_solidus" "c_liquidus"])
plot!(p1[2], title="Wheeler 1992: Concentration Evolution",
      xlabel="Position x", ylabel="c (Cu mole fraction)",
      ylims=(0.3, 0.7))

savefig(p1, "examples/303_wbm_wheeler1992_evolution.png")
println("  Saved: examples/303_wbm_wheeler1992_evolution.png")

# Plot 2: Final state detail
p2 = plot(size=(800, 400))

# Dual y-axis effect using twinx
plot!(p2, x_grid, φ,
      label="φ (phase)",
      color=:black, linewidth=2,
      ylabel="φ (phase field)")
plot!(p2, x_grid, c,
      label="c (concentration)",
      color=:green, linewidth=2,
      linestyle=:solid)

# Mark equilibrium concentrations
hline!(p2, [f_s.c_eq], color=:blue, linestyle=:dot, label="c_solidus=$(f_s.c_eq)")
hline!(p2, [f_l.c_eq], color=:red, linestyle=:dot, label="c_liquidus=$(f_l.c_eq)")

# Mark interface region
if interface_idx !== nothing
    vline!(p2, [x_interface_final], color=:orange, linestyle=:dash, label="interface")
end

plot!(p2, title="Wheeler 1992: Final State (t=$(Nt))",
      xlabel="Position x",
      legend=:right)

savefig(p2, "examples/303_wbm_wheeler1992_final.png")
println("  Saved: examples/303_wbm_wheeler1992_final.png")

# Plot 3: Free energy landscape
p3 = plot(layout=(1,2), size=(1000, 400), left_margin=15Plots.mm)

c_range = 0.2:0.01:0.8
f_s_vals = [free_energy(f_s, c) for c in c_range]
f_l_vals = [free_energy(f_l, c) for c in c_range]
f_l_under_vals = [free_energy(f_l_undercooled, c) for c in c_range]

plot!(p3[1], c_range, f_s_vals, label="f_solid", color=:blue, linewidth=2)
plot!(p3[1], c_range, f_l_vals, label="f_liquid (T=T_liq)", color=:red, linewidth=2, linestyle=:dash)
plot!(p3[1], c_range, f_l_under_vals, label="f_liquid (undercooled)", color=:red, linewidth=2)
vline!(p3[1], [0.50], color=:green, linestyle=:dot, label="c_init=0.50")
plot!(p3[1], title="Free Energy Curves",
      xlabel="Concentration c", ylabel="Free energy f(c)")

# Driving force vs concentration
Δf_vals = [free_energy(f_s, c) - free_energy(f_l_undercooled, c) for c in c_range]
plot!(p3[2], c_range, Δf_vals, label="f_s - f_l", color=:purple, linewidth=2)
hline!(p3[2], [0], color=:gray, linestyle=:dash, label="")
vline!(p3[2], [0.50], color=:green, linestyle=:dot, label="c_init")
plot!(p3[2], title="Driving Force (negative = solid stable)",
      xlabel="Concentration c", ylabel="Δf = f_s - f_l")

savefig(p3, "examples/303_wbm_wheeler1992_freeenergy.png")
println("  Saved: examples/303_wbm_wheeler1992_freeenergy.png")

# -----------------------------------------------------------------------------
# Animation
# -----------------------------------------------------------------------------
println("\nGenerating animation...")

anim = @animate for (i, snap) in enumerate(animation_snapshots)
    # Create two-panel plot
    p = plot(layout=(2,1), size=(800, 500))

    # Phase field
    plot!(p[1], x_grid, snap.φ,
          color=:black, linewidth=2,
          fill=(0, 0.3, :blue),
          label="",
          ylabel="φ (phase field)",
          ylims=(-0.1, 1.1),
          title="Wheeler 1992 Ni-Cu Solidification (t=$(snap.t))")
    hline!(p[1], [0, 1], color=:gray, linestyle=:dash, label="")

    # Add solid/liquid labels
    annotate!(p[1], [(15, 0.8, text("Solid", 10, :blue)),
                     (85, 0.2, text("Liquid", 10, :red))])

    # Concentration
    plot!(p[2], x_grid, snap.c,
          color=:green, linewidth=2,
          fill=(f_l.c_eq, 0.2, :green),
          label="",
          xlabel="Position x",
          ylabel="c (Cu mole fraction)",
          ylims=(0.35, 0.65))
    hline!(p[2], [f_s.c_eq], color=:blue, linestyle=:dash, label="c_solidus")
    hline!(p[2], [f_l.c_eq], color=:red, linestyle=:dash, label="c_liquidus")
    hline!(p[2], [0.50], color=:gray, linestyle=:dot, label="c_init")
end

gif(anim, "examples/303_wbm_wheeler1992_animation.gif", fps=15)
println("  Saved: examples/303_wbm_wheeler1992_animation.gif")

println()
println("=== Wheeler 1992 Key Physics ===")
println("1. Undercooling drives solidification (f_liquid shifted up)")
println("2. Interface moves into liquid (solid grows)")
println("3. Solute (Cu) is rejected from solid (c_s < c_l)")
println("4. Diffusion in solid is slow → concentration buildup at interface")
println()
println("✓ Simulation completed!")
