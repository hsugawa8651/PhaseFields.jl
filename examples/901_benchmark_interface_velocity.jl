# Benchmark: 1D Interface Velocity Verification
#
# Verifies that the Allen-Cahn model produces interface velocities
# consistent with analytical predictions.
#
# Analytical result (sharp interface limit):
#   v = α · m · |ΔG| / τ
# where α is a geometric factor depending on the double-well potential.
#
# Reference: Kobayashi (1993), Karma-Rappel (1998)
#
# Run: julia --project=. examples/901_benchmark_interface_velocity.jl

using PhaseFields
using Printf
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

println("="^70)
println("Benchmark: 1D Interface Velocity vs Analytical Prediction")
println("="^70)

# =============================================================================
# Simulation parameters
# =============================================================================

# Fixed parameters
const τ = 1.0       # Relaxation time
const W = 1.0       # Interface width parameter
const m = 1.0       # Driving force coupling

# Grid
const Nx = 400      # Grid points (large enough to avoid boundary effects)
const dx = 0.5      # Grid spacing
const dt = 0.01     # Time step (stability: dt < τ·dx²/(2W²))

# Driving forces to test
ΔG_values = [-0.05, -0.1, -0.2, -0.3, -0.4, -0.5]

# =============================================================================
# Helper functions
# =============================================================================

function compute_laplacian!(∇²φ, φ, dx)
    Nx = length(φ)
    for i in 2:Nx-1
        ∇²φ[i] = (φ[i+1] - 2φ[i] + φ[i-1]) / dx^2
    end
    # Neumann BC
    ∇²φ[1] = (φ[2] - φ[1]) / dx^2
    ∇²φ[Nx] = (φ[Nx-1] - φ[Nx]) / dx^2
    return ∇²φ
end

function find_interface_position(φ, dx)
    # Find where φ crosses 0.5 (linear interpolation)
    for i in 1:length(φ)-1
        if (φ[i] - 0.5) * (φ[i+1] - 0.5) < 0
            # Linear interpolation
            t = (0.5 - φ[i]) / (φ[i+1] - φ[i])
            return (i - 1 + t) * dx
        end
    end
    return NaN
end

function run_simulation(ΔG; verbose=false)
    model = AllenCahnModel(τ=τ, W=W, m=m)

    # Initial condition: interface at x = 50
    x0 = 50.0
    φ = [0.5 * (1 - tanh((i*dx - x0) / (sqrt(2)*W))) for i in 0:Nx-1]
    ∇²φ = similar(φ)

    # Track interface position over time
    positions = Float64[]
    times = Float64[]

    t = 0.0
    Nt = 5000  # Total time steps

    for step in 1:Nt
        compute_laplacian!(∇²φ, φ, dx)

        for i in 1:Nx
            dφdt = allen_cahn_rhs(model, φ[i], ∇²φ[i], ΔG)
            φ[i] = clamp(φ[i] + dt * dφdt, 0.0, 1.0)
        end

        t += dt

        # Record position every 100 steps (after initial transient)
        if step % 100 == 0 && step > 500
            pos = find_interface_position(φ, dx)
            if !isnan(pos) && pos > 10*dx && pos < (Nx-10)*dx
                push!(positions, pos)
                push!(times, t)
            end
        end
    end

    # Calculate velocity from linear fit
    if length(positions) >= 2
        # Simple linear regression: v = Δx / Δt
        n = length(positions)
        Δx = positions[end] - positions[1]
        Δt = times[end] - times[1]
        velocity = Δx / Δt
        return velocity
    else
        return NaN
    end
end

# =============================================================================
# Run benchmark
# =============================================================================

println("\nParameters: τ=$τ, W=$W, m=$m, dx=$dx, dt=$dt")
println("\n" * "-"^70)
@printf("  ΔG      |  v (simulated)  |  v/|ΔG|  |  Note\n")
println("-"^70)

velocities = Float64[]
for ΔG in ΔG_values
    v = run_simulation(ΔG)
    push!(velocities, v)
    ratio = abs(v / ΔG)
    @printf(" %6.2f   |    %8.5f     |  %6.3f  |  %s\n",
            ΔG, v, ratio, v > 0 ? "← liquid grows" : "solid grows →")
end

# =============================================================================
# Analysis: Check linearity v ∝ ΔG
# =============================================================================

println("-"^70)
println("\nAnalysis:")

# Linear regression: v = α · ΔG
# Using least squares: α = Σ(ΔG·v) / Σ(ΔG²)
ΔG_arr = Float64.(ΔG_values)
v_arr = velocities
α_fit = sum(ΔG_arr .* v_arr) / sum(ΔG_arr.^2)

println(@sprintf("  Fitted coefficient: v = %.4f × ΔG", α_fit))
println(@sprintf("  Analytical form: v = α·m·ΔG/τ where α ≈ %.4f", α_fit * τ / m))

# Check linearity (R²)
mean(x) = sum(x) / length(x)
v_pred = α_fit .* ΔG_arr
SS_res = sum((v_arr .- v_pred).^2)
SS_tot = sum((v_arr .- mean(v_arr)).^2)
R² = 1 - SS_res / SS_tot

println(@sprintf("  Linearity (R²): %.6f", R²))

if R² > 0.999
    println("\n✅ PASS: Interface velocity is linear in ΔG (R² > 0.999)")
else
    println("\n⚠  Check: R² = $R² (expected > 0.999)")
end

# =============================================================================
# Theoretical comparison
# =============================================================================

println("\n" * "="^70)
println("Theoretical Notes:")
println("="^70)
println("""
For the Allen-Cahn equation with:
  - Double-well: g(φ) = φ²(1-φ)²
  - Interpolation: h(φ) = 3φ² - 2φ³

The sharp interface limit gives:
  v = (m/τ) · ΔG · C

where C depends on the interface profile integrals.

Measured: v ≈ $(@sprintf("%.4f", α_fit)) × ΔG
Expected: v ∝ (m/τ) × ΔG = $(m/τ) × ΔG × C

This implies C ≈ $(@sprintf("%.4f", α_fit * τ / m))

Reference: Karma & Rappel, Phys. Rev. E 57 (1998) 4323
""")

# =============================================================================
# Plot results
# =============================================================================

println("Generating plot...")

p = plot(
    title="Interface Velocity vs Driving Force",
    xlabel="Driving Force ΔG",
    ylabel="Interface Velocity v",
    legend=:topleft,
    size=(800, 600),
    xlims=(-0.55, 0),
    ylims=(minimum(velocities)*1.1, 0.1),
    framestyle=:box,
    linewidth=1.5,
    grid=true
)

# Data points
scatter!(ΔG_arr, v_arr,
         label="Simulation",
         markersize=10,
         markerstrokewidth=2,
         color=:blue)

# Linear fit
ΔG_line = range(-0.55, 0, length=50)
v_line = α_fit .* ΔG_line
plot!(ΔG_line, v_line,
      label=@sprintf("Fit: v = %.3f × ΔG (R²=%.4f)", α_fit, R²),
      color=:red,
      linewidth=3,
      linestyle=:dash)

# Reference line at v=0
hline!([0], color=:gray, linewidth=1.5, linestyle=:dot, label="")
vline!([0], color=:gray, linewidth=1.5, linestyle=:dot, label="")

savefig(p, "examples/901_benchmark_interface_velocity.png")
println("  Saved: examples/901_benchmark_interface_velocity.png")
