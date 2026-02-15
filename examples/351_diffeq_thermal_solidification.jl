# DifferentialEquations.jl Integration: Thermal Solidification Example
#
# Demonstrates coupled thermal phase field simulation using OrdinaryDiffEq.jl:
# - Stefan problem with latent heat release
# - Stiff solver (QNDF) for coupled equations
# - Phase field and temperature evolution

using PhaseFields
using OrdinaryDiffEq
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

# =============================================================================
# Problem Setup: 1D Solidification with Thermal Coupling
# =============================================================================

println("=== Thermal Solidification with DifferentialEquations.jl ===\n")

# Physical parameters (Nickel-like)
# Using dimensionless formulation for numerical stability
model = ThermalPhaseFieldModel(
    τ = 1e-6,       # Relaxation time [s]
    W = 1e-6,       # Interface width parameter [m]
    λ = 2.0,        # Coupling strength
    α = 1e-5,       # Thermal diffusivity [m²/s]
    L = 2.35e9,     # Latent heat [J/m³]
    Cp = 5.42e6,    # Heat capacity [J/(m³·K)]
    Tm = 1728.0     # Melting temperature [K]
)

# Spatial discretization
L_domain = 1e-4  # 100 μm domain
grid = UniformGrid1D(N=101, L=L_domain)

# Boundary conditions
bc_φ = NeumannBC()  # Zero-flux for phase field
bc_u = NeumannBC()  # Adiabatic for temperature

println("Model: Thermal Phase Field")
println("  τ = $(model.τ) s")
println("  W = $(model.W) m")
println("  λ = $(model.λ)")
println("  α = $(model.α) m²/s")
println("  L/Cp = $(model.L / model.Cp) K (latent heat in temperature units)")
println()
println("Grid: N = $(grid.N), L = $(grid.L*1e6) μm, dx = $(grid.dx*1e6) μm")
println()

# =============================================================================
# Initial Conditions
# =============================================================================

# Solid seed in the center of the domain
x_center = grid.L / 2
seed_radius = 5 * model.W  # Seed radius = 5 interface widths

# Phase field: solid (φ=1) in center, liquid (φ=0) outside
φ0 = [0.5 * (1 - tanh((abs(x - x_center) - seed_radius) / (sqrt(2) * model.W)))
      for x in grid.x]

# Dimensionless temperature: slight undercooling
undercooling = 0.05  # 5% undercooling
u0 = fill(-undercooling, grid.N)

println("Initial conditions:")
println("  Solid seed: center at x = $(x_center*1e6) μm, radius = $(seed_radius*1e6) μm")
println("  Undercooling: $(undercooling * 100)%")
println()

# =============================================================================
# Time Integration
# =============================================================================

println("--- Time Integration ---\n")

# Time span
t_end = 1e-5  # 10 μs
tspan = (0.0, t_end)

# Create coupled ODE problem
prob = create_thermal_problem(model, grid, bc_φ, bc_u, φ0, u0, tspan)

println("Solving with QNDF (stiff solver)...")
println("  tspan = (0, $(t_end*1e6)) μs")

# Solve with stiff solver
@time sol = solve(prob, QNDF(autodiff=false);
    saveat=t_end/100,
    abstol=1e-8,
    reltol=1e-6)

println("  Return code: $(sol.retcode)")
println("  Time points saved: $(length(sol.t))")

# Extract solution
φ_hist, u_hist = extract_thermal_solution(sol, grid.N)

# =============================================================================
# Results Analysis
# =============================================================================

println("\n--- Results Analysis ---\n")

# Calculate solid fraction over time
solid_fraction = [sum(φ_hist[:, i] .> 0.5) / grid.N for i in 1:length(sol.t)]

println("Solid fraction evolution:")
println("  t = 0: $(round(solid_fraction[1]*100, digits=1))%")
println("  t = $(t_end*1e6) μs: $(round(solid_fraction[end]*100, digits=1))%")

# Temperature at final time
φ_final = φ_hist[:, end]
u_final = u_hist[:, end]
T_final = physical_temperature.(u_final, model.Tm, model.L, model.Cp)

println("\nTemperature at t = $(t_end*1e6) μs:")
println("  Min: $(round(minimum(T_final), digits=1)) K")
println("  Max: $(round(maximum(T_final), digits=1)) K")
println("  At interface: ~$(round(T_final[findfirst(φ_final .< 0.9)], digits=1)) K")

# =============================================================================
# Visualization
# =============================================================================

println("\n--- Visualization ---\n")

# Plot at selected times
n_times = length(sol.t)
time_indices = [1, n_times÷4, n_times÷2, 3*n_times÷4, n_times]

# Phase field evolution
p1 = plot(xlabel="x [μm]", ylabel="φ",
    title="Phase Field Evolution",
    legend=:topright)

for idx in time_indices
    t_μs = sol.t[idx] * 1e6
    plot!(p1, grid.x * 1e6, φ_hist[:, idx],
        label="t = $(round(t_μs, digits=2)) μs",
        linewidth=2)
end

# Temperature evolution
p2 = plot(xlabel="x [μm]", ylabel="T [K]",
    title="Temperature Evolution",
    legend=:bottomright)

for idx in time_indices
    t_μs = sol.t[idx] * 1e6
    T = physical_temperature.(u_hist[:, idx], model.Tm, model.L, model.Cp)
    plot!(p2, grid.x * 1e6, T,
        label="t = $(round(t_μs, digits=2)) μs",
        linewidth=2)
end
hline!(p2, [model.Tm], color=:black, linestyle=:dash, label="Tm")

# Solid fraction vs time
p3 = plot(sol.t * 1e6, solid_fraction * 100,
    xlabel="Time [μs]", ylabel="Solid Fraction [%]",
    title="Solidification Progress",
    linewidth=2, legend=false)

# Combined plot
p_combined = plot(p1, p2, p3, layout=(3, 1), size=(800, 900))
savefig(p_combined, joinpath(@__DIR__, "351_diffeq_thermal_solidification.png"))
println("Saved: 351_diffeq_thermal_solidification.png")

# =============================================================================
# Animation
# =============================================================================

println("\n--- Creating Animation ---\n")

anim = @animate for i in 1:length(sol.t)
    t_μs = sol.t[i] * 1e6
    φ = φ_hist[:, i]
    T = physical_temperature.(u_hist[:, i], model.Tm, model.L, model.Cp)

    # Phase field plot
    p_φ = plot(grid.x * 1e6, φ,
        xlabel="x [μm]", ylabel="φ",
        title="Phase Field",
        ylim=(-0.1, 1.1),
        linewidth=2, legend=false, color=:blue)

    # Temperature plot
    p_T = plot(grid.x * 1e6, T,
        xlabel="x [μm]", ylabel="T [K]",
        title="Temperature",
        linewidth=2, legend=false, color=:red)
    hline!(p_T, [model.Tm], color=:black, linestyle=:dash)

    # Combined with time annotation
    plot(p_φ, p_T, layout=(2, 1), size=(700, 500),
        plot_title="t = $(round(t_μs, digits=2)) μs")
end

gif_path = joinpath(@__DIR__, "351_diffeq_thermal_solidification.gif")
gif(anim, gif_path, fps=15)
println("Saved: 351_diffeq_thermal_solidification.gif")

# =============================================================================
# Interface Velocity
# =============================================================================

println("\n--- Interface Tracking ---\n")

# Track right interface position (solid growing into liquid)
interface_positions = Float64[]
interface_times = Float64[]

for i in 1:length(sol.t)
    φ = φ_hist[:, i]

    # Find right interface (where φ drops below 0.5 from the center)
    center_idx = grid.N ÷ 2
    for j in center_idx:grid.N-1
        if φ[j] >= 0.5 && φ[j+1] < 0.5
            # Linear interpolation
            x_int = grid.x[j] + (0.5 - φ[j]) / (φ[j+1] - φ[j]) * grid.dx
            push!(interface_positions, x_int)
            push!(interface_times, sol.t[i])
            break
        end
    end
end

if length(interface_positions) > 1
    # Calculate instantaneous velocity
    velocities = diff(interface_positions) ./ diff(interface_times)

    println("Interface velocity:")
    println("  Initial: $(round(velocities[1]*1e3, digits=2)) mm/s")
    println("  Final: $(round(velocities[end]*1e3, digits=2)) mm/s")
    println("  Average: $(round(sum(velocities)/length(velocities)*1e3, digits=2)) mm/s")

    # Plot interface position
    p_int = plot(interface_times * 1e6, (interface_positions .- x_center) * 1e6,
        xlabel="Time [μs]", ylabel="Interface displacement [μm]",
        title="Interface Position (from center)",
        linewidth=2, legend=false)
    savefig(p_int, joinpath(@__DIR__, "351_diffeq_thermal_interface.png"))
    println("\nSaved: 351_diffeq_thermal_interface.png")
end

println("\n=== Done ===")
