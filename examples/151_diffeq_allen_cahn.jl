# DifferentialEquations.jl Integration: Allen-Cahn Example
#
# Demonstrates PhaseFields.jl integration with OrdinaryDiffEq.jl:
# - Explicit solver (Tsit5) vs Implicit solver (QNDF)
# - Adaptive time stepping
# - Solution visualization and animation

using PhaseFields
using OrdinaryDiffEq
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

# =============================================================================
# Problem Setup
# =============================================================================

println("=== Allen-Cahn with DifferentialEquations.jl ===\n")

# Model parameters
model = AllenCahnModel(τ=1.0, W=0.05)

# Spatial discretization
grid = UniformGrid1D(N=101, L=1.0)

# Boundary conditions
bc = NeumannBC()

# Initial condition: step function with smooth transition
x0 = 0.3  # Initial interface position
φ0 = [0.5 * (1 - tanh((x - x0) / (2 * model.W))) for x in grid.x]

# Time span
tspan = (0.0, 2.0)

println("Model: Allen-Cahn")
println("  τ = $(model.τ), W = $(model.W)")
println("Grid: N = $(grid.N), L = $(grid.L), dx = $(grid.dx)")
println("Initial interface at x = $x0")
println()

# =============================================================================
# Solver Comparison: Explicit vs Implicit
# =============================================================================

println("--- Solver Comparison ---\n")

# Create ODE problem
prob = create_allen_cahn_problem(model, grid, bc, φ0, tspan)

# Explicit solver (Tsit5 - 5th order Runge-Kutta)
println("Solving with Tsit5 (explicit)...")
@time sol_explicit = solve(prob, Tsit5(); saveat=0.1)
println("  Steps: $(length(sol_explicit.t))")
println("  Return code: $(sol_explicit.retcode)")

# Implicit solver (QNDF - quasi-constant stepsize BDF)
println("\nSolving with QNDF (implicit, autodiff=false)...")
@time sol_implicit = solve(prob, QNDF(autodiff=false); saveat=0.1)
println("  Steps: $(length(sol_implicit.t))")
println("  Return code: $(sol_implicit.retcode)")

# Compare solutions at final time
φ_explicit = sol_explicit.u[end]
φ_implicit = sol_implicit.u[end]
max_diff = maximum(abs.(φ_explicit - φ_implicit))
println("\nMax difference between solvers: $max_diff")

# =============================================================================
# Adaptive Time Stepping Analysis
# =============================================================================

println("\n--- Adaptive Time Stepping ---\n")

# Solve without fixed saveat to see adaptive stepping
sol_adaptive = solve(prob, Tsit5())
println("Tsit5 adaptive stepping:")
println("  Total steps: $(length(sol_adaptive.t))")
println("  dt range: $(minimum(diff(sol_adaptive.t))) to $(maximum(diff(sol_adaptive.t)))")

# Plot time step sizes
dt_values = diff(sol_adaptive.t)
p_dt = plot(sol_adaptive.t[1:end-1], dt_values,
    xlabel="Time", ylabel="dt",
    title="Adaptive Time Step Size (Tsit5)",
    legend=false, linewidth=2,
    yscale=:log10)

savefig(p_dt, joinpath(@__DIR__, "151_diffeq_allen_cahn_timesteps.png"))
println("Saved: 151_diffeq_allen_cahn_timesteps.png")

# =============================================================================
# Solution Visualization
# =============================================================================

println("\n--- Visualization ---\n")

# Plot solution at multiple times
times_to_plot = [0.0, 0.5, 1.0, 1.5, 2.0]
p_evolution = plot(xlabel="x", ylabel="φ",
    title="Allen-Cahn Phase Field Evolution",
    legend=:topright)

for t in times_to_plot
    # Find closest time in solution
    idx = argmin(abs.(sol_explicit.t .- t))
    φ = sol_explicit.u[idx]
    plot!(p_evolution, grid.x, φ, label="t = $(sol_explicit.t[idx])", linewidth=2)
end

savefig(p_evolution, joinpath(@__DIR__, "151_diffeq_allen_cahn_evolution.png"))
println("Saved: 151_diffeq_allen_cahn_evolution.png")

# Compare explicit vs implicit at final time
p_compare = plot(grid.x, φ_explicit, label="Tsit5 (explicit)",
    xlabel="x", ylabel="φ",
    title="Solver Comparison at t = $(tspan[2])",
    linewidth=2)
plot!(p_compare, grid.x, φ_implicit, label="QNDF (implicit)",
    linewidth=2, linestyle=:dash)

savefig(p_compare, joinpath(@__DIR__, "151_diffeq_allen_cahn_comparison.png"))
println("Saved: 151_diffeq_allen_cahn_comparison.png")

# =============================================================================
# Animation
# =============================================================================

println("\n--- Creating Animation ---\n")

# Solve with more time points for smooth animation
sol_anim = solve(prob, Tsit5(); saveat=0.02)

anim = @animate for (i, t) in enumerate(sol_anim.t)
    φ = sol_anim.u[i]

    plot(grid.x, φ,
        xlabel="x", ylabel="φ",
        title="Allen-Cahn Evolution (t = $(round(t, digits=3)))",
        ylim=(-0.1, 1.1),
        linewidth=2,
        legend=false,
        color=:blue)

    # Mark interface position (φ = 0.5)
    interface_idx = findfirst(φ .< 0.5)
    if interface_idx !== nothing && interface_idx > 1
        x_interface = grid.x[interface_idx-1] +
            (0.5 - φ[interface_idx-1]) / (φ[interface_idx] - φ[interface_idx-1]) * grid.dx
        scatter!([x_interface], [0.5], markersize=8, color=:red, label="")
    end
end

gif_path = joinpath(@__DIR__, "151_diffeq_allen_cahn.gif")
gif(anim, gif_path, fps=20)
println("Saved: 151_diffeq_allen_cahn.gif")

# =============================================================================
# Interface Velocity Analysis
# =============================================================================

println("\n--- Interface Velocity ---\n")

# Track interface position over time
interface_positions = Float64[]
interface_times = Float64[]

for (i, t) in enumerate(sol_anim.t)
    φ = sol_anim.u[i]
    # Find where φ crosses 0.5
    idx = findfirst(φ .< 0.5)
    if idx !== nothing && idx > 1
        # Linear interpolation
        x_int = grid.x[idx-1] +
            (0.5 - φ[idx-1]) / (φ[idx] - φ[idx-1]) * grid.dx
        push!(interface_positions, x_int)
        push!(interface_times, t)
    end
end

# Plot interface position
p_interface = plot(interface_times, interface_positions,
    xlabel="Time", ylabel="Interface Position",
    title="Interface Position vs Time",
    linewidth=2, legend=false)

savefig(p_interface, joinpath(@__DIR__, "151_diffeq_allen_cahn_interface.png"))
println("Saved: 151_diffeq_allen_cahn_interface.png")

# Calculate velocity
if length(interface_positions) > 1
    velocities = diff(interface_positions) ./ diff(interface_times)
    avg_velocity = sum(velocities) / length(velocities)
    println("Average interface velocity: $(round(avg_velocity, digits=4))")
end

# =============================================================================
# Callbacks Demo: SavingCallback and TerminateSteadyState
# =============================================================================

println("\n--- Callbacks Demo ---\n")

# 1. SavingCallback for interface tracking
println("1. SavingCallback for interface tracking:")
cb_interface, saved_interface = create_interface_saving_callback(grid; saveat=0.1)

sol_cb1 = solve(prob, Tsit5(); callback=cb_interface)
println("   Saved $(length(saved_interface.t)) interface positions")
println("   Initial: x = $(round(saved_interface.saveval[1], digits=4))")
println("   Final:   x = $(round(saved_interface.saveval[end], digits=4))")

# 2. SavingCallback for solid fraction tracking
println("\n2. SavingCallback for solid fraction:")
cb_frac, saved_frac = create_solid_fraction_callback(; saveat=0.1)

sol_cb2 = solve(prob, Tsit5(); callback=cb_frac)
println("   Saved $(length(saved_frac.t)) solid fraction values")
println("   Initial: f_s = $(round(saved_frac.saveval[1], digits=4))")
println("   Final:   f_s = $(round(saved_frac.saveval[end], digits=4))")

# 3. TerminateSteadyState for automatic stopping
println("\n3. TerminateSteadyState callback:")
prob_long = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 100.0))
cb_ss = create_steady_state_callback(abstol=1e-6, reltol=1e-4, min_t=0.5)

sol_ss = solve(prob_long, Tsit5(); callback=cb_ss)
println("   Long tspan: (0, 100)")
println("   Terminated at t = $(round(sol_ss.t[end], digits=4))")
println("   ReturnCode: $(sol_ss.retcode)")

# 4. Combined callbacks
println("\n4. Combined callbacks:")
result = create_phase_field_callbacks(grid;
    track_interface=true,
    track_solid_fraction=true,
    terminate_steady_state=true,
    saveat=0.1,
    abstol=1e-6, reltol=1e-4, min_t=0.5)

sol_combined = solve(prob_long, Tsit5(); callback=result.callback)
println("   Terminated at t = $(round(sol_combined.t[end], digits=4))")
println("   Interface: $(length(result.interface_data.t)) points saved")
println("   Solid fraction: $(length(result.solid_fraction_data.t)) points saved")

# Plot callback results
p_cb = plot(layout=(2,1), size=(800,600))

# Interface position from callback
plot!(p_cb[1], saved_interface.t, saved_interface.saveval,
    xlabel="Time", ylabel="Interface Position",
    title="Interface Position (SavingCallback)",
    linewidth=2, legend=false, marker=:circle, markersize=5,
    ylims=(0, 1))

# Solid fraction from callback
plot!(p_cb[2], saved_frac.t, saved_frac.saveval,
    xlabel="Time", ylabel="Solid Fraction",
    title="Solid Fraction (SavingCallback)",
    linewidth=2, legend=false, marker=:circle, markersize=5,
    ylims=(0, 1))

savefig(p_cb, joinpath(@__DIR__, "151_diffeq_allen_cahn_callbacks.png"))
println("\nSaved: 151_diffeq_allen_cahn_callbacks.png")

println("\n=== Done ===")
