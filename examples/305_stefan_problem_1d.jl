# 1D Stefan Problem Benchmark
#
# Verification of thermal phase field model against Neumann analytical solution.
#
# Problem setup (solidification):
#   - Semi-infinite domain x ∈ [0, L]
#   - Initial: Liquid at melting temperature T = Tm
#   - Boundary: T(0,t) = Tw < Tm (wall undercooling)
#   - Solid nucleates at wall and grows into liquid
#   - Interface: x = s(t), T = Tm
#
# Neumann analytical solution:
#   s(t) = 2λ√(αt)
#   where λ satisfies: St·exp(-λ²)/(λ·√π·erf(λ)) = 1
#
# Note on accuracy:
#   The phase field model includes interface kinetics that are not present in
#   the sharp interface (Stefan) limit. For quantitative agreement, one would
#   need to use the thin-interface asymptotics (Karma-Rappel 1998) with
#   carefully calibrated parameters. This example demonstrates qualitative
#   agreement: the interface moves with sqrt(t) behavior and the temperature
#   profile evolves correctly.
#
# Reference: Wikipedia "Stefan problem", Karma-Rappel (1998) PRE 57, 4323

using PhaseFields
using Plots
using SpecialFunctions
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

# =============================================================================
# Neumann Analytical Solution
# =============================================================================

"""
    neumann_lambda(St; tol=1e-10, maxiter=100)

Solve transcendental equation for Neumann solution parameter λ.

The equation is: St·exp(-λ²)/(λ·√π·erf(λ)) = 1

Uses Newton's method.
"""
function neumann_lambda(St; tol=1e-10, maxiter=100)
    # Initial guess based on small St approximation
    λ = sqrt(St / 2)

    for _ in 1:maxiter
        erf_λ = erf(λ)
        exp_λ2 = exp(-λ^2)

        # f(λ) = St·exp(-λ²) - λ·√π·erf(λ) = 0
        f = St * exp_λ2 - λ * sqrt(π) * erf_λ

        # f'(λ) = -2λ·St·exp(-λ²) - √π·erf(λ) - λ·√π·(2/√π)·exp(-λ²)
        #       = -2λ·St·exp(-λ²) - √π·erf(λ) - 2λ·exp(-λ²)
        df = -2λ * St * exp_λ2 - sqrt(π) * erf_λ - 2λ * exp_λ2

        Δλ = -f / df
        λ += Δλ

        if abs(Δλ) < tol
            break
        end
    end

    return λ
end

"""
    neumann_interface_position(t, α, St)

Analytical interface position s(t) = 2λ√(αt).
"""
function neumann_interface_position(t, α, St)
    λ = neumann_lambda(St)
    return 2λ * sqrt(α * t)
end

"""
    neumann_temperature_profile(x, t, α, Tw, Tm, St)

Analytical temperature profile in solid region (solidification case).

For solidification from undercooled wall (Tw < Tm):
T(x,t) = Tw + (Tm - Tw)·erf(x/(2√(αt)))/erf(λ)

Valid in the solid region 0 < x < s(t).
In liquid region x > s(t): T = Tm.
"""
function neumann_temperature_profile(x, t, α, Tw, Tm, St)
    if t <= 0
        return x == 0 ? Tw : Tm
    end

    λ = neumann_lambda(St)
    s = neumann_interface_position(t, α, St)

    if x > s
        # Liquid region: T = Tm
        return Tm
    else
        # Solid region: similarity solution
        η = x / (2 * sqrt(α * t))
        return Tw + (Tm - Tw) * erf(η) / erf(λ)
    end
end

# =============================================================================
# Phase Field Simulation
# =============================================================================

"""
    run_stefan_simulation(; kwargs...)

Run 1D Stefan problem with phase field model (solidification).

# Problem setup
Solidification from an undercooled wall:
- Domain starts as liquid at melting temperature
- Wall (x=0) is held at T = Tm - ΔT (undercooling)
- Solid nucleates at wall and grows into liquid

# Keyword Arguments
- `Nx`: Number of grid points (default: 200)
- `L`: Domain length [m] (default: 1e-4)
- `Tm`: Melting temperature [K] (default: 1728, Ni)
- `ΔT`: Undercooling [K] (default: 50)
- `t_end`: End time [s] (default: 1e-3)
- `n_snapshots`: Number of snapshots (default: 10)
"""
function run_stefan_simulation(;
    Nx = 200,
    L_domain = 1e-4,
    Tm = 1728.0,
    ΔT = 50.0,
    t_end = 1e-3,
    n_snapshots = 10
)
    # Physical parameters (Nickel-like)
    L_latent = 2.35e9   # J/m³
    Cp = 5.42e6         # J/(m³·K)
    α = 1e-5            # m²/s (thermal diffusivity)

    # Phase field parameters
    W = L_domain / 100  # Interface width (thinner = closer to sharp interface)
    # τ and λ calibrated for approximate agreement with sharp interface
    # See Karma-Rappel (1998) for thin interface asymptotics
    τ = W^2 / α * 0.5   # Relaxation time
    λ_coupling = 1.5    # Coupling strength

    # Create model
    model = ThermalPhaseFieldModel(
        τ = τ,
        W = W,
        λ = λ_coupling,
        α = α,
        L = L_latent,
        Cp = Cp,
        Tm = Tm
    )

    # Grid
    dx = L_domain / (Nx - 1)
    x = range(0, L_domain, length=Nx)

    # Stefan number
    St = stefan_number(ΔT, L_latent, Cp)
    println("Stefan number: St = $(round(St, digits=4))")

    # Initial conditions - SOLIDIFICATION
    # Wall at T = Tm - ΔT (undercooling), liquid everywhere at Tm
    Tw = Tm - ΔT  # Wall is undercooled
    u_init = dimensionless_temperature(Tm, Tm, L_latent, Cp)  # = 0

    # Initialize with a thin solid layer at x=0 with diffuse interface
    # φ = 1 - 0.5(1 + tanh((x - x0)/(√2 W))) = 0.5(1 - tanh(...))
    x0 = 3 * W  # Initial interface position
    φ = [0.5 * (1 - tanh((xi - x0) / (sqrt(2) * W))) for xi in x]
    u = fill(u_init, Nx)

    # Boundary condition: wall at x=0 with undercooling
    u_wall = dimensionless_temperature(Tw, Tm, L_latent, Cp)  # < 0

    # Time stepping
    dt = thermal_stability_dt(model, dx)
    n_steps = ceil(Int, t_end / dt)
    dt = t_end / n_steps

    println("Grid: Nx=$Nx, dx=$(round(dx*1e6, digits=2))μm")
    println("Time: dt=$(round(dt*1e6, digits=3))μs, n_steps=$n_steps")

    # Storage for snapshots
    snapshot_interval = max(1, n_steps ÷ n_snapshots)
    times = Float64[]
    φ_snapshots = Vector{Float64}[]
    u_snapshots = Vector{Float64}[]
    T_snapshots = Vector{Float64}[]
    interface_positions = Float64[]

    # Time evolution
    for step in 0:n_steps
        t = step * dt

        # Save snapshot
        if step % snapshot_interval == 0
            push!(times, t)
            push!(φ_snapshots, copy(φ))
            push!(u_snapshots, copy(u))

            # Convert to physical temperature
            T = [physical_temperature(ui, Tm, L_latent, Cp) for ui in u]
            push!(T_snapshots, T)

            # Find interface position (φ = 0.5)
            # For solidification, interface is where solid meets liquid
            # φ goes from ~1 (solid, near x=0) to ~0 (liquid, far from wall)
            i_interface = findfirst(φi -> φi < 0.5, φ)
            if isnothing(i_interface)
                # All solid
                push!(interface_positions, L_domain)
            elseif i_interface == 1
                # All liquid
                push!(interface_positions, 0.0)
            else
                # Interpolate for more accurate position
                φ_prev = φ[i_interface - 1]
                φ_curr = φ[i_interface]
                frac = (0.5 - φ_prev) / (φ_curr - φ_prev)
                x_interface = x[i_interface - 1] + frac * dx
                push!(interface_positions, x_interface)
            end
        end

        if step == n_steps
            break
        end

        # Compute Laplacians (central difference)
        ∇²φ = zeros(Nx)
        ∇²u = zeros(Nx)

        for i in 2:Nx-1
            ∇²φ[i] = (φ[i+1] - 2φ[i] + φ[i-1]) / dx^2
            ∇²u[i] = (u[i+1] - 2u[i] + u[i-1]) / dx^2
        end

        # Boundary conditions
        # x=0: Dirichlet for temperature (superheat)
        ∇²u[1] = 2(u[2] - u[1]) / dx^2  # Ghost node approach
        ∇²φ[1] = 2(φ[2] - φ[1]) / dx^2

        # x=L: Zero flux (insulated)
        ∇²u[Nx] = 2(u[Nx-1] - u[Nx]) / dx^2
        ∇²φ[Nx] = 2(φ[Nx-1] - φ[Nx]) / dx^2

        # Compute phase field RHS
        dφdt = zeros(Nx)
        for i in 1:Nx
            dφdt[i] = thermal_phase_rhs(model, φ[i], ∇²φ[i], u[i])
        end

        # Update phase field
        φ_new = φ + dt * dφdt

        # Compute temperature RHS (using updated dφdt)
        dudt = zeros(Nx)
        for i in 1:Nx
            dudt[i] = thermal_heat_rhs(model, u[i], ∇²u[i], dφdt[i])
        end

        # Update temperature
        u_new = u + dt * dudt

        # Apply boundary conditions
        u_new[1] = u_wall  # Fixed wall temperature (undercooled)
        φ_new[1] = 1.0     # Solid at wall

        # Clamp phase field
        φ_new = clamp.(φ_new, 0.0, 1.0)

        φ = φ_new
        u = u_new
    end

    return (
        x = collect(x),
        times = times,
        φ = φ_snapshots,
        u = u_snapshots,
        T = T_snapshots,
        interface = interface_positions,
        model = model,
        St = St,
        Tw = Tw,
        Tm = Tm,
        α = α,
        dx = dx
    )
end

# =============================================================================
# Plotting
# =============================================================================

function plot_stefan_comparison(result)
    x = result.x
    times = result.times
    α = result.α
    St = result.St
    Tw = result.Tw
    Tm = result.Tm

    # Interface position comparison
    p1 = plot(
        title = "Interface Position vs Time",
        xlabel = "Time [s]",
        ylabel = "Interface Position [m]",
        legend = :bottomright,
        framestyle = :box,
        grid = true
    )

    # Analytical solution
    t_ana = range(1e-6, maximum(times), length=100)
    s_ana = [neumann_interface_position(t, α, St) for t in t_ana]
    plot!(p1, t_ana, s_ana, label="Analytical", lw=3, ls=:dash)

    # Phase field results
    scatter!(p1, times[2:end], result.interface[2:end],
             label="Phase Field", markersize=8, markerstrokewidth=2)

    # Temperature profiles at selected times
    p2 = plot(
        title = "Temperature Profiles",
        xlabel = "Position [m]",
        ylabel = "Temperature [K]",
        legend = :topright,
        framestyle = :box,
        grid = true
    )

    # Plot every other snapshot
    colors = palette(:viridis, length(times))
    for (i, t) in enumerate(times)
        if t > 0 && i % 2 == 0
            # Analytical
            T_ana = [neumann_temperature_profile(xi, t, α, Tw, Tm, St) for xi in x]
            plot!(p2, x, T_ana, label="", lw=3, ls=:dash, color=colors[i])

            # Phase field
            plot!(p2, x, result.T[i], label="t=$(round(t*1000, digits=2))ms",
                  lw=2, color=colors[i])
        end
    end

    # Phase field profiles
    p3 = plot(
        title = "Phase Field Evolution",
        xlabel = "Position [m]",
        ylabel = "φ (0=liquid, 1=solid)",
        legend = :topright,
        framestyle = :box,
        grid = true
    )

    for (i, t) in enumerate(times)
        if i % 2 == 0
            plot!(p3, x, result.φ[i],
                  label="t=$(round(t*1000, digits=2))ms",
                  lw=2, color=colors[i])
        end
    end

    # Error analysis
    p4 = plot(
        title = "Interface Position Error",
        xlabel = "Time [s]",
        ylabel = "Relative Error",
        legend = :topright,
        framestyle = :box,
        grid = true
    )

    errors = Float64[]
    for (i, t) in enumerate(times)
        if t > 0
            s_ana = neumann_interface_position(t, α, St)
            s_pf = result.interface[i]
            if s_ana > 0
                push!(errors, abs(s_pf - s_ana) / s_ana)
            end
        end
    end
    plot!(p4, times[2:end], errors, lw=3, marker=:circle, markersize=6, markerstrokewidth=2, label="Rel. Error")

    return plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
end

# =============================================================================
# Main
# =============================================================================

function main()
    println("=" ^ 60)
    println("1D Stefan Problem: Phase Field vs Analytical Solution")
    println("=" ^ 60)

    # Run simulation
    result = run_stefan_simulation(
        Nx = 300,
        L_domain = 2e-4,
        ΔT = 30.0,      # Moderate superheat
        t_end = 5e-3,   # 5 ms
        n_snapshots = 20
    )

    # Print summary
    println("\nResults:")
    println("-" ^ 40)
    for (i, t) in enumerate(result.times)
        if t > 0
            s_ana = neumann_interface_position(t, result.α, result.St)
            s_pf = result.interface[i]
            error = s_ana > 0 ? 100 * abs(s_pf - s_ana) / s_ana : 0.0
            println("t=$(round(t*1000, digits=2))ms: " *
                   "s_ana=$(round(s_ana*1e6, digits=1))μm, " *
                   "s_pf=$(round(s_pf*1e6, digits=1))μm, " *
                   "error=$(round(error, digits=1))%")
        end
    end

    # Create comparison plot
    p = plot_stefan_comparison(result)
    savefig(p, joinpath(@__DIR__, "305_stefan_problem_1d.png"))
    println("\nPlot saved to examples/305_stefan_problem_1d.png")

    # Create animation
    println("\nGenerating animation...")
    animation_snapshots = []
    for (i, t) in enumerate(result.times)
        p_frame = plot(layout=(1, 2), size=(1000, 400))

        # Temperature
        T_min = min(result.Tw, result.Tm) - 10
        T_max = max(result.Tw, result.Tm) + 10
        plot!(p_frame[1], result.x * 1e6, result.T[i],
              xlabel="Position [μm]", ylabel="Temperature [K]",
              title="t = $(round(t*1000, digits=2)) ms",
              lw=3, legend=false, ylims=(T_min, T_max),
              framestyle=:box, grid=true)
        hline!(p_frame[1], [result.Tm], ls=:dash, lw=2, color=:gray)

        # Phase field
        plot!(p_frame[2], result.x * 1e6, result.φ[i],
              xlabel="Position [μm]", ylabel="φ",
              title="Phase Field",
              lw=3, legend=false, ylims=(-0.1, 1.1),
              framestyle=:box, grid=true)

        push!(animation_snapshots, p_frame)
    end

    anim = @animate for p_frame in animation_snapshots
        plot(p_frame)
    end
    gif(anim, joinpath(@__DIR__, "305_stefan_problem_1d.gif"), fps=4)
    println("Animation saved to examples/305_stefan_problem_1d.gif")

    return result
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
