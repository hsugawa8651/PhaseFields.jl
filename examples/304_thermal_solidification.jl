# Thermal Solidification Example
#
# Demonstrates coupled phase field and temperature evolution for
# solidification of a pure substance.
#
# Physical scenario:
#   - 1D domain initially at uniform undercooling
#   - Solid seed at center nucleates and grows
#   - Latent heat release raises temperature near interface
#   - Solidification slows as latent heat diffuses
#
# This example shows the essential physics of thermal phase field coupling:
#   1. Undercooling drives solidification (φ increases)
#   2. Solidification releases latent heat (T increases locally)
#   3. Heat diffuses away, allowing further solidification

using PhaseFields
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

"""
    run_304_thermal_solidification(; kwargs...)

Run 1D thermal solidification with central seed.

# Keyword Arguments
- `Nx`: Number of grid points (default: 200)
- `L`: Domain length [m] (default: 1e-4)
- `ΔT`: Undercooling [K] (default: 20)
- `t_end`: End time [s] (default: 2e-3)
- `n_frames`: Number of animation frames (default: 50)
"""
function run_304_thermal_solidification(;
    Nx = 200,
    L_domain = 1e-4,
    ΔT = 20.0,
    t_end = 2e-3,
    n_frames = 50
)
    # Physical parameters (Nickel-like)
    Tm = 1728.0         # Melting temperature [K]
    L_latent = 2.35e9   # Latent heat [J/m³]
    Cp = 5.42e6         # Heat capacity [J/(m³·K)]
    α = 1e-5            # Thermal diffusivity [m²/s]

    # Phase field parameters
    W = L_domain / 100  # Interface width
    τ = W^2 / α * 0.5   # Relaxation time
    λ_coupling = 2.0    # Thermal coupling strength

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
    x = collect(range(-L_domain/2, L_domain/2, length=Nx))

    # Stefan number
    St = stefan_number(ΔT, L_latent, Cp)
    println("Stefan number: St = $(round(St, digits=4))")
    println("Grid: Nx=$Nx, dx=$(round(dx*1e6, digits=2))μm")

    # Initial conditions
    # Uniform undercooling: T = Tm - ΔT everywhere
    T_init = Tm - ΔT
    u_init = dimensionless_temperature(T_init, Tm, L_latent, Cp)

    # Solid seed at center with diffuse interface
    seed_radius = 5 * W
    φ = [0.5 * (1 - tanh((abs(xi) - seed_radius) / (sqrt(2) * W))) for xi in x]
    u = fill(u_init, Nx)

    println("Initial undercooling: u = $(round(u_init, digits=4))")

    # Time stepping
    dt = thermal_stability_dt(model, dx)
    n_steps = ceil(Int, t_end / dt)
    dt = t_end / n_steps

    println("Time: dt=$(round(dt*1e6, digits=3))μs, n_steps=$n_steps")

    # Storage for animation
    frame_interval = max(1, n_steps ÷ n_frames)
    times = Float64[]
    φ_frames = Vector{Float64}[]
    T_frames = Vector{Float64}[]
    solid_fractions = Float64[]

    # Time evolution
    for step in 0:n_steps
        t = step * dt

        # Save frame
        if step % frame_interval == 0
            push!(times, t)
            push!(φ_frames, copy(φ))

            T = [physical_temperature(ui, Tm, L_latent, Cp) for ui in u]
            push!(T_frames, T)

            # Calculate solid fraction
            solid_frac = sum(φ) / Nx
            push!(solid_fractions, solid_frac)
        end

        if step == n_steps
            break
        end

        # Compute Laplacians (central difference with periodic BC)
        ∇²φ = zeros(Nx)
        ∇²u = zeros(Nx)

        for i in 2:Nx-1
            ∇²φ[i] = (φ[i+1] - 2φ[i] + φ[i-1]) / dx^2
            ∇²u[i] = (u[i+1] - 2u[i] + u[i-1]) / dx^2
        end

        # Zero-flux boundary conditions
        ∇²φ[1] = (φ[2] - φ[1]) / dx^2
        ∇²φ[Nx] = (φ[Nx-1] - φ[Nx]) / dx^2
        ∇²u[1] = (u[2] - u[1]) / dx^2
        ∇²u[Nx] = (u[Nx-1] - u[Nx]) / dx^2

        # Compute phase field RHS
        dφdt = zeros(Nx)
        for i in 1:Nx
            dφdt[i] = thermal_phase_rhs(model, φ[i], ∇²φ[i], u[i])
        end

        # Update phase field
        φ_new = φ + dt * dφdt

        # Compute temperature RHS
        dudt = zeros(Nx)
        for i in 1:Nx
            dudt[i] = thermal_heat_rhs(model, u[i], ∇²u[i], dφdt[i])
        end

        # Update temperature
        u_new = u + dt * dudt

        # Clamp phase field
        φ = clamp.(φ_new, 0.0, 1.0)
        u = u_new
    end

    return (
        x = x,
        times = times,
        φ = φ_frames,
        T = T_frames,
        solid_fraction = solid_fractions,
        Tm = Tm,
        T_init = T_init,
        L_domain = L_domain
    )
end

"""
    create_animation(result)

Create animation GIF from simulation results.
"""
function create_animation(result)
    x_μm = result.x * 1e6

    animation_frames = []

    for (i, t) in enumerate(result.times)
        p = plot(layout=(2, 1), size=(800, 600))

        t_ms = t * 1000

        # Phase field
        plot!(p[1], x_μm, result.φ[i],
              xlabel="", ylabel="φ (phase field)",
              title="Thermal Solidification: t = $(round(t_ms, digits=2)) ms",
              lw=2, legend=false, ylims=(-0.1, 1.1),
              color=:blue, fill=(0, 0.3, :blue))

        # Temperature
        T_min = result.T_init - 5
        T_max = result.Tm + 5
        plot!(p[2], x_μm, result.T[i],
              xlabel="Position [μm]", ylabel="Temperature [K]",
              lw=2, legend=false, ylims=(T_min, T_max),
              color=:red)
        hline!(p[2], [result.Tm], ls=:dash, color=:gray, label="Tm")

        push!(animation_frames, p)
    end

    return animation_frames
end

"""
    plot_summary(result)

Create summary plot showing evolution and solid fraction.
"""
function plot_summary(result)
    x_μm = result.x * 1e6
    times_ms = result.times * 1000

    # Phase field evolution (selected times)
    p1 = plot(
        title="Phase Field Evolution",
        xlabel="Position [μm]",
        ylabel="φ",
        legend=:outertopright
    )

    n_plots = min(5, length(result.times))
    indices = round.(Int, range(1, length(result.times), length=n_plots))
    colors = palette(:viridis, n_plots)

    for (j, i) in enumerate(indices)
        t_ms = round(times_ms[i], digits=2)
        plot!(p1, x_μm, result.φ[i],
              label="t = $(t_ms) ms",
              lw=2, color=colors[j])
    end

    # Temperature evolution
    p2 = plot(
        title="Temperature Evolution",
        xlabel="Position [μm]",
        ylabel="Temperature [K]",
        legend=:outertopright
    )

    for (j, i) in enumerate(indices)
        t_ms = round(times_ms[i], digits=2)
        plot!(p2, x_μm, result.T[i],
              label="t = $(t_ms) ms",
              lw=2, color=colors[j])
    end
    hline!(p2, [result.Tm], ls=:dash, color=:gray, label="Tm")

    # Solid fraction vs time
    p3 = plot(
        title="Solid Fraction vs Time",
        xlabel="Time [ms]",
        ylabel="Solid Fraction",
        legend=false
    )
    plot!(p3, times_ms, result.solid_fraction,
          lw=2, color=:blue, marker=:circle, markersize=3)

    # Combined
    return plot(p1, p2, p3, layout=(3, 1), size=(800, 900))
end

# =============================================================================
# Main
# =============================================================================

function main()
    println("=" ^ 60)
    println("Thermal Solidification Example")
    println("=" ^ 60)

    # Run simulation
    result = run_304_thermal_solidification(
        Nx = 200,
        L_domain = 1e-4,
        ΔT = 20.0,
        t_end = 2e-3,
        n_frames = 40
    )

    # Print summary
    println("\nResults:")
    println("-" ^ 40)
    println("Initial solid fraction: $(round(result.solid_fraction[1], digits=3))")
    println("Final solid fraction: $(round(result.solid_fraction[end], digits=3))")

    # Find interface positions at start and end
    φ_init = result.φ[1]
    φ_final = result.φ[end]

    i_left_init = findfirst(φi -> φi > 0.5, φ_init)
    i_right_init = findlast(φi -> φi > 0.5, φ_init)
    i_left_final = findfirst(φi -> φi > 0.5, φ_final)
    i_right_final = findlast(φi -> φi > 0.5, φ_final)

    if !isnothing(i_left_init) && !isnothing(i_right_init)
        width_init = (result.x[i_right_init] - result.x[i_left_init]) * 1e6
        println("Initial solid width: $(round(width_init, digits=1)) μm")
    end

    if !isnothing(i_left_final) && !isnothing(i_right_final)
        width_final = (result.x[i_right_final] - result.x[i_left_final]) * 1e6
        println("Final solid width: $(round(width_final, digits=1)) μm")
    end

    # Save summary plot
    p_summary = plot_summary(result)
    savefig(p_summary, joinpath(@__DIR__, "304_thermal_solidification.png"))
    println("\nSummary plot saved to examples/304_thermal_solidification.png")

    # Create and save animation
    println("\nGenerating animation...")
    frames = create_animation(result)

    anim = @animate for frame in frames
        plot(frame)
    end
    gif(anim, joinpath(@__DIR__, "304_thermal_solidification.gif"), fps=8)
    println("Animation saved to examples/304_thermal_solidification.gif")

    return result
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
