# Generate phase-field-only animation GIF for pretalx session image
#
# Run: julia --project=. examples/352_thermal_2d_phase_only.jl

using PhaseFields
using OrdinaryDiffEq
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=5Plots.mm, right_margin=5Plots.mm,
    top_margin=5Plots.mm, bottom_margin=5Plots.mm
)

# Model parameters (same as 352_thermal_2d.jl)
model = ThermalPhaseFieldModel(
    τ = 1.0, W = 0.04, λ = 2.0, α = 1.0, L = 1.0, Cp = 1.0, Tm = 0.0
)

Nx, Ny = 80, 80
Lx, Ly = 1.0, 1.0
grid = UniformGrid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
tspan = (0.0, 0.3)

# Initial conditions
R0 = 0.1
cx, cy = 0.5, 0.5
φ0 = [sqrt((x - cx)^2 + (y - cy)^2) < R0 ? 1.0 : 0.0
      for x in grid.x, y in grid.y]
u0 = -0.3 * ones(Nx, Ny)

# Solve
problem = ThermalProblem(model, grid, φ0, u0, tspan)
sol = PhaseFields.solve(problem, Tsit5(), saveat=0.03)

N_total = Nx * Ny

# Phase-field-only animation
println("Generating phase-field-only animation...")
anim = @animate for idx in 1:length(sol.t)
    φ = reshape(sol.u[idx][1:N_total], Nx, Ny)
    heatmap(grid.x, grid.y, φ',
            c=:viridis, clim=(0, 1),
            xlabel="x", ylabel="y",
            title="Thermal Solidification 2D: φ (t=$(round(sol.t[idx], digits=3)))",
            aspect_ratio=:equal,
            size=(600, 550))
end
gif(anim, "examples/352_thermal_2d_phase_only.gif", fps=8)
println("Saved: examples/352_thermal_2d_phase_only.gif")
