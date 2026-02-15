# CALPHAD Coupling Demo
#
# Demonstrates PhaseFields.jl + OpenCALPHAD.jl integration
# for thermodynamically-driven phase field simulation.
#
# Prerequisites:
#   julia> using Pkg
#   julia> Pkg.develop(path="../OpenCALPHAD.jl")
#
# Run: julia --project=. examples/381_calphad_coupling_demo.jl

using PhaseFields
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

# Load OpenCALPHAD.jl (local package)
using Pkg
Pkg.develop(path=joinpath(@__DIR__, "../../OpenCALPHAD.jl"))
using OpenCALPHAD

println("="^70)
println("CALPHAD Coupling Demo: Ag-Cu Solidification")
println("="^70)

# =============================================================================
# 1. Load thermodynamic database
# =============================================================================
println("\n1. Loading Ag-Cu thermodynamic database...")

tdb_path = joinpath(@__DIR__, "../../OpenCALPHAD.jl/reftest/tdb/agcu.TDB")
db = read_tdb(tdb_path)
println("   Database loaded: $(length(db.phases)) phases")

# =============================================================================
# 2. Calculate CALPHAD parameters at different conditions
# =============================================================================
println("\n2. Thermodynamic parameters at T=1000K, x_Cu=0.3:")

T = 1000.0  # K
x = 0.3     # mole fraction Cu

# Get all phase field parameters
params = phase_field_params(db, T, x, "FCC_A1", "LIQUID")

println("   ΔG (driving force) = $(round(params.ΔG, digits=1)) J/mol")
println("   μ_Ag (solid) = $(round(params.μ_solid[1], digits=1)) J/mol")
println("   μ_Cu (solid) = $(round(params.μ_solid[2], digits=1)) J/mol")
println("   d²G/dx² (solid) = $(round(params.d2G_solid, digits=1)) J/mol")

# =============================================================================
# 3. Driving force vs composition
# =============================================================================
println("\n3. Driving force vs composition at T=1000K:")

x_range = 0.1:0.1:0.5
println("   x_Cu   |    ΔG [J/mol]  |  Favorable phase")
println("   " * "-"^45)
for x_val in x_range
    ΔG = driving_force(db, T, x_val, "FCC_A1", "LIQUID")
    phase = ΔG < 0 ? "Solid (FCC)" : "Liquid"
    println("   $(round(x_val, digits=2))    |    $(lpad(round(Int, ΔG), 6))     |  $phase")
end

# =============================================================================
# 4. Driving force vs temperature (fixed composition)
# =============================================================================
println("\n4. Driving force vs temperature at x_Cu=0.3:")

T_range = 900:50:1200
x_fixed = 0.3
println("   T [K]  |    ΔG [J/mol]  |  Favorable phase")
println("   " * "-"^45)
for T_val in T_range
    ΔG = driving_force(db, T_val, x_fixed, "FCC_A1", "LIQUID")
    phase = ΔG < 0 ? "Solid (FCC)" : "Liquid"
    println("   $(T_val)    |    $(lpad(round(Int, ΔG), 6))     |  $phase")
end

# =============================================================================
# 5. 1D Allen-Cahn with CALPHAD driving force
# =============================================================================
println("\n5. 1D Allen-Cahn simulation with CALPHAD driving force...")

# Model parameters
model = AllenCahnModel(τ=1.0, W=1.0, m=1e-4)  # m scales ΔG to dimensionless

# Grid
Nx = 100
dx = 1.0
φ = [0.5 * (1 + tanh((i - 30) / 3)) for i in 1:Nx]

# Laplacian helper
function compute_laplacian!(∇²φ, φ, dx)
    Nx = length(φ)
    for i in 2:Nx-1
        ∇²φ[i] = (φ[i+1] - 2φ[i] + φ[i-1]) / dx^2
    end
    ∇²φ[1] = (φ[2] - φ[1]) / dx^2
    ∇²φ[Nx] = (φ[Nx-1] - φ[Nx]) / dx^2
end

# ASCII visualization
function show_profile(φ, label)
    chars = [" ", "░", "▒", "▓", "█"]
    str = ""
    for v in φ
        idx = clamp(Int(floor(v * 4)) + 1, 1, 5)
        str *= chars[idx]
    end
    println("$label |$str|")
end

# Simulation parameters
T_sim = 1000.0  # K
x_sim = 0.3     # Cu mole fraction
dt = 0.1
Nt = 300

# Get driving force from CALPHAD
ΔG = driving_force(db, T_sim, x_sim, "FCC_A1", "LIQUID")
println("   Using ΔG = $(round(ΔG, digits=1)) J/mol from CALPHAD")

∇²φ = similar(φ)

println("\n   Time evolution (φ=0: liquid, φ=1: solid):")
show_profile(φ, "   t=0   ")

for step in 1:Nt
    compute_laplacian!(∇²φ, φ, dx)
    for i in 1:Nx
        dφdt = allen_cahn_rhs(model, φ[i], ∇²φ[i], ΔG)
        φ[i] = clamp(φ[i] + dt * dφdt, 0.0, 1.0)
    end
    if step % 100 == 0
        show_profile(φ, "   t=$(lpad(step,3))")
    end
end

# =============================================================================
# 6. Summary
# =============================================================================
println("\n" * "="^70)
println("Summary:")
println("="^70)
println("""
✅ OpenCALPHAD.jl provides thermodynamic data:
   - driving_force(db, T, x, solid, liquid) → ΔG
   - chemical_potential(phase, T, x, db) → (μ₁, μ₂)
   - diffusion_potential(phase, T, x, db) → d²G/dx²
   - phase_field_params(...) → all parameters

✅ PhaseFields.jl provides phase field models:
   - AllenCahnModel, allen_cahn_rhs
   - Interpolation functions (h, g)
   - Anisotropy functions

✅ Integration workflow:
   1. Load TDB database with OpenCALPHAD.jl
   2. Calculate ΔG(T, x) at each grid point
   3. Use ΔG in PhaseFields.jl evolution equations
""")

# =============================================================================
# 7. Generate plots
# =============================================================================
println("Generating plots...")

# Plot 1: Driving force vs composition at fixed T
x_plot = 0.05:0.01:0.6
ΔG_x = [driving_force(db, 1000.0, x, "FCC_A1", "LIQUID") for x in x_plot]

p1 = plot(
    title="Ag-Cu: Driving Force vs Composition (T=1000K)",
    xlabel="x_Cu (mole fraction)",
    ylabel="ΔG [J/mol]",
    legend=:topright,
    size=(700, 500)
)
plot!(x_plot, ΔG_x, label="ΔG (FCC - Liquid)", linewidth=2, color=:blue)
hline!([0], color=:gray, linestyle=:dash, label="")
# Shade regions
x_solid = x_plot[ΔG_x .< 0]
x_liquid = x_plot[ΔG_x .>= 0]
if length(x_solid) > 0
    vspan!([minimum(x_solid), maximum(x_solid)], alpha=0.2, color=:blue, label="FCC stable")
end
if length(x_liquid) > 0
    vspan!([minimum(x_liquid), maximum(x_liquid)], alpha=0.2, color=:red, label="Liquid stable")
end
savefig(p1, "examples/381_calphad_driving_force_composition.png")
println("  Saved: examples/381_calphad_driving_force_composition.png")

# Plot 2: Driving force vs temperature at fixed x
T_plot = 850:5:1250
ΔG_T = [driving_force(db, T, 0.3, "FCC_A1", "LIQUID") for T in T_plot]

p2 = plot(
    title="Ag-Cu: Driving Force vs Temperature (x_Cu=0.3)",
    xlabel="Temperature [K]",
    ylabel="ΔG [J/mol]",
    legend=:topleft,
    size=(700, 500)
)
plot!(T_plot, ΔG_T, label="ΔG (FCC - Liquid)", linewidth=2, color=:red)
hline!([0], color=:gray, linestyle=:dash, label="")

# Find and mark liquidus temperature (where ΔG = 0)
for i in 1:length(ΔG_T)-1
    if ΔG_T[i] * ΔG_T[i+1] < 0
        T_liq = T_plot[i] + (0 - ΔG_T[i]) / (ΔG_T[i+1] - ΔG_T[i]) * 5
        vline!([T_liq], color=:green, linestyle=:dot, label="T_liquidus ≈ $(round(Int, T_liq))K")
        break
    end
end
savefig(p2, "examples/381_calphad_driving_force_temperature.png")
println("  Saved: examples/381_calphad_driving_force_temperature.png")
