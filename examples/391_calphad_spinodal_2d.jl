# 2D Spinodal Decomposition Driven by a Thermodynamic Database
#
# One uniform Ag-Cu mixture separates into an Ag-rich and a Cu-rich phase.
# The free energy is read from a TDB file rather than from a model potential:
# nothing about this run is fitted to make the picture come out.
#
# Run: julia --project=. examples/391_calphad_spinodal_2d.jl
#      (OpenCALPHAD.jl must be in the environment; this script does not
#       modify Project.toml)

using PhaseFields
using OpenCALPHAD
using OrdinaryDiffEq
using ForwardDiff
using Plots
default(
    guidefontsize=14, tickfontsize=12, titlefontsize=14, legendfontsize=11,
    left_margin=15Plots.mm, right_margin=10Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm
)

const R = 8.314462618
const T = 900.0          # inside the Ag-Cu miscibility gap
const RT = R * T

println("="^70)
println("Ag-Cu spinodal decomposition, free energy from a TDB file")
println("="^70)

# =============================================================================
# 1. The database
# =============================================================================
tdb = joinpath(pkgdir(OpenCALPHAD), "reftest", "tdb", "agcu.TDB")
db = read_tdb(tdb)
f_db = calphad_free_energy(db, "FCC_A1", T)

# =============================================================================
# 2. Tabulate it once
#
# calphad_free_energy answers to chemical_potential_bulk, so it can be handed
# straight to CahnHilliardProblem. Doing that on a 128x128 grid is impractical:
# one call into the database costs tens of milliseconds, and the solver wants
# one per cell per stage.
#
# So evaluate it on a composition grid once and interpolate afterwards. The
# thermodynamics still comes from the TDB; only the lookup is cached.
#
# The ideal-mixing part is split off rather than tabulated. dG/dc carries an
# RT*ln(c/(1-c)) term that diverges at c = 0 and c = 1, and that divergence is
# the only thing keeping the composition inside [0,1]. Interpolating through it
# would flatten it. The remainder is smooth and tabulates well.
# =============================================================================
const NTAB = 801
const CLO, CHI = 0.001, 0.999
const CGRID = range(CLO, CHI, length = NTAB)
const DC = step(CGRID)

logit(c) = log(c / (1 - c))

print("Tabulating $(NTAB) points ... ")
t0 = time()
# const matters: excess() reads this four times per call, and a non-const
# global would make every one of them type-unstable.
const μ_excess = [ForwardDiff.derivative(x -> free_energy(f_db, x), c) / RT - logit(c)
                  for c in CGRID]
println("$(round(time() - t0, digits=1)) s")

"""
    TabulatedTDB

Free energy for the Cahn-Hilliard slot, backed by a table built once from a
TDB file. Dimensionless: energies are divided by RT.
"""
struct TabulatedTDB end

# Cubic Hermite between table points, with slopes from central differences.
# Written out rather than pulled from an interpolation package to keep the
# examples free of dependencies the package itself does not have.
function excess(c::Real)
    x = (clamp(c, CLO, CHI) - CLO) / DC
    i = clamp(floor(Int, x), 0, NTAB - 2)
    t = x - i
    y0, y1 = μ_excess[i+1], μ_excess[i+2]
    m0 = i == 0 ? (y1 - y0) : (μ_excess[i+2] - μ_excess[i]) / 2
    m1 = i == NTAB - 2 ? (y1 - y0) : (μ_excess[i+3] - μ_excess[i+1]) / 2
    t2 = t * t
    t3 = t2 * t
    return (2t3 - 3t2 + 1) * y0 + (t3 - 2t2 + t) * m0 +
           (-2t3 + 3t2) * y1 + (t3 - t2) * m1
end

# The divergence stays analytic, so the composition cannot leave (0, 1).
PhaseFields.chemical_potential_bulk(::TabulatedTDB, c::Real) = logit(c) + excess(c)

f = TabulatedTDB()

# =============================================================================
# 3. Solve
#
# kappa is chosen so the fastest growing wavelength is about a third of the
# box; it sets the size of the domains, not the thermodynamics.
# =============================================================================
# N = 192 keeps the interface about 5 cells wide (w = sqrt(2k/|f''|) = 7.6,
# dx = 1.56). Coarsening the grid to 128 leaves it at 3.3 cells, which makes the
# fourth-order equation far stiffer: the same run then takes 973 s instead of
# about 160 s.
Nx = Ny = 192
Lx = Ly = 300.0
grid = UniformGrid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
model = CahnHilliardModel(M=5.0, κ=90.0)

c0 = [0.5 + 0.02 * sin(6π * x / Lx) * cos(6π * y / Ly) +
            0.012 * sin(10π * x / Lx) * sin(8π * y / Ly)
      for x in grid.x, y in grid.y]

tspan = (0.0, 400.0)
problem = CahnHilliardProblem(model, grid, c0, tspan, f, bc=PeriodicBC())

# The default tolerances are not tight enough here: the composition leaves
# [0,1] while retcode still comes back Success. reltol 1e-6 and 1e-8 agree to
# six digits, so 1e-6 is converged.
println("Solving $(Nx)x$(Ny) ...")
t0 = time()
sol = PhaseFields.solve(problem, ROCK2();
                        reltol=1e-6, abstol=1e-9, saveat=20.0,
                        maxiters=20_000_000)
println("  $(round(time() - t0, digits=1)) s, retcode = $(sol.retcode)")

c_final = reshape(sol.u[end], Nx, Ny)
println("  composition in [$(round(minimum(c_final), digits=4)), " *
        "$(round(maximum(c_final), digits=4))]")
println("  mean $(round(sum(c_final) / length(c_final), digits=7)) " *
        "(started at $(round(sum(c0) / length(c0), digits=7)))")

# =============================================================================
# 4. Compare with the phase boundary the same database predicts
#
# The two plateaus are not fitted. Solve the common tangent independently and
# see where they land.
# =============================================================================
G(c) = free_energy(f_db, clamp(c, CLO, CHI)) / RT
μ(c) = PhaseFields.chemical_potential_bulk(f, c)

function binodal(ca, cb)
    for _ in 1:400
        F1 = μ(cb) - μ(ca)
        F2 = (G(cb) - G(ca)) - μ(ca) * (cb - ca)
        max(abs(F1), abs(F2)) < 1e-12 && break
        h = 1e-6
        dμa = (μ(ca + h) - μ(ca - h)) / 2h
        det = (-dμa) * (μ(cb) - μ(ca)) - ((μ(cb + h) - μ(cb - h)) / 2h) * (-dμa * (cb - ca))
        abs(det) < 1e-16 && break
        ca = clamp(ca + 0.5 * (((μ(cb) - μ(ca)) * (-F1) -
                               ((μ(cb + h) - μ(cb - h)) / 2h) * (-F2)) / det), 0.002, 0.5)
        cb = clamp(cb + 0.5 * ((dμa * (cb - ca) * (-F1) + (-dμa) * (-F2)) / det), 0.5, 0.998)
    end
    return ca, cb
end

ca, cb = binodal(0.05, 0.95)
println("\nCALPHAD binodal at $(T) K : c_alpha = $(round(ca, digits=4)), " *
        "c_beta = $(round(cb, digits=4))")
println("simulated plateaus        : c_alpha = $(round(minimum(c_final), digits=4)), " *
        "c_beta = $(round(maximum(c_final), digits=4))")

# =============================================================================
# 5. Figures
# =============================================================================
p1 = heatmap(grid.x, grid.y, c_final', c=:RdBu, clims=(0.0, 1.0),
             aspect_ratio=1, xlabel="x", ylabel="y", colorbar_title="x(Cu)",
             title="Ag-Cu at $(Int(T)) K, free energy from the TDB",
             size=(700, 620))
savefig(p1, "examples/391_calphad_spinodal_2d.png")
println("\n  Saved: examples/391_calphad_spinodal_2d.png")

# saveat is uniform: asking ROCK2 to land exactly on a handful of times cuts
# its step down and costs several minutes. Save on a regular grid and pick the
# panels here instead.
want = [0.0, 30.0, 60.0, 150.0, 400.0]
idx = [argmin(abs.(sol.t .- w)) for w in want]
ps = [heatmap(grid.x, grid.y, reshape(sol.u[i], Nx, Ny)', c=:RdBu, clims=(0.0, 1.0),
              aspect_ratio=:equal, ticks=false, colorbar=false, framestyle=:box,
              xlims=(0, Lx), ylims=(0, Ly),
              title="t = $(Int(sol.t[i]))", titlefontsize=14)
      for i in idx]
p2 = plot(ps..., layout=(1, length(ps)), size=(300 * length(ps), 320),
          margin=1Plots.mm)
savefig(p2, "examples/391_calphad_spinodal_2d_evolution.png")
println("  Saved: examples/391_calphad_spinodal_2d_evolution.png")

println("\n✅ Simulation completed!")
