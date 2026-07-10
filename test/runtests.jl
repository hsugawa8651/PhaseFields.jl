using PhaseFields
using Test
using DifferentiationInterface
import DifferentiationInterface as DI

@testset "PhaseFields.jl" begin
    # Type construction and parameter validation
    include("test_types.jl")
    include("test_interpolation.jl")
    include("test_anisotropy.jl")
    include("test_differentiation.jl")

    # Model RHS (no solve)
    include("test_allen_cahn.jl")
    include("test_cahn_hilliard.jl")
    include("test_kks.jl")
    include("test_wbm.jl")
    include("test_thermal.jl")

    # Abstract types, grids, problem types
    include("test_models_abstract.jl")
    include("test_grids.jl")
    include("test_grids_2d.jl")
    include("test_problems.jl")  # Test 1-7 only (solve tests separated)

    # Snapshot types for plotting
    include("test_snapshot.jl")

    # PythonPlot extension: 3-layer plotting API (loads PythonPlot, so must run
    # AFTER test_snapshot.jl which exercises the not-loaded fallbacks)
    include("test_publication.jl")

    # OpenCALPHAD extension (DB calls only, no solve)
    # Skip if OpenCALPHAD is not available (weakdep, not registered yet)
    if Base.find_package("OpenCALPHAD") !== nothing
        include("test_opencalphad_ext.jl")

        # Public API surface: exported names have methods, orphans and clashes are
        # frozen. Runs here so OpenCALPHAD (and the extensions loaded above) is
        # present; the clash check needs it.
        include("test_public_api.jl")
    else
        @warn "Skipping OpenCALPHAD extension tests (OpenCALPHAD not available)"
    end
end
