using Test
using PhaseFields

# Load OpenCALPHAD to trigger extension loading
using OpenCALPHAD

@testset "OpenCALPHAD Extension" begin

    # Find TDB file from the installed OpenCALPHAD package (works in CI too)
    tdb_path = joinpath(pkgdir(OpenCALPHAD), "reftest", "tdb", "agcu.TDB")

    @testset "Extension loading" begin
        # Extension should be loaded when OpenCALPHAD is imported
        @test isdefined(PhaseFields, :calphad_driving_force)
        @test isdefined(PhaseFields, :create_calphad_allen_cahn)
    end

    if isfile(tdb_path)
        db = read_tdb(tdb_path)

        @testset "calphad_driving_force" begin
            T = 1000.0
            x = 0.3

            ΔG = calphad_driving_force(db, T, x, "FCC_A1", "LIQUID")
            @test ΔG isa Float64
            @test isfinite(ΔG)

            # At this condition, solid should be stable (ΔG < 0)
            @test ΔG < 0

            # Higher temperature should make liquid more stable
            ΔG_hot = calphad_driving_force(db, 1200.0, x, "FCC_A1", "LIQUID")
            @test ΔG_hot > ΔG
        end

        @testset "calphad_chemical_potential" begin
            T = 1000.0
            x = 0.3

            μ = calphad_chemical_potential(db, "FCC_A1", T, x)
            @test μ isa Tuple
            @test length(μ) == 2  # Binary system: Ag, Cu
            @test all(isfinite, μ)
        end

        @testset "calphad_diffusion_potential" begin
            T = 1000.0
            x = 0.3

            d2G = calphad_diffusion_potential(db, "FCC_A1", T, x)
            @test d2G isa Real
            @test isfinite(d2G)
            # Note: d²G/dx² can be negative in unstable (spinodal) regions
        end

        @testset "create_calphad_allen_cahn" begin
            T = 1000.0
            x = 0.3

            model = create_calphad_allen_cahn(db, T, x, "FCC_A1", "LIQUID")

            @test model isa PhaseFields.AbstractCALPHADCoupledModel
            @test model.T == T
            @test model.x == x
            @test model.solid_phase == "FCC_A1"
            @test model.liquid_phase == "LIQUID"
            @test isfinite(model.ΔG)

            # Custom parameters
            model2 = create_calphad_allen_cahn(db, T, x, "FCC_A1", "LIQUID";
                                               τ=2.0, W=1.5, m=1e-3)
            @test model2.base_model.τ == 2.0
            @test model2.base_model.W == 1.5
            @test model2.base_model.m == 1e-3
        end

        @testset "allen_cahn_rhs with CALPHAD model" begin
            model = create_calphad_allen_cahn(db, 1000.0, 0.3, "FCC_A1", "LIQUID")

            # Test RHS computation
            φ = 0.5
            ∇²φ = 0.1

            dφdt = allen_cahn_rhs(model, φ, ∇²φ)
            @test isfinite(dφdt)

            # At interface (φ=0.5), driving force should push toward solid (ΔG < 0)
            # So dφdt should have contribution toward φ=1
        end

        @testset "update_conditions" begin
            # Need to use the extension module's function
            using PhaseFields: AbstractCALPHADCoupledModel

            model = create_calphad_allen_cahn(db, 1000.0, 0.3, "FCC_A1", "LIQUID")

            # Import the extension module to access update_conditions
            ext = Base.get_extension(PhaseFields, :OpenCALPHADExt)
            if ext !== nothing
                # Update temperature
                model_T = ext.update_conditions(model; T=1100.0)
                @test model_T.T == 1100.0
                @test model_T.x == 0.3
                @test model_T.ΔG != model.ΔG

                # Update composition
                model_x = ext.update_conditions(model; x=0.4)
                @test model_x.T == 1000.0
                @test model_x.x == 0.4
                @test model_x.ΔG != model.ΔG

                # Update both
                model_both = ext.update_conditions(model; T=1100.0, x=0.4)
                @test model_both.T == 1100.0
                @test model_both.x == 0.4
            else
                @warn "Could not get extension module for update_conditions test"
            end
        end

        @testset "AD compatibility" begin
            using ForwardDiff

            model = create_calphad_allen_cahn(db, 1000.0, 0.3, "FCC_A1", "LIQUID")

            # RHS should be differentiable with respect to φ
            ∇²φ = 0.1
            f(φ) = allen_cahn_rhs(model, φ, ∇²φ)

            dfdφ = ForwardDiff.derivative(f, 0.5)
            @test isfinite(dfdφ)
        end

        @testset "KKS with CALPHAD - create_calphad_kks_model" begin
            T = 1000.0

            model, f_s, f_l = create_calphad_kks_model(db, T, "FCC_A1", "LIQUID")

            @test model isa PhaseFields.KKSModel
            @test model.τ == 1.0
            @test model.W == 1.0
            @test model.m == 1.0
            @test model.M_s == 1.0
            @test model.M_l == 10.0

            # Custom parameters
            model2, _, _ = create_calphad_kks_model(db, T, "FCC_A1", "LIQUID";
                                                     τ=2.0, W=1.5, m=0.5, M_s=0.1, M_l=1.0)
            @test model2.τ == 2.0
            @test model2.W == 1.5
            @test model2.m == 0.5
            @test model2.M_s == 0.1
            @test model2.M_l == 1.0
        end

        @testset "calphad_free_energy (public front door)" begin
            T = 1000.0
            c = 0.3

            # Reachable from the user namespace, without Base.get_extension
            @test isdefined(PhaseFields, :calphad_free_energy)
            @test :calphad_free_energy in names(PhaseFields)

            f_s = calphad_free_energy(db, "FCC_A1", T)
            f_l = calphad_free_energy(db, "LIQUID", T)
            @test f_s.T == T
            @test f_l.T == T

            # Identical to the object the KKS factory already returns
            _, f_s_kks, _ = create_calphad_kks_model(db, T, "FCC_A1", "LIQUID")
            @test typeof(f_s) === typeof(f_s_kks)
            @test PhaseFields.free_energy(f_s, c) ≈ PhaseFields.free_energy(f_s_kks, c)

            # It fills the KKS free energy slot
            @test isfinite(PhaseFields.free_energy(f_s, c))
            @test isfinite(PhaseFields.chemical_potential(f_s, c))
            @test isfinite(PhaseFields.d2f_dc2(f_s, c))

            # It does NOT fill the Cahn-Hilliard slot: no chemical_potential_bulk
            # method. This assertion documents the limitation; see
            # docs/src/integration/calphad.md.
            @test !applicable(PhaseFields.chemical_potential_bulk, f_s, c)
        end

        @testset "CALPHAD front doors are exported and method backed" begin
            for s in (:calphad_free_energy, :calphad_driving_force,
                      :calphad_chemical_potential, :calphad_diffusion_potential,
                      :create_calphad_allen_cahn, :create_calphad_kks_model,
                      :create_calphad_wbm_model)
                @test s in names(PhaseFields)
                @test !isempty(methods(getproperty(PhaseFields, s)))
            end
        end

        @testset "KKS with CALPHAD - CALPHADFreeEnergy" begin
            T = 1000.0
            _, f_s, f_l = create_calphad_kks_model(db, T, "FCC_A1", "LIQUID")

            # Verify the struct was created correctly
            @test f_s.T == T
            @test f_l.T == T

            # Directly exercise the CALPHADFreeEnergy dispatch paths so a missing
            # OpenCALPHAD accessor is caught here (not silently via kks_partition).
            # free_energy delegates to OpenCALPHAD.gibbs_energy (added in OC v0.2.2);
            # d2f_dc2 delegates to diffusion_potential.
            c = 0.3
            @test isfinite(PhaseFields.free_energy(f_s, c))
            @test isfinite(PhaseFields.free_energy(f_l, c))
            @test isfinite(PhaseFields.d2f_dc2(f_s, c))
            @test isfinite(PhaseFields.chemical_potential(f_s, c))
        end

        # The KKS equilibrium partition (iterative Newton solve to convergence,
        # mass-conservation checks to tight tolerance) is reference validation of
        # the CALPHAD thermodynamics, not a lightweight CI sanity check. It is
        # exercised via the manual integration tests, not here. The dispatch paths
        # it relies on (free_energy / chemical_potential / d2f_dc2) are covered by
        # the direct assertions above.
        @testset "KKS with CALPHAD - kks_partition (reference; integration only)" begin
            @test_skip "kks_partition convergence is reference validation"
        end

    else
        @warn "Skipping OpenCALPHAD extension tests: TDB file not found at $tdb_path"
        @test_skip "TDB file not found"
    end

end
