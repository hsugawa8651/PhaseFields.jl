using Test
using PhaseFields

# Load OpenCALPHAD to trigger extension loading
using OpenCALPHAD

@testset "OpenCALPHAD Extension" begin

    # Find TDB file
    tdb_path = joinpath(@__DIR__, "..", "..", "OpenCALPHAD.jl", "reftest", "tdb", "agcu.TDB")

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

        @testset "KKS with CALPHAD - CALPHADFreeEnergy" begin
            T = 1000.0
            _, f_s, f_l = create_calphad_kks_model(db, T, "FCC_A1", "LIQUID")

            # Access extension module for method dispatch
            ext = Base.get_extension(PhaseFields, :OpenCALPHADExt)
            @test ext !== nothing

            # Verify the struct was created correctly
            @test f_s.T == T
            @test f_l.T == T

            # Note: Direct testing of free_energy/chemical_potential/d2f_dc2
            # is done implicitly through kks_partition which uses these functions
        end

        @testset "KKS with CALPHAD - kks_partition" begin
            T = 1000.0
            _, f_s, f_l = create_calphad_kks_model(db, T, "FCC_A1", "LIQUID")

            c_avg = 0.3
            φ = 0.5

            # Try to partition
            c_s, c_l, μ, converged = kks_partition(c_avg, φ, f_s, f_l)

            @test isfinite(c_s)
            @test isfinite(c_l)
            @test isfinite(μ)
            # Note: convergence depends on CALPHAD free energy landscape
            # May not always converge for all conditions

            if converged
                # Mass conservation check
                h = h_polynomial(φ)
                @test h * c_s + (1 - h) * c_l ≈ c_avg atol=1e-8
            end
        end

        @testset "KKS with CALPHAD - integration" begin
            T = 1000.0
            model, f_s, f_l = create_calphad_kks_model(db, T, "FCC_A1", "LIQUID")

            c_avg = 0.3
            φ = 0.5

            # kks_partition uses chemical_potential and d2f_dc2 internally
            c_s, c_l, μ, converged = kks_partition(c_avg, φ, f_s, f_l)

            @test isfinite(c_s)
            @test isfinite(c_l)
            @test isfinite(μ)
            # Note: convergence depends on CALPHAD thermodynamics

            if converged
                # Mass conservation check
                h = h_polynomial(φ)
                @test h * c_s + (1 - h) * c_l ≈ c_avg atol=1e-6

                # Test phase RHS with a mock Δω value
                ∇²φ = 0.0
                Δω = -100.0  # Arbitrary test value
                dφdt = kks_phase_rhs(model, φ, ∇²φ, Δω)
                @test isfinite(dφdt)
            end
        end

    else
        @warn "Skipping OpenCALPHAD extension tests: TDB file not found at $tdb_path"
        @test_skip "TDB file not found"
    end

end
