@testset "Allen-Cahn Model" begin
    @testset "AllenCahnModel construction" begin
        # Positional constructor
        model = AllenCahnModel(3e-8, 1e-6, 1.0)
        @test model.τ == 3e-8
        @test model.W == 1e-6
        @test model.m == 1.0

        # Keyword constructor
        model2 = AllenCahnModel(τ=3e-8, W=1e-6)
        @test model2.m == 1.0  # default

        # From InterfaceParams
        params = InterfaceParams(W₀=1e-6, σ₀=0.3, τ=3e-8, δ=0.04, n_fold=4)
        model3 = AllenCahnModel(params)
        @test model3.τ == params.τ
        @test model3.W == params.W₀
    end

    @testset "allen_cahn_rhs" begin
        model = AllenCahnModel(τ=3e-8, W=1e-6, m=1.0)

        # At equilibrium (φ=0, ∇²φ=0, ΔG=0): ∂φ/∂t = 0
        @test allen_cahn_rhs(model, 0.0, 0.0, 0.0) ≈ 0.0 atol=1e-15

        # At equilibrium (φ=1, ∇²φ=0, ΔG=0): ∂φ/∂t = 0
        @test allen_cahn_rhs(model, 1.0, 0.0, 0.0) ≈ 0.0 atol=1e-15

        # Positive driving force (ΔG > 0) at φ=0.5 should drive φ toward 1
        # h'(0.5) = 1.5 > 0, so positive ΔG gives positive contribution
        rhs_pos = allen_cahn_rhs(model, 0.5, 0.0, 1000.0)
        @test rhs_pos > 0

        # Negative driving force should drive φ toward 0
        rhs_neg = allen_cahn_rhs(model, 0.5, 0.0, -1000.0)
        @test rhs_neg < 0

        # Diffusion term: positive Laplacian should increase φ
        rhs_diff = allen_cahn_rhs(model, 0.5, 1e12, 0.0)
        @test rhs_diff > 0
    end

    @testset "allen_cahn_residual" begin
        model = AllenCahnModel(τ=3e-8, W=1e-6, m=1.0)

        # At equilibrium states, residual should be zero
        @test PhaseFields.allen_cahn_residual(model, 0.0, 0.0, 0.0) ≈ 0.0 atol=1e-15
        @test PhaseFields.allen_cahn_residual(model, 1.0, 0.0, 0.0) ≈ 0.0 atol=1e-15
    end

    @testset "Interface properties" begin
        model = AllenCahnModel(τ=3e-8, W=1e-6, m=1.0)

        # Interface width ≈ 2√2 W
        width = PhaseFields.allen_cahn_interface_width(model)
        @test width ≈ 2 * sqrt(2) * 1e-6 rtol=1e-10

        # Interface energy ≈ W/3
        energy = PhaseFields.allen_cahn_interface_energy(model)
        @test energy ≈ 1e-6 / 3 rtol=1e-10
    end

    @testset "AD compatibility" begin
        model = AllenCahnModel(τ=3e-8, W=1e-6, m=1.0)

        # allen_cahn_rhs should work with AD for sensitivity analysis
        # ∂(∂φ/∂t)/∂φ at fixed ∇²φ and ΔG
        f(φ) = allen_cahn_rhs(model, φ, 0.0, -5000.0)

        φ = 0.5
        dRdφ = DI.derivative(f, DI.AutoForwardDiff(), φ)

        # Should be finite and non-zero
        @test isfinite(dRdφ)
        @test dRdφ != 0.0

        # Also test with Dual numbers directly
        using ForwardDiff
        @test ForwardDiff.derivative(f, φ) ≈ dRdφ rtol=1e-10
    end

    @testset "Type promotion" begin
        model = AllenCahnModel(τ=3e-8, W=1e-6, m=1.0)

        # Mixed Int and Float should work
        result = allen_cahn_rhs(model, 0.5, 0, 0)
        @test isfinite(result)

        # Float32 should work
        result32 = allen_cahn_rhs(model, Float32(0.5), Float32(0.0), Float32(0.0))
        @test isfinite(result32)
    end
end
