@testset "Anisotropy Functions" begin
    @testset "anisotropy_cubic (4-fold)" begin
        δ = 0.04

        # At θ=0: σ = 1 + δ (maximum)
        @test anisotropy_cubic(0.0, δ=δ) ≈ 1 + δ atol=1e-15

        # At θ=π/4: σ = 1 - δ (minimum)
        @test anisotropy_cubic(π/4, δ=δ) ≈ 1 - δ atol=1e-15

        # At θ=π/2: σ = 1 + δ (same as θ=0)
        @test anisotropy_cubic(π/2, δ=δ) ≈ 1 + δ atol=1e-15

        # 4-fold symmetry: σ(θ) = σ(θ + π/2)
        @test anisotropy_cubic(0.3, δ=δ) ≈ anisotropy_cubic(0.3 + π/2, δ=δ) atol=1e-14
    end

    @testset "anisotropy_hcp (6-fold)" begin
        δ = 0.02

        # At θ=0: σ = 1 + δ (maximum)
        @test anisotropy_hcp(0.0, δ=δ) ≈ 1 + δ atol=1e-15

        # At θ=π/6: σ = 1 - δ (minimum)
        @test anisotropy_hcp(π/6, δ=δ) ≈ 1 - δ atol=1e-15

        # 6-fold symmetry: σ(θ) = σ(θ + π/3)
        @test anisotropy_hcp(0.3, δ=δ) ≈ anisotropy_hcp(0.3 + π/3, δ=δ) atol=1e-14
    end

    @testset "anisotropy_custom" begin
        # 4-fold should match cubic
        @test anisotropy_custom(0.5, n=4, δ=0.04) ≈ anisotropy_cubic(0.5, δ=0.04) atol=1e-15

        # 6-fold should match hcp
        @test anisotropy_custom(0.5, n=6, δ=0.02) ≈ anisotropy_hcp(0.5, δ=0.02) atol=1e-15
    end

    @testset "anisotropy derivatives" begin
        δ = 0.04

        # At θ=0: σ'(0) = 0 (extremum)
        @test PhaseFields.anisotropy_cubic_prime(0.0, δ=δ) ≈ 0.0 atol=1e-15

        # At θ=π/4: σ'(π/4) = 0 (extremum)
        @test PhaseFields.anisotropy_cubic_prime(π/4, δ=δ) ≈ 0.0 atol=1e-15

        # AD verification
        f(θ) = anisotropy_cubic(θ, δ=δ)
        for θ in [0.1, 0.3, 0.5, 0.7]
            ad_deriv = DI.derivative(f, DI.AutoForwardDiff(), θ)
            analytical = PhaseFields.anisotropy_cubic_prime(θ, δ=δ)
            @test ad_deriv ≈ analytical rtol=1e-10
        end
    end

    @testset "anisotropy_stiffness" begin
        δ = 0.04
        n = 4

        # σ + σ'' = 1 + δ(1-n²)cos(nθ)
        # For n=4: coefficient = 1 - 16 = -15
        @test PhaseFields.anisotropy_stiffness(0.0, δ=δ, n=n) ≈ 1 + δ*(1-16) atol=1e-14

        # Stiffness can be negative for large anisotropy (dendrite instability)
        large_δ = 0.1
        stiffness_min = PhaseFields.anisotropy_stiffness(0.0, δ=large_δ, n=4)
        @test stiffness_min < 0  # Negative stiffness indicates instability
    end

    @testset "AD compatibility (Dual numbers)" begin
        using ForwardDiff

        # anisotropy_cubic should work with Dual numbers
        f(θ) = anisotropy_cubic(θ, δ=0.04)
        @test ForwardDiff.derivative(f, 0.3) ≈ PhaseFields.anisotropy_cubic_prime(0.3, δ=0.04) rtol=1e-10

        # anisotropy_hcp should work with Dual numbers
        g(θ) = anisotropy_hcp(θ, δ=0.02)
        @test ForwardDiff.derivative(g, 0.3) ≈ PhaseFields.anisotropy_hcp_prime(0.3, δ=0.02) rtol=1e-10
    end
end
