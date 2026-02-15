@testset "Differentiation" begin
    @testset "Default backend" begin
        # Default backend should be AutoForwardDiff
        backend = PhaseFields.get_ad_backend()
        @test backend isa DI.AutoForwardDiff
    end

    @testset "set_ad_backend!" begin
        # Save original
        original = PhaseFields.get_ad_backend()

        # Set new backend
        new_backend = DI.AutoForwardDiff()
        set_ad_backend!(new_backend)
        @test PhaseFields.get_ad_backend() === new_backend

        # Restore
        set_ad_backend!(original)
    end

    @testset "AD operations with default backend" begin
        # Test derivative
        f(x) = x^2 + 3x
        @test DI.derivative(f, PhaseFields.get_ad_backend(), 2.0) ≈ 7.0 atol=1e-14

        # Test second_derivative
        @test DI.second_derivative(f, PhaseFields.get_ad_backend(), 2.0) ≈ 2.0 atol=1e-14

        # Test gradient
        g(x) = sum(x.^2)
        x = [1.0, 2.0, 3.0]
        grad = DI.gradient(g, PhaseFields.get_ad_backend(), x)
        @test grad ≈ 2 .* x atol=1e-14
    end

    @testset "PhaseFields functions with DI" begin
        # h' via DI
        φ = 0.3
        h_prime_di = DI.derivative(h_polynomial, PhaseFields.get_ad_backend(), φ)
        @test h_prime_di ≈ h_prime(φ) rtol=1e-10

        # g' via DI
        g_prime_di = DI.derivative(g_standard, PhaseFields.get_ad_backend(), φ)
        @test g_prime_di ≈ g_prime(φ) rtol=1e-10

        # g'' via DI
        g_pp_di = DI.second_derivative(g_standard, PhaseFields.get_ad_backend(), φ)
        @test g_pp_di ≈ g_double_prime(φ) rtol=1e-10

        # Anisotropy via DI
        θ = 0.5
        σ_func(θ_val) = anisotropy_cubic(θ_val, δ=0.04)
        σ_prime_di = DI.derivative(σ_func, PhaseFields.get_ad_backend(), θ)
        @test σ_prime_di ≈ PhaseFields.anisotropy_cubic_prime(θ, δ=0.04) rtol=1e-10
    end
end
