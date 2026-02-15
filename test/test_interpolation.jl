@testset "Interpolation Functions" begin
    @testset "h_polynomial" begin
        # Boundary conditions
        @test h_polynomial(0.0) ≈ 0.0 atol=1e-15
        @test h_polynomial(1.0) ≈ 1.0 atol=1e-15

        # Midpoint
        @test h_polynomial(0.5) ≈ 0.5 atol=1e-15

        # Monotonicity
        @test h_polynomial(0.25) < h_polynomial(0.5) < h_polynomial(0.75)
    end

    @testset "h_sin" begin
        # Boundary conditions
        @test h_sin(0.0) ≈ 0.0 atol=1e-15
        @test h_sin(1.0) ≈ 1.0 atol=1e-15

        # Midpoint
        @test h_sin(0.5) ≈ 0.5 atol=1e-15
    end

    @testset "g_standard" begin
        # Zeros at boundaries
        @test g_standard(0.0) ≈ 0.0 atol=1e-15
        @test g_standard(1.0) ≈ 0.0 atol=1e-15

        # Maximum at midpoint
        @test g_standard(0.5) ≈ 1/16 atol=1e-15

        # Symmetry
        @test g_standard(0.25) ≈ g_standard(0.75) atol=1e-15
    end

    @testset "g_obstacle" begin
        # Zeros at boundaries
        @test g_obstacle(0.0) ≈ 0.0 atol=1e-15
        @test g_obstacle(1.0) ≈ 0.0 atol=1e-15

        # Maximum at midpoint
        @test g_obstacle(0.5) ≈ 0.25 atol=1e-15
    end

    @testset "h_prime (analytical)" begin
        # h'(φ) = 6φ(1-φ)
        @test h_prime(0.0) ≈ 0.0 atol=1e-15
        @test h_prime(1.0) ≈ 0.0 atol=1e-15
        @test h_prime(0.5) ≈ 1.5 atol=1e-15  # 6 * 0.5 * 0.5 = 1.5
    end

    @testset "g_prime (analytical)" begin
        # g'(φ) = 2φ(1-φ)(1-2φ)
        @test g_prime(0.0) ≈ 0.0 atol=1e-15
        @test g_prime(1.0) ≈ 0.0 atol=1e-15
        @test g_prime(0.5) ≈ 0.0 atol=1e-15  # zero at maximum
    end

    @testset "g_double_prime (analytical)" begin
        # g''(φ) = 2(1 - 6φ + 6φ²)
        @test g_double_prime(0.0) ≈ 2.0 atol=1e-15
        @test g_double_prime(1.0) ≈ 2.0 atol=1e-15
        @test g_double_prime(0.5) ≈ -1.0 atol=1e-15  # 2(1 - 3 + 1.5) = -1
    end

    @testset "AD vs analytical derivatives" begin
        test_points = [0.1, 0.3, 0.5, 0.7, 0.9]

        for φ in test_points
            # h' comparison
            @test PhaseFields.h_prime_ad(φ) ≈ h_prime(φ) rtol=1e-10

            # g' comparison
            @test PhaseFields.g_prime_ad(φ) ≈ g_prime(φ) rtol=1e-10

            # g'' comparison
            @test PhaseFields.g_double_prime_ad(φ) ≈ g_double_prime(φ) rtol=1e-10
        end
    end

    @testset "AD compatibility (Dual numbers)" begin
        using ForwardDiff

        # h_polynomial should work with Dual numbers
        f(x) = h_polynomial(x)
        @test ForwardDiff.derivative(f, 0.5) ≈ h_prime(0.5) rtol=1e-10

        # g_standard should work with Dual numbers
        g(x) = g_standard(x)
        @test ForwardDiff.derivative(g, 0.5) ≈ g_prime(0.5) rtol=1e-10
    end
end
