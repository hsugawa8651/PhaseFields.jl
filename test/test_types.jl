@testset "Types" begin
    @testset "InterfaceParams" begin
        # Positional constructor
        params = InterfaceParams(1e-6, 0.3, 3e-8, 0.04, 4)
        @test params.W₀ == 1e-6
        @test params.σ₀ == 0.3
        @test params.τ == 3e-8
        @test params.δ == 0.04
        @test params.n_fold == 4

        # Keyword constructor
        params2 = InterfaceParams(W₀=1e-6, σ₀=0.3, τ=3e-8)
        @test params2.W₀ == 1e-6
        @test params2.δ == 0.04  # default
        @test params2.n_fold == 4  # default
    end

    @testset "DiffusionParams" begin
        params = DiffusionParams(1e-9, 1e-13, 40e3, 80e3)
        @test params.D_liquid == 1e-9
        @test params.D_solid == 1e-13
        @test params.Q_liquid == 40e3
        @test params.Q_solid == 80e3

        # Keyword constructor with defaults
        params2 = DiffusionParams(D_liquid=1e-9, D_solid=1e-13)
        @test params2.Q_liquid == 0.0  # default
    end

    @testset "MaterialParams" begin
        params = MaterialParams(1e-5, 12e3, 30.0, 200.0)
        @test params.Vm == 1e-5
        @test params.L == 12e3
        @test params.Cp == 30.0
        @test params.k == 200.0
    end

    @testset "diffusion_coefficient" begin
        # Simple test without activation energy
        params = DiffusionParams(D_liquid=1e-9, D_solid=1e-13)
        T = 1000.0  # K

        # At φ=0 (solid): D ≈ D_solid
        D_solid = PhaseFields.diffusion_coefficient(params, T, 0.0)
        @test D_solid ≈ 1e-13 rtol=0.01

        # At φ=1 (liquid): D ≈ D_liquid
        D_liquid = PhaseFields.diffusion_coefficient(params, T, 1.0)
        @test D_liquid ≈ 1e-9 rtol=0.01

        # At φ=0.5: D is between solid and liquid
        D_mid = PhaseFields.diffusion_coefficient(params, T, 0.5)
        @test D_solid < D_mid < D_liquid
    end
end
