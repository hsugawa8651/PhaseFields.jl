using Test
using PhaseFields

@testset "KKS Model" begin

    @testset "KKSModel construction" begin
        # Keyword constructor
        model = KKSModel(τ=1.0, W=1.0, m=1.0, M_s=1.0, M_l=10.0)
        @test model.τ == 1.0
        @test model.W == 1.0
        @test model.m == 1.0
        @test model.M_s == 1.0
        @test model.M_l == 10.0

        # Different mobilities
        model2 = KKSModel(τ=2.0, W=1.5, M_s=0.1, M_l=1.0)
        @test model2.m == 1.0  # Default value
    end

    @testset "ParabolicFreeEnergy construction" begin
        f = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)
        @test f.A == 1000.0
        @test f.c_eq == 0.1

        # Positional constructor
        f2 = ParabolicFreeEnergy(500.0, 0.5)
        @test f2.A == 500.0
        @test f2.c_eq == 0.5
    end

    @testset "ParabolicFreeEnergy functions" begin
        f = ParabolicFreeEnergy(A=1000.0, c_eq=0.3)

        # Minimum at equilibrium
        @test free_energy(f, 0.3) ≈ 0.0 atol=1e-10

        # Away from equilibrium
        @test free_energy(f, 0.4) ≈ 1000.0 * 0.1^2 atol=1e-10
        @test free_energy(f, 0.2) ≈ 1000.0 * 0.1^2 atol=1e-10

        # Chemical potential at equilibrium
        @test chemical_potential(f, 0.3) ≈ 0.0 atol=1e-10

        # Chemical potential slope
        @test chemical_potential(f, 0.4) > 0
        @test chemical_potential(f, 0.2) < 0

        # Second derivative (curvature)
        @test d2f_dc2(f, 0.3) == 2000.0
        @test d2f_dc2(f, 0.5) == 2000.0  # Constant for parabolic
    end

    @testset "kks_partition - basic" begin
        # Solid: equilibrium at c=0.1
        # Liquid: equilibrium at c=0.9
        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)

        # At interface (φ=0.5), c=0.5
        c_s, c_l, μ, converged = kks_partition(0.5, 0.5, f_s, f_l)
        @test converged
        @test isfinite(c_s)
        @test isfinite(c_l)
        @test isfinite(μ)

        # Mass conservation: h·c_s + (1-h)·c_l = c
        h = h_polynomial(0.5)
        @test h * c_s + (1 - h) * c_l ≈ 0.5 atol=1e-10

        # Equal potential: μ_s = μ_l
        μ_s = chemical_potential(f_s, c_s)
        μ_l = chemical_potential(f_l, c_l)
        @test μ_s ≈ μ_l atol=1e-10
    end

    @testset "kks_partition - different φ values" begin
        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)
        c_avg = 0.5

        for φ in [0.1, 0.3, 0.5, 0.7, 0.9]
            c_s, c_l, μ, converged = kks_partition(c_avg, φ, f_s, f_l)
            @test converged

            # Mass conservation
            h = h_polynomial(φ)
            @test h * c_s + (1 - h) * c_l ≈ c_avg atol=1e-10

            # Equal potential
            μ_s = chemical_potential(f_s, c_s)
            μ_l = chemical_potential(f_l, c_l)
            @test μ_s ≈ μ_l atol=1e-10
        end
    end

    @testset "kks_partition - edge cases" begin
        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)

        # Pure liquid (φ ≈ 0)
        c_s, c_l, μ, converged = kks_partition(0.8, 1e-10, f_s, f_l)
        @test converged
        @test c_l ≈ 0.8 atol=1e-6  # c_l should be close to c_avg

        # Pure solid (φ ≈ 1)
        c_s, c_l, μ, converged = kks_partition(0.2, 1.0 - 1e-10, f_s, f_l)
        @test converged
        @test c_s ≈ 0.2 atol=1e-6  # c_s should be close to c_avg
    end

    @testset "kks_grand_potential_diff" begin
        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)

        # Get equilibrium partition
        c_s, c_l, μ, _ = kks_partition(0.5, 0.5, f_s, f_l)

        Δω = kks_grand_potential_diff(f_s, f_l, c_s, c_l, μ)
        @test isfinite(Δω)

        # At equal A and symmetric equilibrium, Δω should be small at c=0.5, φ=0.5
        # (both phases have similar grand potentials)
    end

    @testset "kks_phase_rhs" begin
        model = KKSModel(τ=1.0, W=1.0, m=1.0, M_s=1.0, M_l=10.0)

        # Test at various conditions
        φ = 0.5
        ∇²φ = 0.1
        Δω = -100.0  # Solid is favored

        dφdt = kks_phase_rhs(model, φ, ∇²φ, Δω)
        @test isfinite(dφdt)

        # Negative Δω (solid favored) + h'(φ) > 0 → should push toward solid (φ→1)
        # At φ=0.5, h'(0.5) = 6*0.5*0.5 = 1.5 > 0
        # So m·h'·Δω = 1.0 * 1.5 * (-100) = -150 → negative contribution
        # Actually, the equation is (diffusion + bulk + driving)/τ
    end

    @testset "kks_mobility" begin
        model = KKSModel(τ=1.0, W=1.0, m=1.0, M_s=1.0, M_l=10.0)

        # At φ=0 (liquid), M = M_l
        @test kks_mobility(model, 0.0) ≈ 10.0 atol=1e-10

        # At φ=1 (solid), M = M_s
        @test kks_mobility(model, 1.0) ≈ 1.0 atol=1e-10

        # At φ=0.5, interpolated
        h = h_polynomial(0.5)
        M_expected = h * 1.0 + (1 - h) * 10.0
        @test kks_mobility(model, 0.5) ≈ M_expected atol=1e-10
    end

    @testset "kks_concentration_rhs" begin
        model = KKSModel(τ=1.0, W=1.0, m=1.0, M_s=1.0, M_l=10.0)

        φ = 0.5
        ∇²μ = 100.0

        dcdt = kks_concentration_rhs(model, φ, ∇²μ)
        @test isfinite(dcdt)

        # dcdt = M(φ) * ∇²μ
        M = kks_mobility(model, φ)
        @test dcdt ≈ M * ∇²μ atol=1e-10
    end

    @testset "AD compatibility" begin
        using ForwardDiff

        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)
        model = KKSModel(τ=1.0, W=1.0, m=1.0, M_s=1.0, M_l=10.0)

        # Free energy should be differentiable
        df = ForwardDiff.derivative(c -> free_energy(f_s, c), 0.3)
        @test df ≈ chemical_potential(f_s, 0.3) atol=1e-8

        # Phase RHS should be differentiable with respect to φ
        Δω = -100.0
        ∇²φ = 0.1
        dfdφ = ForwardDiff.derivative(φ -> kks_phase_rhs(model, φ, ∇²φ, Δω), 0.5)
        @test isfinite(dfdφ)

        # Mobility should be differentiable
        dMdφ = ForwardDiff.derivative(φ -> kks_mobility(model, φ), 0.5)
        @test isfinite(dMdφ)
    end

    @testset "kks_partition - convergence" begin
        # Test with different curvatures
        f_s = ParabolicFreeEnergy(A=500.0, c_eq=0.2)
        f_l = ParabolicFreeEnergy(A=2000.0, c_eq=0.8)

        c_s, c_l, μ, converged = kks_partition(0.5, 0.5, f_s, f_l)
        @test converged

        # Verify constraints
        h = h_polynomial(0.5)
        @test h * c_s + (1 - h) * c_l ≈ 0.5 atol=1e-10

        μ_s = chemical_potential(f_s, c_s)
        μ_l = chemical_potential(f_l, c_l)
        @test μ_s ≈ μ_l atol=1e-10
    end

end
