using Test
using PhaseFields

@testset "Cahn-Hilliard Model" begin

    @testset "CahnHilliardModel construction" begin
        # Positional constructor
        model = CahnHilliardModel(5.0, 2.0)
        @test model.M == 5.0
        @test model.κ == 2.0

        # Keyword constructor
        model2 = CahnHilliardModel(M=10.0, κ=3.0)
        @test model2.M == 10.0
        @test model2.κ == 3.0
    end

    @testset "DoubleWellFreeEnergy construction" begin
        # Positional constructor
        f = DoubleWellFreeEnergy(5.0, 0.3, 0.7)
        @test f.ρs == 5.0
        @test f.cα == 0.3
        @test f.cβ == 0.7

        # Keyword constructor (PFHub BM1 parameters)
        f2 = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)
        @test f2.ρs == 5.0
        @test f2.cα == 0.3
        @test f2.cβ == 0.7
    end

    @testset "Free energy density" begin
        f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)

        # Minima at equilibrium concentrations
        @test free_energy_density(f, 0.3) ≈ 0.0 atol=1e-10
        @test free_energy_density(f, 0.7) ≈ 0.0 atol=1e-10

        # Maximum at c = (cα + cβ)/2 = 0.5
        f_mid = free_energy_density(f, 0.5)
        @test f_mid > 0.0

        # Verify it's a maximum (values on either side are smaller)
        @test free_energy_density(f, 0.4) < f_mid
        @test free_energy_density(f, 0.6) < f_mid

        # Symmetry around midpoint
        @test free_energy_density(f, 0.4) ≈ free_energy_density(f, 0.6) atol=1e-10
    end

    @testset "Chemical potential bulk" begin
        f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)

        # df/dc = 0 at equilibrium concentrations
        @test chemical_potential_bulk(f, 0.3) ≈ 0.0 atol=1e-10
        @test chemical_potential_bulk(f, 0.7) ≈ 0.0 atol=1e-10

        # df/dc = 0 at midpoint (inflection point)
        @test chemical_potential_bulk(f, 0.5) ≈ 0.0 atol=1e-10

        # Sign check: df/dc > 0 for cα < c < 0.5 (approaching maximum)
        @test chemical_potential_bulk(f, 0.4) > 0.0

        # Sign check: df/dc < 0 for 0.5 < c < cβ (descending from maximum)
        @test chemical_potential_bulk(f, 0.6) < 0.0
    end

    @testset "Chemical potential with gradient term" begin
        model = CahnHilliardModel(M=5.0, κ=2.0)
        f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)

        # At equilibrium with no curvature
        μ = cahn_hilliard_chemical_potential(model, f, 0.3, 0.0)
        @test μ ≈ 0.0 atol=1e-10

        # Positive Laplacian (concave down) increases μ
        μ_curved = cahn_hilliard_chemical_potential(model, f, 0.3, 1.0)
        @test μ_curved < 0.0  # -κ∇²c = -κ·(+1) < 0

        # Negative Laplacian (concave up) decreases μ
        μ_curved2 = cahn_hilliard_chemical_potential(model, f, 0.3, -1.0)
        @test μ_curved2 > 0.0
    end

    @testset "Cahn-Hilliard RHS" begin
        model = CahnHilliardModel(M=5.0, κ=2.0)

        # dc/dt = M·∇²μ
        ∇²μ = 100.0
        dcdt = cahn_hilliard_rhs(model, ∇²μ)
        @test dcdt == 5.0 * 100.0

        # Zero flux at equilibrium (∇²μ = 0)
        @test cahn_hilliard_rhs(model, 0.0) == 0.0
    end

    @testset "Interface width estimate" begin
        model = CahnHilliardModel(M=5.0, κ=2.0)
        f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)

        W = cahn_hilliard_interface_width(model, f)
        @test W > 0.0
        @test isfinite(W)

        # Larger κ → wider interface
        model2 = CahnHilliardModel(M=5.0, κ=8.0)
        W2 = cahn_hilliard_interface_width(model2, f)
        @test W2 > W
    end

    @testset "Stability time step" begin
        model = CahnHilliardModel(M=5.0, κ=2.0)

        # dt < dx⁴ / (16·M·κ)
        dx = 1.0
        dt_max = cahn_hilliard_stability_dt(model, dx)
        @test dt_max ≈ 1.0 / (16 * 5.0 * 2.0) atol=1e-10

        # Smaller dx → much smaller dt (4th power)
        dx_small = 0.5
        dt_small = cahn_hilliard_stability_dt(model, dx_small)
        @test dt_small ≈ dt_max / 16 atol=1e-10  # (0.5)^4 = 1/16
    end

    @testset "AD compatibility" begin
        using ForwardDiff

        model = CahnHilliardModel(M=5.0, κ=2.0)
        f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)

        # Free energy density should be differentiable
        df_dc = ForwardDiff.derivative(c -> free_energy_density(f, c), 0.5)
        @test isfinite(df_dc)
        @test df_dc ≈ chemical_potential_bulk(f, 0.5) atol=1e-8

        # Chemical potential bulk should be differentiable
        d2f_dc2 = ForwardDiff.derivative(c -> chemical_potential_bulk(f, c), 0.5)
        @test isfinite(d2f_dc2)

        # Full chemical potential should be differentiable
        # Note: ∇²c must have same type as c for AD to work
        μ_deriv = ForwardDiff.derivative(
            c -> cahn_hilliard_chemical_potential(model, f, c, zero(c)), 0.5)
        @test isfinite(μ_deriv)
    end

    @testset "Mass conservation property" begin
        # Verify that Cahn-Hilliard preserves total mass
        # This is a design test: ∂c/∂t = ∇·J implies ∫c dV = const

        model = CahnHilliardModel(M=5.0, κ=2.0)
        f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)

        # For any uniform concentration field, ∇²μ = 0 (no flux)
        c_uniform = 0.5
        ∇²c = 0.0
        μ = cahn_hilliard_chemical_potential(model, f, c_uniform, ∇²c)
        # At uniform c, ∇μ = 0, so ∇²μ = 0
        # This means dc/dt = 0 for uniform field
        @test cahn_hilliard_rhs(model, 0.0) == 0.0
    end

end
