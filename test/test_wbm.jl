using Test
using PhaseFields

@testset "WBM Model" begin

    @testset "WBMModel constructor" begin
        # Basic constructor
        model = WBMModel(M_φ=1.0, κ=2.0, W=1.0, D_s=1e-13, D_l=1e-9)
        @test model.M_φ == 1.0
        @test model.κ == 2.0
        @test model.W == 1.0
        @test model.D_s == 1e-13
        @test model.D_l == 1e-9

        # Partial specification with defaults
        model2 = WBMModel(M_φ=2.0)  # Other parameters use defaults
        @test model2.M_φ == 2.0
        @test model2.κ == 1.0   # default
        @test model2.W == 1.0   # default
        @test model2.D_s == 1.0 # default
        @test model2.D_l == 1.0 # default
    end

    @testset "Bulk free energy" begin
        # Parabolic free energies for testing
        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.2)  # Solid eq at c=0.2
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.8)  # Liquid eq at c=0.8
        W = 100.0

        # Pure liquid (φ=0): f = f_L + 0
        f_liq = wbm_bulk_free_energy(f_s, f_l, 0.0, 0.5, W)
        f_L_expected = free_energy(f_l, 0.5)
        @test f_liq ≈ f_L_expected atol=1e-10

        # Pure solid (φ=1): f = f_S + 0
        f_sol = wbm_bulk_free_energy(f_s, f_l, 1.0, 0.5, W)
        f_S_expected = free_energy(f_s, 0.5)
        @test f_sol ≈ f_S_expected atol=1e-10

        # Interface (φ=0.5): f = 0.5*f_S + 0.5*f_L + W*g(0.5)
        f_int = wbm_bulk_free_energy(f_s, f_l, 0.5, 0.5, W)
        g_half = g_standard(0.5)  # = 0.0625
        h_half = h_polynomial(0.5)  # = 0.5
        f_expected = h_half * free_energy(f_s, 0.5) + (1 - h_half) * free_energy(f_l, 0.5) + W * g_half
        @test f_int ≈ f_expected atol=1e-10
    end

    @testset "Chemical potential" begin
        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.2)
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.8)

        # Pure liquid: μ = μ_L
        μ_liq = wbm_chemical_potential(f_s, f_l, 0.0, 0.5)
        μ_L_expected = chemical_potential(f_l, 0.5)
        @test μ_liq ≈ μ_L_expected atol=1e-10

        # Pure solid: μ = μ_S
        μ_sol = wbm_chemical_potential(f_s, f_l, 1.0, 0.5)
        μ_S_expected = chemical_potential(f_s, 0.5)
        @test μ_sol ≈ μ_S_expected atol=1e-10

        # Interface: μ = h*μ_S + (1-h)*μ_L
        μ_int = wbm_chemical_potential(f_s, f_l, 0.5, 0.5)
        h = h_polynomial(0.5)
        μ_expected = h * chemical_potential(f_s, 0.5) + (1 - h) * chemical_potential(f_l, 0.5)
        @test μ_int ≈ μ_expected atol=1e-10
    end

    @testset "Driving force" begin
        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.2)
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.8)
        W = 100.0

        # At φ=0 and φ=1: g'(φ) = 0, so driving force depends on f_S - f_L
        df_0 = wbm_driving_force(f_s, f_l, 0.0, 0.5, W)
        @test df_0 ≈ 0.0 atol=1e-10  # h'(0) = 0

        df_1 = wbm_driving_force(f_s, f_l, 1.0, 0.5, W)
        @test df_1 ≈ 0.0 atol=1e-10  # h'(1) = 0

        # At φ=0.5: h'(0.5) = 1.5, g'(0.5) = 0
        df_half = wbm_driving_force(f_s, f_l, 0.5, 0.5, W)
        f_S = free_energy(f_s, 0.5)  # 1000*(0.5-0.2)^2 = 90
        f_L = free_energy(f_l, 0.5)  # 1000*(0.5-0.8)^2 = 90
        h_p = h_prime(0.5)  # = 1.5
        g_p = g_prime(0.5)  # = 0
        df_expected = h_p * (f_S - f_L) + W * g_p
        @test df_half ≈ df_expected atol=1e-10

        # Sign check: when solid is more stable (lower f_S), driving force should push toward solid
        f_s_stable = ParabolicFreeEnergy(A=1000.0, c_eq=0.5)  # f_S(0.5) = 0
        f_l_unstable = ParabolicFreeEnergy(A=1000.0, c_eq=0.8)  # f_L(0.5) = 90
        df = wbm_driving_force(f_s_stable, f_l_unstable, 0.3, 0.5, W)
        # f_S - f_L = 0 - 90 = -90, h'(0.3) > 0
        # So df/dφ < 0, meaning system wants to increase φ (toward solid)
        @test df < 0
    end

    @testset "Phase field RHS" begin
        model = WBMModel(M_φ=1.0, κ=1.0, W=100.0, D_s=1e-13, D_l=1e-9)
        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.2)
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.8)

        # Zero Laplacian, at interface
        rhs = wbm_phase_rhs(model, 0.5, 0.0, 0.5, f_s, f_l)
        # ∂φ/∂t = M_φ * [κ∇²φ - ∂f/∂φ] = M_φ * [0 - df]
        df = wbm_driving_force(f_s, f_l, 0.5, 0.5, model.W)
        @test rhs ≈ model.M_φ * (0 - df) atol=1e-10

        # With positive Laplacian (curvature effect)
        rhs_curv = wbm_phase_rhs(model, 0.5, 1.0, 0.5, f_s, f_l)
        @test rhs_curv ≈ model.M_φ * (model.κ * 1.0 - df) atol=1e-10
    end

    @testset "Diffusivity" begin
        model = WBMModel(M_φ=1.0, κ=1.0, W=1.0, D_s=1e-13, D_l=1e-9)

        # Pure liquid (φ=0): D = D_l
        @test wbm_diffusivity(model, 0.0) ≈ 1e-9 atol=1e-20

        # Pure solid (φ=1): D = D_s
        @test wbm_diffusivity(model, 1.0) ≈ 1e-13 atol=1e-20

        # Interface (φ=0.5): D = h*D_s + (1-h)*D_l
        h = h_polynomial(0.5)
        D_expected = h * 1e-13 + (1 - h) * 1e-9
        @test wbm_diffusivity(model, 0.5) ≈ D_expected atol=1e-20

        # Monotonicity: D should decrease as φ increases
        D_values = [wbm_diffusivity(model, φ) for φ in 0:0.1:1]
        @test all(diff(D_values) .<= 0)
    end

    @testset "Concentration RHS" begin
        model = WBMModel(M_φ=1.0, κ=1.0, W=1.0, D_s=1e-13, D_l=1e-9)

        # ∂c/∂t = D(φ) * ∇²c
        ∇²c = 1.0

        # In liquid
        rhs_l = wbm_concentration_rhs(model, 0.0, ∇²c)
        @test rhs_l ≈ 1e-9 * ∇²c atol=1e-20

        # In solid
        rhs_s = wbm_concentration_rhs(model, 1.0, ∇²c)
        @test rhs_s ≈ 1e-13 * ∇²c atol=1e-20

        # Mass conservation: with ∇²c = 0, no change
        @test wbm_concentration_rhs(model, 0.5, 0.0) ≈ 0.0 atol=1e-20
    end

    @testset "Interface properties" begin
        model = WBMModel(M_φ=1.0, κ=4.0, W=1.0, D_s=1e-13, D_l=1e-9)

        # Interface width: δ ≈ √(κ/W) = √4 = 2
        @test wbm_interface_width(model) ≈ 2.0 atol=1e-10

        # Interface energy: σ ≈ √(κ·W) / (6√2) = 2 / (6√2) ≈ 0.2357
        @test wbm_interface_energy(model) ≈ 2.0 / (6 * sqrt(2)) atol=1e-10

        # Edge case: W = 0
        model_zero = WBMModel(M_φ=1.0, κ=1.0, W=0.0, D_s=1e-13, D_l=1e-9)
        @test wbm_interface_width(model_zero) == Inf
        @test wbm_interface_energy(model_zero) == 0.0
    end

    @testset "AD compatibility" begin
        using ForwardDiff

        model = WBMModel(M_φ=1.0, κ=1.0, W=100.0, D_s=1e-13, D_l=1e-9)
        f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.2)
        f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.8)

        # Test derivatives through ForwardDiff
        φ_test = 0.5
        c_test = 0.4

        # d(bulk_free_energy)/dφ
        df_dφ = ForwardDiff.derivative(φ -> wbm_bulk_free_energy(f_s, f_l, φ, c_test, model.W), φ_test)
        @test isfinite(df_dφ)
        # This should equal wbm_driving_force
        @test df_dφ ≈ wbm_driving_force(f_s, f_l, φ_test, c_test, model.W) atol=1e-8

        # d(bulk_free_energy)/dc
        df_dc = ForwardDiff.derivative(c -> wbm_bulk_free_energy(f_s, f_l, φ_test, c, model.W), c_test)
        @test isfinite(df_dc)
        # This should equal wbm_chemical_potential
        @test df_dc ≈ wbm_chemical_potential(f_s, f_l, φ_test, c_test) atol=1e-8

        # d(phase_rhs)/dφ
        drhs_dφ = ForwardDiff.derivative(
            φ -> wbm_phase_rhs(model, φ, 0.0, c_test, f_s, f_l),
            φ_test
        )
        @test isfinite(drhs_dφ)

        # d(diffusivity)/dφ
        dD_dφ = ForwardDiff.derivative(φ -> wbm_diffusivity(model, φ), φ_test)
        @test isfinite(dD_dφ)
    end

end
