using Test
using PhaseFields

@testset "Thermal Phase Field Model" begin

    @testset "ThermalPhaseFieldModel constructor" begin
        # Basic constructor with all parameters
        model = ThermalPhaseFieldModel(
            τ = 1.0,
            W = 1.0,
            λ = 1.0,
            α = 1e-5,
            L = 2.35e9,
            Cp = 5.42e6,
            Tm = 1728.0
        )
        @test model.τ == 1.0
        @test model.W == 1.0
        @test model.λ == 1.0
        @test model.α == 1e-5
        @test model.L == 2.35e9
        @test model.Cp == 5.42e6
        @test model.Tm == 1728.0

        # Derived parameter K = L/Cp
        @test model.K ≈ 2.35e9 / 5.42e6 atol=1e-6

        # All parameters required
        @test_throws UndefKeywordError ThermalPhaseFieldModel(τ=1.0)
    end

    @testset "Dimensionless temperature conversion" begin
        Tm = 1728.0  # K (Nickel melting point)
        L = 2.35e9   # J/m³
        Cp = 5.42e6  # J/(m³·K)

        # At melting point: u = 0
        u = dimensionless_temperature(Tm, Tm, L, Cp)
        @test u ≈ 0.0 atol=1e-10

        # Below melting point (undercooling): u < 0
        T_under = Tm - 10.0  # 10K undercooling
        u_under = dimensionless_temperature(T_under, Tm, L, Cp)
        @test u_under < 0

        # Above melting point (superheating): u > 0
        T_super = Tm + 10.0  # 10K superheating
        u_super = dimensionless_temperature(T_super, Tm, L, Cp)
        @test u_super > 0

        # Round-trip conversion
        T_original = 1700.0
        u = dimensionless_temperature(T_original, Tm, L, Cp)
        T_back = physical_temperature(u, Tm, L, Cp)
        @test T_back ≈ T_original atol=1e-10
    end

    @testset "Stefan number" begin
        L = 2.35e9   # J/m³
        Cp = 5.42e6  # J/(m³·K)

        # St = Cp·ΔT / L
        ΔT = 100.0  # 100K undercooling
        St = stefan_number(ΔT, L, Cp)
        expected = Cp * ΔT / L
        @test St ≈ expected atol=1e-10

        # Small Stefan number means latent heat dominates
        ΔT_small = 10.0
        St_small = stefan_number(ΔT_small, L, Cp)
        @test St_small < 0.1

        # Zero undercooling gives zero Stefan number
        @test stefan_number(0.0, L, Cp) ≈ 0.0 atol=1e-10
    end

    @testset "Phase field RHS with thermal driving" begin
        model = ThermalPhaseFieldModel(
            τ = 1.0,
            W = 1.0,
            λ = 1.0,
            α = 1e-5,
            L = 2.35e9,
            Cp = 5.42e6,
            Tm = 1728.0
        )

        # At φ=0 (liquid), no undercooling (u=0), no Laplacian
        rhs_0 = thermal_phase_rhs(model, 0.0, 0.0, 0.0)
        # g'(0) = 0, h'(0) = 0, so RHS should be 0
        @test rhs_0 ≈ 0.0 atol=1e-10

        # At φ=1 (solid), no undercooling (u=0), no Laplacian
        rhs_1 = thermal_phase_rhs(model, 1.0, 0.0, 0.0)
        # g'(1) = 0, h'(1) = 0, so RHS should be 0
        @test rhs_1 ≈ 0.0 atol=1e-10

        # At interface (φ=0.5) with undercooling (u<0)
        # Undercooling should drive solidification (increase φ)
        u_undercool = -0.1
        rhs_under = thermal_phase_rhs(model, 0.5, 0.0, u_undercool)
        @test rhs_under > 0  # φ should increase (solidify)

        # At interface (φ=0.5) with superheating (u>0)
        # Superheating should drive melting (decrease φ)
        u_super = 0.1
        rhs_super = thermal_phase_rhs(model, 0.5, 0.0, u_super)
        @test rhs_super < 0  # φ should decrease (melt)

        # Positive Laplacian (curvature effect) promotes growth
        rhs_curv = thermal_phase_rhs(model, 0.5, 1.0, 0.0)
        @test rhs_curv > 0  # Curvature promotes solidification
    end

    @testset "Heat equation RHS with latent heat" begin
        model = ThermalPhaseFieldModel(
            τ = 1.0,
            W = 1.0,
            λ = 1.0,
            α = 1e-5,
            L = 2.35e9,
            Cp = 5.42e6,
            Tm = 1728.0
        )

        # No phase change (dφdt=0), no Laplacian: no temperature change
        rhs_0 = thermal_heat_rhs(model, 0.0, 0.0, 0.0)
        @test rhs_0 ≈ 0.0 atol=1e-10

        # Solidification (dφdt > 0) releases latent heat (u increases)
        dφdt_solidify = 0.1
        rhs_solidify = thermal_heat_rhs(model, 0.0, 0.0, dφdt_solidify)
        @test rhs_solidify > 0  # Temperature increases

        # Melting (dφdt < 0) absorbs latent heat (u decreases)
        dφdt_melt = -0.1
        rhs_melt = thermal_heat_rhs(model, 0.0, 0.0, dφdt_melt)
        @test rhs_melt < 0  # Temperature decreases

        # Thermal diffusion (positive Laplacian) increases temperature
        rhs_diff = thermal_heat_rhs(model, 0.0, 1.0, 0.0)
        @test rhs_diff > 0
    end

    @testset "Stability time step" begin
        model = ThermalPhaseFieldModel(
            τ = 1.0,
            W = 1.0,
            λ = 1.0,
            α = 1e-5,
            L = 2.35e9,
            Cp = 5.42e6,
            Tm = 1728.0
        )

        dx = 1e-6  # 1 μm grid spacing

        dt = thermal_stability_dt(model, dx)

        # Should be positive
        @test dt > 0

        # Should satisfy thermal diffusion stability: dt < dx²/(2α)
        dt_thermal_limit = dx^2 / (2 * model.α)
        @test dt < dt_thermal_limit

        # Should satisfy phase field stability: dt < τ·dx²/(2W²)
        dt_phase_limit = model.τ * dx^2 / (2 * model.W^2)
        @test dt < dt_phase_limit

        # Smaller dx should give smaller dt
        dt_small = thermal_stability_dt(model, dx / 2)
        @test dt_small < dt
    end

    @testset "Energy conservation" begin
        model = ThermalPhaseFieldModel(
            τ = 1.0,
            W = 1.0,
            λ = 1.0,
            α = 1e-5,
            L = 2.35e9,
            Cp = 5.42e6,
            Tm = 1728.0
        )

        # Total enthalpy: H = Cp*T + L*φ = Cp*Tm*(1 + u*L/(Cp*Tm)) + L*φ
        # In dimensionless form: h = u + φ (normalized by L/Cp)

        # For adiabatic system, total enthalpy should be conserved
        # d/dt(∫[u + φ/2] dV) = 0 for adiabatic BC

        # Test: latent heat release equals temperature increase
        # When dφdt > 0 (solidification), du/dt from latent heat = (1/2)*dφdt
        dφdt = 0.1
        du_latent = 0.5 * dφdt  # Latent heat contribution
        rhs_u = thermal_heat_rhs(model, 0.0, 0.0, dφdt)

        # The latent heat source term should be (1/2)*dφdt
        # (factor 1/2 comes from h(φ) normalization where solid fraction = h(φ))
        @test rhs_u ≈ du_latent atol=1e-10
    end

    @testset "AD compatibility" begin
        using ForwardDiff

        model = ThermalPhaseFieldModel(
            τ = 1.0,
            W = 1.0,
            λ = 1.0,
            α = 1e-5,
            L = 2.35e9,
            Cp = 5.42e6,
            Tm = 1728.0
        )

        φ_test = 0.5
        u_test = -0.1
        ∇²φ_test = 0.1

        # d(phase_rhs)/dφ
        drhs_dφ = ForwardDiff.derivative(
            φ -> thermal_phase_rhs(model, φ, ∇²φ_test, u_test),
            φ_test
        )
        @test isfinite(drhs_dφ)

        # d(phase_rhs)/du
        drhs_du = ForwardDiff.derivative(
            u -> thermal_phase_rhs(model, φ_test, ∇²φ_test, u),
            u_test
        )
        @test isfinite(drhs_du)

        # d(heat_rhs)/du
        dheat_du = ForwardDiff.derivative(
            u -> thermal_heat_rhs(model, u, 0.1, 0.1),
            u_test
        )
        @test isfinite(dheat_du)

        # d(dimensionless_temperature)/dT
        ddu_dT = ForwardDiff.derivative(
            T -> dimensionless_temperature(T, 1728.0, 2.35e9, 5.42e6),
            1700.0
        )
        @test isfinite(ddu_dT)
        @test ddu_dT ≈ 5.42e6 / 2.35e9 atol=1e-10  # Cp/L
    end

end
