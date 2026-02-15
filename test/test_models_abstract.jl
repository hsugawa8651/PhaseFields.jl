# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Tests for abstract model types

using Test
using PhaseFields

@testset "Abstract Model Types" begin

    # -----------------------------------------------------------------
    # Test 1: AbstractPhaseFieldModel exists
    # -----------------------------------------------------------------
    @testset "AbstractPhaseFieldModel exists" begin
        @test isdefined(PhaseFields, :AbstractPhaseFieldModel)
        @test AbstractPhaseFieldModel isa DataType
    end

    # -----------------------------------------------------------------
    # Test 2: Concrete models inherit from AbstractPhaseFieldModel
    # -----------------------------------------------------------------
    @testset "Model inheritance" begin
        @test AllenCahnModel <: AbstractPhaseFieldModel
        @test ThermalPhaseFieldModel <: AbstractPhaseFieldModel
        @test CahnHilliardModel <: AbstractPhaseFieldModel
        @test KKSModel <: AbstractPhaseFieldModel
        @test WBMModel <: AbstractPhaseFieldModel
    end

    # -----------------------------------------------------------------
    # Test 3: Error fallback for unimplemented model
    # -----------------------------------------------------------------
    @testset "Error fallback for unimplemented model" begin
        struct TestUnimplementedModel <: AbstractPhaseFieldModel end

        model = TestUnimplementedModel()
        φ = 0.5
        ∇²φ = 0.1

        @test_throws ErrorException PhaseFields.model_rhs(model, φ, ∇²φ)
    end

    # -----------------------------------------------------------------
    # Test 4: Implemented model_rhs works
    # -----------------------------------------------------------------
    @testset "Implemented model_rhs" begin
        model = AllenCahnModel(τ=1.0, W=1.0, m=1.0)
        φ = 0.5
        ∇²φ = 0.1
        ΔG = 0.0

        # Should not throw
        rhs = allen_cahn_rhs(model, φ, ∇²φ, ΔG)
        @test isa(rhs, Float64)
        @test isfinite(rhs)
    end

    # -----------------------------------------------------------------
    # Test 5: @kwdef default values for AllenCahnModel
    # -----------------------------------------------------------------
    @testset "@kwdef default values - AllenCahnModel" begin
        # All defaults
        model = AllenCahnModel()
        @test model.τ == 1.0
        @test model.W == 1.0
        @test model.m == 1.0

        # Partial specification
        model2 = AllenCahnModel(τ=2.0)
        @test model2.τ == 2.0
        @test model2.W == 1.0  # default
        @test model2.m == 1.0  # default

        # Full specification still works
        model3 = AllenCahnModel(τ=3.0, W=0.5, m=0.1)
        @test model3.τ == 3.0
        @test model3.W == 0.5
        @test model3.m == 0.1
    end

    # -----------------------------------------------------------------
    # Test 6: @kwdef default values for other models
    # -----------------------------------------------------------------
    @testset "@kwdef default values - CahnHilliardModel" begin
        model = CahnHilliardModel()
        @test model.M == 1.0
        @test model.κ == 1.0
    end

    @testset "@kwdef default values - KKSModel" begin
        model = KKSModel()
        @test model.τ == 1.0
        @test model.W == 1.0
        @test model.m == 1.0
        @test model.M_s == 1.0
        @test model.M_l == 1.0
    end

    @testset "@kwdef default values - WBMModel" begin
        model = WBMModel()
        @test model.M_φ == 1.0
        @test model.κ == 1.0
        @test model.W == 1.0
        @test model.D_s == 1.0
        @test model.D_l == 1.0
    end

end
