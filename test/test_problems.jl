# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Tests for problem types

using Test
using PhaseFields

@testset "Problem Types" begin

    # -----------------------------------------------------------------
    # Test 1: AbstractPhaseFieldProblem exists
    # -----------------------------------------------------------------
    @testset "AbstractPhaseFieldProblem exists" begin
        @test isdefined(PhaseFields, :AbstractPhaseFieldProblem)
        @test AbstractPhaseFieldProblem isa DataType
    end

    # -----------------------------------------------------------------
    # Test 2: AbstractDomain exists
    # -----------------------------------------------------------------
    @testset "AbstractDomain exists" begin
        @test isdefined(PhaseFields, :AbstractDomain)
    end

    # -----------------------------------------------------------------
    # Test 3: UniformGrid1D <: AbstractDomain{1}
    # -----------------------------------------------------------------
    @testset "UniformGrid1D inheritance" begin
        @test UniformGrid1D <: AbstractDomain{1}

        grid = UniformGrid1D(N=10, L=1.0)
        @test grid isa AbstractDomain{1}
    end

    # -----------------------------------------------------------------
    # Test 4: PhaseFieldProblem exists and inherits
    # -----------------------------------------------------------------
    @testset "PhaseFieldProblem type" begin
        @test isdefined(PhaseFields, :PhaseFieldProblem)
        @test PhaseFieldProblem <: AbstractPhaseFieldProblem
    end

    # -----------------------------------------------------------------
    # Test 5: PhaseFieldProblem construction
    # -----------------------------------------------------------------
    @testset "PhaseFieldProblem construction" begin
        model = AllenCahnModel()
        grid = UniformGrid1D(N=10, L=1.0)
        φ0 = zeros(10)
        tspan = (0.0, 1.0)

        problem = PhaseFieldProblem(
            model=model,
            domain=grid,
            φ0=φ0,
            tspan=tspan
        )

        @test problem.model === model
        @test problem.domain === grid
        @test problem.φ0 === φ0
        @test problem.tspan == tspan
        @test problem.bc isa NeumannBC  # default
    end

    # -----------------------------------------------------------------
    # Test 6: Convenience constructors
    # -----------------------------------------------------------------
    @testset "AllenCahnProblem convenience constructor" begin
        model = AllenCahnModel()
        grid = UniformGrid1D(N=10, L=1.0)
        φ0 = zeros(10)
        tspan = (0.0, 1.0)

        problem = AllenCahnProblem(model, grid, φ0, tspan)

        @test problem isa PhaseFieldProblem
        @test problem.model === model
        @test problem.bc isa NeumannBC
    end

    # -----------------------------------------------------------------
    # Test 7: solve fallback error for unsupported domain
    # -----------------------------------------------------------------
    @testset "solve fallback error" begin
        struct UnsupportedTestDomain <: AbstractDomain{1} end

        model = AllenCahnModel()
        domain = UnsupportedTestDomain()
        φ0 = zeros(10)

        problem = PhaseFieldProblem(
            model=model,
            domain=domain,
            φ0=φ0,
            tspan=(0.0, 1.0)
        )

        @test_throws ErrorException PhaseFields.solve(problem, nothing)
    end

end
