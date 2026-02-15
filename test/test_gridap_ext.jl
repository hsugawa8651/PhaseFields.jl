# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Tests for Gridap Extension

using Test
using PhaseFields

# Check if Gridap is available
gridap_available = try
    @eval using Gridap
    true
catch
    false
end

if gridap_available
    @testset "Gridap Extension" begin

        # -----------------------------------------------------------------
        # Test 1: GridapDomain type exists
        # -----------------------------------------------------------------
        @testset "GridapDomain type" begin
            @test isdefined(PhaseFields, :GridapDomain)
        end

        # -----------------------------------------------------------------
        # Test 2: GridapDomain constructor (2D Cartesian)
        # -----------------------------------------------------------------
        @testset "GridapDomain 2D constructor" begin
            domain = GridapDomain((0, 1, 0, 1), (10, 10))

            @test domain isa AbstractDomain{2}
            @test domain.order == 1
            @test domain.model isa Gridap.CartesianDiscreteModel
        end

        # -----------------------------------------------------------------
        # Test 3: Allen-Cahn FEM solve
        # -----------------------------------------------------------------
        @testset "Allen-Cahn FEM solve" begin
            using OrdinaryDiffEq

            model = AllenCahnModel(τ=1.0, W=0.1, m=1.0)
            domain = GridapDomain((0, 1, 0, 1), (20, 20))

            # Initial condition: circle at center
            φ0(x) = 0.5 * (1 - tanh((sqrt((x[1]-0.5)^2 + (x[2]-0.5)^2) - 0.2) / 0.05))

            problem = PhaseFieldProblem(
                model=model,
                domain=domain,
                φ0=φ0,
                tspan=(0.0, 0.1)
            )

            # Solve using FEM
            sol = PhaseFields.solve(problem, :newton; Δt=0.01)

            # Check solution properties
            @test length(sol.times) > 1
            @test sol.times[1] == 0.0
            @test sol.times[end] ≈ 0.1
        end

    end
else
    @warn "Gridap.jl not available - skipping GridapExt tests"
end
