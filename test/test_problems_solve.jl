# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Integration tests for problem solve interface

using Test
using PhaseFields
using OrdinaryDiffEq

@testset "Problem Solve" begin

    # -----------------------------------------------------------------
    # Test 8: solve with FDM (UniformGrid1D)
    # -----------------------------------------------------------------
    @testset "solve with FDM 1D" begin
        model = AllenCahnModel(τ=1.0, W=0.1, m=1.0)
        grid = UniformGrid1D(N=20, L=1.0)

        # Initial condition: step function
        φ0 = [x < 0.5 ? 1.0 : 0.0 for x in grid.x]

        problem = PhaseFieldProblem(
            model=model,
            domain=grid,
            φ0=φ0,
            tspan=(0.0, 0.1)
        )

        # Solve using the unified interface
        sol = PhaseFields.solve(problem, Tsit5())

        # Check that solution exists and has expected properties
        @test sol isa OrdinaryDiffEq.ODESolution
        @test length(sol.t) > 1
        @test sol.t[1] == 0.0
        @test sol.t[end] ≈ 0.1
        @test length(sol.u[end]) == grid.N
    end

    # -----------------------------------------------------------------
    # Test 9: solve with FDM (UniformGrid2D)
    # -----------------------------------------------------------------
    @testset "solve with FDM 2D" begin
        model = AllenCahnModel(τ=1.0, W=0.1, m=1.0)
        grid = UniformGrid2D(Nx=20, Ny=20, Lx=1.0, Ly=1.0)

        # Initial condition: circular interface
        φ0 = [sqrt((x - 0.5)^2 + (y - 0.5)^2) < 0.3 ? 1.0 : 0.0
              for x in grid.x, y in grid.y]

        problem = PhaseFieldProblem(
            model=model,
            domain=grid,
            φ0=φ0,
            tspan=(0.0, 0.01)
        )

        # Solve using the unified interface
        sol = PhaseFields.solve(problem, Tsit5())

        # Check that solution exists and has expected properties
        @test sol isa OrdinaryDiffEq.ODESolution
        @test length(sol.t) > 1
        @test sol.t[1] == 0.0
        @test sol.t[end] ≈ 0.01
        @test length(sol.u[end]) == grid.Nx * grid.Ny

        # Reshape final solution
        φ_final = reshape(sol.u[end], grid.Nx, grid.Ny)
        @test size(φ_final) == (grid.Nx, grid.Ny)
    end

end
