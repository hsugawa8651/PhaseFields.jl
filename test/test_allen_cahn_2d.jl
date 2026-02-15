# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Tests for Allen-Cahn 2D ODE integration

using Test
using PhaseFields
using OrdinaryDiffEq

@testset "Allen-Cahn 2D ODE" begin

    # -----------------------------------------------------------------
    # Test 1: AllenCahnODEParams with UniformGrid2D
    # -----------------------------------------------------------------
    @testset "AllenCahnODEParams 2D constructor" begin
        model = AllenCahnModel(τ=1.0, W=0.1, m=1.0)
        grid = UniformGrid2D(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
        bc = NeumannBC()

        params = AllenCahnODEParams(model, grid, bc)

        @test params.model === model
        @test params.grid === grid
        @test params.bc === bc
        @test size(params.∇²φ) == (10, 10)
    end

    # -----------------------------------------------------------------
    # Test 2: allen_cahn_ode! with 2D array (vectorized)
    # -----------------------------------------------------------------
    @testset "allen_cahn_ode! 2D" begin
        model = AllenCahnModel(τ=1.0, W=0.1, m=1.0)
        grid = UniformGrid2D(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
        bc = NeumannBC()
        params = AllenCahnODEParams(model, grid, bc)

        # Initial condition: constant field
        φ = 0.5 * ones(grid.Nx, grid.Ny)
        φ_vec = vec(φ)
        dφ_vec = similar(φ_vec)

        # Should not error
        allen_cahn_ode!(dφ_vec, φ_vec, params, 0.0)

        @test length(dφ_vec) == grid.Nx * grid.Ny
        @test all(isfinite.(dφ_vec))
    end

    # -----------------------------------------------------------------
    # Test 3: create_allen_cahn_problem with 2D grid
    # -----------------------------------------------------------------
    @testset "create_allen_cahn_problem 2D" begin
        model = AllenCahnModel(τ=1.0, W=0.1, m=1.0)
        grid = UniformGrid2D(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
        bc = NeumannBC()

        # Initial condition: circular interface
        φ0 = [sqrt((x - 0.5)^2 + (y - 0.5)^2) < 0.3 ? 1.0 : 0.0
              for x in grid.x, y in grid.y]

        prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 0.1))

        @test prob isa OrdinaryDiffEq.ODEProblem
        @test length(prob.u0) == grid.Nx * grid.Ny
    end

    # -----------------------------------------------------------------
    # Test 4: Full solve with 2D grid
    # -----------------------------------------------------------------
    @testset "Allen-Cahn 2D solve" begin
        model = AllenCahnModel(τ=1.0, W=0.1, m=1.0)
        grid = UniformGrid2D(Nx=20, Ny=20, Lx=1.0, Ly=1.0)
        bc = NeumannBC()

        # Initial condition: circular interface
        φ0 = [sqrt((x - 0.5)^2 + (y - 0.5)^2) < 0.3 ? 1.0 : 0.0
              for x in grid.x, y in grid.y]

        prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 0.01))
        sol = OrdinaryDiffEq.solve(prob, Tsit5(), saveat=0.005)

        @test length(sol.t) > 1
        @test all(length.(sol.u) .== grid.Nx * grid.Ny)

        # Reshape final solution
        φ_final = reshape(sol.u[end], grid.Nx, grid.Ny)
        @test size(φ_final) == (grid.Nx, grid.Ny)
    end

    # -----------------------------------------------------------------
    # Test 5: Physical behavior - curvature-driven shrinkage
    # -----------------------------------------------------------------
    @testset "Curvature-driven shrinkage" begin
        model = AllenCahnModel(τ=1.0, W=0.05, m=1.0)
        grid = UniformGrid2D(Nx=50, Ny=50, Lx=1.0, Ly=1.0)
        bc = NeumannBC()

        # Initial condition: circular interface (solid inside)
        φ0 = [sqrt((x - 0.5)^2 + (y - 0.5)^2) < 0.3 ? 1.0 : 0.0
              for x in grid.x, y in grid.y]

        prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 0.5))
        sol = OrdinaryDiffEq.solve(prob, Tsit5())

        # Solid fraction should decrease (curvature drives shrinkage)
        φ_initial = reshape(sol.u[1], grid.Nx, grid.Ny)
        φ_final = reshape(sol.u[end], grid.Nx, grid.Ny)

        solid_initial = sum(φ_initial) / (grid.Nx * grid.Ny)
        solid_final = sum(φ_final) / (grid.Nx * grid.Ny)

        @test solid_final < solid_initial
    end

end
