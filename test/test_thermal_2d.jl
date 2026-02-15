# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Tests for Thermal 2D ODE and unified solve

using Test
using PhaseFields
using OrdinaryDiffEq

@testset "Thermal 2D" begin

    # -----------------------------------------------------------------
    # Test 1: ThermalProblem convenience constructor
    # -----------------------------------------------------------------
    @testset "ThermalProblem constructor" begin
        model = ThermalPhaseFieldModel(τ=1.0, W=0.1, λ=1.0, α=1.0, L=1.0, Cp=1.0, Tm=1.0)
        grid = UniformGrid1D(N=20, L=1.0)

        φ0 = [x < 0.5 ? 1.0 : 0.0 for x in grid.x]
        u0 = zeros(grid.N)

        problem = ThermalProblem(model, grid, φ0, u0, (0.0, 0.1))

        @test problem isa PhaseFieldProblem
        @test problem.model === model
        @test problem.domain === grid
    end

    # -----------------------------------------------------------------
    # Test 2: Unified solve for Thermal 1D
    # -----------------------------------------------------------------
    @testset "Thermal 1D unified solve" begin
        model = ThermalPhaseFieldModel(τ=1.0, W=0.1, λ=1.0, α=1.0, L=1.0, Cp=1.0, Tm=1.0)
        grid = UniformGrid1D(N=20, L=1.0)

        φ0 = [x < 0.5 ? 1.0 : 0.0 for x in grid.x]
        u0 = -0.5 * ones(grid.N)  # undercooling

        problem = ThermalProblem(model, grid, φ0, u0, (0.0, 0.1))
        sol = PhaseFields.solve(problem, Tsit5())

        @test sol isa OrdinaryDiffEq.ODESolution
        @test length(sol.t) > 1
        @test length(sol.u[end]) == 2 * grid.N  # [φ; u]
    end

    # -----------------------------------------------------------------
    # Test 3: ThermalODEParams 2D constructor
    # -----------------------------------------------------------------
    @testset "ThermalODEParams 2D" begin
        model = ThermalPhaseFieldModel(τ=1.0, W=0.1, λ=1.0, α=1.0, L=1.0, Cp=1.0, Tm=1.0)
        grid = UniformGrid2D(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
        bc_φ = NeumannBC()
        bc_u = NeumannBC()

        params = ThermalODEParams(model, grid, bc_φ, bc_u)

        @test params.model === model
        @test params.grid === grid
        @test size(params.∇²φ) == (10, 10)
        @test size(params.∇²u) == (10, 10)
    end

    # -----------------------------------------------------------------
    # Test 4: thermal_ode! 2D
    # -----------------------------------------------------------------
    @testset "thermal_ode! 2D" begin
        model = ThermalPhaseFieldModel(τ=1.0, W=0.1, λ=1.0, α=1.0, L=1.0, Cp=1.0, Tm=1.0)
        grid = UniformGrid2D(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
        bc_φ = NeumannBC()
        bc_u = NeumannBC()
        params = ThermalODEParams(model, grid, bc_φ, bc_u)

        N_total = grid.Nx * grid.Ny
        y = vcat(0.5 * ones(N_total), zeros(N_total))  # [φ; u]
        dy = similar(y)

        thermal_ode!(dy, y, params, 0.0)

        @test length(dy) == 2 * N_total
        @test all(isfinite.(dy))
    end

    # -----------------------------------------------------------------
    # Test 5: Unified solve for Thermal 2D
    # -----------------------------------------------------------------
    @testset "Thermal 2D unified solve" begin
        model = ThermalPhaseFieldModel(τ=1.0, W=0.1, λ=1.0, α=1.0, L=1.0, Cp=1.0, Tm=1.0)
        grid = UniformGrid2D(Nx=20, Ny=20, Lx=1.0, Ly=1.0)

        # Circular solid seed
        φ0 = [sqrt((x - 0.5)^2 + (y - 0.5)^2) < 0.2 ? 1.0 : 0.0
              for x in grid.x, y in grid.y]
        u0 = -0.5 * ones(grid.Nx, grid.Ny)  # undercooling

        problem = ThermalProblem(model, grid, φ0, u0, (0.0, 0.01))
        sol = PhaseFields.solve(problem, Tsit5())

        @test sol isa OrdinaryDiffEq.ODESolution
        @test length(sol.t) > 1

        N_total = grid.Nx * grid.Ny
        @test length(sol.u[end]) == 2 * N_total

        # Extract and reshape final fields
        φ_final = reshape(sol.u[end][1:N_total], grid.Nx, grid.Ny)
        u_final = reshape(sol.u[end][N_total+1:end], grid.Nx, grid.Ny)

        @test size(φ_final) == (grid.Nx, grid.Ny)
        @test size(u_final) == (grid.Nx, grid.Ny)
    end

    # -----------------------------------------------------------------
    # Test 6: Physical behavior - latent heat release
    # -----------------------------------------------------------------
    @testset "Latent heat release" begin
        model = ThermalPhaseFieldModel(τ=1.0, W=0.05, λ=2.0, α=1.0, L=1.0, Cp=1.0, Tm=1.0)
        grid = UniformGrid2D(Nx=30, Ny=30, Lx=1.0, Ly=1.0)

        # Circular solid seed
        φ0 = [sqrt((x - 0.5)^2 + (y - 0.5)^2) < 0.15 ? 1.0 : 0.0
              for x in grid.x, y in grid.y]
        u0 = -0.3 * ones(grid.Nx, grid.Ny)  # undercooling

        problem = ThermalProblem(model, grid, φ0, u0, (0.0, 0.1))
        sol = PhaseFields.solve(problem, Tsit5())

        N_total = grid.Nx * grid.Ny
        u_initial = reshape(sol.u[1][N_total+1:end], grid.Nx, grid.Ny)
        u_final = reshape(sol.u[end][N_total+1:end], grid.Nx, grid.Ny)

        # Temperature should increase due to latent heat release
        @test maximum(u_final) > maximum(u_initial)
    end

end
