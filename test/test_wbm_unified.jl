# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - WBM unified solve tests

using Test
using PhaseFields
using OrdinaryDiffEq

@testset "WBM Unified Solve" begin
    # Model parameters
    f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)
    f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)
    model = WBMModel(M_φ=1.0, κ=1.0, W=1.0, D_s=0.1, D_l=1.0)

    @testset "WBMODEParams 1D" begin
        grid = UniformGrid1D(N=50, L=1.0)
        bc = NeumannBC()

        params = PhaseFields.WBMODEParams(model, grid, bc, bc, f_s, f_l)
        @test params.model === model
        @test params.grid === grid
        @test size(params.∇²φ) == (50,)
        @test size(params.∇²c) == (50,)
    end

    @testset "WBMODEParams 2D" begin
        grid = UniformGrid2D(Nx=30, Ny=30, Lx=1.0, Ly=1.0)
        bc = NeumannBC()

        params = PhaseFields.WBMODEParams(model, grid, bc, bc, f_s, f_l)
        @test params.model === model
        @test params.grid === grid
        @test size(params.∇²φ) == (30, 30)
        @test size(params.∇²c) == (30, 30)
    end

    @testset "wbm_ode! 1D" begin
        grid = UniformGrid1D(N=50, L=1.0)
        bc = NeumannBC()
        params = PhaseFields.WBMODEParams(model, grid, bc, bc, f_s, f_l)

        N = 50
        # Initial: solid seed with concentration c = 0.5
        φ0 = [x < 0.3 ? 1.0 : 0.0 for x in grid.x]
        c0 = 0.5 * ones(N)
        y = vcat(φ0, c0)
        dy = similar(y)

        # Should not error
        PhaseFields.wbm_ode!(dy, y, params, 0.0)

        @test length(dy) == 2N
        @test !any(isnan, dy)
        @test !any(isinf, dy)
    end

    @testset "wbm_ode! 2D" begin
        grid = UniformGrid2D(Nx=30, Ny=30, Lx=1.0, Ly=1.0)
        bc = NeumannBC()
        params = PhaseFields.WBMODEParams(model, grid, bc, bc, f_s, f_l)

        Nx, Ny = 30, 30
        N_total = Nx * Ny
        # Initial: solid seed
        φ0 = [sqrt((x-0.5)^2 + (y-0.5)^2) < 0.2 ? 1.0 : 0.0 for x in grid.x, y in grid.y]
        c0 = 0.5 * ones(Nx, Ny)
        y = vcat(vec(φ0), vec(c0))
        dy = similar(y)

        PhaseFields.wbm_ode!(dy, y, params, 0.0)

        @test length(dy) == 2N_total
        @test !any(isnan, dy)
        @test !any(isinf, dy)
    end

    @testset "Unified solve 1D" begin
        grid = UniformGrid1D(N=50, L=1.0)
        N = 50

        # Initial conditions
        φ0 = [x < 0.3 ? 1.0 : 0.0 for x in grid.x]
        c0 = 0.5 * ones(N)

        problem = WBMProblem(model, grid, φ0, c0, (0.0, 0.01), f_s, f_l)

        sol = PhaseFields.solve(problem, Tsit5(), saveat=0.005)
        @test length(sol.t) > 1
        @test length(sol.u[end]) == 2N
    end

    @testset "Unified solve 2D" begin
        grid = UniformGrid2D(Nx=30, Ny=30, Lx=1.0, Ly=1.0)
        Nx, Ny = 30, 30
        N_total = Nx * Ny

        # Initial conditions
        φ0 = [sqrt((x-0.5)^2 + (y-0.5)^2) < 0.2 ? 1.0 : 0.0 for x in grid.x, y in grid.y]
        c0 = 0.5 * ones(Nx, Ny)

        problem = WBMProblem(model, grid, φ0, c0, (0.0, 0.01), f_s, f_l)

        sol = PhaseFields.solve(problem, Tsit5(), saveat=0.005)
        @test length(sol.t) > 1
        @test length(sol.u[end]) == 2N_total
    end

    @testset "Physics validation" begin
        grid = UniformGrid1D(N=100, L=1.0)
        N = 100

        # Initial: liquid (φ=0) with intermediate concentration
        φ0 = zeros(N)
        c0 = 0.5 * ones(N)

        problem = WBMProblem(model, grid, φ0, c0, (0.0, 0.1), f_s, f_l)
        sol = PhaseFields.solve(problem, Tsit5(), saveat=0.05)

        # Concentration should remain bounded [0, 1]
        for y in sol.u
            c = y[N+1:2N]
            @test all(c .>= -0.1)  # Allow small numerical error
            @test all(c .<= 1.1)
        end
    end
end
