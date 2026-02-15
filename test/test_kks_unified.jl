# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - KKS unified solve tests

using Test
using PhaseFields
using OrdinaryDiffEq

@testset "KKS Unified Solve" begin
    # Model parameters
    f_s = ParabolicFreeEnergy(A=1000.0, c_eq=0.1)
    f_l = ParabolicFreeEnergy(A=1000.0, c_eq=0.9)
    model = KKSModel(τ=1.0, W=0.05, m=1.0, M_s=0.1, M_l=1.0)

    @testset "KKSODEParams 1D" begin
        grid = UniformGrid1D(N=50, L=1.0)
        bc = NeumannBC()

        params = PhaseFields.KKSODEParams(model, grid, bc, bc, f_s, f_l)
        @test params.model === model
        @test params.grid === grid
        @test size(params.∇²φ) == (50,)
        @test size(params.∇²μ) == (50,)
        @test size(params.μ) == (50,)
    end

    @testset "KKSODEParams 2D" begin
        grid = UniformGrid2D(Nx=30, Ny=30, Lx=1.0, Ly=1.0)
        bc = NeumannBC()

        params = PhaseFields.KKSODEParams(model, grid, bc, bc, f_s, f_l)
        @test params.model === model
        @test params.grid === grid
        @test size(params.∇²φ) == (30, 30)
        @test size(params.∇²μ) == (30, 30)
        @test size(params.μ) == (30, 30)
    end

    @testset "kks_ode! 1D" begin
        grid = UniformGrid1D(N=50, L=1.0)
        bc = NeumannBC()
        params = PhaseFields.KKSODEParams(model, grid, bc, bc, f_s, f_l)

        N = 50
        # Initial: solid seed with concentration c = 0.5
        φ0 = [x < 0.3 ? 1.0 : 0.0 for x in grid.x]
        c0 = 0.5 * ones(N)
        y = vcat(φ0, c0)
        dy = similar(y)

        PhaseFields.kks_ode!(dy, y, params, 0.0)

        @test length(dy) == 2N
        @test !any(isnan, dy)
        @test !any(isinf, dy)
    end

    @testset "kks_ode! 2D" begin
        grid = UniformGrid2D(Nx=30, Ny=30, Lx=1.0, Ly=1.0)
        bc = NeumannBC()
        params = PhaseFields.KKSODEParams(model, grid, bc, bc, f_s, f_l)

        Nx, Ny = 30, 30
        N_total = Nx * Ny
        # Initial: solid seed
        φ0 = [sqrt((x-0.5)^2 + (y-0.5)^2) < 0.2 ? 1.0 : 0.0 for x in grid.x, y in grid.y]
        c0 = 0.5 * ones(Nx, Ny)
        y = vcat(vec(φ0), vec(c0))
        dy = similar(y)

        PhaseFields.kks_ode!(dy, y, params, 0.0)

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

        problem = KKSProblem(model, grid, φ0, c0, (0.0, 0.01), f_s, f_l)

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

        problem = KKSProblem(model, grid, φ0, c0, (0.0, 0.01), f_s, f_l)

        sol = PhaseFields.solve(problem, Tsit5(), saveat=0.005)
        @test length(sol.t) > 1
        @test length(sol.u[end]) == 2N_total
    end

    @testset "Phase field bounds" begin
        grid = UniformGrid1D(N=100, L=1.0)
        N = 100

        # Initial: step function
        φ0 = [x < 0.5 ? 1.0 : 0.0 for x in grid.x]
        c0 = 0.5 * ones(N)

        problem = KKSProblem(model, grid, φ0, c0, (0.0, 0.1), f_s, f_l)
        sol = PhaseFields.solve(problem, Tsit5(), saveat=0.05)

        # Phase field should stay reasonably bounded
        for y in sol.u
            φ = y[1:N]
            @test all(φ .>= -0.1)  # Allow small numerical error
            @test all(φ .<= 1.1)
        end
    end

    @testset "Solution stability" begin
        grid = UniformGrid1D(N=100, L=1.0)
        N = 100

        # Initial conditions - use smooth interface
        φ0 = [0.5 * (1 - tanh((x - 0.5) / 0.05)) for x in grid.x]
        c0 = 0.5 * ones(N)

        # Short simulation
        problem = KKSProblem(model, grid, φ0, c0, (0.0, 0.001), f_s, f_l)
        sol = PhaseFields.solve(problem, Tsit5())

        # Solution should not blow up (no NaN/Inf)
        # Note: KKS with simplified M∇²μ may have overshoots
        c_final = sol.u[end][N+1:2N]
        @test !any(isnan, c_final)
        @test !any(isinf, c_final)
        @test all(isfinite, c_final)
    end
end
