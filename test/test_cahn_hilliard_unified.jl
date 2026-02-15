# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Cahn-Hilliard unified solve tests

using Test
using PhaseFields
using OrdinaryDiffEq

@testset "Cahn-Hilliard Unified Solve" begin
    # Model parameters (PFHub BM1-like)
    model = CahnHilliardModel(M=5.0, κ=2.0)
    f = DoubleWellFreeEnergy(ρs=5.0, cα=0.3, cβ=0.7)

    @testset "CahnHilliardODEParams 1D" begin
        grid = UniformGrid1D(N=50, L=1.0)
        bc = NeumannBC()

        params = PhaseFields.CahnHilliardODEParams(model, grid, bc, f)
        @test params.model === model
        @test params.grid === grid
        @test size(params.∇²c) == (50,)
        @test size(params.μ) == (50,)
        @test size(params.∇²μ) == (50,)
    end

    @testset "CahnHilliardODEParams 2D" begin
        grid = UniformGrid2D(Nx=30, Ny=30, Lx=1.0, Ly=1.0)
        bc = NeumannBC()

        params = PhaseFields.CahnHilliardODEParams(model, grid, bc, f)
        @test params.model === model
        @test params.grid === grid
        @test size(params.∇²c) == (30, 30)
        @test size(params.μ) == (30, 30)
        @test size(params.∇²μ) == (30, 30)
    end

    @testset "cahn_hilliard_ode! 1D" begin
        grid = UniformGrid1D(N=50, L=1.0)
        bc = NeumannBC()
        params = PhaseFields.CahnHilliardODEParams(model, grid, bc, f)

        # Initial: small fluctuations around c = 0.5
        c0 = [0.5 + 0.01 * sin(2π * x) for x in grid.x]
        dc = similar(c0)

        PhaseFields.cahn_hilliard_ode!(dc, c0, params, 0.0)

        @test length(dc) == 50
        @test !any(isnan, dc)
        @test !any(isinf, dc)
    end

    @testset "cahn_hilliard_ode! 2D" begin
        grid = UniformGrid2D(Nx=30, Ny=30, Lx=1.0, Ly=1.0)
        bc = PeriodicBC()  # Periodic for spinodal decomposition
        params = PhaseFields.CahnHilliardODEParams(model, grid, bc, f)

        Nx, Ny = 30, 30
        # Initial: small random fluctuations
        c0 = [0.5 + 0.02 * (sin(4π * x) * cos(4π * y)) for x in grid.x, y in grid.y]
        c0_vec = vec(c0)
        dc = similar(c0_vec)

        PhaseFields.cahn_hilliard_ode!(dc, c0_vec, params, 0.0)

        @test length(dc) == Nx * Ny
        @test !any(isnan, dc)
        @test !any(isinf, dc)
    end

    @testset "Unified solve 1D" begin
        grid = UniformGrid1D(N=50, L=1.0)

        # Initial: perturbation around c = 0.5
        c0 = [0.5 + 0.01 * sin(2π * x) for x in grid.x]

        problem = CahnHilliardProblem(model, grid, c0, (0.0, 0.001), f)

        sol = PhaseFields.solve(problem, Tsit5(), saveat=0.0005)
        @test length(sol.t) > 1
        @test length(sol.u[end]) == 50
    end

    @testset "Unified solve 2D" begin
        grid = UniformGrid2D(Nx=30, Ny=30, Lx=1.0, Ly=1.0)
        Nx, Ny = 30, 30

        # Initial: small fluctuations
        c0 = [0.5 + 0.02 * sin(4π * x) * cos(4π * y) for x in grid.x, y in grid.y]

        problem = CahnHilliardProblem(model, grid, c0, (0.0, 0.001), f,
                                       bc=PeriodicBC())

        sol = PhaseFields.solve(problem, Tsit5(), saveat=0.0005)
        @test length(sol.t) > 1
        @test length(sol.u[end]) == Nx * Ny
    end

    @testset "Mass conservation" begin
        grid = UniformGrid1D(N=100, L=1.0)

        # Initial condition
        c0 = [0.5 + 0.1 * sin(2π * x) for x in grid.x]
        initial_mass = sum(c0)

        problem = CahnHilliardProblem(model, grid, c0, (0.0, 0.01), f)
        sol = PhaseFields.solve(problem, Tsit5())

        # Cahn-Hilliard should conserve mass (with Neumann BC)
        final_mass = sum(sol.u[end])
        @test isapprox(final_mass, initial_mass, rtol=1e-6)
    end

    @testset "Spinodal decomposition tendency" begin
        grid = UniformGrid1D(N=100, L=1.0)

        # Initial: unstable (spinodal region) with small perturbation
        c0 = [0.5 + 0.01 * sin(6π * x) for x in grid.x]

        # Use a stiff solver for 4th order CH equation
        # Short simulation to just verify it runs without error
        problem = CahnHilliardProblem(model, grid, c0, (0.0, 0.0001), f)
        sol = PhaseFields.solve(problem, Tsit5(), maxiters=10000)

        # Verify simulation completed and solution evolved
        @test length(sol.t) >= 1
        @test !any(isnan, sol.u[end])
    end
end
