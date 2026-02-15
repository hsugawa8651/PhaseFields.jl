# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Tests for 2D grid types

using Test
using PhaseFields

@testset "2D Grid Types" begin

    # -----------------------------------------------------------------
    # Test 1: UniformGrid2D type exists and inherits
    # -----------------------------------------------------------------
    @testset "UniformGrid2D type" begin
        @test isdefined(PhaseFields, :UniformGrid2D)
        @test UniformGrid2D <: AbstractDomain{2}
    end

    # -----------------------------------------------------------------
    # Test 2: UniformGrid2D constructor
    # -----------------------------------------------------------------
    @testset "UniformGrid2D constructor" begin
        grid = UniformGrid2D(Nx=10, Ny=20, Lx=1.0, Ly=2.0)

        @test grid.Nx == 10
        @test grid.Ny == 20
        @test grid.Lx == 1.0
        @test grid.Ly == 2.0
        @test grid.dx ≈ 1.0 / 9  # Lx / (Nx - 1)
        @test grid.dy ≈ 2.0 / 19 # Ly / (Ny - 1)
        @test length(grid.x) == 10
        @test length(grid.y) == 20
        @test grid.x[1] ≈ 0.0
        @test grid.x[end] ≈ 1.0
        @test grid.y[1] ≈ 0.0
        @test grid.y[end] ≈ 2.0
    end

    # -----------------------------------------------------------------
    # Test 3: Grid interface functions for 2D
    # -----------------------------------------------------------------
    @testset "Grid interface functions 2D" begin
        grid = UniformGrid2D(Nx=10, Ny=15, Lx=1.0, Ly=1.5)

        @test gridsize(grid) == (10, 15)

        sp = spacing(grid)
        @test length(sp) == 2
        @test sp[1] ≈ 1.0 / 9
        @test sp[2] ≈ 1.5 / 14

        coords = coordinates(grid)
        @test length(coords) == 2
        @test length(coords[1]) == 10
        @test length(coords[2]) == 15
    end

    # -----------------------------------------------------------------
    # Test 4: laplacian_2d! Neumann BC
    # -----------------------------------------------------------------
    @testset "laplacian_2d! Neumann BC" begin
        grid = UniformGrid2D(Nx=20, Ny=20, Lx=1.0, Ly=1.0)

        # Test function: sin(πx)sin(πy)
        # ∇²f = -2π²sin(πx)sin(πy)
        u = [sin(π * x) * sin(π * y) for x in grid.x, y in grid.y]
        ∇²u = similar(u)

        laplacian_2d!(∇²u, u, grid.dx, grid.dy, NeumannBC())

        # Check interior point (away from boundaries)
        i, j = 10, 10
        expected = -2 * π^2 * sin(π * grid.x[i]) * sin(π * grid.y[j])
        @test ∇²u[i, j] ≈ expected rtol=0.1
    end

    # -----------------------------------------------------------------
    # Test 5: Generic laplacian! for 2D
    # -----------------------------------------------------------------
    @testset "Generic laplacian! 2D" begin
        grid = UniformGrid2D(Nx=15, Ny=15, Lx=1.0, Ly=1.0)
        u = rand(grid.Nx, grid.Ny)
        ∇²u = similar(u)

        # Generic interface should work
        laplacian!(∇²u, u, grid, NeumannBC())

        # Compare with direct call
        ∇²u_direct = similar(u)
        laplacian_2d!(∇²u_direct, u, grid.dx, grid.dy, NeumannBC())

        @test ∇²u ≈ ∇²u_direct
    end

    # -----------------------------------------------------------------
    # Test 6: laplacian_2d! Dirichlet BC
    # -----------------------------------------------------------------
    @testset "laplacian_2d! Dirichlet BC" begin
        grid = UniformGrid2D(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
        u = ones(grid.Nx, grid.Ny)
        ∇²u = similar(u)

        # Dirichlet BC with all boundaries = 0
        bc = DirichletBC2D(0.0, 0.0, 0.0, 0.0)
        laplacian_2d!(∇²u, u, grid.dx, grid.dy, bc)

        @test all(isfinite.(∇²u))
    end

    # -----------------------------------------------------------------
    # Test 7: laplacian_2d! Periodic BC
    # -----------------------------------------------------------------
    @testset "laplacian_2d! Periodic BC" begin
        grid = UniformGrid2D(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
        u = rand(grid.Nx, grid.Ny)
        ∇²u = similar(u)

        laplacian_2d!(∇²u, u, grid.dx, grid.dy, PeriodicBC())

        @test all(isfinite.(∇²u))
    end

end
