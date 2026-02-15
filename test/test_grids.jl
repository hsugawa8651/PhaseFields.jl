# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Tests for grid types and interfaces

using Test
using PhaseFields

@testset "Grid Types and Interfaces" begin

    # -----------------------------------------------------------------
    # Test 1: Grid interface functions
    # -----------------------------------------------------------------
    @testset "Grid interface functions" begin
        grid = UniformGrid1D(N=10, L=2.0)

        # gridsize returns tuple of grid dimensions
        @test gridsize(grid) == (10,)

        # spacing returns tuple of grid spacings
        sp = spacing(grid)
        @test length(sp) == 1
        @test sp[1] ≈ 2.0 / 9  # L / (N-1)

        # coordinates returns tuple of coordinate arrays
        coords = coordinates(grid)
        @test length(coords) == 1
        @test length(coords[1]) == 10
        @test coords[1][1] ≈ 0.0
        @test coords[1][end] ≈ 2.0
    end

    # -----------------------------------------------------------------
    # Test 2: Generic laplacian! interface
    # -----------------------------------------------------------------
    @testset "Generic laplacian! interface" begin
        grid = UniformGrid1D(N=20, L=1.0)
        u = sin.(π .* grid.x)  # sin(πx)
        ∇²u = similar(u)

        # Generic laplacian! should work with grid
        laplacian!(∇²u, u, grid, NeumannBC())

        # Result should match laplacian_1d!
        ∇²u_direct = similar(u)
        laplacian_1d!(∇²u_direct, u, grid.dx, NeumannBC())
        @test ∇²u ≈ ∇²u_direct
    end

    # -----------------------------------------------------------------
    # Test 3: Fallback error for unsupported domain
    # -----------------------------------------------------------------
    @testset "laplacian! fallback error" begin
        struct UnsupportedTestGrid <: AbstractDomain{4} end  # 4D grid

        grid = UnsupportedTestGrid()
        u = zeros(10)
        ∇²u = similar(u)

        # Should throw error for unsupported domain type
        @test_throws ErrorException laplacian!(∇²u, u, grid, NeumannBC())
    end

    # -----------------------------------------------------------------
    # Test 4: laplacian! with different boundary conditions
    # -----------------------------------------------------------------
    @testset "laplacian! with different BCs" begin
        grid = UniformGrid1D(N=20, L=1.0)
        u = rand(grid.N)
        ∇²u = similar(u)

        # Neumann BC
        laplacian!(∇²u, u, grid, NeumannBC())
        @test all(isfinite.(∇²u))

        # Dirichlet BC
        laplacian!(∇²u, u, grid, DirichletBC(0.0, 0.0))
        @test all(isfinite.(∇²u))

        # Periodic BC
        laplacian!(∇²u, u, grid, PeriodicBC())
        @test all(isfinite.(∇²u))
    end

end
