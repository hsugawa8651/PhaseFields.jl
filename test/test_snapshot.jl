# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Tests for snapshot types

@testset "FieldSnapshot1D" begin
    grid = UniformGrid1D(N=50, L=10.0)
    φ = rand(50)

    # Single-field constructor
    snap = FieldSnapshot1D(grid, φ, 1.0; field_name=:φ, title="Test")
    @test snap isa FieldSnapshot1D
    @test snap.x == grid.x
    @test length(snap.fields) == 1
    @test snap.fields[1].first == :φ
    @test snap.fields[1].second ≈ φ
    @test snap.t == 1.0
    @test snap.title == "Test"

    # Multi-field constructor
    fields = Dict(:φ => rand(50), :c => rand(50))
    snap2 = FieldSnapshot1D(grid, fields, 2.0)
    @test length(snap2.fields) == 2

    # Size mismatch error
    @test_throws ArgumentError FieldSnapshot1D(grid, rand(30), 0.0)
end

@testset "SpaceTimeSnapshot1D" begin
    x = collect(range(0, 1, length=30))
    t = [0.0, 1.0, 2.0]
    data = rand(30, 3)

    snap = SpaceTimeSnapshot1D(x, t, data, :c)
    @test snap isa SpaceTimeSnapshot1D
    @test size(snap.data) == (30, 3)
    @test snap.field_name == :c
    @test snap.colormap == :RdBu

    # Size mismatch error
    @test_throws ArgumentError SpaceTimeSnapshot1D(x, t, rand(20, 3), :c)
end

@testset "FieldSnapshot2D" begin
    grid = UniformGrid2D(Nx=20, Ny=20, Lx=1.0, Ly=1.0)
    field = rand(20, 20)

    snap = FieldSnapshot2D(grid, field, 0.5, :φ)
    @test snap isa FieldSnapshot2D
    @test size(snap.field) == (20, 20)
    @test snap.t == 0.5
    @test snap.field_name == :φ
    @test snap.colormap == :viridis

    # Size mismatch error
    @test_throws ArgumentError FieldSnapshot2D(grid, rand(10, 20), 0.0, :φ)
end

@testset "savefig_publication fallback" begin
    # Fallbacks for the 3-layer API when PythonPlot is not loaded.
    # This runs before test_publication.jl loads PythonPlot, so the extension
    # methods are absent and only the src fallbacks dispatch.
    @test_throws ArgumentError savefig_publication(nothing, "x")
    @test_throws ArgumentError PhaseFields.plot_on_axis!(nothing, nothing)
    @test_throws ArgumentError PhaseFields.figure_publication(nothing)
end
