# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - PythonPlot extension tests (3-layer plotting API)

using Test
using PhaseFields
using PythonPlot

# Force a non-interactive backend on CI (must be after `using PythonPlot`)
PythonPlot.matplotlib.use("Agg")

# Read a matplotlib title back as a Julia string (robust to Py wrapping)
_title(ax) = string(ax.get_title())

@testset "3-layer plotting (PythonPlot)" begin
    # ── Fixtures (verified constructors) ──
    g1 = UniformGrid1D(N = 32, L = 1.0)
    φ = sin.(2π .* g1.x)
    c = 0.5 .+ 0.5 .* φ
    s1_single = FieldSnapshot1D(g1, φ, 0.5; field_name = :φ, title = "1D single")
    # multi-field: raw Vector{Pair} constructor keeps field order deterministic
    s1_multi = FieldSnapshot1D(g1.x, Pair{Symbol,Vector{Float64}}[:φ => φ, :c => c],
                               0.5, "x", "1D multi")
    s1_notitle = FieldSnapshot1D(g1, φ, 0.5; field_name = :φ)  # empty title -> recipe default

    t_axis = collect(0.0:0.1:1.0)
    st_data = [sin(2π * x + 0.5 * t) for x in g1.x, t in t_axis]  # (N x M)
    s_st = SpaceTimeSnapshot1D(g1.x, t_axis, st_data, :φ;
                               xlabel = "Time", ylabel = "Position x", title = "ST")

    g2 = UniformGrid2D(Nx = 16, Ny = 16, Lx = 1.0, Ly = 1.0)
    f2 = [sin(2π * x) * cos(2π * y) for x in g2.x, y in g2.y]  # (Nx x Ny)
    s2 = FieldSnapshot2D(g2, f2, 0.5, :φ; title = "2D")

    # ─────────────────────────────── L3: savefig_publication ───────────────────────────────
    @testset "L3 FieldSnapshot1D single" begin
        mktempdir() do tmp
            p = joinpath(tmp, "1d.pdf")
            @test savefig_publication(s1_single, p) == p
            @test isfile(p)
            @test filesize(p) > 0
        end
    end

    @testset "L3 FieldSnapshot1D multi (vertical stack)" begin
        mktempdir() do tmp
            p = joinpath(tmp, "1dm.pdf")
            savefig_publication(s1_multi, p)
            @test isfile(p)
        end
    end

    @testset "L3 SpaceTimeSnapshot1D" begin
        mktempdir() do tmp
            p = joinpath(tmp, "st.pdf")
            savefig_publication(s_st, p)
            @test isfile(p)
        end
    end

    @testset "L3 FieldSnapshot2D" begin
        mktempdir() do tmp
            p = joinpath(tmp, "2d.pdf")
            savefig_publication(s2, p)
            @test isfile(p)
        end
    end

    @testset "L3 Vector{FieldSnapshot2D} layout" begin
        mktempdir() do tmp
            p = joinpath(tmp, "pair.pdf")
            savefig_publication([s2, s2], p; layout = (1, 2))
            @test isfile(p)
        end
    end

    @testset "L3 custom axis size" begin
        mktempdir() do tmp
            p = joinpath(tmp, "custom.pdf")
            savefig_publication(s1_single, p; axis_width_cm = 10.0, axis_height_cm = 5.0)
            @test isfile(p)
        end
    end

    @testset "L3 PNG output" begin
        mktempdir() do tmp
            p = joinpath(tmp, "1d.png")
            savefig_publication(s1_single, p)
            @test isfile(p)
        end
    end

    # ─────────────────────────────── L1: plot_on_axis! ───────────────────────────────
    @testset "L1 composition onto user axis (2D)" begin
        fig = PythonPlot.figure()
        ax = fig.add_subplot()
        @test PhaseFields.plot_on_axis!(ax, s2) === ax
        mktempdir() do tmp
            p = joinpath(tmp, "compose.pdf")
            fig.savefig(p)
            @test isfile(p)
        end
        PythonPlot.close(fig)
    end

    @testset "L1 SpaceTime / 1D single return ax" begin
        for snap in (s_st, s1_single)
            fig = PythonPlot.figure()
            ax = fig.add_subplot()
            @test PhaseFields.plot_on_axis!(ax, snap) === ax
            PythonPlot.close(fig)
        end
    end

    @testset "L1 title resolution (kwarg > snap.title > default)" begin
        # snap.title used when kwarg empty
        fig = PythonPlot.figure(); ax = fig.add_subplot()
        PhaseFields.plot_on_axis!(ax, s2)
        @test occursin("2D", _title(ax))
        PythonPlot.close(fig)
        # kwarg overrides snap.title
        fig = PythonPlot.figure(); ax = fig.add_subplot()
        PhaseFields.plot_on_axis!(ax, s2; title = "override")
        @test occursin("override", _title(ax))
        PythonPlot.close(fig)
        # empty snap.title -> recipe default (field name based)
        fig = PythonPlot.figure(); ax = fig.add_subplot()
        PhaseFields.plot_on_axis!(ax, s1_notitle)
        @test occursin("φ", _title(ax))
        PythonPlot.close(fig)
    end

    # ─────────────────────────────── L2: figure_publication ───────────────────────────────
    @testset "L2 returns (fig, ax)" begin
        fig, ax = PhaseFields.figure_publication(s2)
        @test fig isa PythonPlot.Figure
        PythonPlot.close(fig)
    end

    @testset "L2 FieldSnapshot1D multi overlay (single ax)" begin
        fig, ax = PhaseFields.figure_publication(s1_multi)
        @test fig isa PythonPlot.Figure
        PythonPlot.close(fig)
    end

    @testset "L2 ylims applied" begin
        # set_ylim path executes without error and figure is created
        fig, ax = PhaseFields.figure_publication(s1_single; ylims = (-2.0, 2.0))
        @test fig isa PythonPlot.Figure
        @test occursin("-2", string(ax.get_ylim()))  # lower bound reflected
        PythonPlot.close(fig)
    end
end
