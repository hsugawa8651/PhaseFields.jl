# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hiroharu Sugawara
# Part of PhaseFields.jl - Integration tests (heavy numerical simulations)
#
# Separated from runtests.jl for faster CI.
# Run manually: julia --project test/run_integration_tests.jl

using PhaseFields
using OrdinaryDiffEq
using Test
using DifferentiationInterface
import DifferentiationInterface as DI

@testset "Integration Tests" begin
    # Problem solve tests (from test_problems.jl Test 8-9)
    include("test_problems_solve.jl")

    # DiffEq integration tests (heaviest)
    include("test_diffeq_integration.jl")

    # 2D simulations
    include("test_allen_cahn_2d.jl")
    include("test_thermal_2d.jl")

    # Unified model tests (1D + 2D solve)
    include("test_wbm_unified.jl")
    include("test_cahn_hilliard_unified.jl")
    include("test_kks_unified.jl")

    # FEM extension (Gridap, if available)
    include("test_gridap_ext.jl")
end
