using Test
using PhaseFields
using OrdinaryDiffEq
using DiffEqCallbacks
using SciMLBase: DiscreteCallback, CallbackSet
using SparseArrays

@testset "DifferentialEquations.jl Integration" begin

    @testset "UniformGrid1D" begin
        grid = UniformGrid1D(N=101, L=1.0)

        @test grid.N == 101
        @test grid.L == 1.0
        @test grid.dx ≈ 0.01 atol=1e-10
        @test length(grid.x) == 101
        @test grid.x[1] ≈ 0.0 atol=1e-10
        @test grid.x[end] ≈ 1.0 atol=1e-10

        # Different domain
        grid2 = UniformGrid1D(N=51, L=2.0)
        @test grid2.dx ≈ 0.04 atol=1e-10
    end

    @testset "Boundary Conditions" begin
        # NeumannBC
        bc_n = NeumannBC()
        @test bc_n isa BoundaryCondition

        # DirichletBC
        bc_d = DirichletBC(0.0, 1.0)
        @test bc_d isa BoundaryCondition
        @test bc_d.left == 0.0
        @test bc_d.right == 1.0

        # PeriodicBC
        bc_p = PeriodicBC()
        @test bc_p isa BoundaryCondition
    end

    @testset "Laplacian 1D - Neumann BC" begin
        N = 11
        dx = 0.1
        bc = NeumannBC()

        # Constant field: Laplacian = 0
        u_const = ones(N)
        ∇²u = zeros(N)
        laplacian_1d!(∇²u, u_const, dx, bc)
        @test all(abs.(∇²u) .< 1e-10)

        # Linear field: Laplacian = 0
        u_linear = collect(0.0:0.1:1.0)
        laplacian_1d!(∇²u, u_linear, dx, bc)
        # Interior points should be 0
        @test all(abs.(∇²u[2:N-1]) .< 1e-10)

        # Quadratic field: u = x², Laplacian = 2
        x = collect(0.0:0.1:1.0)
        u_quad = x.^2
        laplacian_1d!(∇²u, u_quad, dx, bc)
        @test all(abs.(∇²u[2:N-1] .- 2.0) .< 0.1)

        # Allocating version
        ∇²u_alloc = laplacian_1d(u_quad, dx, bc)
        @test ∇²u_alloc ≈ ∇²u
    end

    @testset "Laplacian 1D - Dirichlet BC" begin
        N = 11
        dx = 0.1
        # For Dirichlet BC, bc.left and bc.right are ghost node values
        # For u = x², ghost nodes: u(-dx) ≈ 0, u(L+dx) ≈ 1.21
        bc = DirichletBC(dx^2, (1.0 + dx)^2)

        # Quadratic field: u = x²
        x = collect(0.0:0.1:1.0)
        u_quad = x.^2
        ∇²u = zeros(N)
        laplacian_1d!(∇²u, u_quad, dx, bc)

        # Interior points should have Laplacian ≈ 2
        @test all(abs.(∇²u[2:N-1] .- 2.0) .< 0.1)
        # Boundary points also ≈ 2 with correct ghost values
        @test abs(∇²u[1] - 2.0) < 0.2
        @test abs(∇²u[N] - 2.0) < 0.2
    end

    @testset "Laplacian 1D - Periodic BC" begin
        N = 20  # Finer grid for better accuracy
        dx = 1.0 / N
        bc = PeriodicBC()

        # Constant field
        u_const = ones(N)
        ∇²u = zeros(N)
        laplacian_1d!(∇²u, u_const, dx, bc)
        @test all(abs.(∇²u) .< 1e-10)

        # Sine wave: u = sin(2πx), Laplacian = -4π²sin(2πx)
        x = collect(range(0, 1 - dx, length=N))
        u_sin = sin.(2π .* x)
        laplacian_1d!(∇²u, u_sin, dx, bc)
        expected = -4π^2 .* u_sin
        # Second-order FD has error O(dx²), expect < 2 for dx=0.05
        @test maximum(abs.(∇²u .- expected)) < 2.0
    end

    @testset "Laplacian Matrix 1D" begin
        N = 5
        dx = 0.25
        bc = NeumannBC()

        L = laplacian_matrix_1d(N, dx, bc)
        @test size(L) == (N, N)
        @test issparse(L)

        # Apply to constant field: should give zeros
        u_const = ones(N)
        @test all(abs.(L * u_const) .< 1e-10)

        # Compare with in-place version
        u_test = rand(N)
        ∇²u_mat = L * u_test
        ∇²u_func = laplacian_1d(u_test, dx, bc)
        @test ∇²u_mat ≈ ∇²u_func atol=1e-10
    end

    @testset "Allen-Cahn ODE - Setup" begin
        model = AllenCahnModel(τ=1.0, W=0.1)
        grid = UniformGrid1D(N=51, L=1.0)
        bc = NeumannBC()

        # Create parameters
        params = AllenCahnODEParams(model, grid, bc)
        @test params.model === model
        @test params.grid === grid
        @test length(params.∇²φ) == 51

        # Initial condition
        φ0 = [0.5 + 0.1 * sin(2π * x) for x in grid.x]

        # Create ODEProblem
        prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 1.0))
        @test prob isa ODEProblem
        @test prob.u0 ≈ φ0
        @test prob.tspan == (0.0, 1.0)
    end

    @testset "Allen-Cahn ODE - RHS" begin
        model = AllenCahnModel(τ=1.0, W=0.1)
        grid = UniformGrid1D(N=11, L=1.0)
        bc = NeumannBC()
        params = AllenCahnODEParams(model, grid, bc)

        # At φ = 0, g'(0) = 0, so RHS = W²∇²φ / τ
        φ_zero = zeros(11)
        φ_zero[6] = 0.1  # Small perturbation at center
        dφ = zeros(11)

        allen_cahn_ode!(dφ, φ_zero, params, 0.0)

        # Center should have negative dφ (curvature effect)
        @test dφ[6] != 0.0

        # At φ = 0.5 (interface), g'(0.5) = 0
        φ_half = fill(0.5, 11)
        allen_cahn_ode!(dφ, φ_half, params, 0.0)
        # Uniform field: ∇²φ = 0, g'(0.5) = 0, so dφ ≈ 0
        @test all(abs.(dφ) .< 1e-10)
    end

    @testset "Allen-Cahn ODE - Solve" begin
        model = AllenCahnModel(τ=1.0, W=0.1)
        grid = UniformGrid1D(N=51, L=1.0)
        bc = NeumannBC()

        # Step function initial condition
        φ0 = [x < 0.5 ? 1.0 : 0.0 for x in grid.x]

        prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 1.0))

        # Solve with explicit method
        sol = solve(prob, Tsit5(); saveat=0.1)

        @test sol.retcode == ReturnCode.Success
        @test length(sol.t) == 11  # 0.0, 0.1, ..., 1.0

        # Solution should be bounded [0, 1]
        for φ in sol.u
            @test all(φ .>= -0.1)
            @test all(φ .<= 1.1)
        end

        # Interface should have smoothed
        φ_final = sol.u[end]
        @test φ_final[1] > 0.9   # Still solid-like at left
        @test φ_final[end] < 0.1 # Still liquid-like at right
    end

    @testset "Allen-Cahn ODE - Implicit Solver" begin
        model = AllenCahnModel(τ=1.0, W=0.1)
        grid = UniformGrid1D(N=51, L=1.0)
        bc = NeumannBC()

        # Initial condition with interface at x=0.5
        φ0 = [0.5 + 0.4 * cos(π * x) for x in grid.x]

        prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 5.0))

        # Solve with implicit method (disable autodiff due to pre-allocated workspace)
        sol = solve(prob, QNDF(autodiff=false); saveat=1.0)

        @test sol.retcode == ReturnCode.Success

        # Interface forms at x=0.5, bulk regions reach near 0 and 1
        φ_final = sol.u[end]
        @test φ_final[1] > 0.9    # Left side approaches 1
        @test φ_final[end] < 0.1  # Right side approaches 0
        @test 0.4 < φ_final[26] < 0.6  # Interface center near 0.5
    end

    @testset "Thermal ODE - Setup" begin
        model = ThermalPhaseFieldModel(
            τ=1.0, W=0.1, λ=1.0, α=1e-5, L=2.35e9, Cp=5.42e6, Tm=1728.0
        )
        grid = UniformGrid1D(N=51, L=1e-4)
        bc_φ = NeumannBC()
        bc_u = NeumannBC()

        params = ThermalODEParams(model, grid, bc_φ, bc_u)
        @test length(params.∇²φ) == 51
        @test length(params.∇²u) == 51
        @test length(params.dφdt) == 51

        # Initial conditions
        φ0 = ones(51)
        u0 = zeros(51)

        prob = create_thermal_problem(model, grid, bc_φ, bc_u, φ0, u0, (0.0, 1e-3))
        @test prob isa ODEProblem
        @test length(prob.u0) == 102  # [φ; u]
    end

    @testset "Thermal ODE - RHS" begin
        model = ThermalPhaseFieldModel(
            τ=1.0, W=0.1, λ=1.0, α=1e-5, L=2.35e9, Cp=5.42e6, Tm=1728.0
        )
        grid = UniformGrid1D(N=11, L=1.0)
        bc_φ = NeumannBC()
        bc_u = NeumannBC()
        params = ThermalODEParams(model, grid, bc_φ, bc_u)

        N = grid.N

        # Uniform fields
        y = vcat(fill(0.5, N), fill(0.0, N))  # φ=0.5, u=0
        dy = zeros(2N)

        thermal_ode!(dy, y, params, 0.0)

        dφ = dy[1:N]
        du = dy[N+1:2N]

        # At φ=0.5, u=0, uniform: dφ ≈ 0 (no driving force)
        @test all(abs.(dφ) .< 1e-10)
        # du = α∇²u + 0.5*dφdt, with ∇²u=0 and dφdt=0: du ≈ 0
        @test all(abs.(du) .< 1e-10)
    end

    @testset "Thermal ODE - Solve" begin
        model = ThermalPhaseFieldModel(
            τ=1e-6, W=1e-6, λ=2.0, α=1e-5, L=2.35e9, Cp=5.42e6, Tm=1728.0
        )
        grid = UniformGrid1D(N=51, L=1e-4)

        # Solid seed in center, undercooled
        x0 = grid.L / 2
        W = model.W
        φ0 = [0.5 * (1 - tanh((abs(x - x0) - 5W) / (sqrt(2) * W))) for x in grid.x]
        u0 = fill(-0.05, grid.N)  # Slight undercooling

        bc_φ = NeumannBC()
        bc_u = NeumannBC()

        prob = create_thermal_problem(model, grid, bc_φ, bc_u, φ0, u0, (0.0, 1e-6))

        # Solve with stiff solver (disable autodiff due to pre-allocated workspace)
        sol = solve(prob, QNDF(autodiff=false); saveat=1e-7)

        @test sol.retcode == ReturnCode.Success

        # Extract solution
        φ_hist, u_hist = extract_thermal_solution(sol, grid.N)
        @test size(φ_hist, 1) == grid.N
        @test size(φ_hist, 2) == length(sol.t)
    end

    @testset "extract_thermal_solution" begin
        N = 10
        # Mock solution data
        t = [0.0, 0.5, 1.0]
        u_data = [vcat(rand(N), rand(N)) for _ in t]

        # Create a simple struct to mimic ODESolution
        sol = (t=t, u=u_data)

        φ_hist, u_hist = extract_thermal_solution(sol, N)

        @test size(φ_hist) == (N, 3)
        @test size(u_hist) == (N, 3)

        for i in 1:3
            @test φ_hist[:, i] ≈ u_data[i][1:N]
            @test u_hist[:, i] ≈ u_data[i][N+1:2N]
        end
    end

    # =========================================================================
    # Callbacks Tests
    # =========================================================================

    @testset "interface_position_1d" begin
        x = collect(0.0:0.1:1.0)

        # Step function crossing at x=0.5 (actually between 0.4 and 0.5)
        φ = [xi < 0.5 ? 1.0 : 0.0 for xi in x]
        pos = interface_position_1d(φ, x)
        @test 0.4 ≤ pos ≤ 0.5  # Linear interpolation gives 0.45

        # Smooth tanh profile centered at x=0.3
        φ_smooth = [0.5 * (1 - tanh((xi - 0.3) / 0.05)) for xi in x]
        pos_smooth = interface_position_1d(φ_smooth, x)
        @test 0.25 ≤ pos_smooth ≤ 0.35  # Should be near 0.3

        # No crossing (all solid)
        φ_solid = ones(length(x))
        @test isnan(interface_position_1d(φ_solid, x))

        # No crossing (all liquid)
        φ_liquid = zeros(length(x))
        @test isnan(interface_position_1d(φ_liquid, x))

        # Custom contour crossing
        φ_custom = [xi < 0.7 ? 0.8 : 0.2 for xi in x]
        pos_custom = interface_position_1d(φ_custom, x; contour=0.5)
        @test 0.6 ≤ pos_custom ≤ 0.7  # Between 0.6 and 0.7
    end

    @testset "solid_fraction" begin
        # All solid
        φ_solid = ones(100)
        @test solid_fraction(φ_solid) ≈ 1.0

        # All liquid
        φ_liquid = zeros(100)
        @test solid_fraction(φ_liquid) ≈ 0.0

        # Half solid
        φ_half = vcat(ones(50), zeros(50))
        @test solid_fraction(φ_half) ≈ 0.5

        # Custom threshold
        φ_mixed = fill(0.3, 100)
        @test solid_fraction(φ_mixed; threshold=0.5) ≈ 0.0
        @test solid_fraction(φ_mixed; threshold=0.2) ≈ 1.0
    end

    @testset "create_interface_saving_callback" begin
        grid = UniformGrid1D(N=51, L=1.0)

        cb, saved = create_interface_saving_callback(grid)
        @test cb isa DiscreteCallback  # SavingCallback returns DiscreteCallback
        @test saved isa DiffEqCallbacks.SavedValues

        # With saveat
        cb2, saved2 = create_interface_saving_callback(grid; saveat=0.1)
        @test cb2 isa DiscreteCallback
    end

    @testset "create_solid_fraction_callback" begin
        cb, saved = create_solid_fraction_callback()
        @test cb isa DiscreteCallback
        @test saved isa DiffEqCallbacks.SavedValues

        # With custom threshold
        cb2, saved2 = create_solid_fraction_callback(; threshold=0.3, saveat=0.1)
        @test cb2 isa DiscreteCallback
    end

    @testset "create_steady_state_callback" begin
        cb = create_steady_state_callback()
        @test cb isa SciMLBase.DECallback  # TerminateSteadyState returns a callback

        # With custom tolerances
        cb2 = create_steady_state_callback(abstol=1e-6, reltol=1e-4)
        @test cb2 isa SciMLBase.DECallback

        # With min_t
        cb3 = create_steady_state_callback(min_t=1.0)
        @test cb3 isa SciMLBase.DECallback
    end

    @testset "create_phase_field_callbacks" begin
        grid = UniformGrid1D(N=51, L=1.0)

        # Default: track interface only
        result = create_phase_field_callbacks(grid)
        @test result.callback isa DiscreteCallback
        @test result.interface_data isa DiffEqCallbacks.SavedValues
        @test result.solid_fraction_data === nothing

        # Track both
        result2 = create_phase_field_callbacks(grid;
            track_interface=true, track_solid_fraction=true)
        @test result2.callback isa CallbackSet
        @test result2.interface_data isa DiffEqCallbacks.SavedValues
        @test result2.solid_fraction_data isa DiffEqCallbacks.SavedValues

        # With steady state termination
        result3 = create_phase_field_callbacks(grid;
            track_interface=true, terminate_steady_state=true)
        @test result3.callback isa CallbackSet

        # No callbacks
        result4 = create_phase_field_callbacks(grid;
            track_interface=false, track_solid_fraction=false)
        @test result4.callback === nothing
    end

    @testset "Callbacks with Allen-Cahn ODE" begin
        model = AllenCahnModel(τ=1.0, W=0.05)
        grid = UniformGrid1D(N=51, L=1.0)
        bc = NeumannBC()

        # Initial condition
        φ0 = [0.5 * (1 - tanh((x - 0.3) / 0.1)) for x in grid.x]
        prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 1.0))

        # Test SavingCallback for interface position
        cb, saved = create_interface_saving_callback(grid; saveat=0.2)
        sol = solve(prob, Tsit5(); callback=cb)

        @test sol.retcode == ReturnCode.Success
        @test length(saved.t) > 0
        @test length(saved.saveval) == length(saved.t)
        @test all(!isnan, saved.saveval)  # Should have valid interface positions

        # Test TerminateSteadyState
        cb_ss = create_steady_state_callback(abstol=1e-4, reltol=1e-3, min_t=0.1)
        prob_long = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 100.0))
        sol_ss = solve(prob_long, Tsit5(); callback=cb_ss)

        @test sol_ss.retcode == ReturnCode.Terminated
        @test sol_ss.t[end] < 100.0  # Should terminate early
    end

    @testset "Combined Callbacks Integration" begin
        model = AllenCahnModel(τ=1.0, W=0.05)
        grid = UniformGrid1D(N=51, L=1.0)
        bc = NeumannBC()

        φ0 = [0.5 * (1 - tanh((x - 0.3) / 0.1)) for x in grid.x]
        prob = create_allen_cahn_problem(model, grid, bc, φ0, (0.0, 1.0))

        # Combined callbacks
        result = create_phase_field_callbacks(grid;
            track_interface=true,
            track_solid_fraction=true,
            saveat=0.2)

        sol = solve(prob, Tsit5(); callback=result.callback)

        @test sol.retcode == ReturnCode.Success
        @test length(result.interface_data.t) > 0
        @test length(result.solid_fraction_data.t) > 0

        # Solid fraction should be between 0 and 1
        @test all(0.0 .<= result.solid_fraction_data.saveval .<= 1.0)
    end

end
