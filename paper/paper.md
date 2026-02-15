---
title: 'PhaseFields.jl: A Julia Package for Phase Field Simulations with CALPHAD Coupling'
tags:
  - Julia
  - phase field
  - materials science
  - solidification
  - CALPHAD
authors:
  - name: Hiroharu Sugawara
    orcid: 0000-0002-0071-2396
    corresponding: true
    email: hsugawa@tmu.ac.jp
    affiliation: 1
affiliations:
  - name: Graduate School of Systems Design, Tokyo Metropolitan University, Japan
    index: 1
date: 28 January 2026
bibliography: paper.bib
---

# Summary

Understanding and predicting the internal structure of materials
at the microscale (microstructure) is essential in materials science
and engineering. The arrangement of different phases and their
interfaces determines key material properties including mechanical strength
and thermal conductivity.

The phase field method enables simulation of microstructure evolution
during solidification and phase separation. It represents different
phases (e.g., solid and liquid, matrix and precipitates) using a
continuous field variable φ that varies smoothly across interfaces,
avoiding explicit interface tracking.

PhaseFields.jl is a pure Julia package for phase field simulations
that provides:

- Multiple phase field models (Allen-Cahn, Cahn-Hilliard, KKS, WBM, Thermal)
- 1D and 2D simulations with FDM (built-in) and FEM (via Gridap.jl extension)
- Unified interface: `solve(problem, solver)` with automatic backend selection
- Automatic differentiation support via DifferentiationInterface.jl
- Integration with DifferentialEquations.jl for adaptive time stepping
- CALPHAD thermodynamic coupling via OpenCALPHAD.jl

The package is designed for researchers and students who want readable,
well-documented code to understand and extend phase field implementations.

# Statement of Need

The phase field method is widely used for simulating microstructure
evolution in materials science, including solidification, phase separation,
and grain growth.

For quantitative phase field simulations of real alloys, two requirements
are essential:

- **CALPHAD thermodynamic data**: CALPHAD (CALculation of PHAse Diagrams)
  provides assessed Gibbs energy functions for multicomponent alloys,
  enabling simulations with realistic thermodynamic driving forces
  rather than simplified polynomial approximations.

- **Automatic differentiation (AD)**: Derivatives of free energy functions
  are needed both for solving phase field equations (chemical potentials)
  and for parameter optimization. Manual derivation is error-prone and
  limits model flexibility.

However, existing implementations present barriers:

- **C++/Fortran packages** (MOOSE [@Permann2020], PRISMS-PF [@DeWitt2020]):
  manual derivatives, CALPHAD via pycalphad bridge, complex build systems
- **Python packages** (FiPy [@Guyer2009]):
  no CALPHAD integration, manual derivatives
- **Commercial software** (MICRESS[^micress]):
  built-in CALPHAD, but restricts source code access

[^micress]: https://micress.rwth-aachen.de/

| | MOOSE | FiPy | MICRESS | PhaseFields.jl |
|---|---|---|---|---|
| License | LGPL | Custom | Commercial | MIT |
| Language | C++ | Python | Proprietary | Pure Julia |
| CALPHAD coupling | pycalphad | — | Built-in | OpenCALPHAD.jl |
| Differentiation | Manual | Manual | — | AD |
| Time integration | PETSc | PETSc/SciPy | Proprietary | DiffEq.jl / Gridap |
| External dependencies | Many | NumPy/SciPy | — | None |

PhaseFields.jl addresses these requirements:

1. **CALPHAD coupling**: Direct integration with OpenCALPHAD.jl
   [@OpenCALPHADjl] (a pure Julia port of OpenCALPHAD [@Sundman2015])
   enables simulations using assessed thermodynamic databases.

2. **Automatic differentiation**: ForwardDiff.jl computes exact derivatives
   of free energy functions, eliminating manual derivation.

3. **Pure Julia implementation**: No compilation required, no external
   binary dependencies. Code structure closely follows mathematical
   formulations, making it readable and modifiable.

4. **DifferentialEquations.jl integration**: Access to adaptive time stepping,
   stiff solvers, and callbacks from Julia's mature ODE ecosystem.

# CALPHAD Coupling

PhaseFields.jl integrates with OpenCALPHAD.jl [@OpenCALPHADjl] for
thermodynamically-driven simulations using assessed CALPHAD databases.

# Automatic Differentiation

Chemical potentials and other derivatives are computed automatically:

```julia
using PhaseFields, DifferentiationInterface, ForwardDiff

# Custom free energy f(c): c is composition (0 to 1), W > 0
# Double-well potential with minima at c=0 and c=1
f(c) = W * c^2 * (1-c)^2

# Automatic derivatives
μ = derivative(f, AutoForwardDiff(), c)         # chemical potential ∂f/∂c
κ = second_derivative(f, AutoForwardDiff(), c)  # susceptibility ∂²f/∂c²
```

This eliminates manual derivation and enables custom free energy functions.

# Phase Field Models

PhaseFields.jl provides models covering fundamental formulations
(Allen-Cahn, Cahn-Hilliard) and solidification applications
(KKS, WBM, Thermal):

| Application | Model |
|-------------|-------|
| Order-disorder transitions | Allen-Cahn |
| Phase separation (spinodal decomposition) | Cahn-Hilliard |
| Binary alloy solidification | KKS |
| Dilute alloy solidification | WBM |
| Solidification with latent heat | Thermal |

All models share a unified interface for time evolution:

```julia
using PhaseFields, OrdinaryDiffEq

model = AllenCahnModel(τ=1.0, W=1.0)
grid = UniformGrid2D(Nx=50, Ny=50, Lx=1.0, Ly=1.0)  # FDM 2D
problem = AllenCahnProblem(model, grid, φ0, (0.0, 1.0))
sol = solve(problem, ROCK4())  # adaptive time stepping
```

By running simulations with varying driving force ΔG, one can extract
interface velocity from the time evolution of the phase field.
\autoref{fig:interface_velocity} shows the resulting linear relationship
v = M·ΔG.

![Interface velocity vs thermodynamic driving force showing linear
relationship v = M·ΔG with R² = 1.0000, validating the phase field
implementation against sharp-interface theory.
\label{fig:interface_velocity}](figures/interface_velocity.png)

Spatial discretization supports FDM (built-in, 1D/2D) and FEM
(via Gridap.jl extension, 2D).

# Validation

PhaseFields.jl has been validated against analytical solutions such as:

1. **Interface velocity (1D)**: Linear relationship v = M·ΔG between
   interface velocity and driving force, with R² > 0.999 (\autoref{fig:interface_velocity})

2. **Stefan problem (1D)**: Interface position s(t) matches Neumann's
   analytical solution [@Carslaw1959] for thermal solidification (\autoref{fig:stefan})

![Stefan problem validation: (top-left) interface position s(t) from
phase field simulation (circles) compared with analytical solution (dashed line);
(top-right) temperature profiles at different times; (bottom-left) phase field
evolution; (bottom-right) relative error decreasing over time.
\label{fig:stefan}](figures/stefan_problem.png)

The package includes over 600 unit tests covering all models and
numerical integrations.

# Acknowledgments

The author thanks the developers of the Julia packages that PhaseFields.jl
builds upon: DifferentialEquations.jl, Gridap.jl, ForwardDiff.jl, and
DifferentiationInterface.jl.

# References
