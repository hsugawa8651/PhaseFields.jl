# PhaseFields.jl Examples

Last-Modified: 2026-01-26

## Numbering System

| 1st Digit | Physical Phenomenon |
|-----------|---------------------|
| 1 | Allen-Cahn (phase transition basics) |
| 2 | Cahn-Hilliard (spinodal decomposition) |
| 3 | Solidification (KKS, WBM, thermal, Stefan) |
| 4 | Coarsening (Ostwald ripening) |
| 9 | Benchmarks / Validation |

| 2nd Digit | Coupling | Dependencies |
|-----------|----------|--------------|
| 0 | Standalone | PhaseFields.jl only |
| 5 | DiffEq | + DifferentialEquations.jl |
| 8 | CALPHAD | + OpenCALPHAD.jl |

## File List

### 1xx: Allen-Cahn

| File | Coupling | Description |
|------|----------|-------------|
| 101_allen_cahn_1d.jl | Standalone | 1D Allen-Cahn evolution |
| 102_allen_cahn_2d.jl | DiffEq | 2D Allen-Cahn with unified solve API |
| 151_diffeq_allen_cahn.jl | DiffEq | Allen-Cahn with adaptive time stepping |

### 2xx: Cahn-Hilliard

| File | Coupling | Description |
|------|----------|-------------|
| 201_spinodal_1d.jl | Standalone | 1D spinodal decomposition |
| 251_spinodal_2d.jl | DiffEq | 2D spinodal decomposition with unified solve API |

### 3xx: Solidification

| File | Coupling | Description |
|------|----------|-------------|
| 301_kks_solidification.jl | Standalone | KKS model solidification |
| 302_wbm_solidification.jl | Standalone | WBM model solidification |
| 303_wbm_wheeler1992.jl | Standalone | WBM vs Wheeler 1992 paper |
| 304_thermal_solidification.jl | Standalone | Thermal + phase field coupling |
| 305_stefan_problem_1d.jl | Standalone | Classical Stefan problem |
| 351_diffeq_thermal_solidification.jl | DiffEq | Thermal solidification with DiffEq |
| 352_thermal_2d.jl | DiffEq | 2D Thermal phase field with unified solve API |
| 353_wbm_2d.jl | DiffEq | 2D WBM binary alloy with unified solve API |
| 354_kks_2d.jl | DiffEq | 2D KKS binary alloy with unified solve API |
| 381_calphad_coupling_demo.jl | CALPHAD | OpenCALPHAD.jl integration demo |

### 4xx: Coarsening

| File | Coupling | Description |
|------|----------|-------------|
| 401_ostwald_ripening.jl | Standalone | Multi-particle coarsening |

### 9xx: Benchmarks

| File | Coupling | Description |
|------|----------|-------------|
| 901_benchmark_interface_velocity.jl | Standalone | Interface velocity validation |

## How to Run

```bash
cd PhaseFields.jl

# Run a single example
julia --project=. examples/101_allen_cahn_1d.jl

# Run interactively (keeps plot window open)
julia --project=. -i examples/301_kks_solidification.jl
```

## Dependencies

| Coupling | Required Packages |
|----------|-------------------|
| Standalone (x0x) | PhaseFields, Plots |
| DiffEq (x5x) | + OrdinaryDiffEq |
| CALPHAD (x8x) | + OpenCALPHAD |
