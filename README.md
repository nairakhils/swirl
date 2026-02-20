# Swirl

A 1D compressible Euler solver for fluid dynamics — implements fifth-order WENO-JS and WENO-Z reconstruction with Lax-Friedrichs flux splitting.

Built as a learning project to explore high-order hydrodynamics methods on modern GPU backends.

## Features

- **WENO5-JS** and **WENO5-Z** fifth-order spatial reconstruction
- Lax-Friedrichs flux splitting
- RK1, RK3, RK4 time integration
- Outflow and periodic boundary conditions
- Exact Riemann solver for analytical comparison
- Real-time matplotlib plotting with dark theme
- Two backends:
  - **JAX** (`swirl_solver.py`) — XLA-compiled, supports CPU/GPU/Metal/CUDA
  - **CuPy** (`swirl_solver_cccl.py`) — CUDA-native with fused WENO5 RawKernels, CUB reductions via `cuda.compute`, and per-fold timing breakdowns

## Array Layout

Conservative variables stored as `[3, Z]`:
- Row 0: density (rho)
- Row 1: momentum (rho * u)
- Row 2: total energy (E)

## Installation

### JAX backend

```bash
pip install jax jaxlib numpy matplotlib
```

For GPU support, install the appropriate `jaxlib` variant (CUDA, Metal, etc.) per the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

### CuPy backend (CUDA only)

```bash
pip install cupy-cuda12x numpy matplotlib
```

Replace `cupy-cuda12x` with the package matching your CUDA toolkit version (e.g. `cupy-cuda13x`).

Optional — install `cuda-cccl` for CUB device-wide reductions (max wavespeed via `cuda.compute`). The solver falls back to CuPy if unavailable:

```bash
pip install "cuda-cccl[cu12]"    # match your CUDA toolkit version
```

`cuda-cccl` requires Python <= 3.13.

## Usage

### JAX solver

```bash
# Sod shock tube (default), GPU backend, real-time plot
python swirl_solver.py --test 2

# Density wave, CPU backend, 64-bit precision, no plot
python swirl_solver.py --test 1 --backend cpu --precision 64 --no-plot

# WENO5-Z with RK4, custom grid
python swirl_solver.py --solver weno5z --integrator RK4 --grid-size 10000 --cfl 0.3
```

### CuPy solver

```bash
# Sod shock tube, no plot
python swirl_solver_cccl.py --test 2 --no-plot

# Density wave with WENO5-Z
python swirl_solver_cccl.py --test 1 --solver weno5z
```

### CLI options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--test` | `1`, `2` | `2` | Test case (1: density wave, 2: Sod shock tube) |
| `--solver` | `solverf`, `weno5`, `weno5z` | `weno5` | Spatial reconstruction method |
| `--integrator` | `RK1`, `RK3`, `RK4` | `RK1` | Time integration scheme |
| `--flux` | `laxf` | `laxf` | Numerical flux function |
| `--grid-size` | int | `5000` | Number of interior grid cells |
| `--cfl` | float | `0.6` | CFL number for adaptive timestep |
| `--no-plot` | flag | off | Disable real-time plotting |
| `--plot-interval` | int | `1` | Update plot every N iterations |
| `--fold` | int | `20` | Timesteps per timing measurement |
| `--cell-average` | `standard`, `accurate` | `standard` | Cell averaging for initial conditions |

JAX-only flags: `--backend` (`cpu`/`gpu`/`metal`/`cuda`/`auto`), `--precision` (`32`/`64`).

## Test Cases

**Test 1 — Density wave**: Smooth sinusoidal density profile advected with periodic structure. Reports L2 error norm against the exact solution.

**Test 2 — Sod shock tube**: Classic Riemann problem (rho=1, p=1 | rho=0.125, p=0.1) producing a rarefaction, contact discontinuity, and shock. Compared against the exact Riemann solution.

## Performance

Output reports **Mzps** (million zone-updates per second) along with per-fold timing breakdowns (WENO kernel, reduction, and total wall time).

The CuPy backend uses:
- **Fused CUDA RawKernels** for WENO5 reconstruction with shared-memory tiling (256 threads/block, 5-cell halo)
- **CUB device-wide reductions** via `cuda.compute` (`make_reduce_into` with `OpKind.MAXIMUM`) for max wavespeed computation, with cached reducer and pre-allocated temp storage to avoid per-timestep recompilation
