# Swirl

A grid solver for fluid dyanmic equations - includes the fifth order WENO-JS and modified Roe solver.

A test project to learn and document hydrodynamics code.




## Features

- WENO-JS fifth order
- Roe solver with WENO-Z fifth order
- Written in JAX



## Installation

Dependencies:

```bash
  jax
  python
  numpy
  matplotlib
```
It uses the "cpu" version of jax by default. To change that, edit the following code:
```bash
  $ swirl-solver.py
jax.config.update('jax_platform_name', "cpu")

#can change it to "gpu"
```

## Running Tests

To run tests, run the following command

```bash
  python swirl-solver.py
```

At the top of swirl-solver.py
```bash
# For a Density wave solution:
# Set up the test
BC, u0, rd, xlims, Tfinal = setup_test(1)

# For a Riemann solution (Sod Shock Tube):
# Set up the test
BC, u0, rd, xlims, Tfinal = setup_test(2)
```

To change flux method, solver type or time integration method, edit the following code:
```bash
flux_type, slvr, integ = laxf, weno5, RK4

# flux_type: laxf || slvr: solverf, weno5, roe || integ: RK1, RK3, RK4
```

## Plotting

To activate plotting, in swirl-solver.py:
```bash
plots = True
```

## Solver Parameters

In swirl-solver.py:
```bash
Z =  # No. of grid points
Co =  # CFL number
```

