# Swirl

A grid solver for fluid dyanmic equations - includes the fifth order WENO and Roe-averaged flux solver.

A test project to learn and document hydrodynamics code.




## Features

- WENO5 - fifth order
- Roe solver with WENO5
- Written in JAX



## Installation

Dependencies:

```bash
  jax
  python
  numpy
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

In swirl_rd.py
```bash
# For a density wave solution:
# Set up the test
BC, u0, rd, xlims, Tfinal = setup_test(1)

# For a riemann solution (Sod Shock Tube):
# Set up the test
BC, u0, rd, xlims, Tfinal = setup_test(2)
```

To change flux method, solver type or time integration method, edit the following code:
```bash
flux_type, slvr, integ = laxf, weno5, RK4

# flux_type: solverf, laxf || slvr: weno5, roeweno1 || integ: RK3, RK4
```

