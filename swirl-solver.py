#//grid code logic: https://github.com/harold-berjamin
import jax


jax.config.update('jax_platform_name', "cpu")
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import time
from swirl import *

# Initialize tests
# flux_type: laxf || slvr: solverf, weno5, roeweno1 || integ: RK1, RK3, RK4
flux_type, slvr, integ = laxf, weno5, RK4
cellaverage_type = cellaverage # cellaverage || cellaverage_acc

rhoJ = jnp.array([1, 0.125])
uJ = jnp.array([0, 0])
pJ = jnp.array([1, 0.1])

# Test parameters
tests = {
    1: {"BC": outflow, "u0": densityfn, "xlims": np.array([0, 2]), "Tfinal": 0.1, "rd": 'den_wave'},
    2: {"BC": outflow, "u0": riemann(rhoJ, uJ, pJ),
        "xlims": np.array([-0.5, 0.5]), "Tfinal": 0.015, "rd": 'riemann'}
}

def setup_test(test_number):
    test = tests[test_number]
    BC, u0, xlims, Tfinal, rd = test["BC"], test["u0"], test["xlims"], test["Tfinal"], test["rd"]
    return BC, u0, rd, xlims, Tfinal

# Set up the test
BC, u0, rd, xlims, Tfinal = setup_test(2)


# Initialize Grid
plots = True
Z = 5000
Co = 0.6 # CFL

# Data initialisation
t, n, dy, Z = 0, 0, jnp.abs((xlims[1]-xlims[0])/(Z+6)), Z+6
y = jnp.linspace(xlims[0]-2.5*dy, xlims[1]+2.5*dy, Z)
u = BC(cellaverage_type(u0, y, dy)) #use cellaverage_acc for better accuracy
dt = Co * dy / jnp.max(jnp.abs(EigA(u)[0]))
L = diffeq(flux_type, slvr, BC, dt, dy)

def plot_initial_state(u, y, t):
    with plt.style.context('dark_background'):
        figure, axes = plt.subplots(3, 1, figsize=(12, 12))
        M = 1.1 * (jnp.max(jnp.abs(u)) + 0.02)
        colors = ['r', 'g', 'b']  # Red, green, blue
        if cellaverage_type == cellaverage:
            y_midpoints = 0.5 * (y[1:] + y[:-1])
            lines = [ax.plot(y_midpoints, u[i, :], '.', linewidth=0.5, color=colors[i])[0] for i, ax in enumerate(axes)]
        else:
            lines = [ax.plot(y, u[i, :], '.', linewidth=0.5, color=colors[i])[0] for i, ax in enumerate(axes)]    
        for ax, line, label in zip(axes, lines, ['Density', 'Velocity', 'Pressure']):
            ax.set_ylim([-M, M])
            ax.set_xlabel('x')
            ax.set_ylabel(label)
            ax.set_title(f't = {t}')
        plt.tight_layout()
        plt.draw()
    return figure, axes, lines

def update_plot(u, t, lines, axes):
    for i, line in enumerate(lines):
        line.set_ydata(u[i, :])
        axes[i].set_title(f't = {t}')
    plt.draw()
    plt.pause(0.05)
    
figure, axes, lines = plot_initial_state(u, y, t)


def compute_next_step(u, t, dt, n):
    u = integ(L, u, dt)
    t += dt
    n += 1
    eigval, eigvec = EigA(u)
    amax = jnp.max(jnp.abs(eigval))
    dt = Co * dy / amax
    return u, t, dt, n
    

print(f'Initiating loop...')

fold = 20
while t < Tfinal:
    t0 = time.time()
    for i in range(fold):
        u, t, dt, n = compute_next_step(u, t, dt, n)
    t1 = time.time()
    print(f"[{n:04d}] t={t:.3f} Mzps={Z * fold * 1e-6 / (t1 - t0):.4f}")


# Exact solution
analytic_u = RiemannExact(jnp.array([rhoJ, rhoJ*uJ, 0.5*rhoJ*uJ**2 + pJ/(gamma-1)]), gamma, t) if rd == 'riemann' else lambda x: u0(x - t)

# Compare plots
if plots:
    x_coords = jnp.linspace(y[0], y[Z-1], int(max(2 * Z, 1e3)))
    for i, ax in enumerate(axes):
        ax.plot(x_coords, analytic_u(x_coords)[i, :], 'w:')
        ax.set_title(f't = {t}')
    [line.set_ydata(u[i, :]) for i, line in enumerate(lines)]
    plt.show()

# L2 Norm
if rd == 'den_wave' and cellaverage_type == cellaverage_acc:
    bool_mask = (xlims[0] < y) & (y < xlims[1])
    difference = u[:, bool_mask] - cellaverage_type(analytic_u, y, dy)[:, bool_mask]
    print('L2 norm\n', jnp.linalg.norm(difference * dy, axis=1))
    
elif rd == 'den_wave' and cellaverage_type == cellaverage:
    bool_mask = (xlims[0] < y[:-1]) & (y[:-1] < xlims[1])
    difference = u[:, bool_mask] - cellaverage_type(analytic_u, y, dy)[:, bool_mask]
    print('L2 norm\n', jnp.linalg.norm(difference * dy, axis=1))

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

