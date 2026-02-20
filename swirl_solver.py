#!/usr/bin/env python3
import os
import sys
import argparse

def parse_backend_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', type=str, choices=['cpu', 'gpu', 'metal', 'cuda', 'auto'], 
                        default='auto', help='Force JAX backend (cpu/gpu/metal/cuda/auto)')
    parser.add_argument('--precision', type=str, choices=['32', '64'], default='32',
                        help='Float precision: 32 (faster) or 64 (more accurate)')
    args, _ = parser.parse_known_args()
    return args.backend, args.precision

backend_choice, precision = parse_backend_arg()

if backend_choice == 'cpu':
    os.environ['JAX_PLATFORMS'] = 'cpu'
elif backend_choice == 'metal':
    os.environ['JAX_PLATFORMS'] = ''
elif backend_choice == 'cuda':
    os.environ['JAX_PLATFORMS'] = 'cuda,gpu'
elif backend_choice == 'gpu':
    os.environ['JAX_PLATFORMS'] = 'gpu'

import jax
jax.config.update('jax_enable_x64', precision == '64')
import jax.numpy as jnp

actual_backend = jax.default_backend()
print(f"JAX backend: {actual_backend}")
from jax import jit, vmap, lax
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import time

gamma = 1.4

@jit
def flux(x):
    rho = x[0, :]
    m = x[1, :]
    E = x[2, :]
    u = m / rho
    u_sq = u * u
    p = (gamma - 1) * (E - 0.5 * rho * u_sq)
    flux = jnp.stack([m, m * u + p, (E + p) * u])
    return flux

@jit
def compute_values(x):
    rho = x[0, :]
    m = x[1, :]
    E = x[2, :]
    u = m / rho
    p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
    a = jnp.sqrt(gamma * p / rho)
    return jnp.array([rho, m, E, u, p, a])

def Avec(x):
    jac = jax.jacfwd(compute_values)(x)
    return jac

@jit
def EigA(x):
    rho, m, E = x
    u = m / rho
    p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
    a = jnp.sqrt(gamma * p / rho)
    eigval = jnp.array([u - a, u, u + a])
    return eigval, None

def numerical_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2.0 * h)

def newton_raphson(func, x0, tol=1e-5, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        if jnp.abs(fx) < tol:
            return x
        dfx = numerical_derivative(func, x)
        if dfx == 0:
            print("Zero derivative. No solution found.")
            return None
        x = x - fx / dfx
    print("Exceeded maximum iterations. No solution found.")
    return None

def RiemannExact(initial_state, gamma, t):
    density, momentum, energy = initial_state[:, :2]
    velocity = momentum / density
    pressure = (gamma - 1) * (energy - 0.5 * momentum * velocity)
    a = jnp.sqrt(gamma * pressure / density)
    A = 2 / ((gamma + 1) * density)
    B = (gamma - 1) / (gamma + 1) * pressure
    du = jnp.diff(velocity)
    fluxt = lambda p, i: (p - pressure[i]) * jnp.sqrt(A[i] / (p + B[i])) * (p > pressure[i]) + \
                         2 * a[i] / (gamma - 1) * ((p / pressure[i]) ** ((gamma - 1) / (2 * gamma)) - 1) * (p <= pressure[i])
    fluxF = lambda p: fluxt(p, 0) + fluxt(p, 1) + du
    if velocity[1] - velocity[0] > 2 * (a[0] + a[1]) / (gamma - 1):
        print('vacuum condition')
        us, rholeft, uleft, left_pressureeft, rhoright, uright, pright = 0, lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0
    else:
        p0 = max([jnp.finfo(float).eps, 0.5 * (pressure[0] + pressure[1]) - du * (density[0] + density[1]) * (a[0] + a[1]) / 8])
        ps = newton_raphson(fluxF, p0)
        us = 0.5 * (velocity[1] + velocity[0]) + 0.5 * (fluxt(ps, 1) - fluxt(ps, 0))
        def calculate_values(pressure_side, density_side, velocity_side, a_side, ps, us, t, gamma, side):
            if ps > pressure_side:
                rhoS = density_side * ((gamma - 1) / (gamma + 1) + ps / pressure_side) / ((gamma - 1) / (gamma + 1) * ps / pressure_side + 1)
                S = velocity_side - a_side * jnp.sqrt((gamma + 1) / (2 * gamma) * ps / pressure_side + (gamma - 1) / (2 * gamma)) if side == 'left' else velocity_side + a_side * jnp.sqrt((gamma + 1) / (2 * gamma) * ps / pressure_side + (gamma - 1) / (2 * gamma))
                rho = lambda x: density_side * (x < S * t) + rhoS * (x >= S * t) if side == 'left' else density_side * (x > S * t) + rhoS * (x <= S * t)
                u = lambda x: velocity_side * (x < S * t) + us * (x >= S * t) if side == 'left' else velocity_side * (x > S * t) + us * (x <= S * t)
                p = lambda x: pressure_side * (x < S * t) + ps * (x >= S * t) if side == 'left' else pressure_side * (x > S * t) + ps * (x <= S * t)
            else:
                aS = a_side + (velocity_side - us) * (gamma - 1) / 2 if side == 'left' else a_side + (us - velocity_side) * (gamma - 1) / 2
                rhoS = gamma * ps / aS ** 2
                if side == 'left':
                    rho = lambda x: density_side * (x < (velocity_side - a_side) * t) + rhoS * (x >= (us - aS) * t) + density_side * jnp.abs(
                        2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 / (gamma - 1)) * (
                            x >= (velocity_side - a_side) * t) * (x < (us - aS) * t)
                    u = lambda x: velocity_side * (x < (velocity_side - a_side) * t) + us * (x >= (us - aS) * t) + 2 / (gamma + 1) * (
                            a_side + (gamma - 1) / 2 * velocity_side + x / t) * (x >= (velocity_side - a_side) * t) * (x < (us - aS) * t)
                    p = lambda x: pressure_side * (x < (velocity_side - a_side) * t) + ps * (x >= (us - aS) * t) + pressure_side * jnp.abs(
                        2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 * gamma / (gamma - 1)) * (
                            x >= (velocity_side - a_side) * t) * (x < (us - aS) * t)
                else:
                    rho = lambda x: density_side * (x > (velocity_side + a_side) * t) + rhoS * (x <= (us + aS) * t) + density_side * jnp.abs(
                        2 / (gamma + 1) - (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 / (gamma - 1)) * (
                            x <= (velocity_side + a_side) * t) * (x > (us + aS) * t)
                    u = lambda x: velocity_side * (x > (velocity_side + a_side) * t) + us * (x <= (us + aS) * t) + 2 / (gamma + 1) * (
                            -a_side + (gamma - 1) / 2 * velocity_side + x / t) * (x <= (velocity_side + a_side) * t) * (x > (us + aS) * t)
                    p = lambda x: pressure_side * (x > (velocity_side + a_side) * t) + ps * (x <= (us + aS) * t) + pressure_side * jnp.abs(
                        2 / (gamma + 1) - (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 * gamma / (gamma - 1)) * (
                            x <= (velocity_side + a_side) * t) * (x > (us + aS) * t)
            return rho, u, p
        rholeft, uleft, left_pressureeft = calculate_values(pressure[0], density[0], velocity[0], a[0], ps, us, t, gamma, 'left')
        rhoright, uright, pright = calculate_values(pressure[1], density[1], velocity[1], a[1], ps, us, t, gamma, 'right')
    @jit
    def fin(x):
        x = jnp.asarray(x)
        UL = jnp.array([rholeft(x), rholeft(x)*uleft(x), 0.5*rholeft(x)*uleft(x)**2 + left_pressureeft(x)/(gamma-1)])
        UR = jnp.array([rhoright(x), rhoright(x)*uright(x), 0.5*rhoright(x)*uright(x)**2 + pright(x)/(gamma-1)])
        u = jnp.where(x < us * t, UL, UR)
        return u
    return fin

def riemann(r, u, p):
    x1 = jnp.array([r, r*u, 0.5*r*u**2 + p/(gamma-1)])
    @jit
    def arrcheck(y):
        y = jnp.asarray(y)
        y = y if y.ndim > 0 else jnp.expand_dims(y, 0)
        arr = jnp.where(y<0, x1[:,0,None], x1[:,1,None])
        return arr
    return arrcheck

def densityfn(y):
    rho = jnp.sin(jnp.pi*y) + 5
    u = jnp.zeros_like(y)
    p = jnp.ones_like(y)
    e = p / ((gamma - 1) * rho)
    arr = jnp.vstack([rho, rho * u, rho * e])
    return arr

def simpsons_rule(func, a, b, n=100):
    h = (b - a) / n
    x = jnp.linspace(a, b, n+1)
    y = func(x)
    return h / 3 * (y[:,0] + y[:,-1] + 4 * jnp.sum(y[:,1:-1:2], axis=1) + 2 * jnp.sum(y[:,2:-1:2], axis=1))

def cellaverage_acc(arr, y, dy, n=100):
    arravg_vmap = vmap(lambda yi: simpsons_rule(arr, yi-0.5*dy, yi+0.5*dy, n) / dy)
    x = arravg_vmap(y).T
    return x

def cellaverage(arr, y, dy):
    mid_points = 0.5 * (y[:-1] + y[1:])
    x = arr(mid_points).squeeze()
    return x

@jit
def periodic(x):
    Z = x.shape[1]
    y = x.at[:,:3].set(x[:,Z-6:Z-3])
    y = y.at[:,Z-3:].set(x[:,3:6])
    return y

@jit
def outflow(x):
    Z = x.shape[1]
    y = x.at[:,:3].set(x[:,3:4])
    y = y.at[:,Z-3:].set(x[:,Z-4:Z-3])
    return y

@jit
def laxf(x, slvr, dt, dx):
    left, right = slvr(x)
    eigval, _ = EigA(x)
    amax = jnp.max(jnp.abs(eigval))
    flux1 = 0.5 * (flux(left) + flux(right) - amax * (right - left))
    return flux1

@jit
def solverf(x):
    left = x
    right = jnp.roll(x, -1, axis=1)
    return left, right

def diffeq(flux, slvr, BC, dt, dy):
    @jit
    def bcsolver(x):
        right = flux(BC(x), slvr, dt, dy)
        left = jnp.concatenate((right[:, -1:], right[:, :-1]), axis=1)
        df = BC(-(right - left)/dy)
        return df
    return bcsolver

@jit
def wenowind_z(l1, l0, lr, r0, r1):
    eps = 1e-6
    pa1 = (2*l1 - 7*l0 + 11*lr) / 6
    pa2 = (-l0 + 5*lr + 2*r0) / 6
    pa3 = (2*lr + 5*r0 - r1) / 6
    d1 = l1 - 2*l0 + lr
    d2 = l0 - 2*lr + r0
    d3 = lr - 2*r0 + r1
    b1 = 13/12 * d1**2 + 0.25 * (l1 - 4*l0 + 3*lr)**2
    b2 = 13/12 * d2**2 + 0.25 * (l0 - r0)**2
    b3 = 13/12 * d3**2 + 0.25 * (3*lr - 4*r0 + r1)**2
    tau = jnp.abs(b1 - b3)
    b1_inv = 1.0 / (eps + b1)
    b2_inv = 1.0 / (eps + b2)
    b3_inv = 1.0 / (eps + b3)
    tau_factor1 = 1 + (tau * b1_inv)**2
    tau_factor2 = 1 + (tau * b2_inv)**2
    tau_factor3 = 1 + (tau * b3_inv)**2
    w1 = 0.1 * b1_inv**2 * tau_factor1
    w2 = 0.6 * b2_inv**2 * tau_factor2
    w3 = 0.3 * b3_inv**2 * tau_factor3
    ws_inv = 1.0 / (w1 + w2 + w3)
    reconstp = (w1*pa1 + w2*pa2 + w3*pa3) * ws_inv
    return reconstp

@jit
def weno5z(x):
    Z = x.shape[1]
    left = jnp.zeros_like(x)
    right = jnp.roll(x, -1, axis=1)
    j = jnp.arange(2, Z-3)
    wenowind_vmap = vmap(wenowind_z, in_axes=(0,0,0,0,0))
    left = left.at[:,j].set(wenowind_vmap(x[:,j-2],x[:,j-1],x[:,j],x[:,j+1],x[:,j+2]))
    right = right.at[:,j].set(wenowind_vmap(x[:,j+3],x[:,j+2],x[:,j+1],x[:,j],x[:,j-1]))
    return left, right

@jit
def wenowind(l1, l0, lr, r0, r1):
    eps = 1e-6
    pa1 = (2*l1 - 7*l0 + 11*lr) / 6
    pa2 = (-l0 + 5*lr + 2*r0) / 6
    pa3 = (2*lr + 5*r0 - r1) / 6
    d1 = l1 - 2*l0 + lr
    d2 = l0 - 2*lr + r0
    d3 = lr - 2*r0 + r1
    b1 = 13/12 * d1**2 + 0.25 * (l1 - 4*l0 + 3*lr)**2
    b2 = 13/12 * d2**2 + 0.25 * (l0 - r0)**2
    b3 = 13/12 * d3**2 + 0.25 * (3*lr - 4*r0 + r1)**2
    b1_inv_sq = 1.0 / (eps + b1)**2
    b2_inv_sq = 1.0 / (eps + b2)**2
    b3_inv_sq = 1.0 / (eps + b3)**2
    w1 = 0.1 * b1_inv_sq
    w2 = 0.6 * b2_inv_sq
    w3 = 0.3 * b3_inv_sq
    ws_inv = 1.0 / (w1 + w2 + w3)
    reconstp = (w1*pa1 + w2*pa2 + w3*pa3) * ws_inv
    return reconstp

@jit
def weno5(x):
    Z = x.shape[1]
    left = jnp.zeros_like(x)
    right = jnp.roll(x, -1, axis=1)
    j = jnp.arange(2, Z-3)
    wenowind_vmap = vmap(wenowind, in_axes=(0,0,0,0,0))
    left = left.at[:,j].set(wenowind_vmap(x[:,j-2],x[:,j-1],x[:,j],x[:,j+1],x[:,j+2]))
    right = right.at[:,j].set(wenowind_vmap(x[:,j+3],x[:,j+2],x[:,j+1],x[:,j],x[:,j-1]))
    return left, right

def RK1(L, u, dt):
    u_new = u + dt * L(u)
    return u_new

def RK3(L, u, dt):
    u1 = u + dt * L(u)
    u2 = u + 0.25 * dt * L(u1)
    up = (u + 2*u2 + 2*dt*L(u2))/3
    return up

def RK4(L, u, dt):
    u1 = u + 0.5 * dt * L(u)
    u2 = u + 0.5 * dt * L(u1)
    u3 = u + dt * L(u2)
    up = (-u + u1 + 2*u2 + u3 + 0.5*dt*L(u3))/3
    return up

def plot_initial_state(u, y, t, cellaverage_type):
    with plt.style.context('dark_background'):
        figure, axes = plt.subplots(3, 1, figsize=(12, 12))
        M = 1.1 * (jnp.max(jnp.abs(u)) + 0.02)
        colors = ['r', 'g', 'b']
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
    plt.pause(0.01)

def main():
    parser = argparse.ArgumentParser(description='1D Hydrodynamics Solver')
    parser.add_argument('--backend', type=str, choices=['cpu', 'gpu', 'metal', 'cuda', 'auto'], 
                        default='auto', help='Force JAX backend (cpu/gpu/metal/cuda/auto)')
    parser.add_argument('--precision', type=str, choices=['32', '64'], default='32',
                        help='Float precision: 32 (faster) or 64 (more accurate)')
    parser.add_argument('--test', type=int, choices=[1, 2], default=2,
                      help='Test case number (1: density wave, 2: Riemann problem)')
    parser.add_argument('--flux', type=str, choices=['laxf'], default='laxf',
                      help='Flux type')
    parser.add_argument('--solver', type=str, choices=['solverf', 'weno5', 'weno5z'], default='weno5',
                      help='Solver type')
    parser.add_argument('--integrator', type=str, choices=['RK1', 'RK3', 'RK4'], default='RK1',
                      help='Time integration method')
    parser.add_argument('--grid-size', type=int, default=5000,
                      help='Grid size')
    parser.add_argument('--cfl', type=float, default=0.6,
                      help='CFL number')
    parser.add_argument('--cell-average', type=str, choices=['standard', 'accurate'], default='standard',
                      help='Cell averaging method')
    parser.add_argument('--no-plot', action='store_true',
                      help='Disable real-time plotting for better performance')
    parser.add_argument('--plot-interval', type=int, default=1,
                      help='Update plot every N iterations (default: 1)')
    parser.add_argument('--fold', type=int, default=20,
                      help='Number of steps per timing measurement (default: 20)')
    args = parser.parse_args()
    flux_type = globals()[args.flux]
    slvr = globals()[args.solver]
    integ = globals()[args.integrator]
    cellaverage_type = cellaverage_acc if args.cell_average == 'accurate' else cellaverage
    Z = args.grid_size
    Co = args.cfl
    rhoJ = jnp.array([1, 0.125])
    uJ = jnp.array([0, 0])
    pJ = jnp.array([1, 0.1])
    tests = {
        1: {"BC": outflow, "u0": densityfn, "xlims": np.array([0, 2]), "Tfinal": 0.1, "rd": 'den_wave'},
        2: {"BC": outflow, "u0": riemann(rhoJ, uJ, pJ),
            "xlims": np.array([-0.5, 0.5]), "Tfinal": 0.015, "rd": 'riemann'}
    }
    test = tests[args.test]
    BC, u0, xlims, Tfinal, rd = test["BC"], test["u0"], test["xlims"], test["Tfinal"], test["rd"]
    plots = not args.no_plot
    plot_interval = args.plot_interval
    t, n = 0, 0
    dy = jnp.abs((xlims[1]-xlims[0])/(Z+6))
    Z = Z+6
    y = jnp.linspace(xlims[0]-2.5*dy, xlims[1]+2.5*dy, Z)
    u = BC(cellaverage_type(u0, y, dy))
    dt = Co * dy / jnp.max(jnp.abs(EigA(u)[0]))
    if plots:
        figure, axes, lines = plot_initial_state(u, y, t, cellaverage_type)
    else:
        figure, axes, lines = None, None, None
    fold = args.fold
    print(f'Initiating simulation with:')
    print(f'  Backend: {args.backend}')
    print(f'  Precision: {precision}-bit')
    print(f'  Grid size: {Z}')
    print(f'  Time integrator: {args.integrator}')
    print(f'  Flux solver: {args.flux}')
    print(f'  Spatial solver: {args.solver}')
    print(f'  CFL number: {Co}')
    print(f'  Real-time plotting: {plots}')
    if plots:
        print(f'  Plot interval: {plot_interval}')
    print(f'  Steps per timing: {fold}')
    print(f'  Test case: {args.test}')
    print('Pre-compiling simulation kernel...')
    @jit
    def timestep_fold(state):
        u, t, dt_current = state
        def body_fn(i, carry):
            u, t, dt_inner = carry
            def L_operator(u_state):
                u_bc = BC(u_state)
                left, right = slvr(u_bc)
                eigval, _ = EigA(u_bc)
                amax = jnp.max(jnp.abs(eigval))
                flux_left_right = flux(left)
                flux_right_right = flux(right)
                flux_vals = 0.5 * (flux_left_right + flux_right_right - amax * (right - left))
                flux_left = jnp.concatenate((flux_vals[:, -1:], flux_vals[:, :-1]), axis=1)
                df = BC(-(flux_vals - flux_left) / dy)
                return df
            u_new = integ(L_operator, u, dt_inner)
            eigval, _ = EigA(u_new)
            amax = jnp.max(jnp.abs(eigval))
            dt_new = Co * dy / amax
            return u_new, t + dt_inner, dt_new
        return lax.fori_loop(0, fold, body_fn, (u, t, dt_current))
    _ = timestep_fold((u, t, dt))
    print('Compilation complete. Initiating loop...')
    plot_counter = 0
    n = 0
    while t < Tfinal:
        t0 = time.time()
        u, t, dt = timestep_fold((u, t, dt))
        u.block_until_ready()
        t1 = time.time()
        n += fold
        print(f"[{n:04d}] t={t:.3f} Mzps={Z * fold * 1e-6 / (t1 - t0):.4f}")
        if plots and plot_counter % plot_interval == 0:
            update_plot(u, t, lines, axes)
        plot_counter += 1
    analytic_u = RiemannExact(jnp.array([rhoJ, rhoJ*uJ, 0.5*rhoJ*uJ**2 + pJ/(gamma-1)]), gamma, t) if rd == 'riemann' else lambda x: u0(x - t)
    if plots:
        x_coords = jnp.linspace(y[0], y[Z-1], int(max(2 * Z, 1e3)))
        for i, ax in enumerate(axes):
            ax.plot(x_coords, analytic_u(x_coords)[i, :], 'w:')
            ax.set_title(f't = {t}')
        [line.set_ydata(u[i, :]) for i, line in enumerate(lines)]
        plt.show()
    if rd == 'den_wave':
        bool_mask = (xlims[0] < y[:-1]) & (y[:-1] < xlims[1]) if cellaverage_type == cellaverage else (xlims[0] < y) & (y < xlims[1])
        difference = u[:, bool_mask] - cellaverage_type(analytic_u, y, dy)[:, bool_mask]
        l2_norm = jnp.linalg.norm(difference * dy, axis=1)
        print('L2 norm\n', l2_norm)
    class TestResults:
        def __init__(self):
            self.final_state = u
            self.l2_norm = l2_norm if rd == 'den_wave' else None
            self.t = t
            self.grid_size = Z
            self.dt = dt
    return TestResults()

if __name__ == '__main__':
    main()
