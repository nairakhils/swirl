import jax
jax.config.update('jax_platform_name', "cpu")
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np


gamma = 1.4

@jit
def flux(x):
    rho = x[0, :]
    m = x[1, :]
    E = x[2, :]

    u = m / rho
    p = (gamma - 1) * (E - 0.5 * rho * u ** 2)

    flux_rho = m
    flux_m = u * m + p
    flux_E = (E + p) * u

    flux = jnp.concatenate([jnp.array([flux_rho]), jnp.array([flux_m]), jnp.array([flux_E])])
    return flux

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
    Hen = 0.5 * u ** 2 + a ** 2 / (gamma - 1)
    u_squared = u ** 2
    ua = u * a

    eigvec = jnp.stack([jnp.ones_like(u), 
                        jnp.where(jnp.arange(len(u)) % 3 == 0, u - a, jnp.where(jnp.arange(len(u)) % 3 == 1, u, u + a)), 
                        jnp.where(jnp.arange(len(u)) % 3 == 0, Hen - ua, jnp.where(jnp.arange(len(u)) % 3 == 1, 0.5 * u_squared, Hen + ua))])

    return eigval, eigvec

# Newton-Raphson method for the Exact-Riemann Solution 
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

# Exact Riemann Solution
# Reference: Toro, E.F., "Riemann Solvers and Numerical Methods for Fluid Dynamics", Springer-Verlag, 3rd edition, 2009, pp. 136-164.
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
        # solution types
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
                rho = lambda x: density_side * (x < (velocity_side - a_side) * t) + rhoS * (x >= (us - aS) * t) + density_side * jnp.abs(
                    2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 / (gamma - 1)) * (
                                        x >= (velocity_side - a_side) * t) * (x < (us - aS) * t) if side == 'left' else density_side * (x > (velocity_side + a_side) * t) + rhoS * (x <= (us + aS) * t) + density_side * jnp.abs(
                    2 / (gamma + 1) - (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 / (gamma - 1)) * (
                                        x <= (velocity_side + a_side) * t) * (x > (us + aS) * t)
                u = lambda x: velocity_side * (x < (velocity_side - a_side) * t) + us * (x >= (us - aS) * t) + 2 / (gamma + 1) * (
                        a_side + (gamma - 1) / 2 * velocity_side + x / t) * (x >= (velocity_side - a_side) * t) * (x < (us - aS) * t) if side == 'left' else velocity_side * (x > (velocity_side + a_side) * t) + us * (x <= (us + aS) * t) + 2 / (gamma + 1) * (
                        -a_side + (gamma - 1) / 2 * velocity_side + x / t) * (x <= (velocity_side + a_side) * t) * (x > (us + aS) * t)
                p = lambda x: pressure_side * (x < (velocity_side - a_side) * t) + ps * (x >= (us - aS) * t) + pressure_side * jnp.abs(
                    2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 * gamma / (gamma - 1)) * (
                                        x >= (velocity_side - a_side) * t) * (x < (us - aS) * t) if side == 'left' else pressure_side * (x > (velocity_side + a_side) * t) + ps * (x <= (us + aS) * t) + pressure_side * jnp.abs(
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
    # Define the initial conditions
    x1 = jnp.array([r, r*u, 0.5*r*u**2 + p/(gamma-1)])

    @jit
    def arrcheck(y):
        # Ensure y is an array
        y = jnp.asarray(y)
        # Expand dimensions if y is a scalar
        y = y if y.ndim > 0 else jnp.expand_dims(y, 0)
        # Compute the result
        arr = jnp.where(y<0, x1[:,0,None], x1[:,1,None])
        return arr

    return arrcheck

# densityfn function: Returns a function that computes the initial conditions for a density wave problem given an array of positions.

def densityfn(y):
    # Define the density profile
    rho = jnp.sin(jnp.pi*y) + 5 # The "+ 2" ensures that the density is always positive

    # Assume stationary gas (u = 0) and constant pressure (p = 1)
    u = jnp.zeros_like(y)
    p = jnp.ones_like(y)

    # Calculate energy using the equation of state
    e = p / ((gamma - 1) * rho)

    # Stack the arrays
    arr = jnp.vstack([rho, rho * u, rho * e])

    return arr

# Calculating Cell Average
def simpsons_rule(func, a, b, n=100):
    h = (b - a) / n
    x = jnp.linspace(a, b, n+1)
    y = func(x)
    return h / 3 * (y[:,0] + y[:,-1] + 4 * jnp.sum(y[:,1:-1:2], axis=1) + 2 * jnp.sum(y[:,2:-1:2], axis=1))

def cellaverage_acc(arr, y, dy, n=100):
    x = jnp.zeros((3, len(y)))
    arravg = lambda y: simpsons_rule(arr, y-0.5*dy, y+0.5*dy, n) / dy
    for i in range(len(y)):
        x = x.at[:,i].set(arravg(y[i]))
    return x

def cellaverage(arr, y, dy):
    mid_points = 0.5 * (y[:-1] + y[1:])
    x = arr(mid_points).squeeze()
    return x
 
# Periodic Boundary Conditions
@jit
def periodic(x):
    Z = x.shape[1]
    y = x.at[:,:3].set(x[:,Z-6:Z-3])
    y = y.at[:,Z-3:].set(x[:,3:6])
    return y

#Outflow Boundary Conditions
@jit
def outflow(x):
    Z = x.shape[1]
    y = x.at[:,:3].set(x[:,3:4])
    y = y.at[:,Z-3:].set(x[:,Z-4:Z-3])
    return y


# Lax-Friedrichs flux
def laxf(x, slvr, dt, dx): # LLF flux
    left, right = slvr(x)
    eigval, eigvec = EigA(x)
    amax = jnp.max(jnp.abs(eigval[2,:] - eigval[1,:]) + jnp.abs(eigval[1,:]))
    flux1 = 0.5 * (flux(left) + flux(right) - amax * (right - left))
    return flux1

# Basic Grid Solver
@jit
def solverf(x):
    left = x.copy()
    right = jnp.hstack((x[:, 1:], x[:, :1]))
    return left, right

# diffeq function: Constructs a differential equation solver with specified flux, solver, boundary conditions, time step, and spatial step.
def diffeq(flux, slvr, BC, dt, dy):
    @jit
    def bcsolver(x):
        right = flux(BC(x), slvr, dt, dy)
        left = jnp.concatenate((right[:, -1:], right[:, :-1]), axis=1)
        df = BC(-(right - left)/dy)
        return df
    return bcsolver

# WENO-Z Method
@jit
def wenowind_z(l1, l0, lr, r0, r1):
    # polynomial approx
    pa1 = (2*l1 - 7*l0 + 11*lr )/6
    pa2 = ( -l0 + 5*lr   + 2*r0)/6
    pa3 = (2*lr   + 5*r0 -   r1)/6
    # smoothness indicators
    b1 = 13/12*(l1 - 2*l0 + lr  )**2 + 0.25*(l1 - 4*l0 + 3*lr)**2
    b2 = 13/12*(l0 - 2*lr   + r0)**2 + 0.25*(l0 - r0)**2
    b3 = 13/12*(lr   - 2*r0 + r1)**2 + 0.25*(3*lr - 4*r0 + r1)**2
    # weights
    tau = abs(b1 - b3)
    w1 = 0.1 / (1e-6 + b1)**2 * (1 + (tau / (1e-6 + b1))**2)
    w2 = 0.6 / (1e-6 + b2)**2 * (1 + (tau / (1e-6 + b2))**2)
    w3 = 0.3 / (1e-6 + b3)**2 * (1 + (tau / (1e-6 + b3))**2)
    ws = w1 + w2 + w3
    # reconstructed cell-interface value
    reconstp = (w1*pa1 + w2*pa2 + w3*pa3)/ws
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


# Fifth-order WENO Method - WENO-JS
@jit
def wenowind(l1, l0, lr, r0, r1):
    # polynomial approx
    pa1 = (2*l1 - 7*l0 + 11*lr )/6
    pa2 = ( -l0 + 5*lr   + 2*r0)/6
    pa3 = (2*lr   + 5*r0 -   r1)/6
    # smoothness indicators
    b1 = 13/12*(l1 - 2*l0 + lr  )**2 + 0.25*(l1 - 4*l0 + 3*lr)**2
    b2 = 13/12*(l0 - 2*lr   + r0)**2 + 0.25*(l0 - r0)**2
    b3 = 13/12*(lr   - 2*r0 + r1)**2 + 0.25*(3*lr - 4*r0 + r1)**2
    # weights
    w1 = 0.1 / (1e-6 + b1)**2
    w2 = 0.6 / (1e-6 + b2)**2
    w3 = 0.3 / (1e-6 + b3)**2
    ws = w1 + w2 + w3
    # reconstructed cell-interface value
    reconstp = (w1*pa1 + w2*pa2 + w3*pa3)/ws
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

# Roe Solver
def roe_eig(L, R):
    sqrt_dens_L, sqrt_dens_R = jnp.sqrt(L[0]), jnp.sqrt(R[0])
    v_L, v_R = L[1] / L[0], R[1] / R[0]
    v_roe = (sqrt_dens_L * v_L + sqrt_dens_R * v_R) / (sqrt_dens_L + sqrt_dens_R)
    P_L, P_R = (gamma - 1) * (L[2] - 0.5 * L[0] * v_L**2), (gamma - 1) * (R[2] - 0.5 * R[0] * v_R**2)
    c_L, c_R = jnp.sqrt(gamma * P_L / L[0]), jnp.sqrt(gamma * P_R / R[0])
    h_L, h_R = 0.5 * v_L**2 + c_L**2 / (gamma - 1), 0.5 * v_R**2 + c_R**2 / (gamma - 1)
    h_roe = (sqrt_dens_L * h_L + sqrt_dens_R * h_R) / (sqrt_dens_L + sqrt_dens_R)
    e_int = h_roe - 0.5 * v_roe**2
    c = jnp.sqrt((gamma - 1) * e_int)
    r_eig = jnp.array([1, 1, 1, v_roe - c, v_roe, v_roe + c, h_roe - v_roe * c, 0.5 * v_roe**2, h_roe + v_roe * c])
    l_eig = 0.5 / e_int * jnp.array([0.5 * v_roe**2 + v_roe * e_int / c, -e_int / c - v_roe, 1, 2 * e_int - v_roe**2, 2 * v_roe, -2, 0.5 * v_roe**2 - v_roe * e_int / c, e_int / c - v_roe, 1])
    return jnp.reshape(r_eig, (3, 3)), jnp.reshape(l_eig, (3, 3))

#Modified Roe Solver - Roe + WENO-Z
def roe(u):
    Z = u.shape[1]
    u1 = u.copy()
    u2 = jnp.roll(u, -1, axis=1)
    w = u.copy()

    def bf(j, v):
        w, u1, u2 = v
        rm, lm = roe_eig(u[:, j], u[:, j + 1])

        def il(k, w):
            return lax.dynamic_update_slice(w, jnp.dot(lm, u[:, k])[:, None], (0, k))

        w = lax.fori_loop(j - 2, j + 4, il, w)

        wm = w[:, j].copy()
        wp = w[:, j + 1].copy()

        for i in range(3):
            wm = lax.dynamic_update_slice(wm, wenowind_z(w[i, j - 2], w[i, j - 1], w[i, j], w[i, j + 1], w[i, j + 2])[None], (i,))
            wp = lax.dynamic_update_slice(wp, wenowind_z(w[i, j + 3], w[i, j + 2], w[i, j + 1], w[i, j], w[i, j - 1])[None], (i,))

        u1 = lax.dynamic_update_slice(u1, jnp.dot(rm, wm)[:, None], (0, j))
        u2 = lax.dynamic_update_slice(u2, jnp.dot(rm, wp)[:, None], (0, j))

        return w, u1, u2

    w, u1, u2 = lax.fori_loop(2, Z - 3, bf, (w, u1, u2))

    return u1, u2

# RK1 method: Euler method
def RK1(L, u, dt):
    u_new = u + dt * L(u)
    return u_new

# RK3 method: A third-order Runge-Kutta method for approximating the solution to a differential equation.
def RK3(L, u, dt):
    stages = jnp.array([1, 0.25])
    u_values = [u]
    for stage in stages:
        u_new = u + dt*stage*L(u_values[-1])
        u_values.append(u_new)
    up = (u + 2*u_values[-1] + 2*dt*L(u_values[-1]))/3
    return up

# RK4 method: A fourth-order Runge-Kutta method for approximating the solution to a differential equation.
def RK4(L, u, dt):
    stages = jnp.array([0.5, 0.5, 1])
    u_values = [u]
    for stage in stages:
        u_new = u + dt*stage*L(u_values[-1])
        u_values.append(u_new)
    up = (-u + u_values[1] + 2*u_values[2] + u_values[3] + 0.5*dt*L(u_values[-1]))/3
    return up

