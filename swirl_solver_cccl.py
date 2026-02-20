#!/usr/bin/env python3
"""1D Compressible Euler Solver — CuPy (CUDA) Backend
Port of swirl_solver.py from JAX to CuPy.
Uses WENO5 reconstruction with Lax-Friedrichs flux splitting.
Array layout: [3, Z] where row 0=density, 1=momentum, 2=energy.
"""
import argparse
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

gamma = 1.4

# ---------------------------------------------------------------------------
# WENO5 fused CUDA RawKernels (shared-memory tiled, 256 threads/block)
# Both JS and Z variants live in a single compilation unit.
# The source uses REAL / FABS placeholders that are text-replaced per dtype.
# ---------------------------------------------------------------------------
_WENO5_KERNEL_SRC = r"""
#define BLOCK_SIZE 256
#define HALO_L     2
#define HALO_R     3
#define TILE       (BLOCK_SIZE + HALO_L + HALO_R)   /* 261 */

/* ---- shared-memory tile loader (identical for JS & Z) ---- */
__device__ __forceinline__
void load_tile(const REAL* __restrict__ U, REAL* smem, int Z,
               int tid, int gbase)
{
    for (int v = 0; v < 3; v++) {
        REAL*       s  = smem + v * TILE;
        const REAL* Uv = U    + v * Z;
        int gj = gbase + tid + HALO_L;
        if (gj < Z) s[tid + HALO_L] = Uv[gj];
        if (tid < HALO_L) {
            int hj = gbase + tid;
            if (hj < Z) s[tid] = Uv[hj];
        }
        if (tid < HALO_R) {
            int pos = BLOCK_SIZE + HALO_L + tid;
            int hj  = gbase + pos;
            if (hj < Z) s[pos] = Uv[hj];
        }
    }
}

/* ---- smoothness indicators + candidate polynomials (shared) ---- */
__device__ __forceinline__
void weno_polys(REAL a, REAL b, REAL c, REAL d, REAL e,
                REAL* p1, REAL* p2, REAL* p3,
                REAL* B1, REAL* B2, REAL* B3)
{
    *p1 = ((REAL)2*a - (REAL)7*b + (REAL)11*c) / (REAL)6;
    *p2 = (-b + (REAL)5*c + (REAL)2*d)          / (REAL)6;
    *p3 = ((REAL)2*c + (REAL)5*d - e)            / (REAL)6;
    REAL dd1 = a - (REAL)2*b + c;
    REAL dd2 = b - (REAL)2*c + d;
    REAL dd3 = c - (REAL)2*d + e;
    REAL tt1 = a - (REAL)4*b + (REAL)3*c;
    REAL tt2 = b - d;
    REAL tt3 = (REAL)3*c - (REAL)4*d + e;
    *B1 = (REAL)(13.0/12.0)*dd1*dd1 + (REAL)0.25*tt1*tt1;
    *B2 = (REAL)(13.0/12.0)*dd2*dd2 + (REAL)0.25*tt2*tt2;
    *B3 = (REAL)(13.0/12.0)*dd3*dd3 + (REAL)0.25*tt3*tt3;
}

/* ========================== WENO5-JS kernel ========================== */
extern "C" __global__
void weno5_js_kernel(const REAL* __restrict__ U,
                     REAL*       __restrict__ L,
                     REAL*       __restrict__ R,
                     const int Z)
{
    extern __shared__ REAL smem[];
    const int tid   = threadIdx.x;
    const int gbase = blockIdx.x * BLOCK_SIZE;
    const int j     = gbase + tid + 2;

    load_tile(U, smem, Z, tid, gbase);
    __syncthreads();
    if (j >= Z - 3) return;

    const int  li  = tid + HALO_L;
    const REAL eps = (REAL)1e-6;

    for (int v = 0; v < 3; v++) {
        const REAL* s = smem + v * TILE;
        REAL p1, p2, p3, B1, B2, B3;

        /* left reconstruction: stencil (j-2 … j+2) */
        weno_polys(s[li-2], s[li-1], s[li], s[li+1], s[li+2],
                   &p1, &p2, &p3, &B1, &B2, &B3);
        REAL eb1 = eps + B1, eb2 = eps + B2, eb3 = eps + B3;
        REAL w1 = (REAL)0.1 / (eb1*eb1);
        REAL w2 = (REAL)0.6 / (eb2*eb2);
        REAL w3 = (REAL)0.3 / (eb3*eb3);
        L[v*Z + j] = (w1*p1 + w2*p2 + w3*p3) / (w1 + w2 + w3);

        /* right reconstruction: mirror stencil (j+3 … j-1) */
        weno_polys(s[li+3], s[li+2], s[li+1], s[li], s[li-1],
                   &p1, &p2, &p3, &B1, &B2, &B3);
        eb1 = eps + B1; eb2 = eps + B2; eb3 = eps + B3;
        w1 = (REAL)0.1 / (eb1*eb1);
        w2 = (REAL)0.6 / (eb2*eb2);
        w3 = (REAL)0.3 / (eb3*eb3);
        R[v*Z + j] = (w1*p1 + w2*p2 + w3*p3) / (w1 + w2 + w3);
    }
}

/* ========================== WENO5-Z kernel =========================== */
extern "C" __global__
void weno5_z_kernel(const REAL* __restrict__ U,
                    REAL*       __restrict__ L,
                    REAL*       __restrict__ R,
                    const int Z)
{
    extern __shared__ REAL smem[];
    const int tid   = threadIdx.x;
    const int gbase = blockIdx.x * BLOCK_SIZE;
    const int j     = gbase + tid + 2;

    load_tile(U, smem, Z, tid, gbase);
    __syncthreads();
    if (j >= Z - 3) return;

    const int  li  = tid + HALO_L;
    const REAL eps = (REAL)1e-6;

    for (int v = 0; v < 3; v++) {
        const REAL* s = smem + v * TILE;
        REAL p1, p2, p3, B1, B2, B3;

        /* left reconstruction */
        weno_polys(s[li-2], s[li-1], s[li], s[li+1], s[li+2],
                   &p1, &p2, &p3, &B1, &B2, &B3);
        REAL tau = FABS(B1 - B3);
        REAL b1i = (REAL)1 / (eps + B1);
        REAL b2i = (REAL)1 / (eps + B2);
        REAL b3i = (REAL)1 / (eps + B3);
        REAL tf1 = tau * b1i;  tf1 = (REAL)1 + tf1 * tf1;
        REAL tf2 = tau * b2i;  tf2 = (REAL)1 + tf2 * tf2;
        REAL tf3 = tau * b3i;  tf3 = (REAL)1 + tf3 * tf3;
        REAL w1 = (REAL)0.1 * b1i * b1i * tf1;
        REAL w2 = (REAL)0.6 * b2i * b2i * tf2;
        REAL w3 = (REAL)0.3 * b3i * b3i * tf3;
        L[v*Z + j] = (w1*p1 + w2*p2 + w3*p3) / (w1 + w2 + w3);

        /* right reconstruction */
        weno_polys(s[li+3], s[li+2], s[li+1], s[li], s[li-1],
                   &p1, &p2, &p3, &B1, &B2, &B3);
        tau = FABS(B1 - B3);
        b1i = (REAL)1 / (eps + B1);
        b2i = (REAL)1 / (eps + B2);
        b3i = (REAL)1 / (eps + B3);
        tf1 = tau * b1i;  tf1 = (REAL)1 + tf1 * tf1;
        tf2 = tau * b2i;  tf2 = (REAL)1 + tf2 * tf2;
        tf3 = tau * b3i;  tf3 = (REAL)1 + tf3 * tf3;
        w1 = (REAL)0.1 * b1i * b1i * tf1;
        w2 = (REAL)0.6 * b2i * b2i * tf2;
        w3 = (REAL)0.3 * b3i * b3i * tf3;
        R[v*Z + j] = (w1*p1 + w2*p2 + w3*p3) / (w1 + w2 + w3);
    }
}
"""

_WENO5_BLOCK = 256
_weno_kernel_cache = {}


def _get_weno_kernels(dtype):
    """Compile (or fetch cached) JS and Z kernels for the given dtype."""
    if dtype not in _weno_kernel_cache:
        if dtype == cp.float32:
            src = _WENO5_KERNEL_SRC.replace('REAL', 'float').replace('FABS', 'fabsf')
        else:
            src = _WENO5_KERNEL_SRC.replace('REAL', 'double').replace('FABS', 'fabs')
        js = cp.RawKernel(src, 'weno5_js_kernel')
        zk = cp.RawKernel(src, 'weno5_z_kernel')
        _weno_kernel_cache[dtype] = (js, zk)
    return _weno_kernel_cache[dtype]


def _launch_weno_kernel(kernel, x):
    """Allocate left/right, launch *kernel*, return (left, right)."""
    Z = x.shape[1]
    left = cp.zeros_like(x)
    right = cp.roll(x, -1, axis=1)
    n_interior = Z - 5
    if n_interior > 0:
        grid = ((n_interior + _WENO5_BLOCK - 1) // _WENO5_BLOCK,)
        smem = 3 * (_WENO5_BLOCK + 5) * x.dtype.itemsize
        kernel(grid, (_WENO5_BLOCK,),
               (x, left, right, np.int32(Z)),
               shared_mem=smem)
    return left, right


def flux(x):
    rho = x[0, :]
    m = x[1, :]
    E = x[2, :]
    u = m / rho
    u_sq = u * u
    p = (gamma - 1) * (E - 0.5 * rho * u_sq)
    return cp.stack([m, m * u + p, (E + p) * u])


def compute_values(x):
    rho = x[0, :]
    m = x[1, :]
    E = x[2, :]
    u = m / rho
    p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
    a = cp.sqrt(gamma * p / rho)
    return cp.array([rho, m, E, u, p, a])


def EigA(x):
    rho, m, E = x
    u = m / rho
    p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
    a = cp.sqrt(gamma * p / rho)
    eigval = cp.array([u - a, u, u + a])
    return eigval, None


def numerical_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2.0 * h)


def newton_raphson(func, x0, tol=1e-5, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        if abs(float(fx)) < tol:
            return x
        dfx = numerical_derivative(func, x)
        if float(dfx) == 0:
            print("Zero derivative. No solution found.")
            return None
        x = x - fx / dfx
    print("Exceeded maximum iterations. No solution found.")
    return None


def RiemannExact(initial_state, gamma, t):
    density, momentum, energy = initial_state[:, :2]
    velocity = momentum / density
    pressure = (gamma - 1) * (energy - 0.5 * momentum * velocity)
    a = cp.sqrt(gamma * pressure / density)
    A = 2 / ((gamma + 1) * density)
    B = (gamma - 1) / (gamma + 1) * pressure
    du = cp.diff(velocity)[0]
    fluxt = lambda p, i: (p - pressure[i]) * cp.sqrt(A[i] / (p + B[i])) * (p > pressure[i]) + \
                         2 * a[i] / (gamma - 1) * ((p / pressure[i]) ** ((gamma - 1) / (2 * gamma)) - 1) * (p <= pressure[i])
    fluxF = lambda p: fluxt(p, 0) + fluxt(p, 1) + du
    if float(velocity[1] - velocity[0]) > 2 * float(a[0] + a[1]) / (gamma - 1):
        print('vacuum condition')
        us, rholeft, uleft, left_pressureeft, rhoright, uright, pright = 0, lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0, lambda x: 0
    else:
        p0 = max(np.finfo(np.float64).eps, float(0.5 * (pressure[0] + pressure[1]) - du * (density[0] + density[1]) * (a[0] + a[1]) / 8))
        ps = newton_raphson(fluxF, p0)
        us = 0.5 * (velocity[1] + velocity[0]) + 0.5 * (fluxt(ps, 1) - fluxt(ps, 0))
        def calculate_values(pressure_side, density_side, velocity_side, a_side, ps, us, t, gamma, side):
            if ps > pressure_side:
                rhoS = density_side * ((gamma - 1) / (gamma + 1) + ps / pressure_side) / ((gamma - 1) / (gamma + 1) * ps / pressure_side + 1)
                S = velocity_side - a_side * cp.sqrt((gamma + 1) / (2 * gamma) * ps / pressure_side + (gamma - 1) / (2 * gamma)) if side == 'left' else velocity_side + a_side * cp.sqrt((gamma + 1) / (2 * gamma) * ps / pressure_side + (gamma - 1) / (2 * gamma))
                rho = lambda x: density_side * (x < S * t) + rhoS * (x >= S * t) if side == 'left' else density_side * (x > S * t) + rhoS * (x <= S * t)
                u = lambda x: velocity_side * (x < S * t) + us * (x >= S * t) if side == 'left' else velocity_side * (x > S * t) + us * (x <= S * t)
                p = lambda x: pressure_side * (x < S * t) + ps * (x >= S * t) if side == 'left' else pressure_side * (x > S * t) + ps * (x <= S * t)
            else:
                aS = a_side + (velocity_side - us) * (gamma - 1) / 2 if side == 'left' else a_side + (us - velocity_side) * (gamma - 1) / 2
                rhoS = gamma * ps / aS ** 2
                if side == 'left':
                    rho = lambda x: density_side * (x < (velocity_side - a_side) * t) + rhoS * (x >= (us - aS) * t) + density_side * cp.abs(
                        2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 / (gamma - 1)) * (
                            x >= (velocity_side - a_side) * t) * (x < (us - aS) * t)
                    u = lambda x: velocity_side * (x < (velocity_side - a_side) * t) + us * (x >= (us - aS) * t) + 2 / (gamma + 1) * (
                            a_side + (gamma - 1) / 2 * velocity_side + x / t) * (x >= (velocity_side - a_side) * t) * (x < (us - aS) * t)
                    p = lambda x: pressure_side * (x < (velocity_side - a_side) * t) + ps * (x >= (us - aS) * t) + pressure_side * cp.abs(
                        2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 * gamma / (gamma - 1)) * (
                            x >= (velocity_side - a_side) * t) * (x < (us - aS) * t)
                else:
                    rho = lambda x: density_side * (x > (velocity_side + a_side) * t) + rhoS * (x <= (us + aS) * t) + density_side * cp.abs(
                        2 / (gamma + 1) - (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 / (gamma - 1)) * (
                            x <= (velocity_side + a_side) * t) * (x > (us + aS) * t)
                    u = lambda x: velocity_side * (x > (velocity_side + a_side) * t) + us * (x <= (us + aS) * t) + 2 / (gamma + 1) * (
                            -a_side + (gamma - 1) / 2 * velocity_side + x / t) * (x <= (velocity_side + a_side) * t) * (x > (us + aS) * t)
                    p = lambda x: pressure_side * (x > (velocity_side + a_side) * t) + ps * (x <= (us + aS) * t) + pressure_side * cp.abs(
                        2 / (gamma + 1) - (gamma - 1) / ((gamma + 1) * a_side) * (velocity_side - x / t)) ** (2 * gamma / (gamma - 1)) * (
                            x <= (velocity_side + a_side) * t) * (x > (us + aS) * t)
            return rho, u, p
        rholeft, uleft, left_pressureeft = calculate_values(pressure[0], density[0], velocity[0], a[0], ps, us, t, gamma, 'left')
        rhoright, uright, pright = calculate_values(pressure[1], density[1], velocity[1], a[1], ps, us, t, gamma, 'right')
    def fin(x):
        x = cp.asarray(x)
        UL = cp.array([rholeft(x), rholeft(x)*uleft(x), 0.5*rholeft(x)*uleft(x)**2 + left_pressureeft(x)/(gamma-1)])
        UR = cp.array([rhoright(x), rhoright(x)*uright(x), 0.5*rhoright(x)*uright(x)**2 + pright(x)/(gamma-1)])
        u = cp.where(x < us * t, UL, UR)
        return u
    return fin


def riemann(r, u, p):
    x1 = cp.array([r, r*u, 0.5*r*u**2 + p/(gamma-1)])
    def arrcheck(y):
        y = cp.asarray(y)
        y = y if y.ndim > 0 else cp.expand_dims(y, 0)
        arr = cp.where(y < 0, x1[:, 0, None], x1[:, 1, None])
        return arr
    return arrcheck


def densityfn(y):
    rho = cp.sin(cp.pi * y) + 5
    u = cp.zeros_like(y)
    p = cp.ones_like(y)
    e = p / ((gamma - 1) * rho)
    arr = cp.vstack([rho, rho * u, rho * e])
    return arr


def simpsons_rule(func, a, b, n=100):
    h = (b - a) / n
    x = cp.linspace(a, b, n + 1)
    y = func(x)
    return h / 3 * (y[:, 0] + y[:, -1] + 4 * cp.sum(y[:, 1:-1:2], axis=1) + 2 * cp.sum(y[:, 2:-1:2], axis=1))


def cellaverage_acc(arr, y, dy, n=100):
    h = dy / n
    t_vals = cp.linspace(0.0, 1.0, n + 1)
    a = y - 0.5 * dy
    quad_pts = a[:, None] + t_vals[None, :] * dy
    flat_pts = quad_pts.reshape(-1)
    vals = arr(flat_pts)
    vals = vals.reshape(3, len(y), n + 1)
    integral = h / 3 * (vals[:, :, 0] + vals[:, :, -1]
                        + 4 * cp.sum(vals[:, :, 1:-1:2], axis=2)
                        + 2 * cp.sum(vals[:, :, 2:-1:2], axis=2))
    return integral / dy


def cellaverage(arr, y, dy):
    mid_points = 0.5 * (y[:-1] + y[1:])
    x = arr(mid_points).squeeze()
    return x


def periodic(x):
    Z = x.shape[1]
    y = x.copy()
    y[:, :3] = x[:, Z-6:Z-3]
    y[:, Z-3:] = x[:, 3:6]
    return y


def outflow(x):
    Z = x.shape[1]
    y = x.copy()
    y[:, :3] = x[:, 3:4]
    y[:, Z-3:] = x[:, Z-4:Z-3]
    return y


def laxf(x, slvr, dt, dx):
    left, right = slvr(x)
    eigval, _ = EigA(x)
    amax = cp.max(cp.abs(eigval))
    flux1 = 0.5 * (flux(left) + flux(right) - amax * (right - left))
    return flux1


def solverf(x):
    left = x
    right = cp.roll(x, -1, axis=1)
    return left, right


def diffeq(flux_fn, slvr, BC, dt, dy):
    def bcsolver(x):
        right = flux_fn(BC(x), slvr, dt, dy)
        left = cp.concatenate((right[:, -1:], right[:, :-1]), axis=1)
        df = BC(-(right - left) / dy)
        return df
    return bcsolver


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
    tau = cp.abs(b1 - b3)
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


def _weno5z_py(x):
    """Pure-CuPy WENO5-Z reference (for verification)."""
    Z = x.shape[1]
    left = cp.zeros_like(x)
    right = cp.roll(x, -1, axis=1)
    left[:, 2:Z-3] = wenowind_z(x[:, 0:Z-5], x[:, 1:Z-4], x[:, 2:Z-3], x[:, 3:Z-2], x[:, 4:Z-1])
    right[:, 2:Z-3] = wenowind_z(x[:, 5:Z], x[:, 4:Z-1], x[:, 3:Z-2], x[:, 2:Z-3], x[:, 1:Z-4])
    return left, right


def weno5z(x):
    _, z_kern = _get_weno_kernels(x.dtype)
    return _launch_weno_kernel(z_kern, x)


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


def _weno5_py(x):
    """Pure-CuPy WENO5-JS reference (for verification)."""
    Z = x.shape[1]
    left = cp.zeros_like(x)
    right = cp.roll(x, -1, axis=1)
    left[:, 2:Z-3] = wenowind(x[:, 0:Z-5], x[:, 1:Z-4], x[:, 2:Z-3], x[:, 3:Z-2], x[:, 4:Z-1])
    right[:, 2:Z-3] = wenowind(x[:, 5:Z], x[:, 4:Z-1], x[:, 3:Z-2], x[:, 2:Z-3], x[:, 1:Z-4])
    return left, right


def weno5(x):
    js_kern, _ = _get_weno_kernels(x.dtype)
    return _launch_weno_kernel(js_kern, x)


def RK1(L, u, dt):
    u_new = u + dt * L(u)
    return u_new


def RK3(L, u, dt):
    u1 = u + dt * L(u)
    u2 = u + 0.25 * dt * L(u1)
    up = (u + 2*u2 + 2*dt*L(u2)) / 3
    return up


def RK4(L, u, dt):
    u1 = u + 0.5 * dt * L(u)
    u2 = u + 0.5 * dt * L(u1)
    u3 = u + dt * L(u2)
    up = (-u + u1 + 2*u2 + u3 + 0.5*dt*L(u3)) / 3
    return up


def plot_initial_state(u, y, t, cellaverage_type):
    u_np = cp.asnumpy(u)
    y_np = cp.asnumpy(y)
    with plt.style.context('dark_background'):
        figure, axes = plt.subplots(3, 1, figsize=(12, 12))
        M = 1.1 * (float(cp.max(cp.abs(u))) + 0.02)
        colors = ['r', 'g', 'b']
        if cellaverage_type == cellaverage:
            y_midpoints = 0.5 * (y_np[1:] + y_np[:-1])
            lines = [ax.plot(y_midpoints, u_np[i, :], '.', linewidth=0.5, color=colors[i])[0] for i, ax in enumerate(axes)]
        else:
            lines = [ax.plot(y_np, u_np[i, :], '.', linewidth=0.5, color=colors[i])[0] for i, ax in enumerate(axes)]
        for ax, line, label in zip(axes, lines, ['Density', 'Velocity', 'Pressure']):
            ax.set_ylim([-M, M])
            ax.set_xlabel('x')
            ax.set_ylabel(label)
            ax.set_title(f't = {t}')
        plt.tight_layout()
        plt.draw()
    return figure, axes, lines


def update_plot(u, t, lines, axes):
    u_np = cp.asnumpy(u)
    for i, line in enumerate(lines):
        line.set_ydata(u_np[i, :])
        axes[i].set_title(f't = {t}')
    plt.draw()
    plt.pause(0.01)


def main():
    parser = argparse.ArgumentParser(description='1D Hydrodynamics Solver (CuPy/CUDA)')
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

    print(f"CuPy backend: cuda (device {cp.cuda.runtime.getDevice()})")

    flux_type = globals()[args.flux]
    slvr = globals()[args.solver]
    integ = globals()[args.integrator]
    cellaverage_type = cellaverage_acc if args.cell_average == 'accurate' else cellaverage
    Z = args.grid_size
    Co = args.cfl

    rhoJ = cp.array([1.0, 0.125])
    uJ = cp.array([0.0, 0.0])
    pJ = cp.array([1.0, 0.1])

    tests = {
        1: {"BC": outflow, "u0": densityfn, "xlims": np.array([0, 2]), "Tfinal": 0.1, "rd": 'den_wave'},
        2: {"BC": outflow, "u0": riemann(rhoJ, uJ, pJ),
            "xlims": np.array([-0.5, 0.5]), "Tfinal": 0.015, "rd": 'riemann'}
    }
    test = tests[args.test]
    BC, u0, xlims, Tfinal, rd = test["BC"], test["u0"], test["xlims"], test["Tfinal"], test["rd"]

    plots = not args.no_plot
    plot_interval = args.plot_interval

    t = 0.0
    n = 0
    dy = cp.abs((xlims[1] - xlims[0]) / (Z + 6))
    Z = Z + 6
    y = cp.linspace(xlims[0] - 2.5 * float(dy), xlims[1] + 2.5 * float(dy), Z)
    u = BC(cellaverage_type(u0, y, dy))
    dt = Co * dy / cp.max(cp.abs(EigA(u)[0]))

    if plots:
        figure, axes, lines = plot_initial_state(u, y, t, cellaverage_type)
    else:
        figure, axes, lines = None, None, None

    fold = args.fold

    print(f'Initiating simulation with:')
    print(f'  Backend: cuda (CuPy)')
    print(f'  Precision: 32-bit')
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

    # Define L_operator closure once (equivalent to the inlined operator in JAX fori_loop)
    def L_operator(u_state):
        u_bc = BC(u_state)
        left, right = slvr(u_bc)
        eigval, _ = EigA(u_bc)
        amax = cp.max(cp.abs(eigval))
        flux_left_right = flux(left)
        flux_right_right = flux(right)
        flux_vals = 0.5 * (flux_left_right + flux_right_right - amax * (right - left))
        flux_shifted = cp.concatenate((flux_vals[:, -1:], flux_vals[:, :-1]), axis=1)
        df = BC(-(flux_vals - flux_shifted) / dy)
        return df

    # Warmup (equivalent to JAX JIT pre-compilation step)
    print('Warming up CuPy kernels...')
    _ = integ(L_operator, u.copy(), dt)
    cp.cuda.Device().synchronize()
    print('Warmup complete. Initiating loop...')

    plot_counter = 0
    n = 0

    while t < Tfinal:
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()

        # Python for-loop replacing jax.lax.fori_loop
        for _ in range(fold):
            u = integ(L_operator, u, dt)
            t = t + float(dt)
            eigval, _ = EigA(u)
            amax = cp.max(cp.abs(eigval))
            dt = Co * dy / amax

        end_event.record()
        end_event.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        elapsed_s = elapsed_ms / 1000.0

        n += fold
        print(f"[{n:04d}] t={t:.3f} Mzps={Z * fold * 1e-6 / elapsed_s:.4f}")

        if plots and plot_counter % plot_interval == 0:
            update_plot(u, t, lines, axes)
        plot_counter += 1

    analytic_u = RiemannExact(cp.array([rhoJ, rhoJ*uJ, 0.5*rhoJ*uJ**2 + pJ/(gamma-1)]), gamma, t) if rd == 'riemann' else lambda x: u0(x - t)

    if plots:
        x_coords = cp.linspace(float(y[0]), float(y[Z-1]), int(max(2 * Z, 1e3)))
        x_coords_np = cp.asnumpy(x_coords)
        analytic_vals = cp.asnumpy(analytic_u(x_coords))
        for i, ax in enumerate(axes):
            ax.plot(x_coords_np, analytic_vals[i, :], 'w:')
            ax.set_title(f't = {t}')
        u_np = cp.asnumpy(u)
        [line.set_ydata(u_np[i, :]) for i, line in enumerate(lines)]
        plt.show()

    l2_norm = None
    if rd == 'den_wave':
        bool_mask = (xlims[0] < cp.asnumpy(y[:-1])) & (cp.asnumpy(y[:-1]) < xlims[1]) if cellaverage_type == cellaverage else (xlims[0] < cp.asnumpy(y)) & (cp.asnumpy(y) < xlims[1])
        bool_mask = cp.asarray(bool_mask)
        difference = u[:, bool_mask] - cellaverage_type(analytic_u, y, dy)[:, bool_mask]
        l2_norm = cp.linalg.norm(difference * dy, axis=1)
        print('L2 norm\n', cp.asnumpy(l2_norm))

    class TestResults:
        def __init__(self):
            self.final_state = u
            self.l2_norm = l2_norm
            self.t = t
            self.grid_size = Z
            self.dt = dt
    return TestResults()


if __name__ == '__main__':
    main()
