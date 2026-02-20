#!/usr/bin/env python3
"""1D Compressible Euler Solver — CuPy (CUDA) Backend
Port of swirl_solver.py from JAX to CuPy.
Uses WENO5 reconstruction with Lax-Friedrichs flux splitting.
Array layout: [3, Z] where row 0=density, 1=momentum, 2=energy.
"""
import argparse
import subprocess
import sys
import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

gamma = 1.4

# ---------------------------------------------------------------------------
# cuda.compute (CUB) integration for device-wide reductions
# ---------------------------------------------------------------------------
try:
    import cuda.compute as ccl
    _HAS_CCL = True
except ImportError:
    _HAS_CCL = False


class _CUBMaxReducer:
    """Cached CUB MAXIMUM reducer.  Avoids recompilation and temp-storage
    reallocation on every timestep.  Used for max(|eigenvalues|)."""
    __slots__ = ('reducer', 'd_out', 'h_init', 'temp', 'n', 'op')

    def __init__(self, n, dtype):
        self.n = n
        self.op = ccl.OpKind.MAXIMUM
        self.d_out = cp.zeros(1, dtype=dtype)
        self.h_init = np.array(0.0, dtype=np.dtype(dtype))
        proto = cp.empty(n, dtype=dtype)
        self.reducer = ccl.make_reduce_into(
            proto, self.d_out, self.op, self.h_init
        )
        temp_bytes = self.reducer(
            None, proto, self.d_out, self.op, n, self.h_init
        )
        self.temp = cp.empty(max(temp_bytes, 1), dtype=cp.uint8)

    def __call__(self, arr):
        """Return max(|arr.ravel()|) as a CuPy 0-d array (stays on GPU)."""
        flat = cp.abs(arr).ravel()
        self.reducer(
            self.temp, flat, self.d_out, self.op, self.n, self.h_init
        )
        return self.d_out[0]


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


# ---------------------------------------------------------------------------
# Lax-Friedrichs flux divergence fused kernel
# Fuses: Euler flux eval (left & right) + LF combination + spatial diff
# into a single kernel.  Uses shared memory (1-element halo per block)
# so each thread can access the numerical flux at interface j-1.
# ---------------------------------------------------------------------------
_LF_KERNEL_SRC = r"""
#define LF_BLOCK 256

/* ---- Euler physical flux from conservative variables ---- */
__device__ __forceinline__
void euler_flux(REAL rho, REAL m, REAL E, REAL gam,
                REAL* f0, REAL* f1, REAL* f2)
{
    REAL u = m / rho;
    REAL p = (gam - (REAL)1) * (E - (REAL)0.5 * rho * u * u);
    *f0 = m;
    *f1 = m * u + p;
    *f2 = (E + p) * u;
}

extern "C" __global__
void laxfriedrichs_flux_divergence_kernel(
    const REAL* __restrict__ L,          /* left  states [3*Z] row-major */
    const REAL* __restrict__ R,          /* right states [3*Z]           */
    REAL*       __restrict__ dUdt,       /* output RHS   [3*Z]           */
    const REAL* __restrict__ amax_ptr,   /* max wavespeed (1 element)    */
    const int Z,
    const REAL inv_dy,
    const REAL gam)
{
    /* shared memory: 3 * (LF_BLOCK + 1) for numerical fluxes + halo */
    extern __shared__ REAL smem[];
    const int tid = threadIdx.x;
    const int g   = blockIdx.x * LF_BLOCK + tid;
    const REAL amax = amax_ptr[0];

    /* --- compute LF numerical flux at interface g --- */
    REAL fn0 = (REAL)0, fn1 = (REAL)0, fn2 = (REAL)0;
    if (g < Z) {
        REAL Ll0 = L[g], Ll1 = L[Z + g], Ll2 = L[2*Z + g];
        REAL Rr0 = R[g], Rr1 = R[Z + g], Rr2 = R[2*Z + g];
        REAL fL0, fL1, fL2, fR0, fR1, fR2;
        euler_flux(Ll0, Ll1, Ll2, gam, &fL0, &fL1, &fL2);
        euler_flux(Rr0, Rr1, Rr2, gam, &fR0, &fR1, &fR2);
        fn0 = (REAL)0.5 * (fL0 + fR0 - amax * (Rr0 - Ll0));
        fn1 = (REAL)0.5 * (fL1 + fR1 - amax * (Rr1 - Ll1));
        fn2 = (REAL)0.5 * (fL2 + fR2 - amax * (Rr2 - Ll2));
    }

    /* store in shared memory (slot tid+1; slot 0 is left halo) */
    REAL* s0 = smem;
    REAL* s1 = smem + (LF_BLOCK + 1);
    REAL* s2 = smem + 2 * (LF_BLOCK + 1);
    s0[tid + 1] = fn0;
    s1[tid + 1] = fn1;
    s2[tid + 1] = fn2;

    /* thread 0 computes left-halo flux at interface (g-1) */
    if (tid == 0) {
        int h = g - 1;
        if (h < 0) h = Z - 1;          /* wrap like cp.concatenate */
        REAL Ll0 = L[h], Ll1 = L[Z + h], Ll2 = L[2*Z + h];
        REAL Rr0 = R[h], Rr1 = R[Z + h], Rr2 = R[2*Z + h];
        REAL fL0, fL1, fL2, fR0, fR1, fR2;
        euler_flux(Ll0, Ll1, Ll2, gam, &fL0, &fL1, &fL2);
        euler_flux(Rr0, Rr1, Rr2, gam, &fR0, &fR1, &fR2);
        s0[0] = (REAL)0.5 * (fL0 + fR0 - amax * (Rr0 - Ll0));
        s1[0] = (REAL)0.5 * (fL1 + fR1 - amax * (Rr1 - Ll1));
        s2[0] = (REAL)0.5 * (fL2 + fR2 - amax * (Rr2 - Ll2));
    }

    __syncthreads();

    /* dUdt = -(F[g] - F[g-1]) / dy */
    if (g < Z) {
        dUdt[g]       = -(s0[tid + 1] - s0[tid]) * inv_dy;
        dUdt[Z + g]   = -(s1[tid + 1] - s1[tid]) * inv_dy;
        dUdt[2*Z + g] = -(s2[tid + 1] - s2[tid]) * inv_dy;
    }
}
"""

_LF_BLOCK = 256
_lf_kernel_cache = {}


def _get_lf_kernel(dtype):
    """Compile (or fetch cached) LF divergence kernel for the given dtype."""
    if dtype not in _lf_kernel_cache:
        if dtype == cp.float32:
            src = _LF_KERNEL_SRC.replace('REAL', 'float')
        else:
            src = _LF_KERNEL_SRC.replace('REAL', 'double')
        _lf_kernel_cache[dtype] = cp.RawKernel(
            src, 'laxfriedrichs_flux_divergence_kernel')
    return _lf_kernel_cache[dtype]


def _launch_lf_divergence(left, right, amax, inv_dy_val):
    """Fused Lax-Friedrichs flux + spatial divergence.  Returns dUdt[3,Z]."""
    Z = left.shape[1]
    dUdt = cp.empty_like(left)
    kernel = _get_lf_kernel(left.dtype)
    grid = ((Z + _LF_BLOCK - 1) // _LF_BLOCK,)
    smem = 3 * (_LF_BLOCK + 1) * left.dtype.itemsize
    # amax is a CuPy 0-d array; reshape to 1-d so RawKernel passes as pointer
    amax_arr = amax.reshape(1) if amax.ndim == 0 else amax
    real_t = np.float32 if left.dtype == cp.float32 else np.float64
    kernel(grid, (_LF_BLOCK,),
           (left, right, dUdt, amax_arr, np.int32(Z),
            real_t(inv_dy_val), real_t(gamma)),
           shared_mem=smem)
    return dUdt


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
    parser.add_argument('--benchmark', action='store_true',
                      help='Run benchmark: per-component timing, bandwidth, scaling study')
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args)
        return

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

    # Cached cuda.compute CUB max-abs reducer (3 eigenvalue rows × Z cols)
    cub_max = _CUBMaxReducer(3 * Z, u.dtype) if _HAS_CCL else None

    def _max_abs(arr):
        return cub_max(arr) if cub_max is not None else cp.max(cp.abs(arr))

    inv_dy = float(1.0 / dy)  # precompute once (dy is constant)
    dt = Co * dy / _max_abs(EigA(u)[0])

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
    print(f'  cuda.compute (CUB): {"available" if _HAS_CCL else "not available"}')

    # Timing event accumulators for per-fold breakdowns
    _weno_events = []
    _reduce_events = []
    _flux_events = []

    # Define L_operator closure once (equivalent to the inlined operator in JAX fori_loop)
    def L_operator(u_state):
        u_bc = BC(u_state)
        we0 = cp.cuda.Event(); we0.record()
        left, right = slvr(u_bc)
        we1 = cp.cuda.Event(); we1.record()
        eigval, _ = EigA(u_bc)
        re0 = cp.cuda.Event(); re0.record()
        amax = _max_abs(eigval)
        re1 = cp.cuda.Event(); re1.record()
        fe0 = cp.cuda.Event(); fe0.record()
        dUdt = _launch_lf_divergence(left, right, amax, inv_dy)
        fe1 = cp.cuda.Event(); fe1.record()
        _weno_events.append((we0, we1))
        _reduce_events.append((re0, re1))
        _flux_events.append((fe0, fe1))
        df = BC(dUdt)
        return df

    # Warmup (equivalent to JAX JIT pre-compilation step)
    print('Warming up CuPy kernels...')
    _ = integ(L_operator, u.copy(), dt)
    cp.cuda.Device().synchronize()
    _weno_events.clear()
    _reduce_events.clear()
    _flux_events.clear()
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
            re0 = cp.cuda.Event(); re0.record()
            amax = _max_abs(eigval)
            re1 = cp.cuda.Event(); re1.record()
            _reduce_events.append((re0, re1))
            dt = Co * dy / amax

        end_event.record()
        end_event.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        elapsed_s = elapsed_ms / 1000.0

        weno_ms = sum(cp.cuda.get_elapsed_time(a, b) for a, b in _weno_events)
        reduce_ms = sum(cp.cuda.get_elapsed_time(a, b) for a, b in _reduce_events)
        flux_ms = sum(cp.cuda.get_elapsed_time(a, b) for a, b in _flux_events)
        _weno_events.clear()
        _reduce_events.clear()
        _flux_events.clear()

        n += fold
        print(f"[{n:04d}] t={t:.3f} Mzps={Z * fold * 1e-6 / elapsed_s:.4f}"
              f"  weno={weno_ms:.1f}ms  reduce={reduce_ms:.1f}ms"
              f"  flux={flux_ms:.1f}ms  total={elapsed_ms:.1f}ms")

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


def _run_benchmark_single(Z_interior, solver_name, integrator_name, cfl, n_steps=100):
    """Run a single benchmark for a given grid size. Returns dict of timings."""
    slvr = globals()[solver_name]
    integ = globals()[integrator_name]
    BC = outflow

    rhoJ = cp.array([1.0, 0.125])
    uJ = cp.array([0.0, 0.0])
    pJ = cp.array([1.0, 0.1])
    u0 = riemann(rhoJ, uJ, pJ)
    xlims = np.array([-0.5, 0.5])

    dy = abs((xlims[1] - xlims[0]) / (Z_interior + 6))
    Z = Z_interior + 6
    y = cp.linspace(xlims[0] - 2.5 * dy, xlims[1] + 2.5 * dy, Z)
    u = BC(cellaverage(u0, y, dy))

    cub_max = _CUBMaxReducer(3 * Z, u.dtype) if _HAS_CCL else None
    def _max_abs(arr):
        return cub_max(arr) if cub_max is not None else cp.max(cp.abs(arr))

    inv_dy = float(1.0 / dy)
    dt = cfl * dy / _max_abs(EigA(u)[0])

    # Event lists for each component (non-overlapping atomic pieces)
    bc_events, weno_events, eig_events = [], [], []
    reduce_events, flux_events = [], []

    def L_operator_bench(u_state):
        be0 = cp.cuda.Event(); be0.record()
        u_bc = BC(u_state)
        be1 = cp.cuda.Event(); be1.record()
        bc_events.append((be0, be1))

        we0 = cp.cuda.Event(); we0.record()
        left, right = slvr(u_bc)
        we1 = cp.cuda.Event(); we1.record()
        weno_events.append((we0, we1))

        ee0 = cp.cuda.Event(); ee0.record()
        eigval, _ = EigA(u_bc)
        ee1 = cp.cuda.Event(); ee1.record()
        eig_events.append((ee0, ee1))

        re0 = cp.cuda.Event(); re0.record()
        amax = _max_abs(eigval)
        re1 = cp.cuda.Event(); re1.record()
        reduce_events.append((re0, re1))

        fe0 = cp.cuda.Event(); fe0.record()
        dUdt = _launch_lf_divergence(left, right, amax, inv_dy)
        fe1 = cp.cuda.Event(); fe1.record()
        flux_events.append((fe0, fe1))

        be2 = cp.cuda.Event(); be2.record()
        df = BC(dUdt)
        be3 = cp.cuda.Event(); be3.record()
        bc_events.append((be2, be3))
        return df

    # Warmup
    _ = integ(L_operator_bench, u.copy(), dt)
    cp.cuda.Device().synchronize()
    bc_events.clear(); weno_events.clear(); eig_events.clear()
    reduce_events.clear(); flux_events.clear()

    # Timed run
    cp.cuda.Device().synchronize()
    wall_start = time.perf_counter()
    total_e0 = cp.cuda.Event(); total_e0.record()

    for _ in range(n_steps):
        u = integ(L_operator_bench, u, dt)

        ee0 = cp.cuda.Event(); ee0.record()
        eigval, _ = EigA(u)
        ee1 = cp.cuda.Event(); ee1.record()
        eig_events.append((ee0, ee1))

        re0 = cp.cuda.Event(); re0.record()
        amax = _max_abs(eigval)
        re1 = cp.cuda.Event(); re1.record()
        reduce_events.append((re0, re1))
        dt = cfl * dy / amax

    total_e1 = cp.cuda.Event(); total_e1.record()
    total_e1.synchronize()
    wall_elapsed = time.perf_counter() - wall_start

    def sum_ms(evts):
        return sum(cp.cuda.get_elapsed_time(a, b) for a, b in evts) if evts else 0.0

    bc_ms = sum_ms(bc_events)
    weno_ms = sum_ms(weno_events)
    eig_ms = sum_ms(eig_events)
    reduce_ms = sum_ms(reduce_events)
    flux_ms = sum_ms(flux_events)
    total_gpu_ms = cp.cuda.get_elapsed_time(total_e0, total_e1)

    # Bandwidth calculations (bytes per step)
    elem = 4  # float32
    # WENO: reads [3,Z], writes 2x[3,Z]
    weno_bytes = (3 * Z * elem) + 2 * (3 * Z * elem)
    # LF flux: reads 2x[3,Z] + 1 scalar, writes [3,Z]
    lf_bytes = 2 * (3 * Z * elem) + (3 * Z * elem)
    # Reduce: reads [3,Z], writes 1 scalar
    reduce_bytes = 3 * Z * elem

    # L_operator calls per RK step: RK1=1, RK3=3, RK4=4
    rk_calls = {'RK1': 1, 'RK3': 3, 'RK4': 4}.get(integrator_name, 1)

    return {
        'Z': Z, 'n_steps': n_steps, 'rk_calls': rk_calls,
        'bc_ms': bc_ms, 'weno_ms': weno_ms, 'eig_ms': eig_ms,
        'reduce_ms': reduce_ms, 'flux_ms': flux_ms,
        'total_gpu_ms': total_gpu_ms, 'wall_s': wall_elapsed,
        'weno_bytes': weno_bytes, 'lf_bytes': lf_bytes, 'reduce_bytes': reduce_bytes,
    }


def _print_benchmark_report(results):
    """Print formatted benchmark report for a single grid size."""
    Z = results['Z']
    N = results['n_steps']
    total = results['total_gpu_ms']
    rk_calls = results['rk_calls']

    components = [
        ('BC (outflow)',   results['bc_ms']),
        ('WENO5 kernel',   results['weno_ms']),
        ('EigA compute',   results['eig_ms']),
        ('Max reduce',     results['reduce_ms']),
        ('LF flux div',    results['flux_ms']),
    ]
    accounted = sum(v for _, v in components)
    overhead = total - accounted

    print(f"\n{'='*65}")
    print(f"  Grid Z={Z}  |  {N} steps (RK×{rk_calls})  |  GPU: {total:.2f} ms")
    print(f"{'='*65}")
    print(f"  {'Component':<18} {'Total ms':>10} {'Avg us/step':>12} {'%':>7}")
    print(f"  {'-'*50}")
    for name, ms in components:
        pct = 100 * ms / total if total > 0 else 0
        print(f"  {name:<18} {ms:>10.2f} {ms*1000/N:>12.1f} {pct:>6.1f}%")
    pct_oh = 100 * overhead / total if total > 0 else 0
    print(f"  {'RK arith + Python':<18} {overhead:>10.2f} {overhead*1000/N:>12.1f} {pct_oh:>6.1f}%")

    # Bandwidth — account for RK calls per step
    print(f"\n  Effective bandwidth (GB/s):")
    weno_ms = results['weno_ms']
    flux_ms = results['flux_ms']
    reduce_ms = results['reduce_ms']
    n_L_calls = N * rk_calls  # total L_operator invocations
    if weno_ms > 0:
        weno_bw = results['weno_bytes'] * n_L_calls / (weno_ms * 1e-3) / 1e9
        print(f"    WENO5:   {weno_bw:>8.1f} GB/s")
    if flux_ms > 0:
        lf_bw = results['lf_bytes'] * n_L_calls / (flux_ms * 1e-3) / 1e9
        print(f"    LF flux: {lf_bw:>8.1f} GB/s")
    if reduce_ms > 0:
        # reduce: n_L_calls inside L_operator + N for dt update
        n_reduces = n_L_calls + N
        red_bw = results['reduce_bytes'] * n_reduces / (reduce_ms * 1e-3) / 1e9
        print(f"    Reduce:  {red_bw:>8.1f} GB/s")

    mzps = Z * N * 1e-6 / (total * 1e-3)
    print(f"\n  Mzps: {mzps:.2f}")
    return mzps


def run_benchmark(args):
    """Full benchmark: per-component timing, scaling study, optional JAX comparison."""
    print("=" * 65)
    print("  SWIRL BENCHMARK — CuPy/CUDA Backend")
    print("=" * 65)

    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f"  GPU: {props['name'].decode()}")
    print(f"  cuda.compute (CUB): {'available' if _HAS_CCL else 'not available'}")
    print(f"  Solver: {args.solver}  |  Integrator: {args.integrator}  |  CFL: {args.cfl}")

    grid_sizes = [1000, 5000, 10000, 50000, 100000]
    all_results = []

    for gz in grid_sizes:
        print(f"\n>>> Benchmarking grid size {gz} ...")
        r = _run_benchmark_single(gz, args.solver, args.integrator, args.cfl, n_steps=100)
        all_results.append(r)
        _print_benchmark_report(r)

    # Scaling summary table
    print(f"\n{'='*65}")
    print(f"  SCALING SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Grid':>8} {'Total ms':>10} {'Mzps':>10} {'us/step':>10} {'WENO%':>7} {'LF%':>7} {'Red%':>7}")
    print(f"  {'-'*62}")
    for r in all_results:
        Z = r['Z']
        N = r['n_steps']
        t = r['total_gpu_ms']
        mzps = Z * N * 1e-6 / (t * 1e-3) if t > 0 else 0
        w_pct = 100 * r['weno_ms'] / t if t > 0 else 0
        f_pct = 100 * r['flux_ms'] / t if t > 0 else 0
        r_pct = 100 * r['reduce_ms'] / t if t > 0 else 0
        print(f"  {Z:>8} {t:>10.2f} {mzps:>10.2f} {t*1000/N:>10.1f} {w_pct:>6.1f}% {f_pct:>6.1f}% {r_pct:>6.1f}%")

    # JAX comparison
    print(f"\n{'='*65}")
    print(f"  JAX COMPARISON (grid=5000, 100 steps)")
    print(f"{'='*65}")
    try:
        import os, shutil
        # Prefer system python for JAX (may not be in CuPy venv)
        py_candidates = [shutil.which('python3'), shutil.which('python')]
        py_candidates = [p for p in py_candidates if p]
        if not py_candidates:
            py_candidates = [sys.executable]
        solver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'swirl_solver.py')
        # Try cuda first, then gpu, then cpu
        jax_ran = False
        for backend in ['cuda', 'gpu', 'cpu']:
            env = os.environ.copy()
            if backend in ('cuda', 'gpu'):
                env['JAX_PLATFORMS'] = backend
            jax_cmd = [
                py_candidates[0], solver_path,
                '--backend', backend, '--no-plot', '--grid-size', '5000',
                '--solver', args.solver, '--integrator', args.integrator,
                '--cfl', str(args.cfl), '--fold', '100',
            ]
            print(f"  Trying: {' '.join(jax_cmd[-10:])}")
            result = subprocess.run(jax_cmd, capture_output=True, text=True,
                                    timeout=120, env=env)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if 'Mzps' in line:
                        print(f"  JAX ({backend}): {line.strip()}")
                cupy_r = next((r for r in all_results if r['Z'] == 5006), None)
                if cupy_r:
                    cupy_mzps = cupy_r['Z'] * cupy_r['n_steps'] * 1e-6 / (cupy_r['total_gpu_ms'] * 1e-3)
                    print(f"  CuPy fused: {cupy_mzps:.2f} Mzps")
                jax_ran = True
                break
        if not jax_ran:
            print(f"  JAX solver unavailable (tried cuda/gpu/cpu)")
    except FileNotFoundError:
        print("  JAX solver not found (swirl_solver.py)")
    except subprocess.TimeoutExpired:
        print("  JAX solver timed out (120s)")
    except Exception as e:
        print(f"  JAX comparison error: {e}")


if __name__ == '__main__':
    main()
