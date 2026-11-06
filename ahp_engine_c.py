"""
Traduccion 1:1 a C (via ctypes) del nucleo `_evolve` de gas03_analysis.py.

Mismo algoritmo: anillo periodico, tiempo local por particula, arbol de minimos
O(log N) por colision y recalculo de los 3 pares afectados.  Solo sirve para
correr sin numba; con numba instalado, `blast_ahp.py` usa directamente
gas03_analysis._evolve.

Uso:
    from ahp_engine_c import load_c_engine
    evolve = load_c_engine()   # None si no hay compilador de C
"""
import os
import ctypes
import subprocess
import tempfile

import numpy as np

_C_SRC = r"""
#include <math.h>
#include <stdlib.h>

static double pair_time(long i, double t, const double *x, const double *v,
                        const double *tloc, double L, long N) {
    long j = i + 1; double wrap = 0.0;
    if (j == N) { j = 0; wrap = L; }
    double rel = v[i] - v[j];
    if (rel <= 0.0) return INFINITY;
    double xi = x[i] + v[i] * (t - tloc[i]);
    double xj = x[j] + v[j] * (t - tloc[j]) + wrap;
    double gap = xj - xi;
    if (gap < 0.0) gap = 0.0;
    return t + gap / rel;
}

static void refresh(long k, double t, const double *x, const double *v,
                    const double *tloc, double L, long N, double *tc,
                    double *tree, long M) {
    double val = pair_time(k, t, x, v, tloc, L, N);
    tc[k] = val;
    long p = M + k;
    tree[p] = val;
    p >>= 1;
    while (p >= 1) {
        double a = tree[2*p], b = tree[2*p+1];
        tree[p] = a < b ? a : b;
        p >>= 1;
    }
}

long evolve(double *x, double *v, const double *m, long N, double L,
            const double *ts, long S, double *pos, double *vel) {
    long M = 1; while (M < N) M *= 2;
    double *tree = malloc(sizeof(double) * 2 * M);
    double *tloc = calloc(N, sizeof(double));
    double *tc = malloc(sizeof(double) * N);
    for (long p = 0; p < 2*M; p++) tree[p] = INFINITY;
    for (long k = 0; k < N; k++) {
        tc[k] = pair_time(k, 0.0, x, v, tloc, L, N);
        tree[M + k] = tc[k];
    }
    for (long p = M - 1; p > 0; p--) {
        double a = tree[2*p], b = tree[2*p+1];
        tree[p] = a < b ? a : b;
    }
    long n_coll = 0;
    for (long s = 0; s < S; s++) {
        double tsamp = ts[s];
        while (tree[1] < tsamp) {
            long p = 1;
            while (p < M) { if (tree[2*p] <= tree[p]) p = 2*p; else p = 2*p+1; }
            long i = p - M, j = i + 1;
            if (j == N) j = 0;
            double t = tc[i];
            x[i] += v[i] * (t - tloc[i]); tloc[i] = t;
            x[j] += v[j] * (t - tloc[j]); tloc[j] = t;
            double m1 = m[i], m2 = m[j], v1 = v[i], v2 = v[j], mt = m1 + m2;
            v[i] = ((m1 - m2) * v1 + 2.0 * m2 * v2) / mt;
            v[j] = (2.0 * m1 * v1 + (m2 - m1) * v2) / mt;
            n_coll++;
            long a = i - 1; if (a < 0) a = N - 1;
            refresh(a, t, x, v, tloc, L, N, tc, tree, M);
            refresh(i, t, x, v, tloc, L, N, tc, tree, M);
            refresh(j, t, x, v, tloc, L, N, tc, tree, M);
        }
        for (long k = 0; k < N; k++) {
            double xk = x[k] + v[k] * (tsamp - tloc[k]);
            pos[s*N + k] = xk - L * floor(xk / L);
            vel[s*N + k] = v[k];
        }
    }
    free(tree); free(tloc); free(tc);
    return n_coll;
}
"""

_LIB = None


def load_c_engine():
    """Compila (una vez) y devuelve evolve(x, v, m, L, times, positions, velocities)."""
    global _LIB
    if _LIB is None:
        d = os.path.join(tempfile.gettempdir(), "ahp_engine_c")
        os.makedirs(d, exist_ok=True)
        src, so = os.path.join(d, "ahp.c"), os.path.join(d, "ahp.so")
        if not os.path.exists(so):
            with open(src, "w") as f:
                f.write(_C_SRC)
            try:
                subprocess.run(["cc", "-O3", "-march=native", "-shared", "-fPIC",
                                src, "-o", so, "-lm"], check=True)
            except Exception:
                return None
        lib = ctypes.CDLL(so)
        dp = ctypes.POINTER(ctypes.c_double)
        lib.evolve.argtypes = [dp, dp, dp, ctypes.c_long, ctypes.c_double,
                               dp, ctypes.c_long, dp, dp]
        lib.evolve.restype = ctypes.c_long
        _LIB = lib

    def evolve(x, v, m, L, times, positions, velocities):
        arrs = [np.ascontiguousarray(a, dtype=np.float64)
                for a in (x, v, m, times)]
        assert positions.flags.c_contiguous and velocities.flags.c_contiguous
        p = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return int(_LIB.evolve(p(arrs[0]), p(arrs[1]), p(arrs[2]), len(arrs[0]),
                               float(L), p(arrs[3]), len(arrs[3]),
                               p(positions), p(velocities)))

    return evolve
