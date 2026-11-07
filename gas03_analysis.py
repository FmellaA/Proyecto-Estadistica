"""
Gas de puntos duros alternantes (AHP) con perturbacion sinusoidal de temperatura.

Que hace este script
--------------------
1. Dinamica microscopica exacta (dirigida por eventos, anillo periodico).
   - El orden de las particulas en el anillo se conserva (puntos duros), asi que
     los vecinos son siempre (i, i+1) y NO hace falta ordenar posiciones.
   - Cada particula tiene su propio "tiempo local": solo se actualizan las dos
     que chocan y se recalculan 3 tiempos de colision.
   - El siguiente evento se obtiene con un arbol de minimos: O(log N) por colision
     (antes: O(N log N)).  Con numba instalado el nucleo se compila (muy rapido);
     sin numba funciona igual en Python puro (mucho mas lento).
2. Campos hidrodinamicos (T, rho, n, u) vectorizados con np.bincount.
3. Ensemble paralelo sobre semillas + amplitudes de modo de Fourier:
   temperatura A_T, densidad A_n, velocidad A_u y ENTROPIA A_s (modo de calor).
4. Todas las figuras se guardan como PNG en OUTDIR (no se muestran salvo --show).
5. Estudio opcional en funcion de la razon de masas r = mB/mA.

Uso:
    python gas03_analysis.py            # corrida completa
    python gas03_analysis.py --quick    # prueba rapida (N chico, pocas semillas)
    python gas03_analysis.py --mass-ratio   # agrega el estudio en r
    python gas03_analysis.py --show         # ademas muestra las figuras
"""
import os
import sys
import time
import argparse
from multiprocessing import Pool

import warnings
import numpy as np
import matplotlib

# bins con <2 particulas dan NaN a proposito; nanmean sobre celdas vacias avisa en vano
warnings.filterwarnings("ignore", category=RuntimeWarning)

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                       # fallback: Python puro
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


# ============================================================
# 1. Condicion inicial
# ============================================================

def gas(N=5000, L=100.0, mA=1.0, mB=2.0, T0=1.0,
        amplitude=0.2, mode=1, seed=123):
    """
    Particulas equiespaciadas en [0,L) con masas alternantes mA, mB,
    T(x,0) = T0 * [1 + amplitude*cos(2*pi*mode*(x-L/2)/L)],  k_B = 1.
    """
    if N < 4 or N % 2 != 0:
        raise ValueError("N must be an even integer >= 4.")
    if L <= 0 or mA <= 0 or mB <= 0 or T0 <= 0:
        raise ValueError("L, mA, mB and T0 must be positive.")
    if not 0 <= amplitude < 1:
        raise ValueError("amplitude must satisfy 0 <= amplitude < 1.")
    if int(mode) != mode or mode < 1:
        raise ValueError("mode must be a positive integer.")

    rng = np.random.default_rng(seed)

    masses = np.empty(N)
    masses[0::2] = mA
    masses[1::2] = mB

    x = (np.arange(N) + 0.5) * L / N
    k = 2.0 * np.pi * mode / L
    T_local = T0 * (1.0 + amplitude * np.cos(k * (x - L / 2)))

    v = rng.normal(0.0, np.sqrt(T_local / masses), size=N)
    v -= np.sum(masses * v) / np.sum(masses)      # sin movimiento del CM
    return masses, v, x


# ============================================================
# 2. Dinamica dirigida por eventos (nucleo)
# ============================================================

@njit(cache=True)
def _pair_time(i, t, x, v, tloc, L, N):
    """Tiempo absoluto de la proxima colision del par (i, i+1) (anillo)."""
    j = i + 1
    wrap = 0.0
    if j == N:
        j = 0
        wrap = L
    rel = v[i] - v[j]
    if rel <= 0.0:
        return np.inf
    xi = x[i] + v[i] * (t - tloc[i])
    xj = x[j] + v[j] * (t - tloc[j]) + wrap
    gap = xj - xi
    if gap < 0.0:
        gap = 0.0
    return t + gap / rel


@njit(cache=True)
def _refresh(k, t, x, v, tloc, L, N, tc, tree, M):
    """Recalcula el tiempo del par k y actualiza el arbol de minimos."""
    val = _pair_time(k, t, x, v, tloc, L, N)
    tc[k] = val
    p = M + k
    tree[p] = val
    p >>= 1
    while p >= 1:
        a = tree[2 * p]
        b = tree[2 * p + 1]
        tree[p] = a if a < b else b
        p >>= 1


@njit(cache=True)
def _evolve(x, v, m, L, sample_times, positions, velocities):
    """
    Evoluciona el anillo y guarda snapshots en sample_times.

    x se interpreta SIN envolver: x[0] < x[1] < ... < x[N-1] < x[0]+L siempre.
    Devuelve el numero de colisiones.
    """
    N = x.shape[0]
    M = 1
    while M < N:
        M *= 2
    tree = np.full(2 * M, np.inf)
    tloc = np.zeros(N)
    tc = np.full(N, np.inf)

    for k in range(N):
        tc[k] = _pair_time(k, 0.0, x, v, tloc, L, N)
        tree[M + k] = tc[k]
    for p in range(M - 1, 0, -1):
        a = tree[2 * p]
        b = tree[2 * p + 1]
        tree[p] = a if a < b else b

    n_coll = 0
    for s in range(sample_times.shape[0]):
        ts = sample_times[s]

        while tree[1] < ts:
            # hoja con el minimo
            p = 1
            while p < M:
                if tree[2 * p] <= tree[p]:
                    p = 2 * p
                else:
                    p = 2 * p + 1
            i = p - M
            j = i + 1
            if j == N:
                j = 0
            t = tc[i]

            # adelantar solo las dos particulas que chocan
            x[i] += v[i] * (t - tloc[i])
            tloc[i] = t
            x[j] += v[j] * (t - tloc[j])
            tloc[j] = t

            # colision elastica 1D
            m1 = m[i]
            m2 = m[j]
            v1 = v[i]
            v2 = v[j]
            mt = m1 + m2
            v[i] = ((m1 - m2) * v1 + 2.0 * m2 * v2) / mt
            v[j] = (2.0 * m1 * v1 + (m2 - m1) * v2) / mt
            n_coll += 1

            # pares afectados: (i-1,i), (i,j), (j,j+1)
            a = i - 1
            if a < 0:
                a = N - 1
            _refresh(a, t, x, v, tloc, L, N, tc, tree, M)
            _refresh(i, t, x, v, tloc, L, N, tc, tree, M)
            _refresh(j, t, x, v, tloc, L, N, tc, tree, M)

        for k in range(N):
            xk = x[k] + v[k] * (ts - tloc[k])
            positions[s, k] = xk - L * np.floor(xk / L)
            velocities[s, k] = v[k]

    return n_coll


def simulate(N=200, L=100.0, mA=1.0, mB=2.0, T0=1.0,
             amplitude=0.2, mode=1, t_end=100.0,
             n_samples=501, seed=123):
    """Corre el AHP y guarda snapshots microscopicos."""
    if t_end <= 0 or n_samples < 2:
        raise ValueError("t_end must be positive and n_samples >= 2.")

    masses, v, x = gas(N=N, L=L, mA=mA, mB=mB, T0=T0,
                       amplitude=amplitude, mode=mode, seed=seed)

    times = np.linspace(0.0, t_end, n_samples)
    positions = np.empty((n_samples, N))
    velocities = np.empty((n_samples, N))

    n_coll = _evolve(x.copy(), v.copy(), masses, float(L), times,
                     positions, velocities)

    energy = 0.5 * (velocities ** 2) @ masses
    momentum = velocities @ masses

    return {
        "times": times,
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "L": L,
        "energy": energy,
        "momentum": momentum,
        "energy_error": np.max(np.abs(energy - energy[0])) / energy[0],
        "n_collisions": int(n_coll),
    }


# ============================================================
# 3. Campos hidrodinamicos (vectorizado)
# ============================================================

def hydro_profiles(positions, velocities, masses, L, n_bins=40):
    """
    Campos coarse-grained por bins espaciales.
      n: densidad numerica, rho: densidad de masa,
      u: velocidad de flujo (pesada por masa),
      T: temperatura cinetica 1D = sum m (v-u)^2 / (count-1)   (k_B = 1)
    El factor (count-1) vuelve insesgado el estimador (se resto u del bin).
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    S, N = positions.shape
    dx = L / n_bins
    x_bins = (np.arange(n_bins) + 0.5) * dx - L / 2

    idx = np.clip((positions / dx).astype(np.intp), 0, n_bins - 1)
    flat = (idx + n_bins * np.arange(S)[:, None]).ravel()
    size = S * n_bins

    m = np.broadcast_to(masses, (S, N))
    mv = m * velocities

    count = np.bincount(flat, minlength=size).astype(float)
    M0 = np.bincount(flat, weights=m.ravel(), minlength=size)
    M1 = np.bincount(flat, weights=mv.ravel(), minlength=size)
    M2 = np.bincount(flat, weights=(mv * velocities).ravel(), minlength=size)

    with np.errstate(divide="ignore", invalid="ignore"):
        u = M1 / M0
        T = (M2 - M0 * u ** 2) / (count - 1.0)

    T[count < 2] = np.nan
    u[count == 0] = np.nan
    n = count / dx
    rho = M0 / dx
    n[count == 0] = np.nan
    rho[count == 0] = np.nan

    shape = (S, n_bins)
    return {
        "x": x_bins,
        "number_density": n.reshape(shape),
        "mass_density": rho.reshape(shape),
        "velocity": u.reshape(shape),
        "temperature": T.reshape(shape),
    }


# ============================================================
# 4. Amplitudes de modo y ajustes
# ============================================================

def mode_amplitude(x, field, L, mode, basis="cos"):
    """
    Coeficiente de Fourier con signo de `field(x,t)` (shape (S, n_bins)) sobre
    cos(kx) o sin(kx), tras restar la media espacial. Incluye la correccion por
    promedio en bins (sinc), asi para un perfil A*cos(kx) devuelve A.
    """
    k = 2.0 * np.pi * mode / L
    b = np.cos(k * x) if basis == "cos" else np.sin(k * x)
    f = field - np.nanmean(field, axis=1, keepdims=True)
    f = np.nan_to_num(f)
    amp = (f @ b) / (b @ b)
    arg = k * (x[1] - x[0]) / 2.0
    return amp * arg / np.sin(arg)


def entropy_field(hydro):
    """
    Combinacion de entropia (gas ideal 1D, gamma = 3):
        s ~ ln(T) - 2 ln(n)   =>   delta s ~ dT/T - 2 dn/n
    No la afectan las ondas sonoras; solo decae por conduccion de calor.
    """
    T, n = hydro["temperature"], hydro["number_density"]
    Tbar = np.nanmean(T, axis=1, keepdims=True)
    nbar = np.nanmean(n, axis=1, keepdims=True)
    return T / Tbar - 2.0 * n / nbar


def all_amplitudes(hydro, L, mode):
    x = hydro["x"]
    return {
        "A_T": mode_amplitude(x, hydro["temperature"], L, mode, "cos"),
        "A_n": mode_amplitude(x, hydro["number_density"], L, mode, "cos"),
        "A_u": mode_amplitude(x, hydro["velocity"], L, mode, "sin"),
        "A_s": mode_amplitude(x, entropy_field(hydro), L, mode, "cos"),
    }


def estimate_relaxation_time(times, amplitude_t,
                             lower_fraction=0.10, upper_fraction=0.80,
                             min_points=5):
    """Ajuste exponencial exploratorio |A| ~ exp(-t/tau). Devuelve (tau, R^2)."""
    valid = np.isfinite(amplitude_t)
    t = np.asarray(times)[valid]
    a = np.abs(np.asarray(amplitude_t)[valid])
    if len(a) < min_points or a[0] == 0:
        return np.nan, np.nan
    normalized = a / a[0]
    mask = (normalized < upper_fraction) & (normalized > lower_fraction) & (t > t[0])
    if np.count_nonzero(mask) < min_points:
        return np.nan, np.nan
    xfit, yfit = t[mask], np.log(a[mask])
    slope, intercept = np.polyfit(xfit, yfit, 1)
    if slope >= 0:
        return np.nan, np.nan
    ypred = slope * xfit + intercept
    ss_res = np.sum((yfit - ypred) ** 2)
    ss_tot = np.sum((yfit - np.mean(yfit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return -1.0 / slope, r2


def fit_stretched_exp(t, a):
    """
    Ajusta a(t) = exp(-(t/tau)^beta) a la amplitud de entropia normalizada.
    beta < 1 sugiere relajacion no exponencial (conduccion anomala en el AHP).
    Devuelve (tau, beta) o (nan, nan).
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return np.nan, np.nan
    ok = np.isfinite(a) & (t > 0)
    if ok.sum() < 8:
        return np.nan, np.nan
    f = lambda tt, tau, beta: np.exp(-(tt / tau) ** beta)
    try:
        p, _ = curve_fit(f, t[ok], a[ok], p0=(t[-1], 1.0),
                         bounds=([1e-3, 0.2], [1e7, 3.0]))
    except Exception:
        return np.nan, np.nan
    return float(p[0]), float(p[1])


# ============================================================
# 5. Ensemble paralelo
# ============================================================

def _worker(args):
    """Una simulacion completa; devuelve solo campos reducidos (picklable)."""
    seed, n_bins, sim_params, keep_raw = args
    res = simulate(seed=seed, **sim_params)
    L, mode = sim_params["L"], sim_params["mode"]
    h = hydro_profiles(res["positions"], res["velocities"], res["masses"],
                       L, n_bins=n_bins)
    out = {
        "x": h["x"], "times": res["times"],
        "temperature": h["temperature"],
        "number_density": h["number_density"],
        "mass_density": h["mass_density"],
        "velocity": h["velocity"],
        "energy_rel": (res["energy"] - res["energy"][0]) / res["energy"][0],
        "momentum": res["momentum"],
        "v_final": res["velocities"][-1].copy(),
        "n_collisions": res["n_collisions"],
        "masses": res["masses"],
    }
    out.update(all_amplitudes(h, L, mode))
    if keep_raw:
        N = res["positions"].shape[1]
        sel = np.unique(np.concatenate([np.arange(0, N, max(1, N // 40))[:40],
                                        np.arange(1, N, max(1, N // 40))[:40]]))
        out["traj"] = res["positions"][:, sel]
        out["traj_masses"] = res["masses"][sel]
    return out


def ensemble_parallel(seeds, n_bins=40, n_workers=None, **sim_params):
    """Ensemble paralelo (una semilla por proceso)."""
    sim_params = {k: v for k, v in sim_params.items() if k != "seed"}
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    n_workers = max(1, min(n_workers, len(seeds)))
    print(f"  {len(seeds)} semillas, {n_workers} proceso(s), "
          f"numba={'si' if HAVE_NUMBA else 'NO (Python puro, lento)'}")

    args = [(s, n_bins, sim_params, i == 0) for i, s in enumerate(seeds)]
    if n_workers == 1:
        results = [_worker(a) for a in args]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_worker, args, chunksize=1)

    ns = len(results)
    ens = {"x": results[0]["x"], "times": results[0]["times"],
           "masses": results[0]["masses"], "n_seeds": ns,
           "traj": results[0].get("traj"),
           "traj_masses": results[0].get("traj_masses")}

    for key in ("temperature", "number_density", "mass_density", "velocity"):
        arr = np.array([r[key] for r in results])
        ens[key] = np.nanmean(arr, axis=0)
        ens[key + "_sem"] = np.nanstd(arr, axis=0) / np.sqrt(ns)

    for key in ("A_T", "A_n", "A_u", "A_s"):
        arr = np.array([r[key] for r in results])
        ens[key] = np.nanmean(arr, axis=0)
        ens[key + "_sem"] = np.nanstd(arr, axis=0) / np.sqrt(ns)
        ens[key + "_all"] = arr

    ens["energy_rel"] = np.array([r["energy_rel"] for r in results])
    ens["momentum"] = np.array([r["momentum"] for r in results])
    ens["v_final"] = np.array([r["v_final"] for r in results])
    ens["n_collisions"] = np.array([r["n_collisions"] for r in results])
    return ens


# ============================================================
# 6. Teoria lineal (Euler, gamma=3) para comparar
# ============================================================

def euler_theory(ens, params):
    """
    Perturbacion inicial: dT/T = a cos(kx), drho = 0, u = 0.
    Descomposicion: 1/3 modo de entropia (no se propaga) + 2/3 ondas sonoras.
        A_T(t)/A_T(0) = 1/3 + 2/3 cos(wt)
        A_n(t)/n      = -(a/3) (1 - cos(wt))
        A_u(t)        = (a c / 3) sin(wt)
        A_s(t)        = a   (constante: sin conduccion)
    con c = sqrt(3 T / m_medio), w = c k.
    """
    t = ens["times"]
    a = params["amplitude"]
    Tbar = np.nanmean(ens["temperature"][0])
    mbar = 0.5 * (params["mA"] + params["mB"])
    c = np.sqrt(3.0 * Tbar / mbar)
    k = 2.0 * np.pi * params["mode"] / params["L"]
    w = c * k
    return {
        "c": c, "w": w, "period": 2 * np.pi / w, "Tbar": Tbar,
        "A_T": a * Tbar * (1.0 / 3.0 + 2.0 / 3.0 * np.cos(w * t)),
        "A_n_rel": -(a / 3.0) * (1.0 - np.cos(w * t)),
        "A_u": (a * c / 3.0) * np.sin(w * t),
        "A_s": a * np.ones_like(t),
    }


# ============================================================
# 7. Figuras (todas se guardan en PNG)
# ============================================================

def _save(fig, outdir, name, show):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"    guardada: {path}")
    if not show:
        plt.close(fig)


def fig_trajectories(ens, L, outdir, show):
    if ens["traj"] is None:
        return
    t, X, m = ens["times"], ens["traj"] - L / 2, ens["traj_masses"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for lab, mm, col in (("masa mA", m.min(), "tab:blue"),
                         ("masa mB", m.max(), "tab:red")):
        sel = np.isclose(m, mm)
        ax.plot(t, X[:, sel], ".", ms=1.5, color=col, alpha=0.6)
        ax.plot([], [], ".", color=col, label=lab)
    ax.axhline(-L / 2, ls="--", c="k", lw=0.8)
    ax.axhline(L / 2, ls="--", c="k", lw=0.8)
    ax.set_xlabel("t"); ax.set_ylabel("x")
    ax.set_title("Trayectorias (subconjunto, frontera periodica) - semilla 0")
    ax.legend(loc="upper right", markerscale=6)
    _save(fig, outdir, "01_trayectorias", show)


def fig_conservation(ens, outdir, show):
    t = ens["times"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for e in ens["energy_rel"]:
        axes[0].plot(t, e, lw=0.8, alpha=0.7)
    axes[0].set_title("Conservacion de energia")
    axes[0].set_ylabel(r"$(E-E_0)/E_0$")
    for p in ens["momentum"]:
        axes[1].plot(t, p, lw=0.8, alpha=0.7)
    axes[1].set_title("Momento total")
    axes[1].set_ylabel("P")
    for ax in axes:
        ax.set_xlabel("t"); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "02_conservacion", show)


def fig_maps(ens, outdir, show):
    x, t = ens["x"], ens["times"]
    fields = [("temperature", r"$T(x,t)$", "inferno"),
              ("mass_density", r"$\rho(x,t)$", "viridis"),
              ("velocity", r"$u(x,t)$", "coolwarm")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (key, title, cmap) in zip(axes, fields):
        data = np.ma.masked_invalid(ens[key])
        kw = {}
        if key == "velocity":
            vmax = np.nanmax(np.abs(ens[key])); kw = dict(vmin=-vmax, vmax=vmax)
        mesh = ax.pcolormesh(x, t, data, shading="auto", cmap=cmap, **kw)
        ax.set_xlabel("x"); ax.set_title(title)
        fig.colorbar(mesh, ax=ax)
    axes[0].set_ylabel("t")
    fig.suptitle("Mapas espacio-tiempo (promedio de ensemble)")
    fig.tight_layout()
    _save(fig, outdir, "03_mapas_espaciotiempo", show)


def fig_profiles(ens, params, selected_times, outdir, show):
    t, x = ens["times"], ens["x"]
    idxs = [int(np.argmin(np.abs(t - s))) for s in selected_times]
    k = 2.0 * np.pi * params["mode"] / params["L"]
    fields = [("temperature", r"$T(x,t)$", "Temperatura"),
              ("mass_density", r"$\rho(x,t)$", "Densidad de masa"),
              ("velocity", r"$u(x,t)$", "Velocidad de flujo")]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, (key, yl, title) in zip(axes, fields):
        for idx in idxs:
            mean, sem = ens[key][idx], ens[key + "_sem"][idx]
            ax.plot(x, mean, "o-", ms=3, label=f"t = {t[idx]:.0f}")
            ax.fill_between(x, mean - sem, mean + sem, alpha=0.2)
        if key == "temperature":
            Tbar = np.nanmean(ens[key][0])
            ax.plot(x, Tbar * (1 + params["amplitude"] * np.cos(k * x)),
                    "k--", lw=1, label="condicion inicial teorica")
        ax.set_ylabel(yl); ax.set_title(title + " (ensemble, banda = ±1 SEM)")
        ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    _save(fig, outdir, "04_perfiles_hidro", show)


def fig_amplitudes(ens, th, outdir, show):
    t = ens["times"]
    nbar = np.nanmean(ens["number_density"][0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    def band(ax, key, scale=1.0, color="tab:blue", label=""):
        y, e = ens[key] * scale, ens[key + "_sem"] * abs(scale)
        ax.plot(t, y, color=color, label=label)
        ax.fill_between(t, y - e, y + e, color=color, alpha=0.25)

    ax = axes[0, 0]
    band(ax, "A_T", label="simulacion")
    ax.plot(t, th["A_T"], "k--", label="Euler lineal")
    ax.set_title(r"Amplitud de temperatura $A_T$"); ax.set_ylabel(r"$A_T$")

    ax = axes[0, 1]
    band(ax, "A_n", 1.0 / nbar, label="simulacion")
    ax.plot(t, th["A_n_rel"], "k--", label="Euler lineal")
    ax.set_title(r"Amplitud de densidad $A_n/\bar n$"); ax.set_ylabel(r"$A_n/\bar n$")

    ax = axes[1, 0]
    band(ax, "A_u", label="simulacion")
    ax.plot(t, th["A_u"], "k--", label="Euler lineal")
    ax.set_title(r"Amplitud de velocidad $A_u$ (modo seno)"); ax.set_ylabel(r"$A_u$")

    ax = axes[1, 1]
    band(ax, "A_s", label="simulacion")
    ax.plot(t, th["A_s"], "k--", label="Euler (sin calor)")
    ax.set_title(r"Amplitud de entropia $A_s$ (modo de calor)"); ax.set_ylabel(r"$A_s$")

    for ax in axes.ravel():
        ax.axhline(0, c="gray", lw=0.6); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("t")
    fig.suptitle(f"Modos de Fourier (periodo sonoro teorico ≈ {th['period']:.1f}, "
                 f"c ≈ {th['c']:.3f})")
    fig.tight_layout()
    _save(fig, outdir, "05_amplitudes_modos", show)


def fig_entropy_decay(ens, outdir, show):
    t = ens["times"]
    A = ens["A_s"]
    a0 = A[0]
    a = A / a0
    tau_e, r2 = estimate_relaxation_time(t, a)
    tau_s, beta = fit_stretched_exp(t, a)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, ylog in zip(axes, (False, True)):
        ax.plot(t, a, color="tab:blue", label=r"$A_s(t)/A_s(0)$")
        ax.fill_between(t, a - ens["A_s_sem"] / a0, a + ens["A_s_sem"] / a0,
                        color="tab:blue", alpha=0.25)
        if np.isfinite(tau_s):
            ax.plot(t, np.exp(-(t / tau_s) ** beta), "r--",
                    label=fr"exp. estirada: $\tau$={tau_s:.3g}, $\beta$={beta:.2f}")
        if np.isfinite(tau_e):
            ax.plot(t, np.exp(-t / tau_e), "g:", label=fr"exponencial: $\tau$={tau_e:.3g}")
        ax.set_xlabel("t"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if ylog:
            ax.set_yscale("log"); ax.set_ylim(max(1e-2, np.nanmin(a[a > 0]) * 0.8), 1.5)
            ax.set_title("Escala semilog")
        else:
            ax.set_title("Relajacion del modo de calor")
    fig.tight_layout()
    _save(fig, outdir, "06_relajacion_entropia", show)
    print(f"    entropia: tau_exp={tau_e:.4g} (R2={r2:.3f}); "
          f"tau_estirada={tau_s:.4g}, beta={beta:.3f}")
    return tau_e, r2, tau_s, beta


def fig_spectrum(ens, th, outdir, show):
    t = ens["times"]
    dt = t[1] - t[0]
    sig = ens["A_T"] - np.nanmean(ens["A_T"])
    sig = np.nan_to_num(sig) * np.hanning(len(sig))
    nfft = 8 * len(sig)
    spec = np.abs(np.fft.rfft(sig, nfft))
    w = 2 * np.pi * np.fft.rfftfreq(nfft, dt)
    w_peak = w[np.argmax(spec)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(w, spec / spec.max())
    ax.axvline(th["w"], c="k", ls="--", label=fr"$\omega=ck$ teorico = {th['w']:.4f}")
    ax.axvline(w_peak, c="r", ls=":", label=f"pico medido = {w_peak:.4f}")
    ax.set_xlim(0, 4 * th["w"])
    ax.set_xlabel(r"$\omega$"); ax.set_ylabel("espectro normalizado de $A_T(t)$")
    ax.set_title("Frecuencia de la onda sonora"); ax.grid(alpha=0.3); ax.legend()
    _save(fig, outdir, "07_espectro_sonido", show)
    print(f"    omega medido = {w_peak:.4f}  vs  teorico = {th['w']:.4f}")
    return w_peak


def fig_velocity_distribution(ens, outdir, show):
    v, m = ens["v_final"], ens["masses"]
    mA, mB = m.min(), m.max()
    z = v * np.sqrt(m)                          # v*sqrt(m) ~ N(0, T)
    Tbar = np.mean(z ** 2)
    zs = z / np.sqrt(Tbar)
    g = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mm, col, lab in ((mA, "tab:blue", "mA"), (mB, "tab:red", "mB")):
        sel = np.isclose(m, mm)
        ax.hist(zs[:, sel].ravel(), bins=60, range=(-4, 4), density=True,
                histtype="step", color=col, label=f"{lab} (t final)")
    ax.plot(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), "k--", label="Gaussiana")
    ax.set_xlabel(r"$v\sqrt{m}/\sqrt{\bar T}$"); ax.set_ylabel("densidad")
    ax.set_title("Distribucion de velocidades al tiempo final"); ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, outdir, "08_distribucion_velocidades", show)


def fig_snapshots_heat(ens, outdir, show):
    """Evolucion de la desviacion relativa de T respecto a su media (heatmap)."""
    x, t = ens["x"], ens["times"]
    T = ens["temperature"]
    dT = T / np.nanmean(T, axis=1, keepdims=True) - 1.0
    vmax = np.nanmax(np.abs(dT))
    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(x, t, np.ma.masked_invalid(dT), shading="auto",
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("x"); ax.set_ylabel("t")
    ax.set_title(r"$T(x,t)/\langle T\rangle - 1$")
    fig.colorbar(mesh, ax=ax)
    _save(fig, outdir, "09_desviacion_relativa_T", show)


# ============================================================
# 8. Estudio en razon de masas
# ============================================================

def run_mass_ratio_study(mass_ratios, seeds, outdir, show,
                         n_bins=40, **sim_params):
    sim_params = {k: v for k, v in sim_params.items() if k not in ("seed", "mB")}
    curves, taus, betas = {}, [], []
    for r in mass_ratios:
        print(f"\n  r = mB/mA = {r:g}")
        t0 = time.time()
        e = ensemble_parallel(seeds, n_bins=n_bins,
                              mB=sim_params["mA"] * r, **sim_params)
        a = e["A_s"] / e["A_s"][0]
        tau, beta = fit_stretched_exp(e["times"], a)
        curves[r] = (e["times"], e["A_T"], a)
        taus.append(tau); betas.append(beta)
        print(f"    tau={tau:.4g} beta={beta:.3f}  ({time.time() - t0:.0f} s)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r, (t, AT, a) in curves.items():
        axes[0].plot(t, AT, label=fr"$r={r:g}$")
        axes[1].plot(t, a, label=fr"$r={r:g}$")
    axes[0].set_title(r"$A_T(t)$ para distintas razones de masa")
    axes[1].set_title(r"Modo de calor $A_s(t)/A_s(0)$")
    for ax in axes:
        ax.set_xlabel("t"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, outdir, "10_razon_masas_amplitudes", show)

    r_arr, tau_arr, b_arr = (np.asarray(mass_ratios, float),
                             np.asarray(taus), np.asarray(betas))
    good = np.isfinite(tau_arr)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(r_arr[good], tau_arr[good], "o-")
    axes[0].set_ylabel(r"$\tau$ (exp. estirada)")
    axes[1].plot(r_arr[good], b_arr[good], "o-")
    axes[1].set_ylabel(r"$\beta$")
    for ax in axes:
        ax.set_xlabel(r"$r=m_B/m_A$"); ax.grid(alpha=0.3)
    fig.suptitle("Tiempo caracteristico y exponente vs razon de masas")
    fig.tight_layout()
    _save(fig, outdir, "11_razon_masas_tau_beta", show)
    return curves


# ============================================================
# 9. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="prueba rapida")
    ap.add_argument("--mass-ratio", action="store_true", help="estudio en r=mB/mA")
    ap.add_argument("--show", action="store_true", help="mostrar figuras")
    ap.add_argument("--out", default="figuras_gas03", help="carpeta de salida")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    PARAMS = {
        "N": 10000, "L": 100.0, "mA": 1.0, "mB": 2.0, "T0": 1.0,
        "amplitude": 0.2,          # perturbacion grande vs ruido (A_T ruido ~ 2T0/sqrt(N*seeds))
        "mode": 2, "t_end": 400.0, "n_samples": 501,
    }
    N_BINS = 40
    SEEDS = list(range(40))
    SELECTED_TIMES = (0, 100, 200, 300, 400)
    MASS_RATIOS = (1.0, 2.0, 5.0, 10.0)
    MR_SEEDS = list(range(10))

    if args.quick:
        PARAMS.update(N=200, t_end=100.0, n_samples=201)
        SEEDS = list(range(4))
        SELECTED_TIMES = (0, 25, 50, 75, 100)
        MASS_RATIOS = (1.0, 2.0)
        MR_SEEDS = list(range(2))

    os.makedirs(args.out, exist_ok=True)
    print(f"Parametros: {PARAMS}")
    print(f"Salida de figuras: {os.path.abspath(args.out)}")

    # Pre-compilar numba una vez (cache) antes de lanzar los procesos.
    if HAVE_NUMBA:
        simulate(N=8, t_end=1.0, n_samples=2, seed=0)

    t0 = time.time()
    print("\nEnsemble principal:")
    ens = ensemble_parallel(SEEDS, n_bins=N_BINS, n_workers=args.workers, **PARAMS)
    print(f"  tiempo: {time.time() - t0:.1f} s; "
          f"colisiones/semilla ~ {ens['n_collisions'].mean():.3g}")
    print(f"  error maximo de energia: {np.abs(ens['energy_rel']).max():.2e}")

    th = euler_theory(ens, PARAMS)
    print(f"  Euler lineal: c={th['c']:.4f}, omega={th['w']:.4f}, periodo={th['period']:.2f}")

    print("\nFiguras:")
    fig_trajectories(ens, PARAMS["L"], args.out, args.show)
    fig_conservation(ens, args.out, args.show)
    fig_maps(ens, args.out, args.show)
    fig_profiles(ens, PARAMS, SELECTED_TIMES, args.out, args.show)
    fig_amplitudes(ens, th, args.out, args.show)
    fig_entropy_decay(ens, args.out, args.show)
    fig_spectrum(ens, th, args.out, args.show)
    fig_velocity_distribution(ens, args.out, args.show)
    fig_snapshots_heat(ens, args.out, args.show)

    np.savez_compressed(
        os.path.join(args.out, "resultados_ensemble.npz"),
        **{k: v for k, v in ens.items() if isinstance(v, np.ndarray)})

    if args.mass_ratio:
        print("\nEstudio en razon de masas:")
        run_mass_ratio_study(MASS_RATIOS, MR_SEEDS, args.out, args.show,
                             n_bins=N_BINS, **PARAMS)

    if args.show:
        plt.show()
    print("\nListo.")


if __name__ == "__main__":
    main()
