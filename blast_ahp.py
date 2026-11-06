"""
Explosion (blast) en un gas AHP frio: validacion del motor de gas03_analysis.py
contra Chakraborti, Ganapa, Krapivsky & Dhar, PRL 126, 244503 (2021).

Que hace
--------
1. Usa EL MISMO nucleo dirigido por eventos de gas03_analysis.py (`_evolve`):
   anillo periodico, masas alternantes mA, mB.  Solo cambia la condicion
   inicial: gas a T = 0 con densidad uniforme y una energia E depositada en una
   region gaussiana de ancho sigma centrada en x = 0.
2. Campos hidrodinamicos como en la Ec. (4) del paper: se promedian en el
   ensemble los momentos (sum m, sum m v, sum m v^2/2) y luego
       v = p/rho,  eps = e/rho - v^2/2,  T = 2 mu eps,  mu = (mA+mB)/2.
3. Solucion exacta de Taylor-von Neumann-Sedov (TvNS) para gamma = 3, 1D:
   integra las EDOs autosimilares desde el choque (Rankine-Hugoniot) hacia
   el centro y obtiene la constante A de R(t) = (E t^2 / (A rho_inf))^(1/3).
4. Figuras tipo paper:
     B1  mapas espacio-tiempo de rho, |v|, T             (Fig. 1)
     B2  perfiles a 4 tiempos + colapso en xi = x/R(t)   (Fig. 2)
     B3  posicion del frente R(t) vs TvNS (t^(2/3))
     B4  escalamiento del nucleo X ~ t^(38/93)           (Fig. 5)
     B5  conservacion de energia y momento

Motor:
    --engine auto   numba si esta instalado (gas03_analysis._evolve compilado);
                    si no, la traduccion 1:1 a C de ahp_engine_c.py; si no, Python.
    --engine numba|c|python   para forzar uno.

Uso:
    python blast_ahp.py --quick                 # prueba (segundos)
    python blast_ahp.py                         # L = N = 4000, t <= 6400 (escala Fig. 3)
    python blast_ahp.py --paper                 # L = N = 24000, t <= 8e4 (Figs. 1-2)
    python blast_ahp.py --seeds 4000 --workers 8
"""
import os
import sys
import time
import argparse
from multiprocessing import Pool

import numpy as np
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import gas03_analysis as g03          # el motor del analisis principal

_EVOLVE = None                        # se fija en _set_engine()


def _set_engine(name):
    global _EVOLVE
    if name == "auto":
        name = "numba" if g03.HAVE_NUMBA else "c"
    if name == "c":
        from ahp_engine_c import load_c_engine
        ev = load_c_engine()
        if ev is None:
            print("  (sin compilador de C: uso el motor de Python puro)")
            name, ev = "python", g03._evolve
        _EVOLVE = ev
    else:                              # "numba" o "python": el _evolve original
        _EVOLVE = g03._evolve
    return name


def _pool_init(engine_name):
    _set_engine(engine_name)


# ============================================================
# 1. Condicion inicial de explosion
# ============================================================

def blast_initial_condition(N, L, mA, mB, E, sigma, seed):
    """
    Gas frio (v = 0) con N particulas uniformes al azar en [0, L) (densidad
    numerica n = N/L, rho_inf = n mu).  Las particulas cerca de x = L/2 reciben
    velocidades gaussianas con T(x) ~ exp(-(x-L/2)^2 / 2 sigma^2); se quita el
    momento de la region caliente y se reescala para que la energia sea E exacta.
    Devuelve (masses, v, x) con x ordenado: el formato que espera _evolve.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, L, N))
    m = np.empty(N)
    m[0::2], m[1::2] = mA, mB
    v = np.zeros(N)

    dx = x - L / 2
    hot = np.abs(dx) < 6.0 * sigma
    if hot.sum() < 2:                       # al menos 2 particulas calientes
        hot[np.argsort(np.abs(dx))[:2]] = True
    w = np.exp(-dx[hot] ** 2 / (2.0 * sigma ** 2))
    v[hot] = rng.normal(0.0, np.sqrt(w / m[hot]))
    v[hot] -= np.sum(m[hot] * v[hot]) / np.sum(m[hot])
    v *= np.sqrt(E / (0.5 * np.sum(m * v ** 2)))
    return m, v, x


# ============================================================
# 2. Ensemble: momentos por bin y frente de choque
# ============================================================

def _run_batch(args):
    seeds, P, times, n_bins = args
    N, L = P["N"], P["L"]
    S = len(times)
    dx = L / n_bins
    size = S * n_bins
    acc = {k: np.zeros(size) for k in ("count", "M0", "M1", "M2")}
    fronts, e_err, p_tot, n_coll = [], [], [], []
    row = n_bins * np.arange(S)[:, None]

    for seed in seeds:
        m, v, x = blast_initial_condition(N, L, P["mA"], P["mB"], P["E"],
                                          P["sigma"], seed)
        pos = np.empty((S, N))
        vel = np.empty((S, N))
        n_coll.append(_EVOLVE(x.copy(), v.copy(), m, float(L), times, pos, vel))

        idx = np.clip((pos / dx).astype(np.intp), 0, n_bins - 1)
        flat = (idx + row).ravel()
        mm = np.broadcast_to(m, (S, N))
        mv = mm * vel
        acc["count"] += np.bincount(flat, minlength=size)
        acc["M0"] += np.bincount(flat, weights=mm.ravel(), minlength=size)
        acc["M1"] += np.bincount(flat, weights=mv.ravel(), minlength=size)
        acc["M2"] += np.bincount(flat, weights=(0.5 * mv * vel).ravel(),
                                 minlength=size)

        # frente: las particulas frias tienen v == 0 exacto hasta ser golpeadas
        moving = vel != 0.0
        xr = np.where(moving, pos - L / 2, -np.inf).max(axis=1)
        xl = np.where(moving, pos - L / 2, np.inf).min(axis=1)
        fronts.append(0.5 * (xr - xl))

        en = 0.5 * (vel ** 2) @ m
        e_err.append(np.max(np.abs(en - en[0])) / en[0])
        p_tot.append(np.max(np.abs(vel @ m)))

    return acc, np.array(fronts), e_err, p_tot, n_coll


def run_ensemble(P, times, n_bins, seeds, workers, engine):
    batches = [seeds[i::max(1, 4 * workers)] for i in range(max(1, 4 * workers))]
    batches = [b for b in batches if len(b)]
    args = [(b, P, times, n_bins) for b in batches]
    if workers == 1:
        _set_engine(engine)
        out = [_run_batch(a) for a in args]
    else:
        with Pool(workers, initializer=_pool_init, initargs=(engine,)) as pool:
            out = pool.map(_run_batch, args, chunksize=1)

    S, nr = len(times), len(seeds)
    acc = {k: sum(o[0][k] for o in out).reshape(S, n_bins) for k in out[0][0]}
    dxb = P["L"] / n_bins
    mu = 0.5 * (P["mA"] + P["mB"])
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = acc["M0"] / (nr * dxb)
        p = acc["M1"] / (nr * dxb)
        e = acc["M2"] / (nr * dxb)
        v = np.where(rho > 0, p / rho, 0.0)
        eps = np.where(rho > 0, e / rho - 0.5 * v ** 2, 0.0)
    return {
        "x": (np.arange(n_bins) + 0.5) * dxb - P["L"] / 2,
        "times": times,
        "rho": rho, "v": v, "T": 2.0 * mu * eps,
        "front": np.concatenate([o[1] for o in out]),
        "energy_err": np.concatenate([o[2] for o in out]),
        "momentum_max": np.concatenate([o[3] for o in out]),
        "n_coll": np.concatenate([o[4] for o in out]),
        "n_real": nr,
    }


# ============================================================
# 3. Solucion exacta TvNS (gamma = 3, d = 1)
# ============================================================

def tvns_solution(gamma=3.0, lam_min=-20.0, n=20001):
    """
    rho = rho_inf G(xi),  v = (2/3)(x/t) V(xi),  c^2 = 3T/mu = ((2/3) x/t)^2 Z(xi)
    con xi = x/R(t).  En lam = ln xi las ecuaciones de Euler dan un sistema
    lineal para (lnG', V', lnZ'):
        (V-1) lnG' + V'                  = -V
        Z/g lnG' + (V-1) V' + Z/g lnZ'   = V/delta - V^2 - 2Z/g
        (1-g)(V-1) lnG' + (V-1) lnZ'     = 2/delta - 2V
    Rankine-Hugoniot (choque fuerte) en xi = 1:
        G = (g+1)/(g-1),  V = 2/(g+1),  Z = 2g(g-1)/(g+1)^2
    Energia:  A = 2 delta^2 Int_0^1 G xi^2 [V^2/2 + Z/(g(g-1))] dxi
    """
    g, d = gamma, 2.0 / 3.0

    def rhs(lam, y):
        lG, V, lZ = y
        Z = np.exp(lZ)
        Mx = np.array([[V - 1, 1, 0],
                       [Z / g, V - 1, Z / g],
                       [(1 - g) * (V - 1), 0, V - 1]])
        r = np.array([-V, V / d - V * V - 2 * Z / g, 2 / d - 2 * V])
        return np.linalg.solve(Mx, r)

    y0 = [np.log((g + 1) / (g - 1)), 2 / (g + 1),
          np.log(2 * g * (g - 1) / (g + 1) ** 2)]
    sol = solve_ivp(rhs, [0.0, lam_min], y0, rtol=1e-11, atol=1e-12,
                    dense_output=True)
    lam = np.linspace(lam_min, 0.0, n)
    y = sol.sol(lam)
    xi = np.exp(lam)
    G, V, Z = np.exp(y[0]), y[1], np.exp(y[2])
    A = 2 * d * d * np.trapezoid(G * xi ** 2 * (V ** 2 / 2 + Z / (g * (g - 1))), xi)
    return {"xi": xi, "G": G, "V": V, "Z": Z, "A": A}


def tvns_scaled_profiles(sol, P, xi_grid):
    """Campos escalados del paper: rho~ = rho_inf(G-1), v~ = t^{1/3}v, T~ = t^{2/3}T."""
    mu = 0.5 * (P["mA"] + P["mB"])
    rho_inf = P["N"] / P["L"] * mu
    alpha = (P["E"] / (sol["A"] * rho_inf)) ** (1.0 / 3.0)
    a = np.abs(xi_grid)
    inside = (a <= 1.0) & (a > sol["xi"][0])
    G = np.interp(a, sol["xi"], sol["G"])
    V = np.interp(a, sol["xi"], sol["V"])
    Z = np.interp(a, sol["xi"], sol["Z"])
    rt = np.where(inside, rho_inf * (G - 1.0), 0.0)
    vt = np.where(inside, (2 * alpha / 3) * xi_grid * V, 0.0)
    Tt = np.where(inside, (4 * mu * alpha ** 2 / 27) * a ** 2 * Z, 0.0)
    return rt, vt, Tt, alpha, rho_inf


# ============================================================
# 4. Figuras
# ============================================================

def _save(fig, outdir, name, show):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"    guardada: {path}")
    if not show:
        plt.close(fig)


def fig_maps(res, P, outdir, show):
    x, t = res["x"], res["times"]
    fields = [(res["rho"], r"$\rho(x,t)$", "hot", None),
              (np.abs(res["v"]), r"$|v(x,t)|$", "hot", None),
              (res["T"], r"$T(x,t)$", "hot", None)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, (data, title, cmap, _) in zip(axes, fields):
        vmax = np.nanpercentile(data[t > 0.1 * t[-1]], 99.5)
        mesh = ax.pcolormesh(x, t, data, shading="auto", cmap=cmap,
                             vmin=np.nanmin(data), vmax=vmax, rasterized=True)
        ax.set_xlabel("x"); ax.set_title(title)
        fig.colorbar(mesh, ax=ax)
    axes[0].set_ylabel("t")
    fig.suptitle(f"Explosion en gas AHP frio: N = L = {P['N']}, E = {P['E']:g}, "
                 f"$\\mu$ = {0.5 * (P['mA'] + P['mB']):g}, "
                 f"{res['n_real']} realizaciones")
    fig.tight_layout()
    _save(fig, outdir, "B1_mapas_blast", show)


def fig_profiles_and_collapse(res, P, sol, sel_times, outdir, show):
    x, t = res["x"], res["times"]
    idxs = [int(np.argmin(np.abs(t - s))) for s in sel_times]
    rho_inf = P["N"] / P["L"] * 0.5 * (P["mA"] + P["mB"])
    R = lambda tt: (P["E"] * tt ** 2 / (sol["A"] * rho_inf)) ** (1 / 3)
    xi_g = np.linspace(-1.1, 1.1, 2200)   # sin xi = 0 exacto
    rt, vt, Tt, _, _ = tvns_scaled_profiles(sol, P, xi_g)

    styles = [("-", "tab:green"), ("--", "tab:blue"),
              ("-.", "tab:red"), (":", "tab:purple")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    raw = [(res["rho"], r"$\rho(x,t)$"), (res["v"], r"$v(x,t)$"),
           (res["T"], r"$T(x,t)$")]
    scl = [(lambda f, tt: f - rho_inf, r"$\tilde\rho=\rho-\rho_\infty$", rt),
           (lambda f, tt: tt ** (1 / 3) * f, r"$\tilde v=t^{1/3}v$", vt),
           (lambda f, tt: tt ** (2 / 3) * f, r"$\tilde T=t^{2/3}T$", Tt)]
    for r, ((f, yl), (fs, yls, th)) in enumerate(zip(raw, scl)):
        for k, idx in enumerate(idxs):
            ls, c = styles[k % 4]
            lab = f"t = {t[idx]:.4g}"
            axes[r, 0].plot(x, f[idx], ls, c=c, lw=1.3, label=lab)
            axes[r, 1].plot(x / R(t[idx]), fs(f[idx], t[idx]), ls, c=c,
                            lw=1.3, label=lab)
        axes[r, 1].plot(xi_g, th, "k--", lw=1.2, label="TvNS exacta")
        axes[r, 0].set_ylabel(yl); axes[r, 1].set_ylabel(yls)
        axes[r, 1].set_xlim(-1.15, 1.15)
        if r == 2:   # la TvNS diverge en xi -> 0 (T ~ |x|^-1/2): recortar
            axes[r, 1].set_ylim(0, 2.0 * np.nanmax(fs(f[idxs[-1]], t[idxs[-1]])))
        for ax in axes[r]:
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[2, 0].set_xlabel("x"); axes[2, 1].set_xlabel(r"$\xi=x/R(t)$")
    fig.suptitle("Simulacion microscopica (motor gas03) vs solucion de "
                 "Taylor-von Neumann-Sedov", y=1.0)
    fig.tight_layout()
    _save(fig, outdir, "B2_perfiles_y_colapso_TvNS", show)


def fig_front(res, P, sol, outdir, show):
    t = res["times"][1:]
    Rm = res["front"][:, 1:]
    mean, sem = Rm.mean(0), Rm.std(0) / np.sqrt(Rm.shape[0])
    rho_inf = P["N"] / P["L"] * 0.5 * (P["mA"] + P["mB"])
    Rth = (P["E"] * t ** 2 / (sol["A"] * rho_inf)) ** (1 / 3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.loglog(t, mean, "o", ms=3, label="simulacion (particula en movimiento mas lejana)")
    ax.loglog(t, Rth, "k--", label=fr"TvNS: $(Et^2/A\rho_\infty)^{{1/3}}$, A = {sol['A']:.4f}")
    ax.set_xlabel("t"); ax.set_ylabel("R(t)"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8); ax.set_title(r"Frente de choque, $R\sim t^{2/3}$")
    ax = axes[1]
    ax.plot(t, mean / Rth, c="tab:blue")
    ax.fill_between(t, (mean - sem) / Rth, (mean + sem) / Rth, alpha=0.3)
    ax.axhline(1, c="k", ls="--")
    ax.set_xlabel("t"); ax.set_ylabel(r"$R_{sim}/R_{TvNS}$"); ax.grid(alpha=0.3)
    ax.set_title("Cociente (tiende a 1 a tiempos largos)")
    fig.tight_layout()
    _save(fig, outdir, "B3_frente_choque", show)
    return mean[-1] / Rth[-1]


def fig_core(res, P, sel_times, outdir, show):
    x, t = res["x"], res["times"]
    idxs = [int(np.argmin(np.abs(t - s))) for s in sel_times]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, idx in enumerate(idxs):
        tt = t[idx]
        eta = x / tt ** (38 / 93)
        axes[0].plot(eta, tt ** (4 / 31) * res["rho"][idx], label=f"t = {tt:.4g}")
        axes[1].plot(eta, tt ** (50 / 93) * res["T"][idx], label=f"t = {tt:.4g}")
    axes[0].set_ylabel(r"$t^{4/31}\rho$"); axes[1].set_ylabel(r"$t^{50/93}T$")
    win = np.abs(x / t[idxs[-1]] ** (38 / 93)) < 10
    for ax, f, p in ((axes[0], res["rho"], 4 / 31), (axes[1], res["T"], 50 / 93)):
        top = max(np.nanmax(t[i] ** p * f[i][np.abs(x / t[i] ** (38 / 93)) < 10]) for i in idxs)
        ax.set_ylim(0, 1.1 * top)
    for ax in axes:
        ax.set_xlim(-10, 10); ax.set_xlabel(r"$x/t^{38/93}$")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(r"Nucleo dominado por conduccion de calor: $X\sim t^{38/93}$ (Fig. 5 del paper)")
    fig.tight_layout()
    _save(fig, outdir, "B4_escalamiento_nucleo", show)


def fig_conservation(res, outdir, show):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(np.log10(res["energy_err"] + 1e-18), bins=40)
    axes[0].set_xlabel(r"$\log_{10}\max_t |E-E_0|/E_0$")
    axes[0].set_title("Conservacion de energia (por realizacion)")
    axes[1].hist(np.log10(res["momentum_max"] + 1e-18), bins=40, color="tab:orange")
    axes[1].set_xlabel(r"$\log_{10}\max_t |P|$")
    axes[1].set_title("Momento total")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "B5_conservacion", show)


# ============================================================
# 5. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--paper", action="store_true",
                    help="N = L = 24000, t hasta 8e4 (Figs. 1-2 del paper)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--engine", default="auto", choices=["auto", "numba", "c", "python"])
    ap.add_argument("--out", default="figuras_blast")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # mA = 1, mB = 2  ->  mu = 1.5;  n = N/L = 1  ->  rho_inf = 1.5;  E = 32
    P = {"N": 4000, "L": 4000.0, "mA": 1.0, "mB": 2.0, "E": 32.0, "sigma": 2.0}
    t_end, n_samples, bin_w, n_seeds = 6400.0, 161, 20.0, 2000
    sel = (800, 1600, 3200, 6400)
    core_sel = (1600, 3200, 6400)
    if args.paper:
        P.update(N=24000, L=24000.0)
        t_end, n_samples, bin_w, n_seeds = 8.0e4, 201, 60.0, 200
        sel = (1e4, 2e4, 4e4, 8e4)
        core_sel = (2e4, 4e4, 8e4)
    if args.quick:
        P.update(N=1000, L=1000.0)
        t_end, n_samples, bin_w, n_seeds = 600.0, 61, 10.0, 40
        sel = (75, 150, 300, 600)
        core_sel = (150, 300, 600)
    if args.seeds:
        n_seeds = args.seeds
    workers = args.workers or max(1, os.cpu_count() or 1)

    engine = _set_engine(args.engine)
    if engine == "numba":
        g03.simulate(N=8, t_end=1.0, n_samples=2, seed=0)   # compila una vez
    os.makedirs(args.out, exist_ok=True)

    sol = tvns_solution()
    rho_inf = P["N"] / P["L"] * 0.5 * (P["mA"] + P["mB"])
    R_end = (P["E"] * t_end ** 2 / (sol["A"] * rho_inf)) ** (1 / 3)
    print(f"Parametros: {P}; t_end = {t_end:g}; {n_seeds} realizaciones; motor = {engine}")
    print(f"TvNS: A = {sol['A']:.6f};  R(t_end) = {R_end:.1f}  (L/2 = {P['L'] / 2:g})")
    if R_end > 0.45 * P["L"]:
        print("  AVISO: el choque se acerca al borde del anillo; aumenta L o baja t_end.")

    times = np.linspace(0.0, t_end, n_samples)
    n_bins = int(round(P["L"] / bin_w))
    t0 = time.time()
    res = run_ensemble(P, times, n_bins, list(range(n_seeds)), workers, engine)
    print(f"  tiempo: {time.time() - t0:.1f} s; colisiones/realizacion ~ "
          f"{res['n_coll'].mean():.3g}; error max. de energia "
          f"{res['energy_err'].max():.1e}")

    print("Figuras:")
    fig_maps(res, P, args.out, args.show)
    fig_profiles_and_collapse(res, P, sol, sel, args.out, args.show)
    ratio = fig_front(res, P, sol, args.out, args.show)
    fig_core(res, P, core_sel, args.out, args.show)
    fig_conservation(res, args.out, args.show)
    print(f"  R_sim/R_TvNS en t_end = {ratio:.4f}")

    np.savez_compressed(os.path.join(args.out, "resultados_blast.npz"),
                        **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                        tvns_xi=sol["xi"], tvns_G=sol["G"], tvns_V=sol["V"],
                        tvns_Z=sol["Z"], tvns_A=sol["A"])
    if args.show:
        plt.show()
    print("Listo.")


if __name__ == "__main__":
    main()
