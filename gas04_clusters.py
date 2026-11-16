"""
Gas de puntos duros con patron de masas  (mA) - [n * mB] - (mA) - [n * mB] ...
en vez del alternante mA-mB-mA-mB (que corresponde a n = 1).

Reutiliza de gas03_analysis.py: el nucleo dirigido por eventos (_evolve, que
acepta cualquier arreglo de masas), hydro_profiles, all_amplitudes, los ajustes
y las figuras.  Solo cambia:
  - la condicion inicial (patron de masas, periodo n+1; N se ajusta a multiplo
    de n+1 para que el anillo periodico sea consistente),
  - la teoria de Euler: masa media  mbar = (mA + n mB)/(n+1),  c = sqrt(3T/mbar).

Uso:
    python gas04_clusters.py --quick
    python gas04_clusters.py                 # n = 2, 5, 10
    python gas04_clusters.py --n 1 2 5 10    # incluye n = 1 como referencia
    python gas04_clusters.py --seeds 40 --workers 8
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

import gas03_analysis as g03


# ============================================================
# 1. Condicion inicial con patron mA-[n mB]
# ============================================================

def adjust_N(N, n):
    """Multiplo de (n+1) mas cercano a N (y al menos 2 periodos)."""
    p = n + 1
    return max(2 * p, int(round(N / p)) * p)


def mean_mass(mA, mB, n):
    return (mA + n * mB) / (n + 1)


def gas_pattern(N, L, mA, mB, n, T0, amplitude, mode, seed):
    """Particulas equiespaciadas, masas [mA, mB*n] repetidas, T(x,0) sinusoidal."""
    if n < 1 or int(n) != n:
        raise ValueError("n must be a positive integer.")
    if N % (n + 1) != 0:
        raise ValueError("N must be a multiple of n+1.")
    if not 0 <= amplitude < 1:
        raise ValueError("amplitude must satisfy 0 <= amplitude < 1.")

    rng = np.random.default_rng(seed)
    masses = np.tile(np.array([mA] + [mB] * n, dtype=float), N // (n + 1))

    x = (np.arange(N) + 0.5) * L / N
    k = 2.0 * np.pi * mode / L
    T_local = T0 * (1.0 + amplitude * np.cos(k * (x - L / 2)))
    v = rng.normal(0.0, np.sqrt(T_local / masses), size=N)
    v -= np.sum(masses * v) / np.sum(masses)
    return masses, v, x


# ============================================================
# 2. Simulacion + ensemble
# ============================================================

def _worker(args):
    seed, n_bins, P, keep_raw = args
    N, L = P["N"], P["L"]
    masses, v, x = gas_pattern(N, L, P["mA"], P["mB"], P["n"], P["T0"],
                               P["amplitude"], P["mode"], seed)
    times = np.linspace(0.0, P["t_end"], P["n_samples"])
    pos = np.empty((len(times), N))
    vel = np.empty((len(times), N))
    n_coll = g03._evolve(x.copy(), v.copy(), masses, float(L), times, pos, vel)

    energy = 0.5 * (vel ** 2) @ masses
    h = g03.hydro_profiles(pos, vel, masses, L, n_bins=n_bins)
    out = {
        "x": h["x"], "times": times,
        "temperature": h["temperature"],
        "number_density": h["number_density"],
        "mass_density": h["mass_density"],
        "velocity": h["velocity"],
        "energy_rel": (energy - energy[0]) / energy[0],
        "momentum": vel @ masses,
        "v_final": vel[-1].copy(),
        "n_collisions": int(n_coll),
        "masses": masses,
    }
    out.update(g03.all_amplitudes(h, L, P["mode"]))
    if keep_raw:
        sel = np.unique(np.linspace(0, N - 1, 80).astype(int))
        out["traj"] = pos[:, sel]
        out["traj_masses"] = masses[sel]
    return out


def ensemble(seeds, P, n_bins, n_workers):
    n_workers = max(1, min(n_workers, len(seeds)))
    print(f"  {len(seeds)} semillas, {n_workers} proceso(s), "
          f"numba={'si' if g03.HAVE_NUMBA else 'NO (Python puro, lento)'}")
    args = [(s, n_bins, P, i == 0) for i, s in enumerate(seeds)]
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
# 3. Teoria de Euler lineal (gamma = 3) con masa media mbar
# ============================================================

def euler_theory(ens, params):
    """
    Perturbacion inicial: dT/T = a cos(kx), drho = 0, u = 0.
    1/3 modo de entropia (no se propaga) + 2/3 ondas sonoras.
        A_T(t)/A_T(0) = 1/3 + 2/3 cos(wt)
        A_n(t)/n      = -(a/3) (1 - cos(wt))
        A_u(t)        = (a c / 3) sin(wt)
        A_s(t)        = a
    con c = sqrt(3 T / mu), w = c k.
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
# 4. Comparacion entre n
# ============================================================

def fig_compare(summary, outdir, show):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for n, d in summary.items():
        axes[0].plot(d["t"], d["A_T"], label=fr"$n={n}$")
        axes[1].plot(d["t"], d["a_s"], label=fr"$n={n}$")
    axes[0].set_title(r"$A_T(t)$ para distintos $n$")
    axes[1].set_title(r"Modo de calor $A_s(t)/A_s(0)$")
    for ax in axes[:2]:
        ax.set_xlabel("t"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ns = np.array(list(summary), float)
    taus = np.array([summary[int(n)]["tau"] for n in ns])
    betas = np.array([summary[int(n)]["beta"] for n in ns])
    ok = np.isfinite(taus)
    ax = axes[2]
    ax.plot(ns[ok], taus[ok], "o-", c="tab:blue")
    ax.set_xlabel(r"$n$ (particulas $m_B$ entre cada $m_A$)")
    ax.set_ylabel(r"$\tau$ (exp. estirada)", color="tab:blue")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ns[ok], betas[ok], "s--", c="tab:red")
    ax2.set_ylabel(r"$\beta$", color="tab:red")
    ax.set_title(r"$\tau$ y $\beta$ vs $n$")
    fig.tight_layout()
    g03._save(fig, outdir, "gas04_comparacion_n", show)


# ============================================================
# 5. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--n", type=int, nargs="+", default=[2, 5, 10],
                    help="particulas mB entre cada mA (n=1 es el alternante)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="figuras_gas04")
    args = ap.parse_args()

    PARAMS = {"N": 10000, "L": 100.0, "mA": 1.0, "mB": 2.0, "T0": 1.0,
              "amplitude": 0.2, "mode": 2, "t_end": 400.0, "n_samples": 501}
    N_BINS = 40
    N_SEEDS = 40
    SEL = (0, 100, 200, 300, 400)
    if args.quick:
        PARAMS.update(N=200, t_end=100.0, n_samples=201)
        N_SEEDS = 4
        SEL = (0, 25, 50, 75, 100)
    if args.seeds:
        N_SEEDS = args.seeds
    workers = args.workers or max(1, (os.cpu_count() or 1) - 1)
    seeds = list(range(N_SEEDS))

    os.makedirs(args.out, exist_ok=True)
    if g03.HAVE_NUMBA:
        g03.simulate(N=8, t_end=1.0, n_samples=2, seed=0)   # compila una vez

    summary = {}
    for n in args.n:
        P = dict(PARAMS, n=n, N=adjust_N(PARAMS["N"], n))
        outdir = os.path.join(args.out, f"n{n}")
        os.makedirs(outdir, exist_ok=True)
        print(f"\n=== n = {n}  (patron mA-[{n}*mB]); N = {P['N']}, "
              f"mbar = {mean_mass(P['mA'], P['mB'], n):.4f} ===")

        t0 = time.time()
        ens = ensemble(seeds, P, N_BINS, workers)
        print(f"  tiempo: {time.time() - t0:.1f} s; colisiones/semilla ~ "
              f"{ens['n_collisions'].mean():.3g}; error max. de energia "
              f"{np.abs(ens['energy_rel']).max():.2e}")

        th = euler_theory(ens, P)
        print(f"  Euler lineal: c={th['c']:.4f}, omega={th['w']:.4f}, "
              f"periodo={th['period']:.2f}")

        print("  Figuras:")
        g03.fig_trajectories(ens, P["L"], outdir, args.show)
        g03.fig_conservation(ens, outdir, args.show)
        g03.fig_maps(ens, outdir, args.show)
        g03.fig_profiles(ens, P, SEL, outdir, args.show)
        g03.fig_amplitudes(ens, th, outdir, args.show)
        tau_e, r2, tau_s, beta = g03.fig_entropy_decay(ens, outdir, args.show)
        g03.fig_spectrum(ens, th, outdir, args.show)
        g03.fig_velocity_distribution(ens, outdir, args.show)
        g03.fig_snapshots_heat(ens, outdir, args.show)

        np.savez_compressed(
            os.path.join(outdir, "resultados_ensemble.npz"),
            **{k: v for k, v in ens.items() if isinstance(v, np.ndarray)})
        summary[n] = {"t": ens["times"], "A_T": ens["A_T"],
                      "a_s": ens["A_s"] / ens["A_s"][0],
                      "tau": tau_s, "beta": beta}

    print("\nComparacion entre n:")
    fig_compare(summary, args.out, args.show)
    for n, d in summary.items():
        print(f"  n = {n:>3d}: tau = {d['tau']:.4g}, beta = {d['beta']:.3f}")
    if args.show:
        plt.show()
    print("\nListo.")


if __name__ == "__main__":
    main()
