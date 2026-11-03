import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

# ============================================================
# 1. Microscopic dynamics: alternating hard-point gas (AHP)
# ============================================================

def collision(m1, m2, v1, v2):
    """Velocities after a 1D elastic collision."""
    total_mass = m1 + m2
    v1_new = ((m1 - m2) * v1 + 2.0 * m2 * v2) / total_mass
    v2_new = (2.0 * m1 * v1 + (m2 - m1) * v2) / total_mass
    return v1_new, v2_new


def energia(m, v):
    return 0.5 * np.sum(m * v**2)


def momentum_total(m, v):
    return np.sum(m * v)

def gas(N=200, L=100.0, mA=1.0, mB=2.0, T0=1.0,
        amplitude=0.2, mode=1, seed=123):
    """
    Initial condition with a sinusoidal temperature perturbation.

    The particles are equally spaced in [0,L), with alternating masses.
    T(x,0) = T0 * [1 + amplitude*cos(2*pi*mode*(x-L/2)/L)].
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
    xi = x - L / 2
    k = 2.0 * np.pi * mode / L
    T_local = T0 * (1.0 + amplitude * np.cos(k * xi))

    v = rng.normal(0.0, np.sqrt(T_local / masses), size=N)

    # Remove center-of-mass motion.
    v_cm = momentum_total(masses, v) / np.sum(masses)
    v -= v_cm

    return masses, v, x


# ============================================================
# 2. Event-driven dynamics with periodic boundary conditions
# ============================================================

def next_event(x, v, L):
    """
    Find the next collision between spatial nearest neighbors.

    Particle labels remain fixed; sorting is used only to identify
    the neighbors at the current instant.
    """
    N = len(x)
    order = np.argsort(x)

    best_dt = np.inf
    best_event = None

    def consider(dt, event):
        nonlocal best_dt, best_event
        if 0.0 <= dt < best_dt:
            best_dt = dt
            best_event = event

    # Neighbors inside the interval.
    for k in range(N - 1):
        i, j = order[k], order[k + 1]
        relative_velocity = v[i] - v[j]

        if relative_velocity > 0.0:
            dt = (x[j] - x[i]) / relative_velocity
            consider(dt, ("pair", i, j))

    # Neighbor pair across the periodic boundary.
    i, j = order[-1], order[0]
    gap = (x[j] - x[i]) % L
    relative_velocity = v[i] - v[j]

    if relative_velocity > 0.0:
        dt = gap / relative_velocity
        consider(dt, ("periodic", i, j))

    return best_dt, best_event


def simulate(N=200, L=100.0, mA=1.0, mB=2.0, T0=1.0,
             amplitude=0.2, mode=1, t_end=100.0,
             n_samples=501, seed=123):
    """Run the event-driven AHP simulation and store microscopic snapshots."""
    if t_end <= 0 or n_samples < 2:
        raise ValueError("t_end must be positive and n_samples >= 2.")

    masses, v, x = gas(
        N=N, L=L, mA=mA, mB=mB, T0=T0,
        amplitude=amplitude, mode=mode, seed=seed
    )

    sample_times = np.linspace(0.0, t_end, n_samples)
    positions = np.empty((n_samples, N))
    velocities = np.empty((n_samples, N))
    energy = np.empty(n_samples)

    initial_energy = energia(masses, v)
    initial_momentum = momentum_total(masses, v)

    positions[0] = x.copy()
    velocities[0] = v.copy()
    energy[0] = initial_energy

    t = 0.0
    eps = 1e-11 * L

    for s in range(1, n_samples):
        target_time = sample_times[s]

        while t < target_time:
            dt_event, event = next_event(x, v, L)
            dt_sample = target_time - t

            # No collision before the next observation time.
            if event is None or dt_sample < dt_event:
                x = (x + v * dt_sample) % L
                t = target_time
                break

            # Advance exactly to the collision.
            x = (x + v * dt_event) % L
            t += dt_event

            event_type, i, j = event
            v[i], v[j] = collision(masses[i], masses[j], v[i], v[j])

            # Separate the particles by a tiny distance while preserving
            # their spatial order on the ring. This prevents repeated
            # zero-time detection of the same collision.
            if event_type == "pair":
                x[i] = (x[i] - eps) % L
                x[j] = (x[j] + eps) % L
            else:
                # i is the rightmost particle and j the leftmost one.
                x[i] = (L - eps) % L
                x[j] = eps

        positions[s] = x.copy()
        velocities[s] = v.copy()
        energy[s] = energia(masses, v)

    return {
        "times": sample_times,
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "L": L,
        "initial_energy": initial_energy,
        "initial_momentum": initial_momentum,
        "final_momentum": momentum_total(masses, v),
        "energia": energy,
        "energy_error": np.max(np.abs(energy - initial_energy)),
    }


# ============================================================
# 3. Hydrodynamic fields
# ============================================================

def hydro_profiles(positions, velocities, masses, L, n_bins=20):
    """
    Coarse-grained fields in spatial bins.

    rho: mass density
    u: mass-weighted flow velocity
    T: 1D kinetic temperature, with k_B = 1
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")

    n_samples, _ = positions.shape
    edges = np.linspace(0.0, L, n_bins + 1)
    dx = edges[1] - edges[0]
    x_bins = 0.5 * (edges[:-1] + edges[1:]) - L / 2

    number_density = np.full((n_samples, n_bins), np.nan)
    mass_density = np.full((n_samples, n_bins), np.nan)
    mean_velocity = np.full((n_samples, n_bins), np.nan)
    temperature = np.full((n_samples, n_bins), np.nan)

    for s in range(n_samples):
        bin_index = np.floor(positions[s] / dx).astype(int)
        bin_index = np.clip(bin_index, 0, n_bins - 1)

        for b in range(n_bins):
            mask = bin_index == b
            count = np.count_nonzero(mask)
            if count == 0:
                continue

            vb = velocities[s, mask]
            mb = masses[mask]
            total_mass = np.sum(mb)

            ub = np.sum(mb * vb) / total_mass

            number_density[s, b] = count / dx
            mass_density[s, b] = total_mass / dx
            mean_velocity[s, b] = ub

            # In one dimension: T = mean[m_i (v_i-u)^2].
            # Divide by particle number, not by total mass.
            temperature[s, b] = (np.sum(mb * (vb - ub)**2) / count)

    return {
        "x": x_bins,
        "number_density": number_density,
        "mass_density": mass_density,
        "velocity": mean_velocity,
        "temperature": temperature,
    }


# ============================================================
# 3b. Ensemble averaging over initial conditions
# ============================================================

def ensemble_hydro(seeds, n_bins=40, **sim_params):
    """
    Run simulate() with different seeds and average the hydro profiles.

    sim_params must contain: N, L, mA, mB, T0, amplitude, mode,
    t_end, n_samples (WITHOUT 'seed').
    """
    T_list, rho_list, u_list = [], [], []

    for seed in seeds:
        result = simulate(seed=seed, **sim_params)
        hydro = hydro_profiles(
            result["positions"],
            result["velocities"],
            result["masses"],
            sim_params["L"],
            n_bins=n_bins,
        )
        T_list.append(hydro["temperature"])
        rho_list.append(hydro["mass_density"])
        u_list.append(hydro["velocity"])

    T_all   = np.array(T_list)
    rho_all = np.array(rho_list)
    u_all   = np.array(u_list)

    return {
        "x": hydro["x"],
        "times": result["times"],
        "temperature":       np.nanmean(T_all,   axis=0),
        "temperature_std":   np.nanstd(T_all,    axis=0),
        "mass_density":      np.nanmean(rho_all, axis=0),
        "mass_density_std":  np.nanstd(rho_all,  axis=0),
        "velocity":          np.nanmean(u_all,   axis=0),
        "velocity_std":      np.nanstd(u_all,    axis=0),
    }


def ensemble_thermal_amplitude(seeds, n_bins=40, **sim_params):
    """
    Compute A_T(t) for each seed and return mean and std over seeds.

    sim_params must contain: N, L, mA, mB, T0, amplitude, mode,
    t_end, n_samples (WITHOUT 'seed').
    """
    A_list = []

    for seed in seeds:
        result = simulate(seed=seed, **sim_params)
        hydro = hydro_profiles(
            result["positions"],
            result["velocities"],
            result["masses"],
            sim_params["L"],
            n_bins=n_bins,
        )
        A_list.append(
            thermal_mode_amplitude(
                hydro, result["times"], sim_params["L"], sim_params["mode"]
            )
        )

    A_all = np.array(A_list)

    return {
        "times": result["times"],
        "amplitude":     np.nanmean(A_all, axis=0),
        "amplitude_std": np.nanstd(A_all,  axis=0),
        "amplitude_all": A_all,
    }


def _worker_seed(seed, n_bins, sim_params):
    """
    Run one full simulation for a given seed and return only the
    reduced fields needed for ensemble averaging.

    This function must live at module level so it can be pickled
    by multiprocessing on Windows.
    """
    params = dict(sim_params)
    params.pop("seed", None)          # safety: seed is passed separately

    result = simulate(seed=seed, **params)
    hydro = hydro_profiles(
        result["positions"],
        result["velocities"],
        result["masses"],
        params["L"],
        n_bins=n_bins,
    )
    A_T = thermal_mode_amplitude(
        hydro, result["times"], params["L"], params["mode"]
    )

    return {
        "x": hydro["x"],
        "times": result["times"],
        "temperature": hydro["temperature"],
        "mass_density": hydro["mass_density"],
        "velocity": hydro["velocity"],
        "amplitude": A_T,
    }


def ensemble_parallel(seeds, n_bins=40, n_workers=None, **sim_params):
    """
    Run the ensemble in parallel, one seed per core.

    Returns averaged fields, their standard deviations, and the
    per-seed A_T matrix for further analysis.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    n_workers = min(n_workers, len(seeds))

    print(f"Using {n_workers} worker(s) for {len(seeds)} seeds.")

    args = [(seed, n_bins, sim_params) for seed in seeds]

    with Pool(n_workers) as pool:
        # imap_unordered gives progress while workers finish
        results = list(pool.starmap(_worker_seed, args))

    T_all   = np.array([r["temperature"]  for r in results])
    rho_all = np.array([r["mass_density"] for r in results])
    u_all   = np.array([r["velocity"]     for r in results])
    A_all   = np.array([r["amplitude"]    for r in results])

    return {
        "x":   results[0]["x"],
        "times": results[0]["times"],
        "temperature":      np.nanmean(T_all,   axis=0),
        "temperature_std":  np.nanstd(T_all,    axis=0),
        "mass_density":     np.nanmean(rho_all, axis=0),
        "mass_density_std": np.nanstd(rho_all,  axis=0),
        "velocity":         np.nanmean(u_all,   axis=0),
        "velocity_std":     np.nanstd(u_all,    axis=0),
        "amplitude":        np.nanmean(A_all,   axis=0),
        "amplitude_std":    np.nanstd(A_all,    axis=0),
        "amplitude_all":    A_all,
    }
# ============================================================
# 4. Thermal Fourier-mode amplitude and relaxation time
# ============================================================

def thermal_mode_amplitude(hydro, times, L, mode):
    """
    Signed cosine Fourier coefficient of T(x,t), after subtracting
    the spatial mean at each time.

    For a well-resolved profile T = T_mean + A*cos(kx), this returns A.
    """
    x = hydro["x"]
    T = hydro["temperature"]
    k = 2.0 * np.pi * mode / L
    cosine = np.cos(k * x)

    amplitude_t = np.full(len(times), np.nan)

    for s in range(len(times)):
        valid = np.isfinite(T[s])
        if np.count_nonzero(valid) < 2:
            continue

        Tb = T[s, valid]
        cb = cosine[valid]
        Tb = Tb - np.mean(Tb)

        denominator = np.sum(cb**2)
        if denominator > 0:
            amplitude_t[s] = np.sum(Tb * cb) / denominator

    return amplitude_t


def estimate_relaxation_time(times, amplitude_t,
                             lower_fraction=0.10,
                             upper_fraction=0.80,
                             min_points=5):
    """
    Exploratory exponential fit |A_T| ~ exp(-t/tau).

    The fit uses points for which lower_fraction < |A|/|A(0)|
    < upper_fraction. Returns (tau, R_squared), or (nan, nan)
    if a plausible fit cannot be identified.

    This is a diagnostic, not an assumption that relaxation must
    be exponential.
    """
    valid = np.isfinite(amplitude_t)
    t = np.asarray(times)[valid]
    a = np.abs(np.asarray(amplitude_t)[valid])

    if len(a) < min_points or a[0] == 0:
        return np.nan, np.nan

    normalized = a / a[0]
    mask = (
        (normalized < upper_fraction)
        & (normalized > lower_fraction)
        & (t > t[0])
    )

    if np.count_nonzero(mask) < min_points:
        return np.nan, np.nan

    xfit = t[mask]
    yfit = np.log(a[mask])
    slope, intercept = np.polyfit(xfit, yfit, 1)

    if slope >= 0:
        return np.nan, np.nan

    ypred = slope * xfit + intercept
    ss_res = np.sum((yfit - ypred)**2)
    ss_tot = np.sum((yfit - np.mean(yfit))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    tau = -1.0 / slope
    return tau, r_squared


# ============================================================
# 5. Figures for one mass ratio
# ============================================================

def plot_single_run(result, hydro, A_T, mode,
                    selected_times=(0, 20, 50, 100),
                    show=True, save_prefix=None):
    times = result["times"]
    x = hydro["x"]

    # (1) Space-time temperature map.
    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(
        x, times, np.ma.masked_invalid(hydro["temperature"]),
        shading="auto", cmap="inferno"
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_title(r"Space-time temperature map $T(x,t)$")
    fig.colorbar(mesh, ax=ax, label=r"$T(x,t)$")
    fig.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}_temperature_map.png", dpi=300)

    # (2) Hydrodynamic profiles T, rho and u at selected times.
    indices = [
        int(np.argmin(np.abs(times - target)))
        for target in selected_times
    ]
    fields = [
        ("temperature", r"$T(x,t)$", "Temperature"),
        ("mass_density", r"$\rho(x,t)$", "Mass density"),
        ("velocity", r"$u(x,t)$", "Flow velocity"),
    ]

    fig_profiles, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, (key, ylabel, title) in zip(axes, fields):
        for idx in indices:
            ax.plot(
                x, hydro[key][idx],
                marker="o", markersize=3,
                label=f"t = {times[idx]:.1f}"
            )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel(r"$x$")
    fig_profiles.tight_layout()
    if save_prefix:
        fig_profiles.savefig(f"{save_prefix}_hydro_profiles.png", dpi=300)

    # (3) Thermal mode amplitude.
    fig_amp, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, A_T, label=fr"$A_T(t)$, mode {mode}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$A_T(t)$")
    ax.set_title("Thermal perturbation relaxation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig_amp.tight_layout()
    if save_prefix:
        fig_amp.savefig(f"{save_prefix}_thermal_amplitude.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_ensemble(ens, A_ens, mode, selected_times=(0, 20, 50, 100),
                  save_prefix=None, show=True):
    """Plot ensemble-averaged hydro profiles with ±1σ bands."""
    times = ens["times"]
    x = ens["x"]

    indices = [
        int(np.argmin(np.abs(times - t))) for t in selected_times
    ]

    fields = [
        ("temperature",    "temperature_std",  r"$T(x,t)$",  "Temperature"),
        ("mass_density",   "mass_density_std", r"$\rho(x,t)$", "Mass density"),
        ("velocity",       "velocity_std",     r"$u(x,t)$",  "Flow velocity"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, (key, std_key, ylabel, title) in zip(axes, fields):
        for idx in indices:
            mean = ens[key][idx]
            std  = ens[std_key][idx]
            ax.plot(x, mean, marker="o", markersize=3,
                    label=f"t = {times[idx]:.1f}")
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_ylabel(ylabel)
        ax.set_title(title + " (ensemble)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel(r"$x$")
    fig.tight_layout()
    if save_prefix:
        fig.savefig(f"{save_prefix}_ensemble_profiles.png", dpi=300)

    fig_amp, ax = plt.subplots(figsize=(8, 5))
    A     = A_ens["amplitude"]
    A_std = A_ens["amplitude_std"]
    ax.plot(times, A, label=fr"$\langle A_T(t)\rangle$, mode {mode}")
    ax.fill_between(times, A - A_std, A + A_std, alpha=0.2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$A_T(t)$")
    ax.set_title("Thermal perturbation relaxation (ensemble)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig_amp.tight_layout()
    if save_prefix:
        fig_amp.savefig(f"{save_prefix}_ensemble_thermal_amplitude.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close("all")


# ============================================================
# 6. Mass-ratio study
# ============================================================

def run_mass_ratio_study(mass_ratios, N=200, L=100.0, mA=1.0,
                         T0=1.0, amplitude=0.2, mode=1,
                         t_end=100.0, n_samples=501,
                         n_bins=20, seed=123):
    """
    Run one simulation for each r=mB/mA and compare A_T(t).
    The same seed is used to make the initial random-number protocol
    reproducible across the different mass ratios.
    """
    curves = {}
    taus = []
    fit_quality = []

    for r in mass_ratios:
        print(f"Running mass ratio r = {r:g} ...")
        result = simulate(
            N=N, L=L, mA=mA, mB=mA * r,
            T0=T0, amplitude=amplitude, mode=mode,
            t_end=t_end, n_samples=n_samples, seed=seed
        )

        hydro = hydro_profiles(
            result["positions"], result["velocities"],
            result["masses"], L, n_bins=n_bins
        )
        A_T = thermal_mode_amplitude(
            hydro, result["times"], L, mode
        )
        tau, r_squared = estimate_relaxation_time(
            result["times"], A_T
        )

        curves[r] = {
            "times": result["times"],
            "amplitude": A_T,
            "tau": tau,
            "r_squared": r_squared,
        }
        taus.append(tau)
        fit_quality.append(r_squared)

        print(
            f"  energy error = {result['energy_error']:.3e}; "
            f"tau = {tau:.5g}; R^2 = {r_squared:.3f}"
        )

    # Overlay of A_T(t) for all mass ratios.
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, data in curves.items():
        ax.plot(data["times"], data["amplitude"], label=fr"$r={r:g}$")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$A_T(t)$")
    ax.set_title("Thermal-mode relaxation for different mass ratios")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # (4) Characteristic relaxation time versus mass ratio.
    fig_tau, ax_tau = plt.subplots(figsize=(7, 5))
    tau_array = np.asarray(taus, dtype=float)
    r_array = np.asarray(mass_ratios, dtype=float)
    good = np.isfinite(tau_array)

    ax_tau.plot(r_array[good], tau_array[good], "o-")
    ax_tau.set_xlabel(r"Mass ratio $r=m_B/m_A$")
    ax_tau.set_ylabel(r"Characteristic time $\tau$")
    ax_tau.set_title("Relaxation time versus mass ratio")
    ax_tau.grid(alpha=0.3)
    fig_tau.tight_layout()

    plt.show()

    return curves


# ============================================================
# 7. Main analysis
# ============================================================
if __name__ == "__main__":

    PARAMS = {
        "N": 1000,
        "L": 100.0,
        "mA": 1.0,
        "mB": 2.0,
        "T0": 1.0,
        "amplitude": 0.05,
        "mode": 2,
        "t_end": 400.0,
        "n_samples": 501,
        "seed": 123,
    }

    N_BINS = 40
    SELECTED_TIMES = (0, 100, 200, 300, 400)
    SEEDS = list(range(10))

    # Parámetros para simular SIN la semilla.
    sim_params = {k: v for k, v in PARAMS.items() if k != "seed"}

    # ---------- (A) Corrida única (opcional, solo diagnóstico) ----------
    RUN_SINGLE_CHECK = False
    if RUN_SINGLE_CHECK:
        result = simulate(**PARAMS)
        print("Initial energy:", result["initial_energy"])
        print("Maximum energy deviation:", result["energy_error"])
        print("Initial momentum:", result["initial_momentum"])
        print("Final momentum:", result["final_momentum"])

    # ---------- (B) Ensemble paralelo ----------
    print(f"\nRunning ensemble with {len(SEEDS)} seeds ...")
    ens_all = ensemble_parallel(
        SEEDS, n_bins=N_BINS, n_workers=None, **sim_params
    )

    ens = {k: ens_all[k] for k in [
        "x", "times",
        "temperature", "temperature_std",
        "mass_density", "mass_density_std",
        "velocity", "velocity_std",
    ]}
    A_ens = {k: ens_all[k] for k in [
        "times", "amplitude", "amplitude_std", "amplitude_all",
    ]}

    tau, r_squared = estimate_relaxation_time(
        A_ens["times"], A_ens["amplitude"]
    )
    print("Estimated relaxation time (ensemble):", tau)
    print("Exponential-fit R^2 (ensemble):     ", r_squared)

    plot_ensemble(
        ens, A_ens, PARAMS["mode"],
        selected_times=SELECTED_TIMES,
        save_prefix="gas03",
    )