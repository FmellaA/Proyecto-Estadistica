
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Elastic collision
# --------------------------------------------------

def collision(m1, m2, v1, v2):
    total_mass = m1 + m2

    v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / total_mass
    v2_new = (2 * m1 * v1 + (m2 - m1) * v2) / total_mass

    return v1_new, v2_new


def energia(m, v):
    return 0.5 * np.sum(m * v**2)


def momentum_total(m, v):
    return np.sum(m * v)


# --------------------------------------------------


def gas(
    N=200,
    L=100.0,
    mA=1.0,
    mB=1.0,
    T0=1.0,
    amplitude=0.2,
    mode=1,
    seed=123
):
    """
    Initial conditions for a 1D AHP gas with a sinusoidal
    temperature perturbation and periodic boundary conditions.

    Positions are uniformly spaced in [0, L).
    Masses alternate between mA and mB.
    """

    if N < 2 or N % 2 != 0:
        raise ValueError("N must be an even integer >= 2.")

    if L <= 0 or mA <= 0 or mB <= 0 or T0 <= 0:
        raise ValueError("Invalid physical parameters.")

    if not 0 <= amplitude < 1:
        raise ValueError("amplitude must satisfy 0 <= amplitude < 1.")

    if mode < 1 or int(mode) != mode:
        raise ValueError("mode must be a positive integer.")

    rng = np.random.default_rng(seed)

    # Alternating masses
    masses = np.empty(N)
    masses[0::2] = mA
    masses[1::2] = mB

    # Uniform positions in [0, L)
    x = (np.arange(N) + 0.5) * L / N

    # Centered coordinates
    xi = x - L / 2

    # Initial sinusoidal temperature profile
    k = 2 * np.pi * mode / L

    T_local = T0 * (1 + amplitude * np.cos(k * xi))

    # Maxwell-Boltzmann velocity distribution
    v = rng.normal(
        loc=0.0,
        scale=np.sqrt(T_local / masses),
        size=N
    )

    # Remove global center-of-mass velocity
    v_cm = np.sum(masses * v) / np.sum(masses)
    v -= v_cm

    return masses, v, x
# --------------------------------------------------
# 3. Find the next event
# --------------------------------------------------

def next_event(x, v, L):
    """
    Particle indices remain fixed.
    Sorting is used only to identify spatial neighbors.

    Events:
      ("pair", i, j)  : internal collision
      ("periodic", i, j): collision across boundary
    """

    N = len(x)
    order = np.argsort(x)

    best_dt = np.inf
    best_event = None

    def consider(dt, event):
        nonlocal best_dt, best_event

        if dt >= 0 and dt < best_dt:
            best_dt = dt
            best_event = event

    # Internal neighboring pairs
    for k in range(N - 1):
        i = order[k]
        j = order[k + 1]

        relative_velocity = v[i] - v[j]

        if relative_velocity > 0:
            dt = (x[j] - x[i]) / relative_velocity
            consider(dt, ("pair", i, j))

    # Periodic neighboring pair: rightmost and leftmost
    i = order[-1]
    j = order[0]

    # Manejo especial si están exactamente en la misma posición (ej. cruzando el borde)
    if x[i] == x[j]:
        # Comprobamos ambas combinaciones porque argsort puede haberlas ordenado de forma arbitraria
        if v[i] - v[j] > 0:
            consider(0.0, ("pair", i, j))
        elif v[j] - v[i] > 0:
            consider(0.0, ("pair", j, i))
    else:
        relative_velocity = v[i] - v[j]

        if relative_velocity > 0:
            # Usamos módulo para obtener la distancia correcta en el dominio periódico
            gap = (x[j] - x[i] + L) % L
            dt = gap / relative_velocity
            consider(dt, ("periodic", i, j))

    return best_dt, best_event

# --------------------------------------------------
# 4. Event-driven simulation with periodic boundaries
# --------------------------------------------------

def simulate(
    N=200,
    L=100.0,
    mA=1.0,
    mB=1.0,
    T0=1.0,
    amplitude=0.2,
    mode=1,
    t_end=100.0,
    n_samples=501,
    seed=123):

    #masses, v, x = gas(N=N, L=L, mA=mA, mB=mB, T0=T0, seed=seed)
    # En lugar de las 4 partículas manuales, usa:
    masses, v, x = gas(
    N=N,
    L=L,
    mA=mA,
    mB=mB,
    T0=T0,
    amplitude=amplitude,
    mode=mode,
    seed=seed)
    

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

    for s in range(1, n_samples):
        target_time = sample_times[s]

        while t < target_time:
            dt_event, event = next_event(x, v, L)
            dt_sample = target_time - t

            # Observation time occurs before the next event
            if event is None or dt_sample < dt_event:
                x = (x + v * dt_sample) % L
                t = target_time
                break

            # Advance to the event
            x = (x + v * dt_event) % L
            t += dt_event

            event_type = event[0]

            if event_type in ("pair", "periodic"):
                _, i, j = event

                v[i], v[j] = collision(
                    masses[i], masses[j], v[i], v[j]
                )

                # Evitar choque espurio a tiempo cero entre las mismas
                # dos partículas cuando quedan exactamente en la misma posición
                if x[i] == x[j]:
                    eps = 1e-12
                    if v[i] > v[j]:
                        x[i] = (x[i] + eps) % L
                        x[j] = (x[j] - eps) % L
                    else:
                        x[i] = (x[i] - eps) % L
                        x[j] = (x[j] + eps) % L

            # elif event_type == "crossing":
            #     _, i, direction = event

            #     # Avoid repeatedly detecting a zero-time crossing.
            #     if direction == "left":
            #         x[i] = np.nextafter(L, 0.0)
            #     else:
            #         x[i] = 0.0

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
        "energy_error": abs(energy[-1] - initial_energy),
        "initial_momentum": initial_momentum,
        "final_momentum": momentum_total(masses, v),
        "energia": energy,
    }


# --------------------------------------------------
# 5. Hydrodynamic observables
# --------------------------------------------------

def hydro_profiles(positions, velocities, masses, L, n_bins=20):
    n_samples, N = positions.shape

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
            ub = np.mean(vb)

            total_mass = np.sum(mb)

            number_density[s, b] = count / dx
            mass_density[s, b] = total_mass / dx

            ub = np.sum(mb*vb) / total_mass
            mean_velocity[s, b] = ub

            temperature[s, b] = (np.sum(mb * (vb - ub)**2) / total_mass)


    return {
        "x": x_bins,
        "number_density": number_density,
        "mass_density": mass_density,
        "velocity": mean_velocity,
        "temperature": temperature,
    }

def thermal_mode_amplitude(hydro, times, L, mode, T_eq):

    x = hydro["x"]
    T = hydro["temperature"]

    k = 2 * np.pi * mode / L

    cosine = np.cos(k * x)

    amplitude = np.empty(len(times))

    for s in range(len(times)):

        valid = np.isfinite(T[s])

        amplitude[s] = (
            2 / np.count_nonzero(valid)
            * np.sum((T[s, valid] - T_eq) * cosine[valid])
        )

    return amplitude


# --------------------------------------------------
# 6. Run and inspect
# --------------------------------------------------

if __name__ == "__main__":

    result = simulate(
        N = 200, L=100.0,
        mA=1.0,
        mB=2.0,
        T0=1.0,
        t_end=100.0,
        n_samples=501,
        seed=123,
    )

    print("Initial energy:", result["initial_energy"])
    print("Maximum energy deviation:",
          np.max(np.abs(result["energia"] - result["initial_energy"])))
    print("Initial momentum:", result["initial_momentum"])
    print("Final momentum:", result["final_momentum"])

    # Plot wrapped trajectories
    plt.figure(figsize=(9, 5))
    x_centered = result["positions"] - result["L"] / 2

    for i in range(x_centered.shape[1]):
        plt.plot(result["times"], x_centered[:, i],'.', linewidth=0.5)

    plt.axhline(-result["L"] / 2, linestyle="--")
    plt.axhline(result["L"] / 2, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("AHP gas: periodic boundaries")
    plt.tight_layout()
    plt.show()

    # Energy conservation
    plt.figure()
    plt.plot(result["times"], result["energia"])
    plt.axhline(result["initial_energy"], linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Total kinetic energy")
    plt.title("Energy conservation")
    plt.tight_layout()
    #plt.show()

    # Hydrodynamic fields
    hydro = hydro_profiles(
        result["positions"],
        result["velocities"],
        result["masses"],
        result["L"],
        n_bins=20
    )  

    
    T_eq = 2 * result["initial_energy"] / len(result["masses"])

    A_T = thermal_mode_amplitude(
        hydro,
        result["times"],
        result["L"],
        mode=1,
        T_eq=T_eq
    )

    plt.figure()
    plt.plot(result["times"], A_T)
    plt.xlabel("Time")
    plt.ylabel("Thermal mode amplitude")
    plt.title("Thermal relaxation")
    plt.grid()
    plt.tight_layout()
    plt.show() 
    

    # selected_times = [0, 20, 50, 100]
    # indices = [
    #     np.argmin(np.abs(result["times"] - target))
    #     for target in selected_times
    # ]

    # fields = [
    #     ("number_density", "Number density"),
    #     ("mass_density", "Mass density"),
    #     ("velocity", "Mean velocity"),
    #     ("temperature", "Kinetic temperature"),
    # ]

    # for key, ylabel in fields:
    #     plt.figure()

    #     for idx in indices:
    #         plt.plot(
    #             hydro["x"],
    #             hydro[key][idx],
    #             label=f"t = {result['times'][idx]:.1f}",
    #         )

    #     plt.xlabel("Position x")
    #     plt.ylabel(ylabel)
    #     plt.title(ylabel)
    #     plt.legend()
    #     plt.tight_layout()
    #     #plt.show()

    # # Temperature space-time map
    # plt.figure(figsize=(9, 5))
    # plt.pcolormesh(
    #     hydro["x"],
    #     result["times"],
    #     hydro["temperature"],
    #     shading="auto",
    # )
    # plt.xlabel("Position x")
    # plt.ylabel("Time t")
    # plt.title("Temperature evolution")
    # plt.colorbar(label="Temperature")
    # plt.tight_layout()
    # #plt.show()
