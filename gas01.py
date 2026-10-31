import numpy as np 
import matplotlib.pyplot as plt 

m1 = 1.0 
m2 = 2*m1

tmax = 999
N = 1000

def collision(m1,m2,v1,v2): 
    total_mass = m1 + m2
    v1_final = (v1*(m1 - m2) + 2*m2*v2) / total_mass 
    v2_final = (2*m1*v1 + (m2-m1)*v2) / total_mass
    return v1_final, v2_final

def energia(m, v): 
    return 0.5 *np.sum(m * v**2)

def momentum_total(m, v):
    return np.sum(m * v)

def gas(N = 100, L = 100.0, mA = 1.0, mB = 2.0, T0 = 1.0, seed= 123):
    rng = np.random.default_rng(seed)  
    masa = np.empty(N)
    masa[0::2] = mA
    masa[1::2] = mB

    x = (np.arange(N) + 0.5) * L / N
    vel = rng.normal(0.0, scale = np.sqrt(T0/masa), size=N)

    return masa, vel, x


def next_event(x, v, L):

    N = len(x)
    best_dt = np.inf
    best_event = None

    for i in range(N - 1):
        relative_velocity = v[i] - v[i + 1]

        if relative_velocity > 0:
            dt = (x[i + 1] - x[i]) / relative_velocity

            if dt < best_dt:
                best_dt = dt
                best_event = ("pair", i)

    relative_velocity_left = v[-1] - v[0]

    if relative_velocity_left > 0: 
        dt = (L - x[-1] + x[0]) / relative_velocity_left

        if dt < best_dt:
            best_dt = dt
            best_event = ("periodic", N -1)
        
    return best_dt, best_event


def simulate(
    N=3,
    L=100.0,
    mA=1.0,
    mB=2.0,
    T0=1.0,
    t_end=100.0,
    n_samples=501,
    seed=123
):
    masses, v, x = gas(
        N=N, L=L, mA=mA, mB=mB, T0=T0, seed=seed
    )
    energy = np.empty(n_samples)

    sample_times = np.linspace(0.0, t_end, n_samples)
    positions = np.empty((n_samples, N))
    velocities = np.empty((n_samples, N))

    t = 0.0
    sample_index = 0

    positions[0] = x.copy()
    velocities[0] = v.copy()
    sample_index = 1

    initial_energy = energia(masses, v)
    initial_momentum = momentum_total(masses, v)

    energy[0] = initial_energy

    while t < t_end and sample_index < n_samples:

        dt_event, event = next_event(x, v, L)

        target_time = sample_times[sample_index]
        dt_sample = target_time - t

        # Record the state before the next collision if the next
        # requested observation time occurs first.
        if dt_sample < dt_event:
            x += v * dt_sample
            x %= L
            t = target_time

            positions[sample_index] = x.copy()
            velocities[sample_index] = v.copy()
            energy[sample_index] = energia(masses, v)
            sample_index += 1
            continue

        # Advance to the next collision or wall reflection.
        if event is None or not np.isfinite(dt_event):
            x += v * dt_sample
            x %= L
            t = target_time
            positions[sample_index] = x.copy()
            velocities[sample_index] = v.copy()
            energy[sample_index] = energia(masses, v)
            sample_index += 1
            continue

        x += v * dt_event
        t += dt_event

        event_type, i = event

        if event_type == "pair":
            v[i], v[i + 1] = collision(
                masses[i], masses[i + 1], v[i], v[i + 1]
            )

        elif event_type == "periodic":
            v[-1], v[0] = collision(
                masses[-1], masses[0], v[-1], v[0]
            )

    energy_error = abs(energia(masses, v) - initial_energy)
    momentum_final = momentum_total(masses, v)

    return {
        "times": sample_times,
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "L": L,
        "initial_energy": initial_energy,
        "energy_error": energy_error,
        "initial_momentum": initial_momentum,
        "final_momentum": momentum_final,
        "energia": energy
    }


if __name__ == "__main__":
    result = simulate()

    print("Initial energy:", result["initial_energy"])
    print("Absolute energy error:", result["energy_error"])
    print("Initial momentum:", result["initial_momentum"])
    print("Final momentum:", result["final_momentum"])

    import matplotlib.pyplot as plt

    x = result["positions"]
    t = result["times"]

    x_centered = result["positions"] - result["L"] / 2

    plt.figure()
    for i in range(x_centered.shape[1]):
        plt.plot(result["times"], x_centered[:, i],'.', linewidth=0.6)

    plt.axhline(-result["L"] / 2, linestyle="--")
    plt.axhline( result["L"] / 2, linestyle="--")

    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("AHP gas trajectories")
    plt.tight_layout()
    plt.show()

    plt.plot(t, result["energia"])
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy of the AHP gas")
    plt.tight_layout()
    plt.show()


def hydro_profiles(positions, velocities, masses, L, n_bins=10):
    """
    Compute hydrodynamic fields from microscopic snapshots.

    positions, velocities: arrays of shape (n_samples, N)
    masses: array of shape (N,)
    L: box length; microscopic coordinates lie in [0, L]
    n_bins: number of spatial bins

    Uses k_B = 1.
    """

    n_samples, N = positions.shape

    edges = np.linspace(0.0, L, n_bins + 1)
    dx = edges[1] - edges[0]
    x_bins = 0.5 * (edges[:-1] + edges[1:]) - L / 2

    number_density = np.full((n_samples, n_bins), np.nan)
    mass_density = np.full((n_samples, n_bins), np.nan)
    mean_velocity = np.full((n_samples, n_bins), np.nan)
    temperature = np.full((n_samples, n_bins), np.nan)

    for s in range(n_samples):

        # Bin index of each particle
        bin_index = np.floor(positions[s] / dx).astype(int)
        bin_index = np.clip(bin_index, 0, n_bins - 1)

        for b in range(n_bins):
            mask = bin_index == b
            count = np.count_nonzero(mask)

            if count == 0:
                continue

            v_local = velocities[s, mask]
            m_local = masses[mask]

            number_density[s, b] = count / dx
            mass_density[s, b] = np.sum(m_local) / dx

            u = np.mean(v_local)
            mean_velocity[s, b] = u

            temperature[s, b] = np.mean(
                m_local * (v_local - u)**2
            )

    return {
        "x": x_bins,
        "number_density": number_density,
        "mass_density": mass_density,
        "velocity": mean_velocity,
        "temperature": temperature,
    }


hydro = hydro_profiles(
    positions=result["positions"],
    velocities=result["velocities"],
    masses=result["masses"],
    L=result["L"],
    n_bins=50
)

t = result["times"]
x = hydro["x"]

# Select several snapshots
selected_times = [0, 20, 50, 100]
indices = [
    np.argmin(np.abs(t - target))
    for target in selected_times
]

fields = [
    ("number_density", "Number density n(x,t)"),
    ("mass_density", "Mass density rho(x,t)"),
    ("velocity", "Mean velocity u(x,t)"),
    ("temperature", "Kinetic temperature T(x,t)"),
]

for key, ylabel in fields:
    plt.figure()

    for idx in indices:
        plt.plot(
            x,
            hydro[key][idx],
            label=f"t = {t[idx]:.1f}"
        )

    plt.xlabel("Position x")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xlim(-result["L"] / 2, result["L"] / 2)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


plt.plot(x/N, np.sum(fields["temperature"] - fields["temperature"], axis=1))
plt.show()


plt.pcolormesh(
    hydro["x"],
    result["times"],
    hydro["temperature"],
    shading="auto"
)

plt.xlabel("Position x")
plt.ylabel("Time t")
plt.title("Temperature evolution T(x,t)")
plt.colorbar(label="Temperature")
plt.tight_layout()
plt.show()