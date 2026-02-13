import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SciPy FFT (fast)
from scipy.fft import fft2, ifft2, fftfreq


# ============================================================
# 1) FUNCTIONS
# ============================================================

def make_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, dx, dy, X, Y


def make_wavepacket(X, Y, x0, y0, sigma, k0):
    psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sigma**2)) * np.exp(1j * k0 * X)
    return psi


def normalize(psi, dx, dy):
    prob = np.abs(psi)**2
    norm = np.sum(prob) * dx * dy
    return psi / np.sqrt(norm)


def make_k_space(Nx, Ny, dx, dy):
    kx = 2*np.pi * fftfreq(Nx, d=dx)
    ky = 2*np.pi * fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    return K2


def make_absorber(X, Y, Lx, Ly, strength=4.0, power=10):
    rx = np.abs(X) / (Lx/2)
    ry = np.abs(Y) / (Ly/2)
    r = np.maximum(rx, ry)
    mask = np.exp(-strength * (r**power))
    return mask


def make_double_slit_potential(X, Y, x_wall=0.0, thickness=0.25,
                              slit_width=0.7, slit_sep=2.0, V0=500.0):
    """
    Creates a vertical barrier wall at x = x_wall with two slits.
    - thickness: wall thickness in x direction
    - slit_width: slit opening size in y direction
    - slit_sep: distance between the centers of the slits
    - V0: barrier height
    """

    V = np.zeros_like(X, dtype=float)

    # Wall region: where x is close to x_wall
    wall = np.abs(X - x_wall) < (thickness / 2)

    # Define the two slit centers
    y1 = +slit_sep / 2
    y2 = -slit_sep / 2

    # Slit openings: allow wave through (potential = 0) in these y ranges
    slit1 = np.abs(Y - y1) < (slit_width / 2)
    slit2 = np.abs(Y - y2) < (slit_width / 2)

    # Where wall is present BUT not inside slits -> barrier
    barrier = wall & (~(slit1 | slit2))

    V[barrier] = V0
    return V


def make_operators(V, K2, dt, hbar=1.0, m=1.0):
    U_V = np.exp(-1j * V * dt / (2*hbar))
    U_K = np.exp(-1j * (hbar * K2) * dt / (2*m))
    return U_V, U_K


def step_split_step(psi, U_V, U_K, absorber):
    psi = U_V * psi
    psi_k = fft2(psi)
    psi_k *= U_K
    psi = ifft2(psi_k)
    psi = U_V * psi
    psi *= absorber
    return psi


# ============================================================
# 2) PARAMETERS (CONTROL PANEL)
# ============================================================

# Grid
Nx, Ny = 300, 300
Lx, Ly = 24.0, 24.0

# Time
dt = 0.0025
steps_per_frame = 6

# Physics constants (natural units)
hbar = 1.0
m = 1.0

# Wave packet
x0, y0 = -8.0, 0.0
sigma = 0.9
k0 = 8.0

# Double slit
x_wall = 0.0
thickness = 0.35
slit_width = 0.9
slit_sep = 3.0
V0 = 1200.0

# Absorber
absorber_strength = 4.5
absorber_power = 12


# ============================================================
# 3) SETUP
# ============================================================

x, y, dx, dy, X, Y = make_grid(Nx, Ny, Lx, Ly)
psi = make_wavepacket(X, Y, x0, y0, sigma, k0)
psi = normalize(psi, dx, dy)

K2 = make_k_space(Nx, Ny, dx, dy)
V = make_double_slit_potential(X, Y, x_wall, thickness, slit_width, slit_sep, V0)
absorber = make_absorber(X, Y, Lx, Ly, absorber_strength, absorber_power)

U_V, U_K = make_operators(V, K2, dt, hbar, m)


# ============================================================
# 4) VISUALIZATION
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

prob0 = np.abs(psi)**2

img = ax.imshow(
    prob0.T,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    interpolation="nearest",
    aspect="equal"
)

ax.set_xlabel("x")
ax.set_ylabel("y")

# Overlay the barrier (semi-transparent)
Vmask = (V > 0).astype(float)
barrier_img = ax.imshow(
    Vmask.T,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    alpha=0.25,
    interpolation="nearest",
    aspect="equal"
)

title = ax.set_title("Quantum Double Slit — |psi|^2")


# ============================================================
# 5) ANIMATION LOOP
# ============================================================

frame_counter = 0

def update(frame):
    global psi, frame_counter
    frame_counter += 1

    for _ in range(steps_per_frame):
        psi = step_split_step(psi, U_V, U_K, absorber)

    prob = np.abs(psi)**2
    img.set_data(prob.T)

    title.set_text(f"Quantum Double Slit — |psi|^2   (frame={frame_counter})")
    return [img, barrier_img, title]


ani = FuncAnimation(fig, update, frames=800, interval=25, blit=False)
plt.show()
