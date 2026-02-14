import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, ifft2, fftfreq

# ----------------------------
# 1) PARAMETERS
# ----------------------------
Nx, Ny = 256, 256
Lx, Ly = 20.0, 20.0

dt = 0.0015
steps_per_frame = 4

hbar = 1.0
m = 1.0

# Wave packet parameters
x0, y0 = -8.0, 0.0
sigma = 0.6
k0 = 12.0

# Slit parameters
barrier_x = -1.0        # x-position of barrier wall
barrier_width = 0.5   # thickness of barrier
slit_sep = 2.0          # separation between slit centers
slit_width = 0.5        # width of each slit
V0 = 5000.0             # barrier potential

# Screen (detector)
x_screen = 7.0

# Absorbing boundary
absorber_strength = 2.0
absorber_power = 10

# ----------------------------
# 2) GRID SETUP
# ----------------------------
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = np.meshgrid(x, y, indexing="ij")

# ----------------------------
# 3) INITIAL WAVEFUNCTION
# ----------------------------
def make_wavepacket():
    psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sigma**2)) * np.exp(1j * k0 * X)

    # Normalize
    prob = np.abs(psi)**2
    norm = np.sum(prob) * dx * dy
    psi = psi / np.sqrt(norm)
    return psi

psi = make_wavepacket()
psi0 = psi.copy()

# ----------------------------
# 4) DOUBLE SLIT POTENTIAL
# ----------------------------
V = np.zeros((Nx, Ny), dtype=float)

# Barrier region in x
barrier_mask = (np.abs(X - barrier_x) < barrier_width/2)

# Define slits (openings)
slit1_center = +slit_sep/2
slit2_center = -slit_sep/2

slit1 = (np.abs(Y - slit1_center) < slit_width/2)
slit2 = (np.abs(Y - slit2_center) < slit_width/2)

# Barrier everywhere except where slits are
V[barrier_mask] = V0
V[barrier_mask & (slit1 | slit2)] = 0.0

# ----------------------------
# 5) K-SPACE SETUP
# ----------------------------
kx = 2*np.pi * fftfreq(Nx, d=dx)
ky = 2*np.pi * fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2

# ----------------------------
# 6) PRECOMPUTE OPERATORS
# ----------------------------
U_V = np.exp(-1j * V * dt / (2*hbar))
U_K = np.exp(-1j * (hbar * K2) * dt / (2*m))

# ----------------------------
# 7) ABSORBING BOUNDARY MASK
# ----------------------------
def make_absorber(X, Y, Lx, Ly, strength=2.0, power=10):
    rx = np.abs(X) / (Lx/2)
    ry = np.abs(Y) / (Ly/2)
    r = np.maximum(rx, ry)
    return np.exp(-strength * (r**power))

absorber = make_absorber(X, Y, Lx, Ly, strength=absorber_strength, power=absorber_power)

# ----------------------------
# 8) TIME STEP FUNCTION
# ----------------------------
def step(psi):
    psi = U_V * psi
    psi_k = fft2(psi)
    psi_k *= U_K
    psi = ifft2(psi_k)
    psi = U_V * psi
    psi *= absorber
    return psi

# ----------------------------
# 9) EXPECTATION VALUE <x>
# ----------------------------
def x_expectation(psi):
    prob = np.abs(psi)**2
    return np.sum(prob * X) * dx * dy

# ----------------------------
# 10) SCREEN DETECTOR ACCUMULATION
# ----------------------------
screen = np.zeros(Ny)
shots = 0

# find x index for screen
screen_idx = np.argmin(np.abs(x - x_screen))

# ----------------------------
# 11) VISUALIZATION SETUP
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# left panel: wave
prob0 = np.abs(psi)**2
display0 = np.log(prob0 + 1e-12)
display0 = np.nan_to_num(display0, neginf=-30, posinf=0)

img = ax1.imshow(
    display0.T,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    interpolation="nearest",
    aspect="equal"
)
barrier_visual = (V > 0).astype(float)

ax1.imshow(
    barrier_visual.T,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap="Reds",       # white wall, black slits
    alpha=0.8,           # visibility strength
    interpolation="nearest"
)

ax1.set_title("Wave (log(|psi|^2))")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# draw screen line
ax1.axvline(x_screen, color="white", linestyle="--", linewidth=1)

# right panel: accumulated pattern
line, = ax2.plot(y, screen)
ax2.set_xlim(y.min(), y.max())
ax2.set_ylim(0, 1)
ax2.set_title("Detector Screen Accumulation")
ax2.set_xlabel("y")
ax2.set_ylabel("Intensity (normalized)")

# ----------------------------
# 12) ANIMATION UPDATE
# ----------------------------
def update(frame):
    global psi, shots, screen

    # evolve wave
    for _ in range(steps_per_frame):
        psi = step(psi)

    # detector screen: add intensity at x_screen
    I = np.abs(psi[screen_idx, :])**2
    screen += I

    # restart wavepacket when it passes screen
    x_mean = x_expectation(psi)
    if x_mean > x_screen:
        psi = make_wavepacket()
        shots += 1

    # display wave (log scale)
    prob = np.abs(psi)**2
    display = np.log(prob + 1e-12)
    display = np.nan_to_num(display, neginf=-30, posinf=0)
    img.set_data(display.T)

    # normalize detector plot
    if screen.max() > 0:
        line.set_ydata(screen / screen.max())

    ax1.set_title(f"Wave (log(|psi|^2)) | shots = {shots}")

    return img, line

# IMPORTANT: blit=False for Windows stability
ani = FuncAnimation(fig, update, frames=2000, interval=30, blit=False)

plt.tight_layout()
plt.show()
