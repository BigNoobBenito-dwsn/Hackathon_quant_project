import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SciPy FFT (faster than numpy FFT on many machines)
from scipy.fft import fft2, ifft2, fftfreq

# ----------------------------
# 1) PARAMETERS (Control Panel)
# ----------------------------
Nx, Ny = 256, 256
Lx, Ly = 20.0, 20.0
dt = 0.002
steps_per_frame = 10

hbar = 1.0
m = 1.0

# Wave packet parameters
x0, y0 = -6.0, 0.0
sigma = 0.8
k0 = 12.0

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
psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sigma**2)) * np.exp(1j * k0 * X)

# Normalize
prob = np.abs(psi)**2
norm = np.sum(prob) * dx * dy
psi = psi / np.sqrt(norm)

# ----------------------------
# 4) POTENTIAL (for now: zero)
# ----------------------------
V = np.zeros((Nx, Ny), dtype=float)

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
def make_absorber(X, Y, Lx, Ly, strength=4.0, power=10):
    rx = np.abs(X) / (Lx/2)
    ry = np.abs(Y) / (Ly/2)
    r = np.maximum(rx, ry)
    mask = np.exp(-strength * (r**power))
    return mask

absorber = make_absorber(X, Y, Lx, Ly)

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
# 9) VISUALIZATION
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 6))

prob0 = np.abs(psi)**2
img = ax.imshow(
    prob0.T,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    interpolation="nearest",
    aspect="equal",
    vmin=0,
    vmax=prob0.max()
)


ax.set_title("2D Quantum Wave Packet (|psi|^2) — SciPy FFT")
ax.set_xlabel("x")
ax.set_ylabel("y")

# ----------------------------
# 10) ANIMATION UPDATE
# ----------------------------
def update(frame):
    global psi

    for _ in range(steps_per_frame):
        psi = step(psi)

    prob = np.abs(psi)**2
    img.set_data(prob.T)
    ax.set_title(f"2D Quantum Wave Packet (SciPy FFT) | frame={frame}")
    return [img]

ani = FuncAnimation(fig, update, frames=400, interval=30, blit=True)
plt.show()
