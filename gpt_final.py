# app.py
# Streamlit interactive N-slit quantum simulator (2D split-step Fourier Schrödinger)
# Run:  streamlit run app.py

import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

# ----------------------------
# Core physics helpers
# ----------------------------
def make_grid(Nx, Ny, Lx, Ly):
    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, dx, dy, X, Y

def make_wavepacket(X, Y, dx, dy, x0, y0, sigma, k0):
    psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sigma**2)) * np.exp(1j * k0 * X)
    prob = np.abs(psi)**2
    norm = np.sum(prob) * dx * dy
    return psi / np.sqrt(norm + 1e-30)

def make_barrier(X, Y, Nx, Ny, barrier_x, barrier_width, num_slits, slit_sep, slit_width, V0):
    V = np.zeros((Nx, Ny), dtype=float)
    barrier_mask = (np.abs(X - barrier_x) < barrier_width/2)
    V[barrier_mask] = V0

    centers = (np.arange(num_slits) - (num_slits - 1)/2) * slit_sep
    slits_mask = np.zeros_like(V, dtype=bool)
    for c in centers:
        slits_mask |= (np.abs(Y - c) < slit_width/2)

    V[barrier_mask & slits_mask] = 0.0
    return V

def make_kspace(Nx, Ny, dx, dy):
    kx = 2*np.pi * fftfreq(Nx, d=dx)
    ky = 2*np.pi * fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    return K2

def make_absorber(X, Y, Lx, Ly, strength=2.0, power=10):
    rx = np.abs(X) / (Lx/2)
    ry = np.abs(Y) / (Ly/2)
    r = np.maximum(rx, ry)
    return np.exp(-strength * (r**power))

def x_expectation(psi, X, dx, dy):
    prob = np.abs(psi)**2
    return np.sum(prob * X) * dx * dy

def step_split(psi, U_V, U_K, absorber):
    psi = U_V * psi
    psi_k = fft2(psi)
    psi_k *= U_K
    psi = ifft2(psi_k)
    psi = U_V * psi
    psi *= absorber
    return psi

def render_figure(x, y, psi, V, x_screen, screen_accum, shots, num_slits, cmap_name="viridis"):
    prob = np.abs(psi)**2
    display = np.log(prob + 1e-12)
    display = np.nan_to_num(display, neginf=-30, posinf=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    im = ax1.imshow(
        display.T,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        interpolation="nearest",
        aspect="equal",
        cmap=cmap_name
    )

    barrier_visual = (V > 0).astype(float)
    ax1.imshow(
        barrier_visual.T,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="Reds",
        alpha=0.7,
        interpolation="nearest"
    )

    ax1.axvline(x_screen, color="white", linestyle="--", linewidth=1)
    ax1.set_title(f"log(|ψ|²) | shots={shots} | slits={num_slits}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    if screen_accum.max() > 0:
        ax2.plot(y, screen_accum / screen_accum.max())
        ax2.set_ylim(0, 1)
    else:
        ax2.plot(y, screen_accum)
        ax2.set_ylim(0, 1)

    ax2.set_title("Detector accumulation (normalized)")
    ax2.set_xlabel("y")
    ax2.set_ylabel("Intensity")

    plt.tight_layout()
    return fig

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Quantum Slits Lab", layout="wide")
st.title("Quantum Slits Lab (Interactive)")

with st.sidebar:
    st.header("Controls")

    # Display controls
    cmap_name = st.selectbox("Blob colormap", ["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "coolwarm"], index=0)

    # Physics / grid (keep Nx,Ny moderate; Streamlit loops can be slow)
    Nx = st.selectbox("Nx (grid x)", [128, 192, 256], index=0)
    Ny = st.selectbox("Ny (grid y)", [128, 192, 256], index=0)
    Lx = st.slider("Lx (domain width)", 10.0, 30.0, 20.0, 0.5)
    Ly = st.slider("Ly (domain height)", 10.0, 30.0, 20.0, 0.5)

    dt = st.slider("dt (timestep)", 0.0005, 0.0050, 0.0015, 0.0001)
    steps_per_frame = st.slider("Steps per frame", 1, 12, 4, 1)
    fps = st.slider("FPS (approx)", 5, 60, 25, 1)

    st.divider()
    st.subheader("Wave packet")
    x0 = st.slider("Start x0", -Lx/2 + 1.0, -1.0, -8.0, 0.5)
    y0 = st.slider("Start y0", -Ly/2 + 1.0, Ly/2 - 1.0, 0.0, 0.5)
    sigma = st.slider("Sigma (width)", 0.2, 2.0, 0.6, 0.05)
    k0 = st.slider("k0 (momentum)", 2.0, 25.0, 12.0, 0.5)

    st.divider()
    st.subheader("Barrier / slits")
    num_slits = st.radio("Number of slits", [1, 2, 3, 4, 5], index=1, horizontal=True)
    barrier_x = st.slider("Barrier x", -5.0, 5.0, -1.0, 0.1)
    barrier_width = st.slider("Barrier thickness", 0.1, 2.0, 0.5, 0.1)
    slit_sep = st.slider("Slit separation", 0.5, 5.0, 2.0, 0.1)
    slit_width = st.slider("Slit width", 0.1, 2.0, 0.5, 0.1)
    V0 = st.slider("Barrier height V0", 500.0, 10000.0, 5000.0, 500.0)

    st.divider()
    st.subheader("Detector / absorber")
    x_screen = st.slider("Screen x", -Lx/2 + 1.0, Lx/2 - 1.0, 7.0, 0.5)
    absorber_strength = st.slider("Absorber strength", 0.0, 5.0, 2.0, 0.1)
    absorber_power = st.slider("Absorber power", 2, 20, 10, 1)

    st.divider()
    mode = st.selectbox("Run mode", ["Live animation", "Batch (fast accumulation)"], index=0)
    batch_shots = st.slider("Batch shots", 1, 200, 40, 1)
    batch_steps_per_shot = st.slider("Batch steps/shot", 50, 1500, 450, 25)

# ----------------------------
# Session state (keeps sim running across reruns)
# ----------------------------
def params_signature():
    # Any change here triggers a reset
    return (
        Nx, Ny, Lx, Ly, dt, steps_per_frame,
        x0, y0, sigma, k0,
        num_slits, barrier_x, barrier_width, slit_sep, slit_width, V0,
        x_screen, absorber_strength, absorber_power
    )

def init_sim():
    hbar = 1.0
    m = 1.0

    x, y, dx, dy, X, Y = make_grid(Nx, Ny, Lx, Ly)
    K2 = make_kspace(Nx, Ny, dx, dy)
    V = make_barrier(X, Y, Nx, Ny, barrier_x, barrier_width, num_slits, slit_sep, slit_width, V0)

    U_V = np.exp(-1j * V * dt / (2*hbar))
    U_K = np.exp(-1j * (hbar * K2) * dt / (2*m))
    absorber = make_absorber(X, Y, Lx, Ly, strength=absorber_strength, power=absorber_power)

    psi = make_wavepacket(X, Y, dx, dy, x0, y0, sigma, k0)

    screen = np.zeros(Ny, dtype=float)
    shots = 0
    screen_idx = int(np.argmin(np.abs(x - x_screen)))

    return {
        "hbar": hbar, "m": m,
        "x": x, "y": y, "dx": dx, "dy": dy, "X": X, "Y": Y,
        "K2": K2, "V": V,
        "U_V": U_V, "U_K": U_K, "absorber": absorber,
        "psi": psi,
        "screen": screen, "shots": shots,
        "screen_idx": screen_idx,
        "sig": params_signature(),
        "running": False
    }

if "sim" not in st.session_state:
    st.session_state.sim = init_sim()

# Reset if params changed
if st.session_state.sim["sig"] != params_signature():
    st.session_state.sim = init_sim()

sim = st.session_state.sim

# ----------------------------
# Buttons (Start/Stop/Step/Reset)
# ----------------------------
colA, colB, colC, colD = st.columns([1, 1, 1, 2])
with colA:
    if st.button("▶ Start", use_container_width=True):
        sim["running"] = True
with colB:
    if st.button("⏸ Stop", use_container_width=True):
        sim["running"] = False
with colC:
    if st.button("⏭ Step once", use_container_width=True):
        sim["running"] = False
        # one visual update worth of physics
        for _ in range(steps_per_frame):
            sim["psi"] = step_split(sim["psi"], sim["U_V"], sim["U_K"], sim["absorber"])
        I = np.abs(sim["psi"][sim["screen_idx"], :])**2
        sim["screen"] += I
with colD:
    if st.button("🔄 Reset (clear detector)", use_container_width=True):
        st.session_state.sim = init_sim()
        sim = st.session_state.sim

st.caption(
    r"Detector math:  $I(y) = |\psi(x_{\mathrm{screen}}, y, t)|^2$  (accumulated over many shots)."
)

# ----------------------------
# Main display
# ----------------------------
plot_placeholder = st.empty()
info_col1, info_col2 = st.columns(2)

def draw_once():
    fig = render_figure(
        sim["x"], sim["y"], sim["psi"], sim["V"], x_screen,
        sim["screen"], sim["shots"], num_slits, cmap_name=cmap_name
    )
    plot_placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)

def do_live_tick():
    # evolve physics
    for _ in range(steps_per_frame):
        sim["psi"] = step_split(sim["psi"], sim["U_V"], sim["U_K"], sim["absorber"])

    # add detector intensity
    I = np.abs(sim["psi"][sim["screen_idx"], :])**2
    sim["screen"] += I

    # if wave passed screen, fire a new one (new "shot")
    x_mean = x_expectation(sim["psi"], sim["X"], sim["dx"], sim["dy"])
    if x_mean > x_screen:
        sim["psi"] = make_wavepacket(sim["X"], sim["Y"], sim["dx"], sim["dy"], x0, y0, sigma, k0)
        sim["shots"] += 1

def run_batch():
    # Batch mode: repeatedly "fire" shots, accumulate pattern faster (no live animation loop)
    for _ in range(batch_shots):
        # reset wavepacket for each shot
        sim["psi"] = make_wavepacket(sim["X"], sim["Y"], sim["dx"], sim["dy"], x0, y0, sigma, k0)

        for _ in range(batch_steps_per_shot):
            sim["psi"] = step_split(sim["psi"], sim["U_V"], sim["U_K"], sim["absorber"])

        I = np.abs(sim["psi"][sim["screen_idx"], :])**2
        sim["screen"] += I
        sim["shots"] += 1

# Batch controls
if mode == "Batch (fast accumulation)":
    if st.button("⚡ Run batch now", use_container_width=True):
        sim["running"] = False
        run_batch()

# Draw current frame
draw_once()

with info_col1:
    st.write(
        f"**Shots:** {sim['shots']}  \n"
        f"**Grid:** {Nx}×{Ny}  \n"
        f"**dt:** {dt}  | **steps/frame:** {steps_per_frame}  \n"
        f"**Slits:** {num_slits}  | **sep:** {slit_sep}  | **width:** {slit_width}  \n"
        f"**k0:** {k0}  | **sigma:** {sigma}"
    )

with info_col2:
    st.write(
        "Tips:\n"
        "- More slits → sharper, more frequent peaks (grating).\n"
        "- Smaller slit width → stronger diffraction envelope.\n"
        "- Larger k0 → shorter wavelength → tighter fringes.\n"
        "- If it’s slow: reduce Nx/Ny or use Batch mode."
    )

# ----------------------------
# Live animation loop (Streamlit style)
# ----------------------------
# Streamlit reruns the script; to animate we do: tick -> draw -> sleep -> rerun.
if mode == "Live animation" and sim["running"]:
    do_live_tick()
    draw_once()
    time.sleep(1.0 / max(1, fps))
    st.rerun()
