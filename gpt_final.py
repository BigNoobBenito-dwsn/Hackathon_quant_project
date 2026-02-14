import math
import numpy as np
import pygame

def sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    m = np.abs(x) > 1e-12
    out[m] = np.sin(x[m]) / x[m]
    return out

def intensity_multislit(y: np.ndarray, L: float, lam: float, a: float, d: float, N: int) -> np.ndarray:
    s = y / max(L, 1e-9)  # sin(theta) ~ y/L
    beta = math.pi * a * s / max(lam, 1e-9)
    alpha = math.pi * d * s / max(lam, 1e-9)

    envelope = sinc(beta) ** 2

    denom = np.sin(alpha)
    numer = np.sin(N * alpha)

    inter = np.ones_like(alpha)
    mask = np.abs(denom) > 1e-12
    inter[mask] = (numer[mask] / denom[mask]) ** 2
    inter[~mask] = float(N**2)

    return np.clip(envelope * inter, 0, None)

def build_sampler(height_px: int, px_to_world: float, L: float, lam: float, a: float, d: float, N: int):
    ys_world = (np.arange(height_px) - height_px / 2) * px_to_world
    I = intensity_multislit(ys_world, L=L, lam=lam, a=a, d=d, N=N)
    if I.max() <= 0:
        I[:] = 1.0
    pdf = I / I.sum()
    cdf = np.cumsum(pdf)
    cdf[-1] = 1.0
    return ys_world, I, pdf, cdf

def sample_y(cdf: np.ndarray, rng: np.random.Generator, n: int):
    r = rng.random(n)
    return np.searchsorted(cdf, r, side="right")

# =========================================================
# Pygame setup (FULLSCREEN)
# =========================================================
pygame.init()
pygame.display.set_caption("Superposition X")

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
W, H = screen.get_size()
clock = pygame.time.Clock()

FONT   = pygame.font.SysFont("consolas", 18)
FONT_S = pygame.font.SysFont("consolas", 15)
TINY   = pygame.font.SysFont("consolas", 13)
BIG    = pygame.font.SysFont("consolas", 52)
MID    = pygame.font.SysFont("consolas", 24)

# Layout
LEFT_W = int(W * 0.33)
CENTER_X = LEFT_W
DETECTOR_X = int(W * 0.82)
PX_TO_WORLD = 0.02

# Colors / theme
PLAY_BG = (22, 22, 28)
PANEL   = (28, 28, 34)
TXT     = (235, 235, 235)
MUTED   = (185, 185, 185)
ACCENT  = (80, 180, 255)
ACCENT2 = (70, 220, 150)
WARN    = (255, 210, 120)
RED     = (190, 70, 70)

rng = np.random.default_rng()

# =========================================================
# CRT overlays (bright)
# =========================================================
def build_scanlines_surface(w, h, line_gap=3, alpha=14):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    y = 0
    while y < h:
        a = alpha + (y % 9 == 0) * 5
        pygame.draw.line(s, (160, 160, 160, a), (0, y), (w, y))
        y += line_gap
    return s

def build_vignette_surface(w, h, strength=55):
    v = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w / 2, h / 2
    maxr = math.hypot(cx, cy)
    for i in range(16):
        t = i / 15
        a = int(strength * (t ** 2))
        r = int(maxr * (0.60 + 0.50 * t))
        pygame.draw.circle(v, (0, 0, 0, a), (int(cx), int(cy)), r, width=0)
    return v

SCANLINES = build_scanlines_surface(W, H, line_gap=3, alpha=14)
VIGNETTE  = build_vignette_surface(W, H, strength=55)

NOISE = pygame.Surface((W, H), pygame.SRCALPHA)
def refresh_noise(surf, density=2200):
    surf.fill((0, 0, 0, 0))
    xs = rng.integers(0, W, size=density)
    ys = rng.integers(0, H, size=density)
    for x, y in zip(xs, ys):
        a = int(rng.integers(2, 8))
        surf.set_at((int(x), int(y)), (220, 220, 220, a))

BRIGHTNESS_BOOST = 10

def draw_crt_vibe():
    screen.blit(SCANLINES, (0, 0))
    screen.blit(NOISE, (0, 0))
    screen.blit(VIGNETTE, (0, 0))
    if BRIGHTNESS_BOOST > 0:
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((BRIGHTNESS_BOOST, BRIGHTNESS_BOOST, BRIGHTNESS_BOOST, 0))
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

# =========================================================
# UI helpers
# =========================================================
class Button:
    def __init__(self, rect, text, onclick, *,
                 bg=(40, 40, 50), hover=(55, 55, 70), fg=TXT, font=MID):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.onclick = onclick
        self.bg = bg
        self.hover = hover
        self.fg = fg
        self.font = font

    def draw(self, surf):
        mx, my = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(mx, my)
        pygame.draw.rect(surf, self.hover if is_hover else self.bg, self.rect, border_radius=12)
        pygame.draw.rect(surf, (95, 95, 120), self.rect, width=2, border_radius=12)
        t = self.font.render(self.text, True, self.fg)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.onclick()

def draw_text(text, x, y, font=FONT, color=TXT):
    screen.blit(font.render(text, True, color), (x, y))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def wrap_text(text, font, max_w):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def clamp_rect(rect, bounds):
    if rect.left < bounds.left:
        rect.left = bounds.left
    if rect.right > bounds.right:
        rect.right = bounds.right
    if rect.top < bounds.top:
        rect.top = bounds.top
    if rect.bottom > bounds.bottom:
        rect.bottom = bounds.bottom
    return rect

# =========================================================
# States
# =========================================================
STATE_MENU    = "menu"
STATE_PLAY    = "play"
STATE_WAVE    = "wave"
STATE_EXHIBIT = "exhibit"
state = STATE_MENU

def set_state(s):
    global state
    state = s

# =========================================================
# QUANTUM MODE (Particles)
# =========================================================
q_defaults = {
    "N_slits": 2,
    "a": 0.35,
    "d": 1.25,
    "lam": 0.20,
    "L": 12.0,
    "shots_per_frame": 300,
}
q = dict(q_defaults)
q["paused"] = False
q["show_help"] = False

counts = np.zeros(H, dtype=np.float64)
hits_surf = pygame.Surface((W, H), pygame.SRCALPHA)

ys_world, I_world, pdf, cdf = build_sampler(H, PX_TO_WORLD, q["L"], q["lam"], q["a"], q["d"], q["N_slits"])
last_q_sig = None

def q_signature():
    return (q["N_slits"], round(q["a"], 4), round(q["d"], 4), round(q["lam"], 4), round(q["L"], 4))

def q_rebuild():
    global ys_world, I_world, pdf, cdf, last_q_sig
    ys_world, I_world, pdf, cdf = build_sampler(H, PX_TO_WORLD, q["L"], q["lam"], q["a"], q["d"], q["N_slits"])
    last_q_sig = q_signature()

def q_reset_pattern():
    counts[:] = 0
    hits_surf.fill((0, 0, 0, 0))

def q_add_hits(n_hits: int):
    ys = sample_y(cdf, rng, n_hits)
    np.add.at(counts, ys, 1.0)
    for y in ys:
        x = DETECTOR_X - int(rng.integers(0, 10))
        pygame.draw.circle(hits_surf, (80, 180, 255, 35), (int(x), int(y)), 2)

def q_apply_keys(keys):
    if keys[pygame.K_1]: q["N_slits"] = 1
    if keys[pygame.K_2]: q["N_slits"] = 2
    if keys[pygame.K_3]: q["N_slits"] = 3
    if keys[pygame.K_4]: q["N_slits"] = 4
    if keys[pygame.K_5]: q["N_slits"] = 5

    if keys[pygame.K_LEFTBRACKET]:  q["a"] -= 0.01
    if keys[pygame.K_RIGHTBRACKET]: q["a"] += 0.01

    if keys[pygame.K_COMMA]:  q["d"] -= 0.02
    if keys[pygame.K_PERIOD]: q["d"] += 0.02

    if keys[pygame.K_MINUS]:  q["lam"] -= 0.005
    if keys[pygame.K_EQUALS]: q["lam"] += 0.005

    if keys[pygame.K_SEMICOLON]: q["L"] -= 0.1
    if keys[pygame.K_QUOTE]:     q["L"] += 0.1

    if keys[pygame.K_DOWN]: q["shots_per_frame"] = max(10, q["shots_per_frame"] - 30)
    if keys[pygame.K_UP]:   q["shots_per_frame"] = min(7000, q["shots_per_frame"] + 30)

    q["a"]   = clamp(q["a"],   0.05, 1.50)
    q["d"]   = clamp(q["d"],   0.10, 6.00)
    q["lam"] = clamp(q["lam"], 0.02, 1.50)
    q["L"]   = clamp(q["L"],   2.00, 60.0)

def q_draw_slits():
    barrier_x = CENTER_X - 40
    pygame.draw.rect(screen, RED, (barrier_x, 0, 20, H))
    centers = (np.arange(q["N_slits"]) - (q["N_slits"] - 1)/2) * q["d"]
    for c in centers:
        y_center_px = int(H/2 + c / PX_TO_WORLD)
        half_w_px = int((q["a"]/2) / PX_TO_WORLD)
        pygame.draw.rect(screen, PLAY_BG, (barrier_x, y_center_px - half_w_px, 20, 2*half_w_px))

def q_draw_detector():
    pygame.draw.aaline(screen, (200, 200, 200), (DETECTOR_X, 0), (DETECTOR_X, H))

    maxc = counts.max()
    if maxc <= 0:
        maxc = 1.0

    plot_x0 = DETECTOR_X + 16
    plot_w = max(120, W - plot_x0 - 20)

    for y in range(H):
        v = counts[y] / maxc
        bar = int(v * plot_w)
        if bar > 0:
            screen.fill(ACCENT2, (plot_x0, y, bar, 1))

    pygame.draw.rect(screen, (210, 210, 210), (plot_x0, 0, plot_w, H), 1)

def q_draw_hud():
    hud = pygame.Surface((W, 72), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 105))
    screen.blit(hud, (0, 0))
    status = "PAUSED" if q["paused"] else "RUNNING"
    line1 = f"N={q['N_slits']}   a={q['a']:.2f}   d={q['d']:.2f}   λ={q['lam']:.3f}   L={q['L']:.1f}   shots/frame={q['shots_per_frame']}   {status}"
    line2 = "H help   M menu   SPACE pause   R reset"
    draw_text("Quantum Mode (Particles)", 18, 10, font=MID, color=TXT)
    draw_text(line1, 18, 38, font=FONT, color=ACCENT)
    draw_text(line2, 18, 56, font=FONT_S, color=MUTED)

def q_draw_help():
    pad = 18
    w = int(W * 0.52)
    h = int(H * 0.62)
    x = int(W * 0.24)
    y = int(H * 0.18)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((20, 20, 26, 240))
    pygame.draw.rect(panel, (110, 110, 130), (0, 0, w, h), width=2, border_radius=16)
    lines = [
        "HELP (H toggles this)",
        "",
        "Quantum controls:",
        "  1..5     number of slits N",
        "  [ / ]    slit width a",
        "  , / .    separation d",
        "  - / =    wavelength λ",
        "  ; / '    distance L",
        "  ↑ / ↓    shots/frame",
        "",
        "General:",
        "  SPACE    pause/run",
        "  R        reset detector accumulation",
        "  M        menu",
        "  ESC      quit",
    ]
    yy = pad
    for i, line in enumerate(lines):
        f = MID if i == 0 else FONT
        c = WARN if i == 0 else TXT
        panel.blit(f.render(line, True, c), (pad, yy))
        yy += 26 if i == 0 else 22
    screen.blit(panel, (x, y))

def draw_quantum():
    screen.fill(PLAY_BG)
    screen.blit(hits_surf, (0, 0))
    q_draw_slits()
    q_draw_detector()
    q_draw_hud()

    pygame.draw.aaline(screen, (55, 55, 70), (LEFT_W, 0), (LEFT_W, H))
    draw_crt_vibe()
    if q["show_help"]:
        q_draw_help()

# =========================================================
# LIGHT MODE (Waves)
# =========================================================
WF_W, WF_H = 420, 240
wave_field_surf = pygame.Surface((WF_W, WF_H))
wave_rgb = np.zeros((WF_W, WF_H, 3), dtype=np.uint8)

XMAX, YMAX = 16.0, 6.0
xf = np.linspace(0.0, XMAX, WF_W, dtype=np.float32)
yf = np.linspace(-YMAX, YMAX, WF_H, dtype=np.float32)
XF, YF = np.meshgrid(xf, yf, indexing="ij")

wave_defaults = {"slit_sep": 2.0, "lam": 0.8, "phase": 0.0}
wave = dict(wave_defaults)
wave["show_amplitude"] = False
wave["paused"] = False
wave["show_help"] = False
t_wave = 0.0

def wave_reset():
    global t_wave
    for k_, v_ in wave_defaults.items():
        wave[k_] = v_
    wave["show_amplitude"] = False
    wave["paused"] = False
    wave["show_help"] = False
    t_wave = 0.0

def wave_key_controls(keys):
    if keys[pygame.K_MINUS]:  wave["lam"] -= 0.01
    if keys[pygame.K_EQUALS]: wave["lam"] += 0.01
    wave["lam"] = clamp(wave["lam"], 0.2, 2.5)

    if keys[pygame.K_COMMA]:  wave["slit_sep"] -= 0.03
    if keys[pygame.K_PERIOD]: wave["slit_sep"] += 0.03
    wave["slit_sep"] = clamp(wave["slit_sep"], 0.5, 4.5)

    if keys[pygame.K_LEFT]:  wave["phase"] -= 0.05
    if keys[pygame.K_RIGHT]: wave["phase"] += 0.05

def colorize_field(val, intensity=True):
    if intensity:
        v = val / (val.max() + 1e-9)
        v = np.clip(v, 0, 1)
        v = np.sqrt(v)
        r = (30 + 170 * v).astype(np.uint8)
        g = (40 + 200 * v).astype(np.uint8)
        b = (60 + 220 * v).astype(np.uint8)
        return r, g, b
    else:
        a = np.clip(val, -1, 1)
        u = (a + 1) * 0.5
        r = (30 + 220 * u).astype(np.uint8)
        g = (30 + 120 * (1 - np.abs(a))).astype(np.uint8)
        b = (30 + 220 * (1 - u)).astype(np.uint8)
        return r, g, b

def render_wave_field(dt_sec):
    global t_wave
    if not wave["paused"]:
        t_wave += dt_sec

    k = 2 * math.pi / max(wave["lam"], 1e-6)
    omega = 2 * math.pi * 1.2

    y1 = +wave["slit_sep"] / 2
    y2 = -wave["slit_sep"] / 2

    r1 = np.sqrt((XF - 0.0)**2 + (YF - y1)**2) + 1e-6
    r2 = np.sqrt((XF - 0.0)**2 + (YF - y2)**2) + 1e-6

    A1 = (1.0 / np.sqrt(r1)) * np.sin(k * r1 - omega * t_wave)
    A2 = (1.0 / np.sqrt(r2)) * np.sin(k * r2 - omega * t_wave + wave["phase"])
    A = A1 + A2

    if wave["show_amplitude"]:
        r, g, b = colorize_field(A, intensity=False)
    else:
        I = A * A
        r, g, b = colorize_field(I, intensity=True)

    wave_rgb[..., 0] = r
    wave_rgb[..., 1] = g
    wave_rgb[..., 2] = b
    pygame.surfarray.blit_array(wave_field_surf, wave_rgb)

def wave_help_overlay():
    pad = 18
    w = int(W * 0.52)
    h = int(H * 0.60)
    x = int(W * 0.24)
    y = int(H * 0.20)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((20, 20, 26, 240))
    pygame.draw.rect(panel, (110, 110, 130), (0, 0, w, h), width=2, border_radius=16)
    lines = [
        "HELP (H toggles this)",
        "",
        "Light Mode controls:",
        "  V        toggle view (Intensity ↔ Amplitude)",
        "  , / .    slit separation",
        "  - / =    wavelength",
        "  ← / →    phase shift φ",
        "",
        "General:",
        "  SPACE    pause/run",
        "  R        reset Light Mode defaults",
        "  M        menu",
        "  ESC      quit",
        "",
        "Math idea:",
        "  A = A1 + A2   and   I ∝ A²",
    ]
    yy = pad
    for i, line in enumerate(lines):
        f = MID if i == 0 else FONT
        c = WARN if i == 0 else TXT
        panel.blit(f.render(line, True, c), (pad, yy))
        yy += 26 if i == 0 else 22
    screen.blit(panel, (x, y))

def draw_wave(dt_sec):
    screen.fill(PLAY_BG)

    box_x = int(W * 0.10)
    box_y = int(H * 0.16)
    box_w = int(W * 0.62)
    box_h = int(H * 0.70)

    pygame.draw.rect(screen, (20, 20, 26), (box_x, box_y, box_w, box_h), border_radius=16)
    pygame.draw.rect(screen, (90, 90, 110), (box_x, box_y, box_w, box_h), width=2, border_radius=16)

    render_wave_field(dt_sec)
    scaled = pygame.transform.smoothscale(wave_field_surf, (box_w - 40, box_h - 40))
    screen.blit(scaled, (box_x + 20, box_y + 20))

    bar_x = box_x + 20
    bar_w = 10
    pygame.draw.rect(screen, RED, (bar_x - bar_w, box_y + 20, bar_w, box_h - 40))

    def y_to_px(y_world):
        u = (y_world + YMAX) / (2 * YMAX)
        return int((box_y + 20) + u * (box_h - 40))

    slit_half = int((box_h - 40) * 0.045)
    y1 = y_to_px(+wave["slit_sep"] / 2)
    y2 = y_to_px(-wave["slit_sep"] / 2)
    pygame.draw.rect(screen, PLAY_BG, (bar_x - bar_w, y1 - slit_half, bar_w, 2 * slit_half))
    pygame.draw.rect(screen, PLAY_BG, (bar_x - bar_w, y2 - slit_half, bar_w, 2 * slit_half))

    hud = pygame.Surface((W, 92), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 105))
    screen.blit(hud, (0, 0))

    mode_txt = "AMPLITUDE" if wave["show_amplitude"] else "INTENSITY"
    status = "PAUSED" if wave["paused"] else "RUNNING"
    line1 = f"sep={wave['slit_sep']:.2f}   λ={wave['lam']:.2f}   φ={wave['phase']:.2f}   view={mode_txt}   {status}"
    line2 = "H help   V view   M menu   SPACE pause   R reset"

    draw_text("Light Mode (Waves)", 18, 10, font=MID, color=TXT)
    draw_text(line1, 18, 42, font=FONT, color=ACCENT)
    draw_text(line2, 18, 64, font=FONT_S, color=MUTED)

    # right caption / instructions (restored)
    cap_x = int(W * 0.76)
    cap_y = int(H * 0.20)
    tips = [
        "Bright = constructive",
        "Dark = destructive",
        "",
        "Try:",
        "• V for amplitude view",
        "• ← → shifts fringes",
        "• - = changes λ",
        "• , . changes spacing",
        "",
        "Goal:",
        "Build intuition for",
        "fringe spacing + phase.",
    ]
    for i, t in enumerate(tips):
        draw_text(t, cap_x, cap_y + i * 20, font=FONT_S, color=TXT)

    draw_crt_vibe()
    if wave["show_help"]:
        wave_help_overlay()

# =========================================================
# Exhibition X (shifted UP + hover text appears under name/title)
# =========================================================
exhibit = {"show_help": False}

EXHIBIT_EVENTS = [
    (1801, "Thomas Young", "Double-slit interference",
     "Young showed that light through two slits forms bright/dark bands. This is the first clean demonstration of interference—waves can add and cancel.",
     "Light Mode: fringe pattern origin."),
    (1865, "J.C. Maxwell", "Light is an EM wave",
     "Maxwell proved light is an electromagnetic wave. That gives a physical model for wave crests/troughs that can interfere.",
     "Light Mode: waves are real fields."),
    (1900, "Max Planck", "Energy comes in quanta",
     "Planck introduced quantization: energy exchanges happen in discrete packets. This breaks classical continuity and starts modern quantum theory.",
     "Quantum Mode: discreteness begins."),
    (1905, "Albert Einstein", "Photons (particle light)",
     "Einstein explained the photoelectric effect by treating light as photons. Light sometimes behaves like individual hits, not a smooth wave.",
     "Quantum Mode: particle detection idea."),
    (1924, "Louis de Broglie", "Matter has wavelength",
     "De Broglie proposed matter waves: particles have λ = h/p. So electrons should diffract and interfere like light does.",
     "Bridge: why particles form fringes."),
    (1927, "Davisson–Germer", "Electron diffraction proven",
     "They observed interference peaks from electrons scattered off crystals. This confirmed that particles truly behave like waves.",
     "Quantum Mode: validates your pattern."),
    (1927, "Werner Heisenberg", "Uncertainty (no exact paths)",
     "Uncertainty limits knowing position and momentum simultaneously. It supports predicting probabilities instead of single exact trajectories.",
     "Quantum Mode: why distributions matter."),
    (1928, "E. Schrödinger", "Wavefunction dynamics",
     "Schrödinger’s equation evolves ψ over time. The predicted screen pattern comes from |ψ|² (probability density).",
     "Core: |ψ|² → intensity."),
    (1961, "Claus Jönsson", "Modern electron double-slit",
     "Jönsson built a clean electron double-slit apparatus with clear fringes. It’s the modern lab version of what we’re simulating here.",
     "Quantum Mode: classic confirmation."),
    (2020, "Modern Era", "Interference in tech",
     "Interference is now a tool: precision sensing, interferometers, holography, and the logic behind quantum computing uses controlled superposition.",
     "Why Superposition X matters."),
]

def exhibit_header_bounds():
    header_top = int(H * 0.08)
    header_bottom = header_top + 115
    margin = 18
    safe = pygame.Rect(margin, header_bottom + margin, W - 2*margin, H - (header_bottom + 2*margin))
    return header_top, safe

def draw_exhibit_help():
    pad = 18
    w = int(W * 0.54)
    h = int(H * 0.46)
    x = int(W * 0.23)
    y = int(H * 0.26)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((20, 20, 26, 240))
    pygame.draw.rect(panel, (110, 110, 130), (0, 0, w, h), width=2, border_radius=16)
    lines = [
        "HELP (H toggles this)",
        "",
        "Exhibition X:",
        "  • Hover a card to expand and reveal what it contributed.",
        "  • 2×5 layout = 10 milestones.",
        "",
        "Keys:",
        "  M   menu",
        "  ESC quit",
    ]
    yy = pad
    for i, line in enumerate(lines):
        f = MID if i == 0 else FONT
        c = WARN if i == 0 else TXT
        panel.blit(f.render(line, True, c), (pad, yy))
        yy += 26 if i == 0 else 22
    screen.blit(panel, (x, y))

class ExhibitCard:
    def __init__(self, rect, year, name, title, story, hint):
        self.base_rect = pygame.Rect(rect)
        self.year = year
        self.name = name
        self.title = title
        self.story = story
        self.hint = hint

    def is_hover(self, mx, my):
        return self.base_rect.collidepoint(mx, my)

    def draw(self, surf, hover=False, *, safe_bounds=None):
        rect = self.base_rect.copy()

        if hover:
            rect = rect.inflate(int(rect.w * 0.10), int(rect.h * 0.10))
            rect.center = self.base_rect.center
            if safe_bounds is not None:
                rect = clamp_rect(rect, safe_bounds)

        bg_col = (36, 36, 46) if not hover else (46, 48, 66)
        border_col = (95, 95, 120) if not hover else ACCENT

        pygame.draw.rect(surf, bg_col, rect, border_radius=16)
        pygame.draw.rect(surf, border_col, rect, 2, border_radius=16)

        bubble_r = 18 if not hover else 20
        bx = rect.x + 22
        by = rect.y + 26
        pygame.draw.circle(surf, (70, 70, 90), (bx, by), bubble_r)
        pygame.draw.circle(surf, (120, 120, 150), (bx, by), bubble_r, 2)
        initials = "".join([p[0] for p in self.name.replace("&", "").split()[:2]]).upper()
        it = FONT_S.render(initials, True, (230, 230, 240))
        surf.blit(it, it.get_rect(center=(bx, by)))

        year_t = MID.render(str(self.year), True, WARN if hover else TXT)
        surf.blit(year_t, (rect.x + 52, rect.y + 12))

        name_t = FONT.render(self.name, True, TXT)
        surf.blit(name_t, (rect.x + 20, rect.y + 46))

        title_t = FONT_S.render(self.title, True, (215, 215, 230) if hover else MUTED)
        surf.blit(title_t, (rect.x + 20, rect.y + 70))

        if hover:
            # ===== KEY CHANGE: story box starts right under header (under name/title) =====
            header_h = 92
            gap = 10
            inner = pygame.Rect(rect.x + 10, rect.y + header_h, rect.w - 20, rect.h - header_h - gap)
            inner.h = max(140, inner.h)

            # subtle shadow so it feels "over" the card
            shadow = inner.copy()
            shadow.x += 4
            shadow.y += 4
            pygame.draw.rect(surf, (0, 0, 0), shadow, border_radius=12)

            pygame.draw.rect(surf, (18, 18, 24), inner, border_radius=12)
            pygame.draw.rect(surf, (75, 75, 100), inner, 1, border_radius=12)

            padding = 10
            max_w = inner.w - 2 * padding
            line_h = 17

            footer_h = 20
            usable_h = inner.h - 2 * padding - footer_h
            max_lines = max(3, usable_h // line_h)

            story_lines = wrap_text(self.story, FONT_S, max_w)

            if len(story_lines) > max_lines:
                story_lines = story_lines[:max_lines]
                last = story_lines[-1]
                ell = "…"
                while FONT_S.size(last + ell)[0] > max_w and len(last) > 0:
                    last = last[:-1]
                story_lines[-1] = (last + ell) if last else ell

            yy = inner.y + padding
            for ln in story_lines:
                surf.blit(FONT_S.render(ln, True, TXT), (inner.x + padding, yy))
                yy += line_h

            hint_lines = wrap_text(self.hint, TINY, max_w)
            if hint_lines:
                surf.blit(TINY.render(hint_lines[0], True, ACCENT2),
                          (inner.x + padding, inner.bottom - padding - 14))

def build_exhibit_cards():
    cols, rows = 5, 2
    _, safe = exhibit_header_bounds()

    gap_x = int(W * 0.015)
    gap_y = int(H * 0.025)

    card_w = (safe.w - (cols - 1) * gap_x) // cols
    card_h = (safe.h - (rows - 1) * gap_y) // rows
    card_h = min(card_h + int(H * 0.015), (safe.h - (rows - 1) * gap_y) // rows)

    grid_w = cols * card_w + (cols - 1) * gap_x
    grid_h = rows * card_h + (rows - 1) * gap_y

    start_x = safe.x + (safe.w - grid_w) // 2
    start_y = safe.y + (safe.h - grid_h) // 2

    cards = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            year, name, title, story, hint = EXHIBIT_EVENTS[idx]
            x = start_x + c * (card_w + gap_x)
            y = start_y + r * (card_h + gap_y)
            cards.append(ExhibitCard((x, y, card_w, card_h), year, name, title, story, hint))
            idx += 1
    return cards

EXHIBIT_CARDS = build_exhibit_cards()

def draw_menu_background():
    screen.fill((28, 30, 38))

    glow_y = int((pygame.time.get_ticks() * 0.05) % H)
    glow = pygame.Surface((W, 140), pygame.SRCALPHA)
    glow.fill((80, 160, 255, 30))
    screen.blit(glow, (0, glow_y - 70))

    for x in range(0, W, 90):
        pygame.draw.line(screen, (45, 48, 60), (x, 0), (x, H))
    for y in range(0, H, 70):
        pygame.draw.line(screen, (45, 48, 60), (0, y), (W, y))

def draw_exhibit():
    draw_menu_background()

    header_top, safe = exhibit_header_bounds()

    title = "Exhibition X"
    sub   = "Ten milestones explaining how interference became real physics"

    tx = W // 2 - BIG.size(title)[0] // 2
    draw_text(title, tx + 3, header_top + 3, font=BIG, color=(10, 40, 70))
    draw_text(title, tx,     header_top,     font=BIG, color=ACCENT)
    draw_text(sub,  W//2 - MID.size(sub)[0]//2,  header_top + 62, font=MID,  color=ACCENT2)

    pygame.draw.line(screen, ACCENT, (W//2 - 420, header_top + 104), (W//2 + 420, header_top + 104), 2)

    mx, my = pygame.mouse.get_pos()
    hovered = None

    for i, card in enumerate(EXHIBIT_CARDS):
        if card.is_hover(mx, my):
            hovered = i
        else:
            card.draw(screen, hover=False)

    if hovered is not None:
        EXHIBIT_CARDS[hovered].draw(screen, hover=True, safe_bounds=safe)

    screen.blit(SCANLINES, (0, 0))
    if BRIGHTNESS_BOOST > 0:
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((BRIGHTNESS_BOOST, BRIGHTNESS_BOOST, BRIGHTNESS_BOOST, 0))
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    if exhibit["show_help"]:
        draw_exhibit_help()

def draw_menu():
    draw_menu_background()

    title = "Superposition X"
    sub = "10th Hackathon Edition"
    hint = "Choose a mode • M returns here anytime • ESC quits"

    title_y = int(H * 0.18)
    tx = W // 2 - BIG.size(title)[0] // 2

    draw_text(title, tx + 3, title_y + 3, font=BIG, color=(10, 40, 70))
    draw_text(title, tx, title_y, font=BIG, color=ACCENT)

    draw_text(sub,  W//2 - MID.size(sub)[0]//2,   title_y + 66, font=MID,  color=ACCENT2)
    draw_text(hint, W//2 - FONT.size(hint)[0]//2, title_y + 100, font=FONT, color=MUTED)
    pygame.draw.line(screen, ACCENT, (W//2 - 290, title_y + 125), (W//2 + 290, title_y + 125), 2)

    panel_w = 520
    panel_h = 360
    panel_x = W//2 - panel_w//2
    panel_y = int(H * 0.44)
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((20, 20, 26, 160))
    screen.blit(panel, (panel_x, panel_y))
    pygame.draw.rect(screen, (95, 95, 120), (panel_x, panel_y, panel_w, panel_h), 2, border_radius=14)

    for b in menu_buttons:
        b.draw(screen)

    screen.blit(SCANLINES, (0, 0))
    if BRIGHTNESS_BOOST > 0:
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((BRIGHTNESS_BOOST, BRIGHTNESS_BOOST, BRIGHTNESS_BOOST, 0))
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

def build_buttons():
    btn_w, btn_h = 420, 64
    gap = 18

    panel_h = 360
    panel_y = int(H * 0.44)

    total_h = 4 * btn_h + 3 * gap
    start_y = panel_y + (panel_h - total_h) // 2
    center_x = W // 2 - btn_w // 2

    return [
        Button((center_x, start_y + 0*(btn_h+gap), btn_w, btn_h),
               "Quantum Mode (Particles)", lambda: set_state(STATE_PLAY)),
        Button((center_x, start_y + 1*(btn_h+gap), btn_w, btn_h),
               "Light Mode (Waves)", lambda: set_state(STATE_WAVE)),
        Button((center_x, start_y + 2*(btn_h+gap), btn_w, btn_h),
               "Exhibition X", lambda: set_state(STATE_EXHIBIT)),
        Button((center_x, start_y + 3*(btn_h+gap), btn_w, btn_h),
               "Quit", lambda: pygame.event.post(pygame.event.Event(pygame.QUIT))),
    ]

menu_buttons = build_buttons()

# =========================================================
# Main loop
# =========================================================
running = True
noise_timer = 0.0

while running:
    dt_sec = clock.tick(60) / 1000.0

    noise_timer += dt_sec
    if noise_timer > 0.12:
        refresh_noise(NOISE, density=int((W * H) / 520))
        noise_timer = 0.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if state == STATE_MENU:
            for b in menu_buttons:
                b.handle(event)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if event.key == pygame.K_m:
                set_state(STATE_MENU)

            if state == STATE_PLAY:
                if event.key == pygame.K_SPACE:
                    q["paused"] = not q["paused"]
                elif event.key == pygame.K_r:
                    q_reset_pattern()
                elif event.key == pygame.K_h:
                    q["show_help"] = not q["show_help"]

            if state == STATE_WAVE:
                if event.key == pygame.K_SPACE:
                    wave["paused"] = not wave["paused"]
                elif event.key == pygame.K_r:
                    wave_reset()
                elif event.key == pygame.K_h:
                    wave["show_help"] = not wave["show_help"]
                elif event.key == pygame.K_v:
                    wave["show_amplitude"] = not wave["show_amplitude"]

            if state == STATE_EXHIBIT:
                if event.key == pygame.K_h:
                    exhibit["show_help"] = not exhibit["show_help"]

    # Update + draw
    if state == STATE_PLAY:
        keys = pygame.key.get_pressed()
        q_apply_keys(keys)

        sig = q_signature()
        if sig != last_q_sig:
            q_rebuild()

        if not q["paused"]:
            q_add_hits(q["shots_per_frame"])

        draw_quantum()

    elif state == STATE_WAVE:
        keys = pygame.key.get_pressed()
        wave_key_controls(keys)
        draw_wave(dt_sec)

    elif state == STATE_EXHIBIT:
        draw_exhibit()

    elif state == STATE_MENU:
        draw_menu()

    pygame.display.flip()

pygame.quit()
