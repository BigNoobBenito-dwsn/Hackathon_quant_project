import math
import numpy as np
import pygame

# ============================
# Physics (Quantum: Fraunhofer N-slit sampling)
# ============================
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


# ============================
# Pygame setup (FULLSCREEN)
# ============================
pygame.init()
pygame.display.set_caption("Quantum Slits")

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
W, H = screen.get_size()
clock = pygame.time.Clock()

FONT   = pygame.font.SysFont("consolas", 18)
FONT_S = pygame.font.SysFont("consolas", 15)
BIG    = pygame.font.SysFont("consolas", 48)
MID    = pygame.font.SysFont("consolas", 24)

LEFT_W = int(W * 0.33)
CENTER_X = LEFT_W
DETECTOR_X = int(W * 0.82)

PX_TO_WORLD = 0.02

# Theme
BG = (18, 18, 22)
PLAY_BG = (22, 22, 28)
PANEL = (28, 28, 34)
TXT = (235, 235, 235)
MUTED = (185, 185, 185)
ACCENT = (80, 180, 255)
ACCENT2 = (70, 220, 150)
WARN = (255, 210, 120)
RED = (190, 70, 70)

rng = np.random.default_rng()

# ============================
# CRT / vibe overlays (bright)
# ============================
def build_scanlines_surface(w, h, line_gap=3, alpha=14):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    y = 0
    while y < h:
        a = alpha + (y % 9 == 0) * 5
        pygame.draw.line(s, (160, 160, 160, a), (0, y), (w, y))
        y += line_gap
    return s

def build_vignette_surface(w, h, strength=70):
    v = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w / 2, h / 2
    maxr = math.hypot(cx, cy)
    for i in range(18):
        t = i / 17
        a = int(strength * (t ** 2))
        r = int(maxr * (0.55 + 0.55 * t))
        pygame.draw.circle(v, (0, 0, 0, a), (int(cx), int(cy)), r, width=0)
    return v

SCANLINES = build_scanlines_surface(W, H, line_gap=3, alpha=14)
VIGNETTE = build_vignette_surface(W, H, strength=70)

NOISE = pygame.Surface((W, H), pygame.SRCALPHA)
def refresh_noise(surf, density=2200):
    surf.fill((0, 0, 0, 0))
    xs = rng.integers(0, W, size=density)
    ys = rng.integers(0, H, size=density)
    for x, y in zip(xs, ys):
        a = int(rng.integers(3, 10))
        surf.set_at((int(x), int(y)), (220, 220, 220, a))

BRIGHTNESS_BOOST = 12  # 0..30 recommended

def draw_crt_vibe():
    screen.blit(SCANLINES, (0, 0))
    screen.blit(NOISE, (0, 0))
    screen.blit(VIGNETTE, (0, 0))
    if BRIGHTNESS_BOOST > 0:
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((BRIGHTNESS_BOOST, BRIGHTNESS_BOOST, BRIGHTNESS_BOOST, 0))
        screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

# ============================
# Starfield (Menu background)
# ============================
class Star:
    __slots__ = ("x", "y", "z", "speed", "size")
    def __init__(self):
        self.x = rng.uniform(-1, 1)
        self.y = rng.uniform(-1, 1)
        self.z = rng.uniform(0.2, 1.0)
        self.speed = rng.uniform(0.18, 0.60)
        self.size = rng.uniform(0.8, 2.2)

    def step(self):
        self.z -= self.speed * 0.01
        if self.z <= 0.08:
            self.__init__()

    def draw(self, surf):
        px = int(W/2 + (self.x / self.z) * (W * 0.18))
        py = int(H/2 + (self.y / self.z) * (H * 0.18))
        if 0 <= px < W and 0 <= py < H:
            bright = int(140 + 110 * (1 - self.z))
            rad = int(self.size * (1.2 + (1 - self.z)))
            pygame.draw.circle(surf, (bright, bright, bright), (px, py), max(1, rad))

stars = [Star() for _ in range(180)]

def draw_starfield_base():
    screen.fill((10, 10, 16))
    for s in stars:
        s.step()
        s.draw(screen)
    screen.blit(VIGNETTE, (0, 0))

# ============================
# UI helpers
# ============================
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
        pygame.draw.rect(surf, (90, 90, 110), self.rect, width=2, border_radius=12)
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

# ============================
# App state
# ============================
STATE_MENU = "menu"
STATE_PLAY = "play"   # quantum dots
STATE_WAVE = "wave"   # classical wave field
STATE_ABOUT = "about"
state = STATE_MENU

def set_state(new_state):
    global state
    state = new_state

# ============================
# Quantum Mode state
# ============================
q_defaults = {"N_slits": 2, "a": 0.35, "d": 1.25, "lam": 0.20, "L": 12.0, "shots_per_frame": 300}
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
        x = DETECTOR_X - rng.integers(0, 10)
        pygame.draw.circle(hits_surf, (80, 180, 255, 35), (int(x), int(y)), 2)

def q_apply_keys(keys):
    if keys[pygame.K_1]: q["N_slits"] = 1
    if keys[pygame.K_2]: q["N_slits"] = 2
    if keys[pygame.K_3]: q["N_slits"] = 3
    if keys[pygame.K_4]: q["N_slits"] = 4
    if keys[pygame.K_5]: q["N_slits"] = 5

    if keys[pygame.K_LEFTBRACKET]:  q["a"] -= 0.01
    if keys[pygame.K_RIGHTBRACKET]: q["a"] += 0.01
    if keys[pygame.K_COMMA]:        q["d"] -= 0.02
    if keys[pygame.K_PERIOD]:       q["d"] += 0.02

    if keys[pygame.K_MINUS]:        q["lam"] -= 0.005
    if keys[pygame.K_EQUALS]:       q["lam"] += 0.005

    if keys[pygame.K_SEMICOLON]:    q["L"] -= 0.1
    if keys[pygame.K_QUOTE]:        q["L"] += 0.1

    if keys[pygame.K_DOWN]: q["shots_per_frame"] = max(10, q["shots_per_frame"] - 30)
    if keys[pygame.K_UP]:   q["shots_per_frame"] = min(7000, q["shots_per_frame"] + 30)

    q["a"]   = clamp(q["a"],   0.05, 1.50)
    q["d"]   = clamp(q["d"],   0.10, 6.00)
    q["lam"] = clamp(q["lam"], 0.02, 1.50)
    q["L"]   = clamp(q["L"],   2.00, 60.0)

# Quantum draw
def q_draw_slits():
    barrier_x = CENTER_X - 40
    pygame.draw.rect(screen, RED, (barrier_x, 0, 20, H))
    centers = (np.arange(q["N_slits"]) - (q["N_slits"] - 1) / 2) * q["d"]
    for c in centers:
        y_center_px = int(H / 2 + c / PX_TO_WORLD)
        half_w_px = int((q["a"] / 2) / PX_TO_WORLD)
        pygame.draw.rect(screen, PLAY_BG, (barrier_x, y_center_px - half_w_px, 20, 2 * half_w_px))

def q_draw_detector_hist():
    pygame.draw.line(screen, TXT, (DETECTOR_X, 0), (DETECTOR_X, H), 2)
    maxc = counts.max()
    if maxc <= 0: maxc = 1.0
    plot_x0 = DETECTOR_X + 16
    plot_w = max(120, W - plot_x0 - 20)
    for y in range(H):
        v = counts[y] / maxc
        bar = int(v * plot_w)
        if bar > 0:
            screen.fill(ACCENT2, (plot_x0, y, bar, 1))
    pygame.draw.rect(screen, (210, 210, 210), (plot_x0, 0, plot_w, H), 1)

def q_draw_hud():
    hud_h = 72
    hud = pygame.Surface((W, hud_h), pygame.SRCALPHA)
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

def draw_quantum_play():
    screen.fill(PLAY_BG)
    screen.blit(hits_surf, (0, 0))
    q_draw_slits()
    q_draw_detector_hist()
    q_draw_hud()

    pygame.draw.line(screen, (55, 55, 70), (LEFT_W, 0), (LEFT_W, H), 2)
    pygame.draw.line(screen, (55, 55, 70), (DETECTOR_X, 0), (DETECTOR_X, H), 2)

    draw_crt_vibe()
    if q["show_help"]:
        q_draw_help()

# ============================
# Wave Lab state (NEW)
# ============================
WF_W, WF_H = 420, 240
wave_field_surf = pygame.Surface((WF_W, WF_H))
wave_rgb = np.zeros((WF_W, WF_H, 3), dtype=np.uint8)

XMAX, YMAX = 16.0, 6.0
xf = np.linspace(0.0, XMAX, WF_W, dtype=np.float32)
yf = np.linspace(-YMAX, YMAX, WF_H, dtype=np.float32)
XF, YF = np.meshgrid(xf, yf, indexing="ij")

wave_defaults = {"slit_sep": 2.0, "lam": 0.8, "phase": 0.0}
wave = dict(wave_defaults)
wave["show_amplitude"] = False  # False=intensity, True=amplitude
wave["paused"] = False
wave["show_help"] = False
t_wave = 0.0

def wave_reset():
    global t_wave
    for k, v in wave_defaults.items():
        wave[k] = v
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

def colorize_field(val, mode_intensity=True):
    if mode_intensity:
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
    omega = 2 * math.pi * 1.2  # visual animation speed

    y1 = +wave["slit_sep"] / 2
    y2 = -wave["slit_sep"] / 2

    r1 = np.sqrt((XF - 0.0) ** 2 + (YF - y1) ** 2) + 1e-6
    r2 = np.sqrt((XF - 0.0) ** 2 + (YF - y2) ** 2) + 1e-6

    A1 = (1.0 / np.sqrt(r1)) * np.sin(k * r1 - omega * t_wave)
    A2 = (1.0 / np.sqrt(r2)) * np.sin(k * r2 - omega * t_wave + wave["phase"])

    A = A1 + A2

    if wave["show_amplitude"]:
        r, g, b = colorize_field(A, mode_intensity=False)
    else:
        I = A * A
        r, g, b = colorize_field(I, mode_intensity=True)

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
        "Wave Lab controls:",
        "  V        toggle view (Intensity ↔ Amplitude)",
        "  , / .    slit separation",
        "  - / =    wavelength",
        "  ← / →    phase shift φ (moves fringes)",
        "",
        "General:",
        "  SPACE    pause/run",
        "  R        reset Wave Lab to defaults",
        "  M        menu",
        "  ESC      quit",
        "",
        "Math idea:",
        "  A = A1 + A2    and    I ∝ A²",
    ]

    yy = pad
    for i, line in enumerate(lines):
        f = MID if i == 0 else FONT
        c = WARN if i == 0 else TXT
        panel.blit(f.render(line, True, c), (pad, yy))
        yy += 26 if i == 0 else 22

    screen.blit(panel, (x, y))

def draw_wave_lab(dt_sec):
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

    # barrier + slits (visual)
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

    # HUD
    hud_h = 92
    hud = pygame.Surface((W, hud_h), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 105))
    screen.blit(hud, (0, 0))

    mode_txt = "AMPLITUDE" if wave["show_amplitude"] else "INTENSITY"
    status = "PAUSED" if wave["paused"] else "RUNNING"
    line1 = f"sep={wave['slit_sep']:.2f}   λ={wave['lam']:.2f}   phase φ={wave['phase']:.2f}   view={mode_txt}   {status}"
    line2 = "H help   V view   M menu   SPACE pause   R reset"

    draw_text("Wave Lab (Classical Interference)", 18, 10, font=MID, color=TXT)
    draw_text(line1, 18, 42, font=FONT, color=ACCENT)
    draw_text(line2, 18, 64, font=FONT_S, color=MUTED)

    # right caption
    cap_x = int(W * 0.76)
    cap_y = int(H * 0.20)
    lines = [
        "Bright = constructive",
        "Dark = destructive",
        "",
        "Try:",
        "• V for amplitude view",
        "• ← → to shift fringes",
        "• - = to change λ",
        "• , . to change spacing",
    ]
    for i, t in enumerate(lines):
        draw_text(t, cap_x, cap_y + i * 20, font=FONT_S, color=TXT)

    draw_crt_vibe()
    if wave["show_help"]:
        wave_help_overlay()

# ============================
# Menu / About screens
# ============================
def build_buttons():
    btn_w, btn_h = 440, 66
    cx = W // 2 - btn_w // 2
    top = int(H * 0.50)
    gap = 16
    menu = [
        Button((cx, top + 0*(btn_h+gap), btn_w, btn_h), "Quantum Mode (Particles)", lambda: set_state(STATE_PLAY)),
        Button((cx, top + 1*(btn_h+gap), btn_w, btn_h), "Light Mode (Waves)", lambda: set_state(STATE_WAVE)),
        Button((cx, top + 2*(btn_h+gap), btn_w, btn_h), "About", lambda: set_state(STATE_ABOUT)),
        Button((cx, top + 3*(btn_h+gap), btn_w, btn_h), "Quit", lambda: pygame.event.post(pygame.event.Event(pygame.QUIT))),
    ]
    about = [Button((60, H-84, 200, 58), "Back", lambda: set_state(STATE_MENU))]
    return menu, about

menu_buttons, about_buttons = build_buttons()

def draw_menu():
    draw_starfield_base()
    screen.blit(SCANLINES, (0, 0))

    title = "Quantum Slits"
    sub = "Two ways to see interference: particles and waves."
    hint = "Pick a mode • In-game: M = menu"

    title_y = int(H * 0.18)
    tx = W//2 - BIG.size(title)[0]//2

    draw_text(title, tx+3, title_y+3, font=BIG, color=(10, 40, 70))
    draw_text(title, tx,   title_y,   font=BIG, color=ACCENT)

    draw_text(sub,  W//2 - MID.size(sub)[0]//2,   title_y + 64, font=MID,  color=TXT)
    draw_text(hint, W//2 - FONT.size(hint)[0]//2, title_y + 96, font=FONT, color=MUTED)
    pygame.draw.line(screen, ACCENT, (W//2 - 280, title_y + 118), (W//2 + 280, title_y + 118), 2)

    for b in menu_buttons:
        b.draw(screen)

    screen.blit(VIGNETTE, (0, 0))

def draw_about():
    draw_starfield_base()
    pygame.draw.rect(screen, PANEL, (60, 120, W-120, H-220), border_radius=18)
    pygame.draw.rect(screen, (90, 90, 110), (60, 120, W-120, H-220), width=2, border_radius=18)
    draw_text("About", 60, 50, font=BIG, color=ACCENT)

    lines = [
        "Quantum Mode:",
        "• Dots accumulate into an N-slit interference pattern.",
        "",
        "Wave Lab:",
        "• Two coherent waves emerge from the slit openings.",
        "• Amplitudes add, intensity is I ∝ A².",
        "",
        "Universal:",
        "• H help (in-mode), M menu, ESC quit",
    ]
    y0 = 160
    for i, t in enumerate(lines):
        draw_text(t, 90, y0 + i * 26, font=FONT, color=TXT)

    for b in about_buttons:
        b.draw(screen)

    screen.blit(VIGNETTE, (0, 0))

# ============================
# Main loop
# ============================
running = True
noise_timer = 0.0

while running:
    dt_sec = clock.tick(60) / 1000.0

    noise_timer += dt_sec
    if noise_timer > 0.12:
        refresh_noise(NOISE, density=int((W * H) / 420))
        noise_timer = 0.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if state == STATE_MENU:
            for b in menu_buttons:
                b.handle(event)
        elif state == STATE_ABOUT:
            for b in about_buttons:
                b.handle(event)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            # universal menu
            if event.key == pygame.K_m:
                set_state(STATE_MENU)

            if state == STATE_PLAY:
                if event.key == pygame.K_SPACE:
                    q["paused"] = not q["paused"]
                if event.key == pygame.K_r:
                    q_reset_pattern()
                if event.key == pygame.K_h:
                    q["show_help"] = not q["show_help"]

            if state == STATE_WAVE:
                if event.key == pygame.K_SPACE:
                    wave["paused"] = not wave["paused"]
                if event.key == pygame.K_r:
                    wave_reset()
                if event.key == pygame.K_h:
                    wave["show_help"] = not wave["show_help"]
                if event.key == pygame.K_v:
                    wave["show_amplitude"] = not wave["show_amplitude"]

    # ---- Update + Draw ----
    if state == STATE_PLAY:
        keys = pygame.key.get_pressed()
        q_apply_keys(keys)

        sig = q_signature()
        if sig != last_q_sig:
            q_rebuild()

        if not q["paused"]:
            q_add_hits(q["shots_per_frame"])

        draw_quantum_play()

    elif state == STATE_WAVE:
        keys = pygame.key.get_pressed()
        wave_key_controls(keys)
        draw_wave_lab(dt_sec)

    elif state == STATE_MENU:
        draw_menu()

    elif state == STATE_ABOUT:
        draw_about()

    pygame.display.flip()

pygame.quit()
