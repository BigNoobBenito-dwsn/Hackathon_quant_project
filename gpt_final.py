# quantum_slits_pygame.py
# Run:
#   pip install pygame numpy
#   python quantum_slits_pygame.py

import math
import numpy as np
import pygame

# ============================
# Physics (Fraunhofer N-slit)
# ============================
def sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    m = np.abs(x) > 1e-12
    out[m] = np.sin(x[m]) / x[m]
    return out

def intensity_multislit(y: np.ndarray, L: float, lam: float, a: float, d: float, N: int) -> np.ndarray:
    s = y / max(L, 1e-9)  # small-angle sin(theta) ~ y/L
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
BIG    = pygame.font.SysFont("consolas", 46)
MID    = pygame.font.SysFont("consolas", 24)

# Layout (scale with fullscreen size)
LEFT_W = int(W * 0.33)
CENTER_X = LEFT_W
DETECTOR_X = int(W * 0.82)

# Visual scaling: world units per pixel in y
PX_TO_WORLD = 0.02

# Theme
BG = (18, 18, 22)
PANEL = (28, 28, 34)
TXT = (235, 235, 235)
MUTED = (185, 185, 185)
ACCENT = (80, 180, 255)
ACCENT2 = (70, 220, 150)
WARN = (255, 210, 120)
RED = (190, 70, 70)

rng = np.random.default_rng()

# ============================
# CRT / Vibe overlays
# ============================
def build_scanlines_surface(w, h, line_gap=3, alpha=28):
    """Gray horizontal scanlines like old TVs."""
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    # slightly varying alpha per line for organic feel
    y = 0
    while y < h:
        a = alpha + (y % 9 == 0) * 8
        pygame.draw.line(s, (140, 140, 140, a), (0, y), (w, y))
        y += line_gap
    return s

def build_vignette_surface(w, h, strength=150):
    """Soft dark edges."""
    v = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w / 2, h / 2
    maxr = math.hypot(cx, cy)
    # radial gradient approximation with circles
    for i in range(18):
        t = i / 17
        a = int(strength * (t ** 2))
        r = int(maxr * (0.55 + 0.55 * t))
        pygame.draw.circle(v, (0, 0, 0, a), (int(cx), int(cy)), r, width=0)
    return v

SCANLINES = build_scanlines_surface(W, H, line_gap=3, alpha=26)
VIGNETTE = build_vignette_surface(W, H, strength=145)

# tiny “noise” sparkle (very light)
NOISE = pygame.Surface((W, H), pygame.SRCALPHA)
def refresh_noise(surf, density=2200):
    surf.fill((0, 0, 0, 0))
    # random faint pixels
    xs = rng.integers(0, W, size=density)
    ys = rng.integers(0, H, size=density)
    for x, y in zip(xs, ys):
        a = int(rng.integers(6, 18))
        surf.set_at((int(x), int(y)), (210, 210, 210, a))

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
        # move "towards" camera
        self.z -= self.speed * 0.01
        if self.z <= 0.08:
            self.__init__()

    def draw(self, surf):
        # project to screen
        px = int(W/2 + (self.x / self.z) * (W * 0.18))
        py = int(H/2 + (self.y / self.z) * (H * 0.18))
        if 0 <= px < W and 0 <= py < H:
            bright = int(120 + 135 * (1 - self.z))
            rad = int(self.size * (1.2 + (1 - self.z)))
            pygame.draw.circle(surf, (bright, bright, bright), (px, py), max(1, rad))

STAR_COUNT = 180
stars = [Star() for _ in range(STAR_COUNT)]

def draw_starfield():
    # subtle gradient base
    screen.fill((10, 10, 16))
    # draw stars
    for s in stars:
        s.step()
        s.draw(screen)
    # slight vignette even on menu
    screen.blit(VIGNETTE, (0, 0))

# ============================
# App state
# ============================
STATE_MENU = "menu"
STATE_SETTINGS = "settings"
STATE_ABOUT = "about"
STATE_PLAY = "play"
state = STATE_MENU

# ============================
# Sim parameters
# ============================
params = {
    "N_slits": 2,
    "a": 0.35,
    "d": 1.25,
    "lam": 0.20,
    "L": 12.0,
    "shots_per_frame": 300,
    "paused": False,
    "reset_on_change": False,
    "show_help": False,   # H toggles help overlay
}

# Accumulation arrays / surfaces
counts = np.zeros(H, dtype=np.float64)
hits_surf = pygame.Surface((W, H), pygame.SRCALPHA)

ys_world, I_world, pdf, cdf = build_sampler(H, PX_TO_WORLD, params["L"], params["lam"], params["a"], params["d"], params["N_slits"])
last_dist_sig = None

def dist_signature():
    return (params["N_slits"], round(params["a"], 4), round(params["d"], 4), round(params["lam"], 4), round(params["L"], 4))

def rebuild_distribution():
    global ys_world, I_world, pdf, cdf, last_dist_sig
    ys_world, I_world, pdf, cdf = build_sampler(
        H, PX_TO_WORLD,
        params["L"], params["lam"], params["a"], params["d"], params["N_slits"]
    )
    last_dist_sig = dist_signature()

def reset_pattern():
    counts[:] = 0
    hits_surf.fill((0, 0, 0, 0))

def add_hits(n_hits: int):
    ys = sample_y(cdf, rng, n_hits)
    np.add.at(counts, ys, 1.0)
    for y in ys:
        x = DETECTOR_X - rng.integers(0, 10)
        pygame.draw.circle(hits_surf, (80, 180, 255, 35), (int(x), int(y)), 2)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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

def draw_text(surf, text, x, y, font=FONT, color=TXT):
    surf.blit(font.render(text, True, color), (x, y))

def draw_multiline(surf, lines, x, y, font=FONT, color=TXT, line_h=22):
    yy = y
    for line in lines:
        draw_text(surf, line, x, yy, font=font, color=color)
        yy += line_h

# ============================
# Quantum screen (clean + CRT vibe)
# ============================
def draw_slits_diagram():
    barrier_x = CENTER_X - 40
    pygame.draw.rect(screen, RED, (barrier_x, 0, 20, H))

    centers = (np.arange(params["N_slits"]) - (params["N_slits"] - 1) / 2) * params["d"]
    for c in centers:
        y_center_px = int(H / 2 + c / PX_TO_WORLD)
        half_w_px = int((params["a"] / 2) / PX_TO_WORLD)
        pygame.draw.rect(screen, BG, (barrier_x, y_center_px - half_w_px, 20, 2 * half_w_px))

def draw_detector_and_hist():
    pygame.draw.line(screen, TXT, (DETECTOR_X, 0), (DETECTOR_X, H), 2)

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

    pygame.draw.rect(screen, (200, 200, 200), (plot_x0, 0, plot_w, H), 1)

def draw_minimal_hud():
    hud_h = 74
    hud = pygame.Surface((W, hud_h), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 120))
    screen.blit(hud, (0, 0))

    status = "PAUSED" if params["paused"] else "RUNNING"
    line1 = f"N={params['N_slits']}   a={params['a']:.2f}   d={params['d']:.2f}   λ={params['lam']:.3f}   L={params['L']:.1f}   shots/frame={params['shots_per_frame']}   {status}"
    line2 = "Keys: H help   M menu   SPACE pause   R reset"

    draw_text(screen, "Quantum Slits", 18, 10, font=MID, color=TXT)
    draw_text(screen, line1, 18, 38, font=FONT, color=ACCENT)
    draw_text(screen, line2, 18, 56, font=FONT_S, color=MUTED)

def draw_help_overlay():
    pad = 18
    w = int(W * 0.52)
    h = int(H * 0.62)
    x = int(W * 0.24)
    y = int(H * 0.18)

    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((20, 20, 26, 240))
    pygame.draw.rect(panel, (110, 110, 130), (0, 0, w, h), width=2, border_radius=16)

    lines = [
        "HELP (toggle with H)",
        "",
        "Simulation controls:",
        "  1..5        number of slits (N)",
        "  [ / ]       slit width a",
        "  , / .       slit separation d",
        "  - / =       wavelength λ",
        "  ; / '       screen distance L",
        "  ↑ / ↓       shots per frame",
        "",
        "General:",
        "  SPACE       pause / run",
        "  R           reset pattern",
        "  M           back to menu",
        "  ESC         quit",
        "",
        "Model:",
        "  I(y) ∝ sinc²(π a y/(λ L)) × (sin(Nα)/sin α)²",
        "  α = π d y/(λ L)   (Fraunhofer / far-field)"
    ]

    yy = pad
    for i, line in enumerate(lines):
        f = MID if i == 0 else FONT
        c = WARN if i == 0 else TXT
        panel.blit(f.render(line, True, c), (pad, yy))
        yy += 26 if i == 0 else 22

    screen.blit(panel, (x, y))

def draw_crt_vibe():
    # scanlines + tiny noise + vignette on top
    screen.blit(SCANLINES, (0, 0))
    screen.blit(NOISE, (0, 0))
    screen.blit(VIGNETTE, (0, 0))

def draw_play_screen():
    # Slightly different base than menu so it feels “lab”
    screen.fill((14, 14, 18))

    # Content
    screen.blit(hits_surf, (0, 0))
    draw_slits_diagram()
    draw_detector_and_hist()
    draw_minimal_hud()

    # separators (subtle)
    pygame.draw.line(screen, (55, 55, 70), (LEFT_W, 0), (LEFT_W, H), 2)
    pygame.draw.line(screen, (55, 55, 70), (DETECTOR_X, 0), (DETECTOR_X, H), 2)

    # CRT overlay (requested vibe)
    draw_crt_vibe()

    if params["show_help"]:
        draw_help_overlay()

# ============================
# Menu / Settings / About
# ============================
def set_state(new_state):
    global state
    state = new_state

def toggle_reset_on_change():
    params["reset_on_change"] = not params["reset_on_change"]

def build_menus():
    btn_w, btn_h = 340, 64
    cx = W // 2 - btn_w // 2
    top = int(H * 0.48)
    gap = 16

    menu_buttons = [
        Button((cx, top + 0*(btn_h+gap), btn_w, btn_h), "Play",     lambda: set_state(STATE_PLAY)),
        Button((cx, top + 1*(btn_h+gap), btn_w, btn_h), "Settings", lambda: set_state(STATE_SETTINGS)),
        Button((cx, top + 2*(btn_h+gap), btn_w, btn_h), "About",    lambda: set_state(STATE_ABOUT)),
        Button((cx, top + 3*(btn_h+gap), btn_w, btn_h), "Quit",     lambda: pygame.event.post(pygame.event.Event(pygame.QUIT))),
    ]

    settings_buttons = [
        Button((60, H-80, 200, 56), "Back", lambda: set_state(STATE_MENU)),
        Button((280, H-80, 260, 56), "Reset Pattern", reset_pattern),
        Button((560, H-80, 320, 56), "Toggle Reset-on-Change", toggle_reset_on_change),
    ]

    about_buttons = [
        Button((60, H-80, 200, 56), "Back", lambda: set_state(STATE_MENU)),
        Button((280, H-80, 260, 56), "Reset Pattern", reset_pattern),
    ]
    return menu_buttons, settings_buttons, about_buttons

STATE_MENU = "menu"
STATE_SETTINGS = "settings"
STATE_ABOUT = "about"
STATE_PLAY = "play"
menu_buttons, settings_buttons, about_buttons = build_menus()

def draw_menu():
    # Starry animated background (requested)
    draw_starfield()

    # subtle scanlines even on menu (lighter)
    screen.blit(SCANLINES, (0, 0))

    # title + glow-ish underline
    title_y = int(H * 0.18)
    title = "Quantum Slits"
    sub = "Build interference patterns shot-by-shot."
    hint = "Click Play • In-game: H = help, M = menu"

    tx = W//2 - BIG.size(title)[0]//2
    # soft glow by drawing 2 offsets
    draw_text(screen, title, tx+2, title_y+2, font=BIG, color=(20, 60, 90))
    draw_text(screen, title, tx,   title_y,   font=BIG, color=ACCENT)

    draw_text(screen, sub,  W//2 - MID.size(sub)[0]//2,  title_y + 64, font=MID,  color=TXT)
    draw_text(screen, hint, W//2 - FONT.size(hint)[0]//2, title_y + 98, font=FONT, color=MUTED)

    # accent line
    pygame.draw.line(screen, ACCENT, (W//2 - 220, title_y + 118), (W//2 + 220, title_y + 118), 2)

    for b in menu_buttons:
        b.draw(screen)

    # gentle vignette
    screen.blit(VIGNETTE, (0, 0))

def draw_settings():
    draw_starfield()
    pygame.draw.rect(screen, PANEL, (60, 120, W-120, H-220), border_radius=18)
    pygame.draw.rect(screen, (90, 90, 110), (60, 120, W-120, H-220), width=2, border_radius=18)
    draw_text(screen, "Settings", 60, 50, font=BIG, color=ACCENT)

    lines = [
        "Adjust values in-game (fast):",
        "  1..5 slits | [ ] width | , . separation | - = wavelength | ; ' distance | ↑↓ shots",
        "",
        f"Current:  N={params['N_slits']}   a={params['a']:.2f}   d={params['d']:.2f}   λ={params['lam']:.3f}   L={params['L']:.1f}",
        f"Shots/frame: {params['shots_per_frame']}",
        f"Reset-on-change: {'ON' if params['reset_on_change'] else 'OFF'}",
        "",
        "Tip: More slits -> sharper peaks. Bigger λ -> wider fringes."
    ]
    draw_multiline(screen, lines, 90, 160, font=FONT, color=TXT, line_h=26)

    for b in settings_buttons:
        b.draw(screen)

    screen.blit(VIGNETTE, (0, 0))

def draw_about():
    draw_starfield()
    pygame.draw.rect(screen, PANEL, (60, 120, W-120, H-220), border_radius=18)
    pygame.draw.rect(screen, (90, 90, 110), (60, 120, W-120, H-220), width=2, border_radius=18)
    draw_text(screen, "About", 60, 50, font=BIG, color=ACCENT)

    lines = [
        "Real-time multi-slit interference (Fraunhofer / far-field).",
        "",
        "Each dot is a detection event sampled from:",
        "  I(y) ∝ sinc²(π a y/(λ L)) × (sin(Nα)/sin α)²,  α = π d y/(λ L)",
        "",
        "Idea upgrades:",
        "  • overlay theoretical curve on histogram",
        "  • add decoherence slider (blend toward single-slit envelope)",
        "  • export screenshot / GIF",
    ]
    draw_multiline(screen, lines, 90, 160, font=FONT, color=TXT, line_h=26)

    for b in about_buttons:
        b.draw(screen)

    screen.blit(VIGNETTE, (0, 0))

# ============================
# In-game key controls
# ============================
def apply_play_keys(keys):
    if keys[pygame.K_1]: params["N_slits"] = 1
    if keys[pygame.K_2]: params["N_slits"] = 2
    if keys[pygame.K_3]: params["N_slits"] = 3
    if keys[pygame.K_4]: params["N_slits"] = 4
    if keys[pygame.K_5]: params["N_slits"] = 5

    if keys[pygame.K_LEFTBRACKET]:  params["a"] -= 0.01
    if keys[pygame.K_RIGHTBRACKET]: params["a"] += 0.01

    if keys[pygame.K_COMMA]:  params["d"] -= 0.02
    if keys[pygame.K_PERIOD]: params["d"] += 0.02

    if keys[pygame.K_MINUS]:  params["lam"] -= 0.005
    if keys[pygame.K_EQUALS]: params["lam"] += 0.005

    if keys[pygame.K_SEMICOLON]: params["L"] -= 0.1
    if keys[pygame.K_QUOTE]:     params["L"] += 0.1

    if keys[pygame.K_DOWN]: params["shots_per_frame"] = max(10, params["shots_per_frame"] - 30)
    if keys[pygame.K_UP]:   params["shots_per_frame"] = min(7000, params["shots_per_frame"] + 30)

    params["a"]   = clamp(params["a"],   0.05, 1.50)
    params["d"]   = clamp(params["d"],   0.10, 6.00)
    params["lam"] = clamp(params["lam"], 0.02, 1.50)
    params["L"]   = clamp(params["L"],   2.00, 60.0)

# ============================
# Main loop
# ============================
running = True
noise_timer = 0.0

while running:
    dt = clock.tick(60) / 1000.0

    # refresh tiny noise ~8 fps (cheap)
    noise_timer += dt
    if noise_timer > 0.12:
        refresh_noise(NOISE, density=int((W * H) / 420))  # scales with resolution
        noise_timer = 0.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if state == STATE_MENU:
            for b in menu_buttons:
                b.handle(event)
        elif state == STATE_SETTINGS:
            for b in settings_buttons:
                b.handle(event)
        elif state == STATE_ABOUT:
            for b in about_buttons:
                b.handle(event)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if state == STATE_PLAY:
                if event.key == pygame.K_SPACE:
                    params["paused"] = not params["paused"]
                if event.key == pygame.K_r:
                    reset_pattern()
                if event.key == pygame.K_h:
                    params["show_help"] = not params["show_help"]
                if event.key == pygame.K_m:
                    set_state(STATE_MENU)
            else:
                if event.key == pygame.K_m:
                    set_state(STATE_MENU)

    # ---- State update/draw ----
    if state == STATE_PLAY:
        keys = pygame.key.get_pressed()
        apply_play_keys(keys)

        sig = dist_signature()
        if sig != last_dist_sig:
            rebuild_distribution()
            if params["reset_on_change"]:
                reset_pattern()

        if not params["paused"]:
            add_hits(params["shots_per_frame"])

        draw_play_screen()

    elif state == STATE_MENU:
        draw_menu()

    elif state == STATE_SETTINGS:
        draw_settings()

    elif state == STATE_ABOUT:
        draw_about()

    pygame.display.flip()

pygame.quit()
