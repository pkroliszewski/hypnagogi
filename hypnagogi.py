import math
import random
import sys
import pygame

W, H = 800, 600
FPS = 60

# >>> Wydajność:
SCALE = 5                       # było 3
BG_UPDATE_EVERY = 4             # aktualizuj tło co N klatek
USE_SMOOTHSCALE = False         # smoothscale jest drogi
NOISE_OCTAVES = 5               # było 4

PLASMA_EDGE_SHARPNESS = 4.0
PLASMA_INTENSITY = 2.2
CENTER_DETAIL_POWER = 2.2

ELLIPSE_DRIFT_SPEED = 0.25
ELLIPSE_MORPH_SPEED = 0.00512

SQUARE_COUNT = 80
SQUARE_MIN = 8
SQUARE_MAX = 24
SQUARE_DRIFT = 0.00725


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)

def hash2(ix: int, iy: int) -> float:
    n = (ix * 374761393 + iy * 668265263) ^ (ix * 1274126177)
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & 0xFFFFFFFF) / 4294967296.0

def value_noise(x: float, y: float) -> float:
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy

    v00 = hash2(ix, iy)
    v10 = hash2(ix + 1, iy)
    v01 = hash2(ix, iy + 1)
    v11 = hash2(ix + 1, iy + 1)

    sx = smoothstep(fx)
    sy = smoothstep(fy)

    a = v00 + (v10 - v00) * sx
    b = v01 + (v11 - v01) * sx
    return a + (b - a) * sy

def fbm(x: float, y: float, octaves=3) -> float:
    amp = 0.5
    freq = 1.0
    total = 0.0
    for _ in range(octaves):
        total += amp * (value_noise(x * freq, y * freq) * 2 - 1)
        freq *= 2.0
        amp *= 0.5
    return total

def ellipse_level(x: float, y: float, cx: float, cy: float, a: float, b: float) -> float:
    nx = (x - cx) / a
    ny = (y - cy) / b
    return (nx * nx + ny * ny) - 1.0

def ellipse_point(cx: float, cy: float, a: float, b: float, ang: float):
    return (cx + a * math.cos(ang), cy + b * math.sin(ang))

class Square:
    def __init__(self):
        self.ang = random.random() * math.tau
        self.ang_vel = (random.random() * 2 - 1) * SQUARE_DRIFT
        self.base = random.uniform(SQUARE_MIN, SQUARE_MAX)
        self.phase = random.random() * math.tau
        self.phase_vel = random.uniform(0.008, 0.02)
        self.life = random.random()
        self.life_vel = random.uniform(0.002, 0.008)
        self.size_wobble = random.uniform(0.15, 0.45)

    def update(self):
        self.ang += self.ang_vel
        self.phase += self.phase_vel
        self.life += self.life_vel
        if self.life > 1.0:
            self.life = 0.0
            self.ang = random.random() * math.tau
            self.ang_vel = (random.random() * 2 - 1) * SQUARE_DRIFT
            self.base = random.uniform(SQUARE_MIN, SQUARE_MAX)
            self.phase = random.random() * math.tau

    def alpha(self) -> float:
        return math.sin(math.pi * self.life)

    def size(self) -> float:
        return self.base * (1.0 + self.size_wobble * math.sin(self.phase))

def plasma_color(intensity: float, tint_shift: float = 0.0):
    intensity = max(0.0, min(1.2, intensity))
    b = int(40 + 200 * intensity)
    g = int(10 + 140 * intensity)
    r = int(0 + 35 * intensity)
    g = max(0, min(255, int(g + 40 * tint_shift)))
    b = max(0, min(255, int(b + 20 * tint_shift)))
    return (r, g, b)

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Hypnagogi — płynniejsza wersja")
    clock = pygame.time.Clock()

    bg_w = W // SCALE
    bg_h = H // SCALE
    bg = pygame.Surface((bg_w, bg_h))

    cx, cy = W * 0.5, H * 0.52
    base_a, base_b = W * 0.23, H * 0.18
    drift_tx = random.random() * 999
    drift_ty = random.random() * 999
    morph_t = random.random() * 999
    t = 0.0

    squares = [Square() for _ in range(SQUARE_COUNT)]

    frame = 0
    cached_bg_scaled = None

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        t += dt
        frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        drift_tx += 0.22
        drift_ty += 0.19
        dx = (fbm(drift_tx * 0.01, 0.0, 3)) * 2.0
        dy = (fbm(0.0, drift_ty * 0.01, 3)) * 2.0

        cx += dx * (ELLIPSE_DRIFT_SPEED * 60 * dt)
        cy += dy * (ELLIPSE_DRIFT_SPEED * 60 * dt)

        margin = 80
        cx = max(margin + base_a, min(W - margin - base_a, cx))
        cy = max(margin + base_b, min(H - margin - base_b, cy))

        morph_t += ELLIPSE_MORPH_SPEED * 60 * dt
        m = 0.5 + 0.5 * math.sin(morph_t)
        a = base_a * (1.75 + 0.55 * m)
        b = base_b * (1.75 + 0.55 * (1.0 - m))

        # --- Aktualizuj tło rzadziej ---
        if frame % BG_UPDATE_EVERY == 0 or cached_bg_scaled is None:
            for y in range(bg_h):
                py = (y + 0.5) * SCALE
                ry = (py - H * 0.5) / (H * 0.5)
                for x in range(bg_w):
                    px = (x + 0.5) * SCALE
                    rx = (px - W * 0.5) / (W * 0.5)
                    r = math.sqrt(rx * rx + ry * ry)
                    center_factor = max(0.0, 1.0 - r)
                    center_factor = center_factor ** CENTER_DETAIL_POWER

                    lvl = ellipse_level(px, py, cx, cy, a, b)
                    edge = math.exp(-PLASMA_EDGE_SHARPNESS * abs(lvl))

                    #n = fbm(px * 0.02 + t * 0.9, py * 0.02 - t * 0.7, NOISE_OCTAVES)
                    tx = 0.8 * math.sin(t*0.2)
                    ty = 0.8 * math.cos(t*0.3)
                    n = fbm(px * 0.008+tx, py * 0.008 + ty, NOISE_OCTAVES)

                    shimmer = -0.55 + 2.45 * math.sin((n *  6.0 + t * 0.5))

                    intensity = PLASMA_INTENSITY * edge * shimmer * center_factor
                    tint = 0.45 * math.sin(t * 1.1 + n * 3.0)
                    bg.set_at((x, y), plasma_color(intensity * tint, tint))

            if USE_SMOOTHSCALE:
                cached_bg_scaled = pygame.transform.smoothscale(bg, (W, H))
            else:
                cached_bg_scaled = pygame.transform.scale(bg, (W, H))

        screen.blit(cached_bg_scaled, (0, 0))

        # czarna elipsa
        #pygame.draw.ellipse(screen, (0, 0, 0), pygame.Rect(cx - a, cy - b, 2 * a, 2 * b))

        # kwadraty - bez tworzenia surfów per-kwadrat (taniej)
#        for s in squares:
#            s.update()
#            alpha = s.alpha()
#            if alpha < 0.03:
#                continue
#
#            ang = s.ang
#            px, py = ellipse_point(cx, cy, a, b, ang)
#
#            size = s.size()
#            half = size * 0.5

            # zamiast alfa-surface: prosty trick — kilka cienkich obrysów
            # (wizualnie przypomina "mignięcia", a jest szybkie)
#            rect = pygame.Rect(px - half, py - half, size, size)
            #pygame.draw.rect(screen, (0, 0, 0), rect)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
