import math
import random
import sys
import pygame
import os
from pathlib import Path
import numpy as np

W, H = 1024, 1024
FPS = 60

# >>> Wydajność:
SCALE = 3                       # było 3
BG_UPDATE_EVERY = 1             # aktualizuj tło co N klatek
NOISE_OCTAVES = 12               # było 4

PLASMA_EDGE_SHARPNESS = 3.0
PLASMA_INTENSITY = 1.2
CENTER_DETAIL_POWER = 3.2


def _seed_offsets(seeds):
    """Zamienia listę seedów na stabilne offsety (ox, oy) dla fbm."""
    out = []
    for s in seeds:
        # deterministycznie rozbijamy seed na 2 floaty
        # (tu prosto; możesz podmienić na coś bardziej fancy)
        ox = (s * 37.13) % 1000.0
        oy = (s * 91.70) % 1000.0
        out.append((ox, oy))
    return out


def generate_noise_maps(bg_w: int, bg_h: int, scale: int, freq: float, octaves: int, seeds) -> list[np.ndarray]:
    """
    Generuje listę map noise (float32) o rozmiarze [bg_h, bg_w].
    freq: częstotliwość próbkowania (np. 0.008)
    """
    maps: list[np.ndarray] = []
    offsets = _seed_offsets(seeds)

    for (ox, oy) in offsets:
        m = np.zeros((bg_h, bg_w), dtype=np.float32)
        for y in range(bg_h):
            py = (y + 0.5) * scale
            for x in range(bg_w):
                px = (x + 0.5) * scale
                m[y, x] = fbm(px * freq + ox, py * freq + oy, octaves)
        maps.append(m)

    return maps


def save_noise_cache(cache_dir: str | Path, base_maps: list[np.ndarray], shimmer_maps: list[np.ndarray]) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i, m in enumerate(base_maps):
        np.save(cache_dir / f"base_{i}.npy", m)

    for i, m in enumerate(shimmer_maps):
        np.save(cache_dir / f"shimmer_{i}.npy", m)


def load_noise_cache(cache_dir: str | Path, count: int = 3) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cache_dir = Path(cache_dir)

    base_maps = []
    shimmer_maps = []
    for i in range(count):
        base_maps.append(np.load(cache_dir / f"base_{i}.npy"))
        shimmer_maps.append(np.load(cache_dir / f"shimmer_{i}.npy"))

    return base_maps, shimmer_maps


def ensure_noise_cache(
    cache_dir: str | Path,
    bg_w: int,
    bg_h: int,
    scale: int,
    base_freq: float,
    shimmer_freq: float,
    octaves: int,
    base_seeds=(1, 2, 3),
    shimmer_seeds=(101, 102, 103),
    count: int = 3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Jeśli cache istnieje i pasuje rozmiarami → wczytaj.
    Jeśli nie → wygeneruj 3 + 3 mapy i zapisz.
    """
    cache_dir = Path(cache_dir)
    expected = [cache_dir / f"base_{i}.npy" for i in range(count)] + [cache_dir / f"shimmer_{i}.npy" for i in range(count)]
    have_all = all(p.exists() for p in expected)

    if have_all:
        print("Reading noise and shimmer from cache")
        base_maps, shimmer_maps = load_noise_cache(cache_dir, count=count)

        # szybka walidacja rozmiaru (żeby nie użyć starego cache po zmianie SCALE/W/H)
        ok = True
        for m in base_maps + shimmer_maps:
            if m.shape != (bg_h, bg_w):
                ok = False
                break

        if ok:
            return base_maps, shimmer_maps

    # jeśli nie ma cache albo nie pasuje → generuj
    print("Generating noise cache")
    base_maps = generate_noise_maps(bg_w, bg_h, scale, base_freq, octaves, seeds=base_seeds)
    print("Generating shimmer cache")
    shimmer_maps = generate_noise_maps(bg_w, bg_h, scale, shimmer_freq, octaves, seeds=shimmer_seeds)

    save_noise_cache(cache_dir, base_maps, shimmer_maps)
    return base_maps, shimmer_maps


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

def plasma_color(intensity: float, tint_shift: float = 0.0):
    intensity = max(0.0, min(1.2, intensity))
    b = int(40 + 200 * intensity)
    g = int(10 + 140 * intensity)
    r = int(0 + 35 * intensity)
    g = max(0, min(255, int(g + 40 * tint_shift)))
    b = max(0, min(255, int(b + 20 * tint_shift)))
    return r, g, b


def main():
    pygame.init()
    font = pygame.font.SysFont("consolas", 8)

    pygame.display.set_caption("Hypnagogi — płynniejsza wersja")
    clock = pygame.time.Clock()

    bg_w = W // SCALE
    bg_h = H // SCALE

    screen = pygame.display.set_mode((bg_w, bg_h), pygame.SCALED)

    bg = pygame.Surface((bg_w, bg_h))

    CACHE_DIR = ".noise_cache"

    # Ustal częstotliwości osobno:
    BASE_FREQ = 0.0065      # „większe plamy”
    SHIMMER_FREQ = 0.013    # „drobniejszy detal do jarzenia”

    base_maps, shimmer_maps = ensure_noise_cache(
        CACHE_DIR,
        bg_w=bg_w,
        bg_h=bg_h,
        scale=SCALE,
        base_freq=BASE_FREQ,
        shimmer_freq=SHIMMER_FREQ,
        octaves=NOISE_OCTAVES,  # użyj swojego parametru
        base_seeds=(11, 22, 33),
        shimmer_seeds=(111, 222, 333),
        count=3,
    )

    cx, cy = W * 0.5, H * 0.52
    base_a, base_b = W * 0.23, H * 0.18
    drift_tx = random.random() * 999
    drift_ty = random.random() * 999
    morph_t = random.random() * 999
    t = 0.0
    rgb_buf = bytearray(bg_w * bg_h * 3)

    frame = 0

    # --- stała mapa centrum (raz) ---
    xs = (np.arange(bg_w) + 0.5) * SCALE
    ys = (np.arange(bg_h) + 0.5) * SCALE
    PX, PY = np.meshgrid(xs, ys)  # shape: (bg_h, bg_w)

    RX = (PX - W * 0.5) / (W * 0.5)
    RY = (PY - H * 0.5) / (H * 0.5)
    R = np.sqrt(RX * RX + RY * RY)

    center_map = np.clip(1.0 - R, 0.0, 1.0).astype(np.float32) ** CENTER_DETAIL_POWER

    # widok na bufor (tworzysz raz!)
    buf_view = np.frombuffer(rgb_buf, dtype=np.uint8).reshape(bg_h, bg_w, 3)

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

        phase = (t * 0.10) % 3.0
        i0 = int(phase) % 3
        i1 = (i0 + 1) % 3
        f = phase - int(phase)
        f = f * f * (3 - 2 * f)  # smoothstep
        invf = 1.0 - f

        base0 = base_maps[i0]
        base1 = base_maps[i1]
        shim0 = shimmer_maps[i0]
        shim1 = shimmer_maps[i1]

        # crossfade (raz na klatkę)
        phase = (t * 0.10) % 3.0
        i0 = int(phase)
        i1 = (i0 + 1) % 3
        f = phase - i0
        f = f * f * (3.0 - 2.0 * f)     # smoothstep
        invf = 1.0 - f

        # blend map (wektorowo)
        n_base = invf * base_maps[i0] + f * base_maps[i1]
        n_shim = invf * shimmer_maps[i0] + f * shimmer_maps[i1]

        # shimmer/tint (wektorowo)
        shimmer = 0.55 + 0.45 * np.sin(n_shim * 6.0 + t * 1.7).astype(np.float32)
        tint    = 0.25 * np.sin(t * 0.7 + n_base * 2.0).astype(np.float32)

        # intensywność plazmy
        intensity = (PLASMA_INTENSITY * shimmer * center_map).astype(np.float32)
        ii = np.clip(intensity, 0.0, 1.2)

        # paleta (RGB uint8)
        b = (40 + 200 * ii).clip(0, 255).astype(np.uint8)
        g = (10 + 140 * ii + 40 * tint).clip(0, 255).astype(np.uint8)
        r = (0  + 35  * ii).clip(0, 255).astype(np.uint8)

        # wpis do bufora (bez kopiowania „po pikselu”)
        buf_view[..., 0] = r
        buf_view[..., 1] = g
        buf_view[..., 2] = b

        screen.blit(pygame.image.frombuffer(rgb_buf, (bg_w, bg_h), "RGB"), (0, 0))
        fps = clock.get_fps()
        fps_text = font.render(f"{fps:5.1f} FPS", True, (40, 200, 255))
        screen.blit(fps_text, (10, 10))
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
