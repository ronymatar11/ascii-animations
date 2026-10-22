"""
The Curse of Being Human  |  9:16, red and black, every frame is text, cut to a spoken-word track.
One continuous camera, no cuts: each scene turns into the next.

the curse of being human: a red rim light traces a classical bust (from a reference photo) ->
humans pursue peace: dots fill it in, the camera moves in on the head ->
yet manufacture chaos: pixel flames (from a reference) climb over it ->
they fear death: the fire burns away to an x-ray skull with a cassette for a brain (from a reference) ->
yet waste their lives: the tape unspools, the camera follows it down to money on a string (from a reference) ->
they demand freedom: the string snaps, the money drops out of frame -> iron bars slam down one by one,
the first over the string -> then construct their: hands grab them -> own prisons: pull back, the prison
is inside an eye -> perhaps the greatest flaw in your design: the eye is one of a wall of eyes, all turning
to look at you -> is not your mortality: they all close -> it is your extraordinary ability: they give way to two hands
holding a closed ring box (from a reference) -> to remain alive: on the hit the lid springs open, an Esc key
inside, its light pulsing with the heart -> without ever truly living: Esc is pressed, the screen switches off.

Export:   python curse_of_being_human.py --music voice.mp4 --export curse.mp4
"""
import argparse, math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from code_to_monalisa import W, H, CI, RAMP, BLOCK, PRINTABLE, sstep, smoother, fbm, find_font, ffmpeg_bin
from second_sun import (FCW, FCH, FCOLS, FROWS, FOY, FGX, FGY, FATLAS, SOFT, render_fine, glyphs,
                        stroke_glyph, smin, typed)

FPS = 30
DUR = 30.55
HITS = [3.0, 14.0, 17.0, 26.5]            # the heaviest hits in the track
LYRICS = [
    (0.0, 2.5, "The curse of being human"),
    (2.7, 5.03, "Humans pursue peace,"),
    (5.03, 7.43, "yet manufacture chaos."),
    (7.43, 9.23, "They fear death,"),
    (9.23, 11.8, "yet waste their lives."),
    (11.83, 13.23, "They demand freedom,"),
    (13.8, 14.87, "then construct their"),
    (14.87, 16.67, "own prisons."),
    (16.67, 17.9, "Perhaps the greatest"),
    (17.9, 19.97, "flaw in your design"),
    (19.97, 22.5, "is not your mortality."),
    (22.53, 25.23, "It is your extraordinary ability"),
    (26.0, 27.27, "to remain alive"),
    (27.27, 29.53, "without ever truly living."),
]
T_B = 12.6                                # scene handoff: the string is all that's on screen

INK = np.array([0.40, 0.02, 0.03], np.float32)
RED = np.array([1.0, 0.13, 0.08], np.float32)
HOT = np.array([1.0, 0.60, 0.52], np.float32)
PALE = np.array([1.0, 0.86, 0.82], np.float32)
DOTS = np.array(sorted([CI[c] for c in ".:oO0@"], key=lambda i: FATLAS[i].mean()))
SHADE = np.array(sorted([CI[c] for c in ".:-=+*#%@"], key=lambda i: FATLAS[i].mean()))   # smooth shading, no letters

POOL = [
    "def human(): return curse", "while alive: pursue(peace)", "chaos = manufacture(peace)",
    "if fear(death): waste(life)", "freedom = demand()", "prison = build(freedom)",
    "assert alive and not living", "for day in life: repeat(day)", "mortality.flaw = False",
    "ability = extraordinary(remain_alive)", "try: live()", "except Fear: exist()",
    "heart.beat() if alive else None", "eyes = [Eye() for _ in range(n)]", "while True: watch(you)",
    "skull.load(Cassette('memories'))", "money.hang(string)", "bars.extend(hands)",
    "# the curse of being human", "import mortality", "return alive and not living", "pulse = 0",
    "design.flaws.append(you)", "time.sleep(life)", "if not living: continue", "0x00 0xDE 0xAD",
]


def redmap(b):
    b = np.clip(b, 0, 1)[..., None]
    lo = INK + (RED - INK) * np.clip(b * 2.2, 0, 1)
    hi = np.clip((b - 0.72) / 0.28, 0, 1)
    return lo * (1 - hi) + HOT * hi

def ckey(t, keys):
    """C1 curve through (time, value) keys, catmull-rom tangents: it passes every key without stopping"""
    ts = [k[0] for k in keys]; vs = [k[1] for k in keys]
    if t <= ts[0]: return vs[0]
    if t >= ts[-1]: return vs[-1]
    i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
    def tan(j):
        a, b = max(j - 1, 0), min(j + 1, len(ts) - 1)
        return (vs[b] - vs[a]) / (ts[b] - ts[a])
    h = ts[i + 1] - ts[i]; s = (t - ts[i]) / h
    m0, m1 = tan(i) * h, tan(i + 1) * h
    return ((2 * s ** 3 - 3 * s ** 2 + 1) * vs[i] + (s ** 3 - 2 * s ** 2 + s) * m0 +
            (-2 * s ** 3 + 3 * s ** 2) * vs[i + 1] + (s ** 3 - s ** 2) * m1)

def mono(size, bold=True):
    for f in (["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Bold.ttf"] if bold else []):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return find_font(size)

# ------------------------------------------------------------------ 3D helpers
def _ell(x, y, z, rx, ry, rz):
    k0 = np.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
    k1 = np.sqrt((x / rx ** 2) ** 2 + (y / ry ** 2) ** 2 + (z / rz ** 2) ** 2)
    return k0 * (k0 - 1) / (k1 + 1e-9)

def _cap(x, y, z, a, b, ra, rb=None):
    rb = ra if rb is None else rb
    bx, by, bz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    px, py, pz = x - a[0], y - a[1], z - a[2]
    h = np.clip((px * bx + py * by + pz * bz) / (bx * bx + by * by + bz * bz), 0, 1)
    return np.sqrt((px - bx * h) ** 2 + (py - by * h) ** 2 + (pz - bz * h) ** 2) - (ra + (rb - ra) * h)

def rot(yaw, pitch):
    """head -> camera"""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    return np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])

def march(sdf, o, dvec, t0, t1, steps=46, eps=0.004):
    n = len(o)
    t = np.full(n, t0); hit = np.zeros(n, bool); act = np.arange(n)
    for _ in range(steps):
        p = o[act] + t[act, None] * dvec
        d = sdf(p[:, 0], p[:, 1], p[:, 2])
        t[act] += d * 0.85
        done = d < eps
        hit[act[done]] = True
        act = act[~done & (t[act] < t1)]
        if len(act) == 0: break
    return t, hit

def normals(sdf, p, e=0.004):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    n = np.stack([sdf(x + e, y, z) - sdf(x - e, y, z), sdf(x, y + e, z) - sdf(x, y - e, z),
                  sdf(x, y, z + e) - sdf(x, y, z - e)], 1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)

def unit(*v):
    v = np.array(v, float); return v / np.linalg.norm(v)


# ------------------------------------------------------------------ 2D helpers
def _seg2(x, y, a, b, ra, rb=None):
    rb = ra if rb is None else rb
    bx, by = b[0] - a[0], b[1] - a[1]
    px, py = x - a[0], y - a[1]
    h = np.clip((px * bx + py * by) / (bx * bx + by * by), 0, 1)
    return np.hypot(px - bx * h, py - by * h) - (ra + (rb - ra) * h)

def _rbox2(x, y, hx, hy, r):
    qx, qy = np.abs(x) - hx + r, np.abs(y) - hy + r
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r

def splat(idx, col, solid, X, Y, color, b=None, bright=0.9, glyph=None):
    """draw a polyline (screen px) into the fine grid as strokes (or one glyph)"""
    if len(X) < 2: return
    c = np.round(X / FCW - 0.5).astype(int); r = np.round((Y - FOY) / FCH - 0.5).astype(int)
    g = stroke_glyph(np.arctan2(np.gradient(Y) / FCH, np.gradient(X) / FCW)) if glyph is None else np.full(len(X), glyph)
    ok = (c >= 0) & (c < FCOLS) & (r >= 0) & (r < FROWS)
    idx[r[ok], c[ok]] = g[ok]
    col[r[ok], c[ok]] = color[ok] if np.ndim(color) == 2 else color
    solid[r[ok], c[ok]] = True
    if b is not None:
        b[r[ok], c[ok]] = bright

EX = 1.3
def lids(xl, op):
    sx = np.clip(1 - (xl / EX) ** 2, 0, 1)
    y_low = -0.40 * sx ** 0.85 - 0.03 + 0.03 * xl
    y_up = y_low + (0.58 * sx ** 0.7 + 0.02 - 0.04 * xl - y_low) * op
    return sx, y_low, y_up

FS = 0.2                                          # fist units -> eye units
def fist_sdf(x, y, z, clench):
    """a hand gripping a vertical bar (axis = y, radius BAR_R / FS) from behind and to the left;
    fingers wrap round the front of the bar, the thumb crosses over them"""
    rg = BAR_R / FS + 0.08
    d = _cap(x, y, z, (-0.33, -0.22, -0.30), (-0.62, -1.6, -1.0), 0.17, 0.2)             # forearm
    pal = _ell(x + 0.30, y + 0.02, z + 0.24, 0.13, 0.36, 0.24)                          # palm, behind
    d = smin(d, pal, 0.1)
    end = 0.75 + 0.25 * clench
    for i, (r, ln) in enumerate(((0.082, 1.0), (0.086, 1.06), (0.08, 1.0), (0.07, 0.86))):
        yi = 0.25 - 0.165 * i
        ph = np.array([-2.5, -1.45, -0.45, -0.45 + end * ln])                          # around the bar
        pts = [(rg * math.sin(a_), yi - 0.02 * k, rg * math.cos(a_)) for k, a_ in enumerate(ph)]
        pts[0] = (-0.27, yi + 0.01, -0.33)
        f = _cap(x, y, z, pts[0], pts[1], r * 1.08, r)
        f = smin(f, _cap(x, y, z, pts[1], pts[2], r, r * 0.95), 0.03)
        f = smin(f, _cap(x, y, z, pts[2], pts[3], r * 0.95, r * 0.85), 0.03)
        d = smin(d, f, 0.025)
    th = smin(_cap(x, y, z, (-0.32, -0.3, -0.05), (-0.16, -0.12, rg + 0.05), 0.1, 0.09),
              _cap(x, y, z, (-0.16, -0.12, rg + 0.05), (0.06, 0.05, rg + 0.07), 0.09, 0.075), 0.03)
    return smin(d, th, 0.04)

# ------------------------------------------------------------------ scene A camera (photo px)
# the bust photo: screen px per photo px, where it sits, the point between the eyes
# the head is already moving at the first frame: it drifts in from up and to the left, tilted, and
# settles into its pose by "Humans pursue peace"
BUST_K = [(0, 1.62), (1.4, 1.88), (2.7, 2.1), (5.0, 2.65), (7.4, 2.9), (9.0, 3.1)]
BUST_X = [(0, 330), (1.4, 470), (2.7, 540), (9.0, 540)]
HY = [(0, 560), (1.4, 665), (2.7, 725), (5.0, 860), (7.4, 900), (9.0, 920)]
ROLL = [(0, -0.2), (1.4, -0.05), (2.7, 0.02), (5.0, -0.02), (7.4, 0.02), (9.2, 0.0), (12.6, 0.0)]
BUST_E = (255.0, 290.0)
# the x-ray: screen px per image px, the head's center, the reels, where the drawn tape ends
SKULL_K = [(6.8, 2.25), (7.4, 2.1), (9.2, 1.55), (12.6, 1.5)]
SKULL_HY = [(6.8, 980), (9.2, 900), (12.6, 900)]
SKULL_E = (300.0, 240.0)
REELS = [((243.0, 120.0), 40.0), ((368.0, 146.0), 62.0)]
TAPE0 = (250.0, 415.0)
TAPE_L = [(9.2, 0), (9.45, 400), (9.7, 1200), (9.95, 2080), (10.15, 2260), (10.4, 2140), (10.7, 2190), (11.2, 2175), (12.6, 2175)]
FIRE_K = 2.0                                     # fire pattern, screen px per image px
# the money photo: the string's top, the point that ends up at the centre of the screen, scale and tilt
MONEY_KNOT, MONEY_C = (159.0, 0.0), (200.0, 350.0)   # money photo px: the string's top, the stack's middle
K_M0 = 1.9                                       # screen px per photo px
T_SNAP, MONEY_G = 11.83, 5200.0                  # "They demand freedom": the string snaps, the money drops

def sample(img, PX, PY):
    """bilinear sample of an (h, w) or (h, w, c) image at float px; zero outside"""
    h, w = img.shape[:2]
    ok = (PX >= 0) & (PX < w - 1) & (PY >= 0) & (PY < h - 1)
    x0 = np.clip(PX, 0, w - 1.001); y0 = np.clip(PY, 0, h - 1.001)
    i, j = x0.astype(int), y0.astype(int)
    fx, fy = x0 - i, y0 - j
    if img.ndim == 3:
        fx, fy, ok = fx[..., None], fy[..., None], ok[..., None]
    v = (img[j, i] * (1 - fx) * (1 - fy) + img[j, i + 1] * fx * (1 - fy) +
         img[j + 1, i] * (1 - fx) * fy + img[j + 1, i + 1] * fx * fy)
    return np.where(ok, v, 0.0)

# ------------------------------------------------------------------ scene B camera (eye units)
EU = 330.0
LOGZ = [(12.6, math.log(6.5)), (13.3, math.log(6.4)), (14.6, math.log(5.6)), (15.6, math.log(2.6)),
        (16.7, math.log(1.0)), (17.6, math.log(0.5)), (18.8, math.log(0.36)), (20.0, math.log(0.33)),
        (21.5, math.log(0.31)), (22.5, math.log(0.38)), (23.6, math.log(0.98)), (24.6, math.log(1.06)),
        (25.3, math.log(1.1)), (26.5, math.log(1.25)), (27.8, math.log(1.2)), (30.55, math.log(1.2))]
CX = [(12.6, 0.08), (14.6, 0.08), (16.0, 0.05), (17.6, 0.0), (19.0, 0.8), (20.5, 0.3), (21.8, -0.6),
      (22.6, -0.2), (23.6, 0.3), (25.0, 0.32)]
CY = [(12.6, 0.08), (14.6, 0.06), (16.0, 0.0), (17.6, 0.2), (19.0, 0.9), (20.5, -0.4), (21.8, -0.9),
      (22.6, -0.75), (23.6, -0.95), (25.0, -0.85)]
ROLLB = [(12.6, 0), (14.6, 0.0), (16.0, 0.04), (17.6, 0.1), (19.5, -0.06), (21.0, 0.08), (22.6, 0.02),
         (23.6, 0.0), (30.55, 0.0)]
BAR_DX, BAR_R = 0.16, 0.016
BAR_TIP, BAR_FALL = -0.30, 0.17                  # where the bars' ends land, how long each takes to fall
# the bust comes back at the end: bust photo px per eye unit, and the px of its left eye (where it cries)
BEATS = [(26.5, 0.8), (27.45, 0.48)]             # heart hits: the box's light pulses
# the ring box (reference image px): lid hinge, how deep it lies closed, how tall it stands open, its sides
HINGE, LID_D, LID_H, LID_TOP = 510, 135, 280, 230
LID_X, BASE_X = (372, 612), (368, 630)
KEY_BOX, KEY_C = (395, 600, 505, 645), (497, 560)
T_OPEN, T_PRESS, T_OFF = 26.45, 29.0, 29.12      # the lid opens on the hit; Esc is pressed; the screen goes off
BOXCAM_X = [(22.6, 500), (24.5, 497), (26.4, 498), (27.3, 492), (27.9, 495), (28.9, 497), (30.55, 497)]
BOXCAM_Y = [(22.6, 650), (24.5, 645), (26.4, 640), (27.3, 500), (27.9, 520), (28.9, 568), (30.55, 570)]
BOXCAM_Z = [(22.6, math.log(3.3)), (24.5, math.log(3.0)), (26.4, math.log(2.85)), (27.3, math.log(1.05)),
            (27.9, math.log(1.15)), (28.9, math.log(2.6)), (30.55, math.log(2.75))]


class Film:
    def __init__(self, handle="@rony_matar", lyrics=True, seed=5):
        rng = np.random.default_rng(seed)
        self.handle, self.lyrics = handle, lyrics
        self.rand = rng.random((FROWS, FCOLS))
        self.setup_bust()
        here = os.path.dirname(os.path.abspath(__file__))
        self.skull = np.asarray(Image.open(os.path.join(here, "assets", "skull_xray.png")).convert("L"), np.float32) / 255.0
        self.fire = np.asarray(Image.open(os.path.join(here, "assets", "fire.png")).convert("RGB"), np.float32) / 255.0
        self.money = np.asarray(Image.open(os.path.join(here, "assets", "money.png")).convert("RGBA"), np.float32) / 255.0
        self.build_code(rng)
        # the fire eats the face from the bottom up
        self.T_burn = 7.25 + 1.2 * np.clip((1750 - FGY) / 1500, 0, 1) + 0.35 * (fbm(FGX / 150, FGY / 150, 3, 19) - 0.5)
        self.setup_tape()
        self.setup_bars(rng)
        self.setup_wall(rng)
        self.setup_box()

    # ---------- the page of code behind everything
    def build_code(self, rng):
        idx = np.zeros((FROWS, FCOLS), np.int64); tt = np.full((FROWS, FCOLS), np.inf)
        for r in range(FROWS):
            x = int(rng.integers(0, 4))
            while x < FCOLS - 6:
                e = min(FCOLS, x + int(rng.integers(12, 42)))
                txt = ""
                while len(txt) < e - x:
                    txt += POOL[int(rng.integers(len(POOL)))] + " "
                idx[r, x:e] = [CI.get(c, 0) for c in txt[:e - x]]
                tt[r, x:e] = 0.05 + 1.6 * rng.random() ** 1.4 + np.arange(e - x) / rng.uniform(60, 140)
                x = e + int(rng.integers(2, 6))
        idx[idx == CI[" "]] = 0
        self.code_idx, self.code_t = idx, tt
        self.code_v = rng.choice([-1, 1], FROWS) * rng.uniform(1.5, 5.0, FROWS)

    def code_layer(self, t, fi):
        ci = (np.arange(FCOLS)[None, :] + np.floor(self.code_v * t).astype(int)[:, None]) % FCOLS
        idx = np.take_along_axis(self.code_idx, ci, 1)
        on = (t >= np.take_along_axis(self.code_t, ci, 1)) & (idx > 0)
        rng = np.random.default_rng(fi)
        flick = on & (rng.random((FROWS, FCOLS)) < 0.004)
        idx = np.where(flick, rng.choice(PRINTABLE, (FROWS, FCOLS)), idx)
        lv = 0.12 + 0.04 * np.sin(FGY * 0.01 + t * 1.3) + 0.25 * flick
        col = RED[None, None, :] * lv[..., None]
        return np.where(on, idx, 0), col.astype(np.float32)

    # ---------- scene A: bust, fire, x-ray, tape, money
    def setup_bust(self):
        """the rim-lit bust from the reference photo (red light only), and its filled silhouette"""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bust_rim.png")
        im = Image.open(path).convert("L")
        rim = np.asarray(im.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255.0
        lit = rim > 0.16
        sil = np.zeros_like(lit)
        rows = np.nonzero(lit.any(1))[0]
        for r in range(rows.min(), rows.max() + 1):
            xs = np.nonzero(lit[max(r - 6, 0):r + 7].any(0))[0]
            if len(xs) > 1:
                sil[r, xs.min():xs.max() + 1] = True
        blur = np.asarray(Image.fromarray((sil * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(28)), np.float32) / 255
        self.bust_rim = rim
        self.bust_glow = np.asarray(im.filter(ImageFilter.GaussianBlur(7)), np.float32) / 255.0
        self.bust_in = np.clip((blur - 0.45) * 2.2, 0, 1) * sil

    def bust_sample(self, img, PX, PY):
        h, w = img.shape
        ok = (PX >= 0) & (PX < w - 1) & (PY >= 0) & (PY < h - 1)
        x0 = np.clip(PX, 0, w - 1.001); y0 = np.clip(PY, 0, h - 1.001)
        i, j = x0.astype(int), y0.astype(int)
        fx, fy = x0 - i, y0 - j
        v = (img[j, i] * (1 - fx) * (1 - fy) + img[j, i + 1] * fx * (1 - fy) +
             img[j + 1, i] * (1 - fx) * fy + img[j + 1, i + 1] * fx * fy)
        return np.where(ok, v, 0.0)

    def setup_tape(self):
        # the tape hangs straight; once the camera has panned, the money hangs in the middle of the frame
        k = ckey(11.5, SKULL_K)
        off = (MONEY_C[0] - MONEY_KNOT[0], MONEY_C[1] - MONEY_KNOT[1])
        hxf = 540 - off[0] * K_M0 - (TAPE0[0] - SKULL_E[0]) * k
        self.HXS = [(6.8, 540), (8.6, 545), (10.6, hxf), (12.6, hxf)]
        self.PAN = [(0, 0)]
        y0 = ckey(11.5, SKULL_HY) + (TAPE0[1] - SKULL_E[1]) * k
        P = 960 - (y0 + TAPE_L[-1][1] + off[1] * K_M0)
        self.PAN = [(0, 0), (9.35, 0), (9.9, 0.12 * P), (10.6, 0.58 * P), (11.3, 0.93 * P), (12.6, P)]
        # the money only exists on the tape once its end is out of frame
        self.t_att = 12.0
        for tt in np.arange(9.2, 10.4, 0.01):
            _, hang = self.tape(tt)
            if hang is not None and hang[1] > 2010:
                self.t_att = tt; break

    def setup_bars(self, rng):
        # the string the money hung from becomes the line the first bar comes down over
        _, self.snap_hang = self.tape(T_SNAP)
        X, Y = self.tape(T_B)[0]
        vis = (Y > 0) & (Y < H)
        sx = float(X[vis].mean()) if vis.any() else self.snap_hang[0]
        Z0 = math.exp(ckey(T_B, LOGZ))
        cx0 = (540 - sx) / (EU * Z0)
        self.CXK = [(T_B, cx0), (13.7, BAR_DX / 2)] + [k for k in CX if k[0] >= 14.6]
        cy0 = ckey(T_B, CY)
        self.string_y0 = cy0 - (self.snap_hang[1] - 960) / (EU * Z0)
        self.string_k = 1.0 / (EU * Z0)
        # the bars slam down: the one over the string first, then its neighbours, then the rest off-screen
        order = {0: 12.72, 1: 12.97, -1: 13.2, 2: 13.42}
        self.bar_t = [order.get(k, 13.5 + 0.03 * abs(k)) for k in range(-8, 9)]
        self.spark_rng = [[(rng.uniform(-0.35, 0.35), rng.uniform(0.15, 0.7), rng.uniform(0.25, 0.55)) for _ in range(18)]
                          for _ in range(17)]
        self.impacts = [(T_SNAP, 6.0)] + [(order[k] + BAR_FALL, 9.0) for k in order]

    def bar_tip(self, k, t):
        tk = self.bar_t[k + 8]
        x = min(max((t - tk) / BAR_FALL, 0.0), 1.0)
        top = 0.8
        if x < 1:
            return top - (top - BAR_TIP) * x * x
        a = t - tk - BAR_FALL
        return BAR_TIP + 0.022 * math.exp(-a / 0.07) * abs(math.sin(a * 45))

    def skull_cam(self, t):
        return ckey(t, SKULL_K), ckey(t, self.HXS), ckey(t, SKULL_HY) + ckey(t, self.PAN), ckey(t, ROLL)

    def skull_to_screen(self, PX, PY, t):
        k, hx, hy, roll = self.skull_cam(t)
        dx, dy = (PX - SKULL_E[0]) * k, (PY - SKULL_E[1]) * k
        cr, sr = math.cos(roll), math.sin(roll)
        return hx + dx * cr - dy * sr, hy + dx * sr + dy * cr

    def tape(self, t):
        """the tape running on from the curls in the x-ray, down out of frame: screen points, hanging end"""
        L = ckey(t, TAPE_L) if t > 9.2 else 0.0
        if L <= 5:
            return None, None
        s = np.concatenate([np.arange(0, min(L, 700), 0.6), np.arange(700, L, 2.5)])
        k = ckey(t, SKULL_K)
        A = 34 * k * np.exp(-s / 260)
        ph = s / (17 * k) + 1.2 + 0.6 * math.sin(t * 1.3)
        grow = 1 - np.exp(-s / 12)
        X0, Y0 = self.skull_to_screen(np.array([TAPE0[0]]), np.array([TAPE0[1]]), t)
        X = X0[0] + A * np.sin(ph) * grow
        Y = Y0[0] + s + A * (np.cos(ph) - math.cos(1.2)) * grow
        return (X, Y), (X[-1], Y[-1])

    def money_pose(self, t, hang):
        """the money photo on screen (centre, scale, tilt) and where its string is tied on: it swings on
        the tape, then the string snaps and it drops out of frame, tumbling"""
        tw = t - 9.95
        phi = 0.07 * math.sin(2.4 * tw) * math.exp(-max(tw, 0) / 1.8) + 0.025 * math.sin(t * 1.1)
        ox, oy = MONEY_C[0] - MONEY_KNOT[0], MONEY_C[1] - MONEY_KNOT[1]
        tau = t - T_SNAP
        if tau > 0:
            hang = self.snap_hang
            phi += 0.55 * tau ** 2 + 0.25 * tau
        c, s = math.cos(phi), math.sin(phi)
        cx, cy = hang[0] + (ox * c - oy * s) * K_M0, hang[1] + (ox * s + oy * c) * K_M0
        if tau > 0:
            cx += 70 * tau
            cy += 0.5 * MONEY_G * tau ** 2 + 120 * tau
        return cx, cy, K_M0, phi, hang

    def money_texture(self, AX, AY):
        """brightness, colour, coverage of the money photo at its own px"""
        rgba = sample(self.money, AX, AY)
        a = np.clip((rgba[..., 3] - 0.35) / 0.3, 0, 1)
        rgb = rgba[..., :3]
        lum = 0.5 * rgb[..., 0] + 0.35 * rgb[..., 1] + 0.15 * rgb[..., 2]
        b = np.clip((lum - 0.06) / 0.7, 0, 1) ** 1.15
        cc = rgb / (rgb.max(-1, keepdims=True) + 0.02)
        return b, cc, a > 0.5

    def money_layer(self, b, col, gl, solid, t, SX, SY, hang):
        # while it falls, a few earlier positions trail behind it (motion blur)
        subs = [0.0] if t < T_SNAP + 0.02 else [0.045, 0.03, 0.015, 0.0]
        for k, dt in enumerate(subs):
            cx, cy, K, phi, knot = self.money_pose(t - dt, hang)
            c, s = math.cos(phi), math.sin(phi)
            dx, dy = SX - cx, SY - cy
            AX = MONEY_C[0] + (dx * c + dy * s) / K
            AY = MONEY_C[1] + (-dx * s + dy * c) / K
            mb, mc, on = self.money_texture(AX, AY)
            fade = 0.35 + 0.65 * (k + 1) / len(subs)
            b[on] = mb[on] * fade; col[on] = mc[on]; gl[on] = 0; solid[on] = True

    def string_end(self, t):
        """after the snap the tape's end springs back up: how far, in screen px"""
        tau = t - T_SNAP
        if tau <= 0: return 0.0
        return 120 * (1 - math.exp(-tau / 0.09) * math.cos(tau * 19))

    def scene_a(self, t, SX, SY, fi):
        shape = (FROWS, FCOLS)
        b = np.zeros(shape); col = np.zeros(shape + (3,), np.float32)
        gl = np.zeros(shape, np.int64); solid = np.zeros(shape, bool)
        # small, slow head movement on top of the camera path
        hm = 1 - float(smoother(6.5, 7.6, t))
        roll = ckey(t, ROLL) + hm * (0.022 * math.sin(t * 1.25) + 0.01 * math.sin(t * 2.9 + 1))
        cr, sr = math.cos(roll), math.sin(roll)
        dx = SX - ckey(t, BUST_X) - hm * (9 * math.sin(t * 0.85) + 4 * math.sin(t * 2.1))
        dy = SY - ckey(t, HY) - hm * (6 * math.sin(t * 1.05 + 1))
        burnt = t >= self.T_burn

        # the bust: a red rim light traces it, dots fill it in, then firelight
        if t < 8.9:
            k = ckey(t, BUST_K) * (1 + 0.012 * math.sin(t * 1.6))          # breathing
            PX = BUST_E[0] + (dx * cr + dy * sr) / k
            PY = BUST_E[1] + (-dx * sr + dy * cr) / k
            # the head turns a little, left and right, nodding: a cylindrical warp of the photo, the head
            # turning more than the shoulders, so it reads as a head in space and not a flat picture
            hm = 1 - float(smoother(6.5, 7.6, t))
            yaw = hm * (0.17 * math.sin(t * 0.75 + 0.6) + 0.05 * math.sin(t * 1.9)) - 0.18 * (1 - float(smoother(0.0, 2.7, t)))
            pitch = hm * 0.035 * math.sin(t * 1.1 + 2)
            w = np.clip((560 - PY) / 200, 0.25, 1.0)
            u = np.clip((PX - BUST_E[0]) / 300, -0.999, 0.999)
            PX = BUST_E[0] + 300 * np.sin(np.arcsin(u) - yaw * w)
            PY = PY - pitch * 150 * np.sqrt(1 - u ** 2) * w
            rim = self.bust_sample(self.bust_rim, PX, PY)
            inner = self.bust_sample(self.bust_in, PX, PY)
            glow = self.bust_sample(self.bust_glow, PX, PY)
            rimI = 1.0                                                  # lit from the first frame, no fade in
            keyI = float(smoother(2.9, 3.8, t))
            fireI = float(smoother(5.0, 6.0, t))
            flick = 0.75 + 0.25 * np.sin(t * 23 + PY * 0.05) * np.sin(t * 17 + PX * 0.04)
            # the light breathes and travels slowly around the rim
            ang = np.arctan2(PY - 330, PX - BUST_E[0])
            travel = 0.72 + 0.4 * np.cos(ang - t * 0.9) ** 2 * hm + 0.28 * (1 - hm)
            rimI = rimI * travel
            rb = (np.clip(rim * 1.8, 0, 1) ** 0.7 + 0.55 * np.clip(glow * 2.2, 0, 1) ** 0.9) * rimI
            rb = rb * (1 + 0.3 * keyI + 0.6 * fireI * flick)
            fill = inner * (keyI * (0.07 + 0.16 * np.clip((900 - PY) / 700, 0, 1)) * (0.85 + 0.15 * math.sin(t * 2.2))
                            + fireI * 0.35 * flick * np.clip((PY - 250) / 600, 0, 1))
            on = ~burnt & ((rb > 0.04) | (fill > 0.035))
            bb = np.clip(np.maximum(rb, fill), 0, 1)
            b = np.where(on, bb, b)
            col = np.where(on[..., None], redmap(bb * (1 + 0.3 * fireI)), col)
            jr = np.random.default_rng(fi).integers(-1, 2, shape)
            dots = DOTS[np.clip((fill * 2.6 * (len(DOTS) - 1)).astype(int) + jr, 0, len(DOTS) - 1)]
            gl = np.where(on & (rb <= fill), dots, gl)
            solid |= ~burnt & (on | (inner > 0.05))

        # the x-ray, revealed where the fire has burnt through; the reels spin, a scan line runs down it
        if t > 7.0:
            k, hx, hy, rl = self.skull_cam(t)
            c2, s2 = math.cos(rl), math.sin(rl)
            ex, ey = SX - hx, SY - hy
            PX = SKULL_E[0] + (ex * c2 + ey * s2) / k
            PY = SKULL_E[1] + (-ex * s2 + ey * c2) / k
            ink = self.bust_sample(self.skull, PX, PY)
            scan = ((t - 7.3) * 520) % 1500 - 150
            bb = ink * (1.05 + 0.1 * math.sin(t * 9)) + 0.35 * ink * np.exp(-((PY - scan) / 22) ** 2)
            spin = t * 2.5 + (0.004 * ckey(t, TAPE_L) if t > 9.2 else 0.0)
            for (rx, ry), rr in REELS:
                d = np.hypot(PX - rx, PY - ry)
                a_ = np.arctan2(PY - ry, PX - rx) - spin
                spoke = (d < rr * 0.9) & (d > 11) & (np.cos(3 * a_) > 0.93)
                bb = np.where(spoke, 0.85, bb)
            on = burnt & (bb > 0.035)
            bb = np.clip(bb, 0, 1)
            b = np.where(on, bb, b)
            col = np.where(on[..., None], redmap(bb), col)
            solid |= burnt & (ink > 0.03)

        # fire: the pixel flames climb from the bottom, scrolling up, burn the face off, then die
        if 4.95 < t < 9.0:
            front = 1980 - 2200 * float(smoother(5.0, 6.6, t)) + 160 * (fbm(SX / 260, np.full(SX.shape, t * 0.6), 2, 3) - 0.5)
            live = (SY > front) & (t < self.T_burn + 0.4)
            if live.any():
                fh, fw, _ = self.fire.shape
                FY = SY / FIRE_K + 250 * float((t - 5.0) / 3.6)
                FX = (SX - 540) / FIRE_K + fw / 2 + 5 * np.sin(FY / 14 + t * 7)
                i = np.clip(FX.astype(int), 0, fw - 1); j = np.clip(FY.astype(int), 0, fh - 1)
                rgb = self.fire[j, i]
                hot = np.clip((rgb[..., 1] - 0.22) / 0.35, 0, 1) ** 0.8
                fb = np.clip((0.8 * rgb[..., 0] + 0.65 * hot) * (0.9 + 0.15 * np.sin(t * 12 + FY / 30)), 0, 1)
                fade = 1 - sstep(self.T_burn - 0.05, self.T_burn + 0.4, t)
                dark = live & (rgb[..., 0] < 0.28)
                fl = live & ~dark
                b = np.where(fl, np.maximum(fb * fade, b * (1 - fade)), b)
                col = np.where(fl[..., None], RED * (1 - hot[..., None]) + HOT * hot[..., None], col)
                b = np.where(dark, b * (1 - fade), b)
                gl = np.where(live & (fade > 0.3), 0, gl)
                solid |= live & (fade > 0.3)
            ember = (t > self.T_burn - 0.12) & (t < self.T_burn + 0.06)
            if ember.any():
                rng = np.random.default_rng(fi + 3)
                gl = np.where(ember, rng.choice(PRINTABLE, (FROWS, FCOLS)), gl)
                b = np.where(ember, 0.95, b)
                col = np.where(ember[..., None], np.where((self.rand > 0.5)[..., None], HOT, RED), col)

        # money glow: a red spotlight from the lower left while it hangs
        gm = float(sstep(10.2, 10.9, t)) * (1 - float(sstep(12.1, 12.5, t)))
        if gm > 0:
            glow = 0.55 * np.exp(-(((SX - 260) / 620) ** 2 + ((SY - 1350) / 780) ** 2)) * gm
            on = (b < 0.04) & ~solid & (glow > 0.11)
            jr = np.random.default_rng(fi + 7).integers(-1, 2, shape)
            gl = np.where(on, SOFT[np.clip((glow * 1.8 * (len(SOFT) - 1)).astype(int) + jr, 0, len(SOFT) - 1)], gl)
            b = np.where(on, glow, b); col = np.where(on[..., None], redmap(glow * 1.4), col)

        # the tape with the money on its end; at the snap the money drops and the tape springs back up
        pts, hang = self.tape(t)
        if pts is not None:
            X, Y = pts
            if t >= self.t_att:
                self.money_layer(b, col, gl, solid, t, SX, SY, hang)
            if t > T_SNAP:
                ye = self.snap_hang[1] - self.string_end(t)
                keep = Y < ye
                X, Y = X[keep], Y[keep]
                tau = t - T_SNAP
                if len(X) > 2:                                   # the frayed end curls as it recoils
                    curl = np.clip((Y - (ye - 60)) / 60, 0, 1) ** 2 * 25 * math.sin(tau * 23) * math.exp(-tau / 0.25)
                    X = X + curl
                fl = math.exp(-tau / 0.09)
                if fl > 0.05:                                    # the snap: a flash where it broke
                    d = np.hypot(SX - self.snap_hang[0], (SY - self.snap_hang[1]) * 0.8)
                    on = d < 90 * fl
                    b = np.where(on, np.maximum(b, fl * (1 - d / (90 * fl + 1e-6))), b)
                    col = np.where(on[..., None], PALE, col); gl = np.where(on & (d < 30), CI["*"], gl); solid |= on
            splat(gl, col, solid, X, Y, HOT, b, 0.95)
        return b, col, gl, solid

    # ---------- scene B: bars, the eye, the wall of eyes, the red square, the line
    def setup_wall(self, rng):
        eyes = [(0.0, 0.0, 1.0, 0.0)]
        tries = 0
        while len(eyes) < 40 and tries < 6000:
            tries += 1
            s = rng.uniform(0.55, 1.3)
            x, y = rng.uniform(-9, 9), rng.uniform(-15, 12)
            if all(math.hypot((x - ex) / (1.3 * (s + es)), (y - ey) / (0.85 * (s + es))) > 1.0 for ex, ey, es, _ in eyes):
                eyes.append((x, y, s, rng.uniform(-0.55, 0.55)))
        self.eyes = eyes
        dist = np.array([math.hypot(x, y) for x, y, _, _ in eyes])
        self.eye_open = 16.85 + 0.05 * dist
        self.eye_close = 20.05 + 0.9 * (1 - dist / dist.max())
        self.eye_ph = rng.uniform(0, 6.3, len(eyes))
        self.eye_blink = rng.uniform(18.3, 19.8, len(eyes))

    def cam_b(self, t):
        Z = math.exp(ckey(t, LOGZ))
        cx, cy = ckey(t, self.CXK), ckey(t, CY)
        return cx, cy, Z, ckey(t, ROLLB)

    def to_eye(self, SX, SY, cam):
        cx, cy, Z, r = cam
        dx, dy = SX - 540, SY - 960
        cr, sr = math.cos(r), math.sin(r)
        return cx + (dx * cr + dy * sr) / (EU * Z), cy - (-dx * sr + dy * cr) / (EU * Z)

    def from_eye(self, X, Y, cam):
        cx, cy, Z, r = cam
        rx, ry = (X - cx) * EU * Z, -(Y - cy) * EU * Z
        cr, sr = math.cos(r), math.sin(r)
        return 540 + rx * cr - ry * sr, 960 + rx * sr + ry * cr

    def hero_open(self, t):
        return 1 - float(smoother(self.eye_close[0], self.eye_close[0] + 0.2, t))

    def cell(self, X, Y, t, fi):
        """inside the prison: the snapped string, then iron bars slam down one by one, a crossbar, two fists"""
        shape = X.shape
        b = np.zeros(shape); col = np.zeros(shape + (3,), np.float32)
        gl = np.zeros(shape, np.int64); solid = np.zeros(shape, bool)
        Z = math.exp(ckey(t, LOGZ))
        px = 1.0 / (EU * Z)                                       # eye units per screen px
        # the string, still swinging back from the snap, until the first bar comes down over it
        if t < self.bar_t[0] + 0.3:
            ye = self.string_y0 + self.string_end(t) * self.string_k
            sw = 0.004 * math.sin(t * 9) * math.exp(-(t - T_SNAP) / 0.6)
            on = (np.abs(X - sw) < max(0.0016, 0.5 * FCW * px)) & (Y > ye)
            b = np.where(on, 0.9, b); col = np.where(on[..., None], HOT, col); gl = np.where(on, CI["|"], gl); solid |= on
        rr = max(BAR_R, 0.55 * FCW * px)
        for k in range(-8, 9):
            tk = self.bar_t[k + 8]
            if t < tk: continue
            xk = k * BAR_DX
            if np.abs(X - xk).min() > rr: continue
            tip = self.bar_tip(k, t)
            dxn = (X - xk) / rr
            nz = np.sqrt(np.clip(1 - dxn ** 2, 0, 1))
            cap = tip + rr * (1 - nz) * 0.9                       # rounded end
            on = (np.abs(dxn) < 1) & (Y > cap)
            if not on.any(): continue
            rust = fbm(Y * 22 + k * 7, dxn * 0.6 + k, 3, 41)
            diff = np.clip(-0.5 * dxn + 0.85 * nz, 0, 1)
            spec = np.exp(-((dxn + 0.32) / 0.13) ** 2)
            bb = np.clip((0.2 + 0.72 * diff) * (0.72 + 0.45 * rust) + 0.6 * spec * (rust > 0.35), 0, 1)
            falling = t < tk + BAR_FALL
            g = SHADE[np.clip((bb * len(SHADE)).astype(int), 0, len(SHADE) - 1)]
            g = np.where(spec > 0.5, CI["|"], g)
            b = np.where(on, bb, b); col = np.where(on[..., None], redmap(bb), col)
            col = np.where((on & (spec > 0.5))[..., None], np.array([1.0, 0.75, 0.68]), col)
            gl = np.where(on, g, gl); solid |= on
            if falling:                                           # speed lines beside a falling bar
                sl = ((np.abs(np.abs(X - xk) - rr * 1.7) < 0.6 * FCW * px) & (Y > tip + 0.05) &
                      (Y < tip + 0.35) & (self.rand > 0.35))
                b = np.where(sl, 0.45, b); col = np.where(sl[..., None], RED, col); gl = np.where(sl, CI["|"], gl); solid |= sl
        # impact: sparks and dust where each bar lands
        for k in range(-8, 9):
            tl = self.bar_t[k + 8] + BAR_FALL
            age = t - tl
            if not (0 <= age < 0.6): continue
            xk = k * BAR_DX
            if abs(xk - X.mean()) > (X.max() - X.min()): continue
            pr = self.spark_rng[k + 8]
            for (vx, vy, life) in pr:
                if age > life: continue
                xs = xk + vx * age
                ys = BAR_TIP + vy * age - 1.6 * age ** 2
                d = np.hypot((X - xs) / (FCW * px * 0.8), (Y - ys) / (FCH * px * 0.8))
                on = d < 1
                if on.any():
                    lv = 1 - age / life
                    b = np.where(on, np.maximum(b, lv), b); col = np.where(on[..., None], np.where(lv > 0.5, PALE, HOT), col)
                    gl = np.where(on, CI["*"] if lv > 0.5 else CI["."], gl); solid |= on
            dust = (np.abs(Y - BAR_TIP) < 0.012 + 0.05 * age) & (np.abs(X - xk) < 0.02 + 0.25 * age) & (self.rand < 0.5 * (1 - age / 0.6))
            b = np.where(dust & ~solid, 0.35, b); col = np.where((dust & ~solid)[..., None], RED, col)
            gl = np.where(dust & ~solid, CI["~"], gl); solid |= dust
        # the crossbar slams in on the hit
        xe = -1.6 + 3.1 * float(np.clip((t - 13.95) / 0.09, 0, 1)) ** 2
        if xe > -1.6:
            dyn = (Y + 0.24) / 0.014
            on = (np.abs(dyn) < 1) & (X < xe)
            nz = np.sqrt(np.clip(1 - dyn ** 2, 0, 1))
            bb = np.clip(0.2 + 0.6 * np.clip(0.55 * dyn + 0.8 * nz, 0, 1) + 0.35 * np.exp(-((dyn - 0.35) / 0.18) ** 2), 0, 1)
            b = np.where(on, bb, b); col = np.where(on[..., None], redmap(bb), col)
            gl = np.where(on & (np.abs(dyn - 0.35) < 0.2), CI["-"], np.where(on, 0, gl))
            solid |= on
        # two fists rise and grab the bars, the right one mirrored
        yf = -0.95 + 0.93 * float(np.clip((t - 13.65) / 0.38, 0, 1)) ** 0.5
        clench = float(smoother(13.9, 14.05, t))
        if t > 13.65:
            for xf, mir in ((0.0, 1.0), (BAR_DX, -1.0)):
                xl, yl = mir * (X - xf) / FS, (Y - yf) / FS
                m = (np.abs(xl) < 0.75) & (yl > -1.9) & (yl < 0.65)
                if not m.any(): continue
                o = np.stack([xl[m], yl[m], np.full(m.sum(), 2.0)], 1)
                dv = np.array([0.0, 0.0, -1.0])
                sdf = lambda a_, b_, c_: fist_sdf(a_, b_, c_, clench)
                tt, hit = march(sdf, o, dv, 0.5, 4.5, steps=40, eps=0.003)
                if not hit.any(): continue
                ph = o[hit] + tt[hit, None] * dv
                behind = (ph[:, 2] < 0) & (np.abs(ph[:, 0]) < BAR_R / FS)      # the bar is in front of the palm
                n = normals(sdf, ph)
                n[:, 0] *= mir
                q = ph + normals(sdf, ph) * 0.06
                ao = np.clip(sdf(q[:, 0], q[:, 1], q[:, 2]) / 0.06, 0, 1)
                lam = np.clip(n @ unit(-0.55, 0.6, 0.6), 0, 1)
                rim = np.clip(n @ unit(0.8, 0.2, -0.5), 0, 1) ** 2
                bb = np.clip((0.06 + 0.85 * lam ** 1.2) * (0.4 + 0.6 * ao) + 0.5 * rim, 0, 1)
                bb *= np.clip(1 + (ph[:, 1] + 0.4) * 0.8, 0.12, 1)               # the arm sinks into the dark
                ix = np.nonzero(m)
                hr, hc = ix[0][hit][~behind], ix[1][hit][~behind]
                bb = bb[~behind]
                b[hr, hc] = bb; col[hr, hc] = redmap(bb); gl[hr, hc] = 0; solid[hr, hc] = True
        return b, col, gl, solid

    def flesh(self, X, Y, t, Z=1.0):
        """skin between the eyes: red, mottled, scratched with ink like an etching"""
        n = fbm(X * 1.1, Y * 1.1, 3, 23)
        ridge = np.abs(fbm(X * 0.9 + 3, Y * 1.6, 3, 29) - 0.5)
        b = 0.44 + 0.26 * n + 0.3 * sstep(0.66, 0.8, fbm(X * 0.7 + t * 0.03, Y * 0.7, 2, 31))
        thin = min(1.0, 0.7 / Z)
        folds = ridge < 0.02 * thin
        hatch = ((np.sin(X * 46 + 7 * fbm(X * 0.7, Y * 0.5, 2, 91)) > 1 - 0.07 * thin) &
                 (fbm(X * 1.4 + 5, Y * 0.35, 2, 93) > 0.52))
        dark = folds | hatch
        return np.where(dark, 0.0, b), dark

    def paint_eye(self, xl, yl, op, gx, gy, t, rnd):
        """an eye in its own units: shaded sclera with veins, a fibrous iris, pupil, wet highlights,
        inked lids with a crease and curved lashes"""
        sx, y_low, y_up = lids(xl, op)
        inside = (np.abs(xl) < EX) & (yl > y_low) & (yl < y_up)
        b = np.zeros(xl.shape); col = np.zeros(xl.shape + (3,), np.float32)
        gl = np.zeros(xl.shape, np.int64); solid = np.zeros(xl.shape, bool)
        d = np.hypot(xl - gx, yl - gy)
        ball = np.clip(1 - (xl / 1.45) ** 2 - 0.4 * (yl / 0.7) ** 2, 0, 1)
        shadow = sstep(0.0, 0.18, y_up - yl)
        corner = sstep(0.0, 0.4, EX - np.abs(xl))
        scl = (0.5 + 0.42 * ball) * (0.4 + 0.6 * shadow) * (0.55 + 0.45 * corner)
        vein = (np.abs(fbm(xl * 5 + 7, yl * 5, 3, 61) - 0.5) < 0.013) & (np.abs(xl - gx) > 0.4)
        b = np.where(inside, np.where(vein, 0.5, scl), b)
        scol = PALE * corner[..., None] + HOT * (1 - corner[..., None])
        col = np.where(inside[..., None], np.where(vein[..., None], RED, scol), col)
        gl = np.where(inside, SHADE[np.clip((scl * 1.15 * len(SHADE)).astype(int), 0, len(SHADE) - 1)], gl)
        a = np.arctan2(yl - gy, xl - gx)
        fn = fbm(np.cos(a) * 5 + d * 6, np.sin(a) * 5 + d * 6, 3, 21)
        fib = 0.5 + 0.5 * np.sin(a * 70 + 9 * fn)
        iris = inside & (d < 0.36)
        ib = (0.16 + 0.34 * fib * (0.6 + 0.4 * fn)) * (1 - 0.85 * sstep(0.29, 0.36, d))
        ib = ib + 0.22 * np.exp(-((d - 0.2) / 0.025) ** 2)                      # collarette
        ib = ib * (0.55 + 0.45 * shadow)
        b = np.where(iris, ib, b); col = np.where(iris[..., None], redmap(ib * 1.3 + 0.12), col)
        gl = np.where(iris & (fib > 0.66) & (d > 0.15), stroke_glyph(np.arctan2(np.sin(a) / FCH, np.cos(a) / FCW)), gl)
        pup = inside & (d < 0.14)
        b = np.where(pup, 0.0, b); solid |= pup
        hl1 = np.hypot((xl - gx + 0.11) / 1.2, yl - gy - 0.12) < 0.065
        hl2 = np.hypot(xl - gx - 0.1, yl - gy + 0.08) < 0.03
        hl = inside & (hl1 | hl2)
        b = np.where(hl, 1.0, b); col = np.where(hl[..., None], PALE, col)
        gl = np.where(hl, CI["@"], gl)
        wl = inside & (yl - y_low < 0.035) & (np.abs(xl) < EX * 0.9)          # wet lower waterline
        b = np.where(wl, np.maximum(b, 0.75), b); col = np.where(wl[..., None], PALE, col)
        # inked lids: thick upper line, thin lower line, a crease above, curved lashes
        line = (np.abs(xl) < EX + 0.04) & ((np.abs(yl - y_up) < 0.045) | (np.abs(yl - y_low) < 0.025))
        crease = (np.abs(yl - (y_up + 0.16 * sx ** 0.5)) < 0.018) & (np.abs(xl) < EX * 0.9)
        lh = 0.22 * sx ** 0.6 * (0.45 + 0.55 * op)
        h = yl - y_up
        above = (h > 0) & (h < lh) & (np.abs(xl) < EX)
        lean = xl * 0.9 + 0.6 * h / np.maximum(lh, 1e-3) * np.sign(xl)       # lashes fan out and curl
        lash = above & (((xl - h * lean) * 15) % 1.0 < 0.3 - 0.2 * h / np.maximum(lh, 1e-3))
        hb = y_low - yl
        below = (hb > 0) & (hb < 0.07 * sx) & (((xl + hb * xl * 2) * 24) % 1.0 < 0.25)
        dark = line | crease | lash | below
        b = np.where(dark, 0.0, b); solid |= dark
        return b, col, gl, solid, inside | dark                             # skin shows between the lashes

    def scene_b(self, t, SX, SY, fi):
        shape = (FROWS, FCOLS)
        cam = self.cam_b(t)
        X, Y = self.to_eye(SX, SY, cam)
        Z = cam[2]
        if t < 24.3:
            b, folds = self.flesh(X, Y, t, Z)
            col = redmap(b); gl = np.zeros(shape, np.int64); solid = folds | (b > 0.03)
            look = float(smoother(17.9, 18.02, t))                      # "flaw in your design": they all look at you
            for i, (ex, ey, es, er) in enumerate(self.eyes):
                pxc, pyc = self.from_eye(np.array([ex]), np.array([ey]), cam)
                rad = 1.9 * es * EU * Z
                r0, r1 = int((pyc[0] - rad - FOY) / FCH), int((pyc[0] + rad - FOY) / FCH) + 2
                c0, c1 = int((pxc[0] - rad) / FCW), int((pxc[0] + rad) / FCW) + 2
                r0, r1, c0, c1 = max(r0, 0), min(r1, FROWS), max(c0, 0), min(c1, FCOLS)
                if r0 >= r1 or c0 >= c1: continue
                sl = (slice(r0, r1), slice(c0, c1))
                cs, sn = math.cos(-er), math.sin(-er)
                xl0, yl0 = (X[sl] - ex) / es, (Y[sl] - ey) / es
                xl, yl = xl0 * cs - yl0 * sn, xl0 * sn + yl0 * cs
                e = (xl / 1.5) ** 2 + (yl / 0.85) ** 2
                ring = (e > 1.05) & (e < 2.8) & (((e * 2.2 + 0.2 * fbm(xl * 2, yl * 2, 2, 71)) % 1.0) < 0.09 * min(1.0, 0.8 / (Z * es)))
                b[sl] = np.where(ring, 0.0, b[sl])
                if i == 0:
                    op = self.hero_open(t)
                    gx = 0.04 * math.sin(t * 0.6); gy = 0.02 * math.sin(t * 0.8)
                else:
                    op = float(smoother(self.eye_open[i], self.eye_open[i] + 0.25, t)) * (1 - float(smoother(self.eye_close[i], self.eye_close[i] + 0.2, t)))
                    blink = math.exp(-((t - self.eye_blink[i]) / 0.06) ** 2)
                    op *= 1 - blink
                    ph = self.eye_ph[i]
                    gx = (0.12 * math.sin(t * 0.7 + ph) + 0.05 * (math.sin(t * 3.1 + ph) > 0.7)) * (1 - look)
                    gy = 0.05 * math.sin(t * 0.5 + 2 * ph) * (1 - look)
                eb, ec, eg, es_, reg = self.paint_eye(xl, yl, op, gx, gy, t, self.rand[sl])
                if i == 0 and t < 22.5:
                    inside_h = (np.abs(xl) < EX) & (yl > lids(xl, op)[1]) & (yl < lids(xl, op)[2])
                    if inside_h.any():
                        cb, cc, cg, csd = self.cell(X[sl], Y[sl], t, fi)
                        eb = np.where(inside_h, cb, eb); ec = np.where(inside_h[..., None], cc, ec)
                        eg = np.where(inside_h, cg, eg); es_ = np.where(inside_h, csd, es_ & ~inside_h)
                b[sl] = np.where(reg, eb, b[sl]); col[sl] = np.where(reg[..., None], ec, col[sl])
                gl[sl] = np.where(reg, eg, gl[sl]); solid[sl] = np.where(reg, es_ | (eb > 0.03), solid[sl])
            col = np.where((b > 0.03)[..., None] & (col.max(-1, keepdims=True) == 0), redmap(b), col)
        else:
            b = np.zeros(shape); col = np.zeros(shape + (3,), np.float32)
            gl = np.zeros(shape, np.int64); solid = np.zeros(shape, bool)

        # the wall of closed eyes gives way, in code, to hands holding a closed ring box
        if t > 22.6:
            rb, rc, rg, rsolid = self.box_scene(SX, SY, t)
            sw = 22.85 + 0.8 * np.clip(np.hypot(SX - 540, (SY - 960) * 1.2) / 1100, 0, 1) + 0.25 * self.rand
            on = t >= sw
            b = np.where(on, rb, b); col = np.where(on[..., None], rc, col)
            gl = np.where(on, rg, gl); solid = np.where(on, rsolid, solid)
            dec = (t > sw - 0.09) & (t < sw)
            rng = np.random.default_rng(fi + 11)
            gl = np.where(dec, rng.choice(PRINTABLE, shape), gl)
            b = np.where(dec, 0.85, b); col = np.where(dec[..., None], HOT, col); solid |= dec
        return b, col, gl, solid

    # ---------- the ending: a ring box (from a reference) opens, an Esc key inside; it's pressed, the screen goes off
    def setup_box(self):
        here = os.path.dirname(os.path.abspath(__file__))
        img = np.asarray(Image.open(os.path.join(here, "assets", "esc_box.png")).convert("RGB"), np.float32) / 255.0
        self.box_img = img
        # the plate: the open lid painted out, so the lid can be drawn opening over it
        plate = img.copy()
        x0, x1, y0, y1 = LID_X[0] - 6, LID_X[1] + 6, LID_TOP - 4, HINGE + 3
        reg = plate[y0:y1, x0:x1]
        red = (reg.max(-1) - reg.min(-1)) > 0.2
        reg[red] = 0.1
        self.box_plate = plate

    def box_cam(self, t):
        cx, cy = ckey(t, BOXCAM_X), ckey(t, BOXCAM_Y)
        z = math.exp(ckey(t, BOXCAM_Z)) * 1080 / self.box_img.shape[1]     # screen px per image px
        return cx, cy, z, 0.02 * math.sin(t * 0.8)

    def lid_angle(self, t):
        x = (t - T_OPEN) / 0.42
        if x <= 0: return 0.0
        if x < 1: return math.pi / 2 * (1 - (1 - x) ** 3)
        a = t - T_OPEN - 0.42
        return math.pi / 2 + 0.07 * math.exp(-a / 0.12) * math.sin(a * 30)

    def box_scene(self, SX, SY, t):
        shape = SX.shape
        cx, cy, z, rl = self.box_cam(t)
        c, s = math.cos(rl), math.sin(rl)
        dx, dy = SX - 540, SY - 960
        AX = cx + (dx * c + dy * s) / z
        AY = cy + (-dx * s + dy * c) / z
        # the key goes down when it's pressed
        press = float(smoother(T_PRESS, T_PRESS + 0.07, t))
        on_key = (AX > KEY_BOX[0]) & (AX < KEY_BOX[1]) & (AY > KEY_BOX[2]) & (AY < KEY_BOX[3])
        rgb = sample(self.box_plate, AX, np.where(on_key, AY - 11 * press, AY))
        rgb = np.where((on_key & (press > 0))[..., None], rgb * (1 - 0.18 * press), rgb)
        # the lid: closed it lies over the opening (we see its top), opening it swings up on its hinge
        th = self.lid_angle(t)
        ye = HINGE + LID_D * math.cos(th) - LID_H * math.sin(th)
        if ye > HINGE + 1:                                       # its outside
            m = (AX > BASE_X[0]) & (AX < BASE_X[1]) & (AY > HINGE) & (AY < ye)
            f = np.clip((AY - HINGE) / (ye - HINGE), 0, 1)
            face = np.stack([0.86 - 0.12 * f, 0.1 + 0.04 * (1 - f), 0.15 + 0.03 * (1 - f)], -1)
            front = m & (AY > ye - 16)
            face = np.where(front[..., None], np.array([0.62, 0.06, 0.1]), face)
            edge = m & ((np.abs(AX - BASE_X[0]) < 2.5) | (np.abs(AX - BASE_X[1]) < 2.5) | (np.abs(AY - ye) < 2.5) |
                        (np.abs(AY - (ye - 16)) < 1.5))
            face = np.where(edge[..., None], 0.05, face)
            rgb = np.where(m[..., None], face, rgb)
        elif ye < HINGE - 1:                                     # its inside, squashed until it's fully up
            m = (AX > LID_X[0]) & (AX < LID_X[1]) & (AY > ye) & (AY < HINGE)
            IY = HINGE - (HINGE - AY) * (HINGE - LID_TOP) / (HINGE - ye)
            lid = sample(self.box_img, AX, IY)
            rgb = np.where(m[..., None], lid, rgb)
        # light spills out of the opened box and pulses with the heart
        o = math.sin(min(th, math.pi / 2))
        pulse = sum(math.exp(-(t - tb) / 0.22) for tb, _ in BEATS if t >= tb)
        d = np.hypot(AX - KEY_C[0], (AY - KEY_C[1]) * 1.3)
        glow = o * (0.5 + 0.5 * pulse) * np.exp(-d / 85) * (1 - press * 0.6)
        mx = rgb.max(-1); mn = rgb.min(-1)
        b = np.clip((mx - 0.13) / 0.8, 0, 1) ** 1.05
        red = (mx - mn) > 0.22
        white = mn > 0.72
        col = np.where(red[..., None], rgb / (mx[..., None] + 0.02), redmap(b * 0.95))
        col = np.where(white[..., None], PALE, col)
        b = np.clip(b + 0.6 * glow * (on_key | (b < 0.2)), 0, 1)
        col = np.where(((glow > 0.12) & ~red & ~white)[..., None], HOT, col)
        # soften the edges of the picture into the dark
        h, w = self.box_img.shape[:2]
        edge = np.clip(np.minimum.reduce([AX, w - AX, AY, h - AY]) / 45, 0, 1)
        b = b * edge
        solid = b > 0.05
        return np.where(solid, b, 0.0), col.astype(np.float32), np.zeros(shape, np.int64), solid

    def crt_off(self, idx, col, t):
        """Esc: the picture collapses to a line, the line to a dot, the dot goes out"""
        a = float(smoother(T_OFF, T_OFF + 0.2, t))
        bx = float(smoother(T_OFF + 0.2, T_OFF + 0.36, t))
        if a <= 0:
            return idx, col
        sy, sx = max(1 - a, 0.004), max(1 - bx, 0.004)
        rr = np.arange(FROWS); cc = np.arange(FCOLS)
        r0, c0 = (FROWS - 1) / 2, (FCOLS - 1) / 2
        rs = np.round(r0 + (rr - r0) / sy).astype(int); cs = np.round(c0 + (cc - c0) / sx).astype(int)
        okr = (rs >= 0) & (rs < FROWS); okc = (cs >= 0) & (cs < FCOLS)
        ni = np.zeros_like(idx); nc = np.zeros_like(col)
        sub_i = idx[np.clip(rs, 0, FROWS - 1)][:, np.clip(cs, 0, FCOLS - 1)]
        sub_c = col[np.clip(rs, 0, FROWS - 1)][:, np.clip(cs, 0, FCOLS - 1)]
        ok = okr[:, None] & okc[None, :]
        ni[ok] = sub_i[ok]; nc[ok] = sub_c[ok]
        boost = 1 + 2.5 * a + 2 * bx
        nc = np.clip(nc * boost + (a > 0.7) * 0.3, 0, 1)
        fade = 1 - float(smoother(T_OFF + 0.3, T_OFF + 0.55, t))
        return ni, (nc * fade).astype(np.float32)

    # ---------- compose
    def lyric_overlay(self, img, t):
        if not self.lyrics and t < DUR - 1.1:
            return img
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        items = LYRICS if self.lyrics else []
        items = items + [(29.6, DUR + 1, ">>> " + self.handle)] if self.handle else items
        for k, (t0, t1, txt) in enumerate(items):
            if not (t0 <= t < t1 + 0.15): continue
            title = k == 0 and self.lyrics
            f = mono(56 if title else 38)
            s, typing = typed(txt, t, t0, 16 if title else 26)
            fade = 1 - float(sstep(t1, t1 + 0.15, t))
            shown = s + ("█" if typing or (txt.startswith(">>>") and (t * 2.4) % 2 < 1) else "")
            w = f.getlength(txt)
            y = 1560 if title else 1690
            c = (255, 205, 195) if not txt.startswith(">>>") else (255, 95, 75)
            dr.text(((W - w) / 2, y), shown, font=f, fill=tuple(int(v * fade) for v in c),
                    stroke_width=4, stroke_fill=(0, 0, 0))
        return np.asarray(im)

    def frame(self, t, fi):
        # one handheld camera float plus a kick on the heaviest hits, shared by every scene
        fx = 7 * math.sin(t * 0.7) + 4 * math.sin(t * 1.9 + 1)
        fy = 6 * math.sin(t * 0.9 + 2) + 3 * math.sin(t * 2.3)
        for th, amp in [(h, 14.0) for h in HITS] + self.impacts:
            if 0 <= t - th < 0.5:
                k = math.exp(-(t - th) / 0.12)
                fx += amp * k * math.sin((t - th) * 70); fy += amp * 0.75 * k * math.cos((t - th) * 61)
        SX, SY = FGX + fx, FGY + fy
        if t < T_B:
            b, col, gl, solid = self.scene_a(t, SX, SY, fi)
        else:
            b, col, gl, solid = self.scene_b(t, SX, SY, fi)
        idx = np.where(gl > 0, gl, glyphs(np.clip(b, 0, 1), int(t * 12), (FROWS, FCOLS)))
        col = np.clip(col * np.clip(0.4 + 0.9 * b, 0, 1)[..., None], 0, 1)
        empty = ~solid & (b < 0.035) & (gl == 0)
        cidx, ccol = self.code_layer(t, fi)
        pulse = sum(math.exp(-(t - tb) / 0.2) for tb, _ in BEATS if t >= tb)     # the heartbeat lights the room
        ccol = ccol * (1 + 2.5 * pulse)
        idx = np.where(empty, cidx, idx); col = np.where(empty[..., None], ccol, col)
        idx, col = self.crt_off(idx, col.astype(np.float32), t)
        img = render_fine(idx, col.astype(np.float32))
        return self.lyric_overlay(img, t)


def export(path, music, handle, lyrics):
    film = Film(handle, lyrics)
    ff = ffmpeg_bin()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if music:
        cmd += ["-t", str(DUR), "-i", music, "-map", "0:v", "-map", "1:a",
                "-af", f"afade=t=out:st={DUR - 0.6}:d=0.6", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", "-shortest", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for fi in range(n):
        p.stdin.write(film.frame(fi / FPS, fi).tobytes())
        if fi % 15 == 0 or fi == n - 1:
            print(f"\rrendering {fi + 1}/{n}", end="", flush=True)
    p.stdin.close(); p.wait()
    print(f"\nsaved {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="curse_of_being_human.mp4")
    ap.add_argument("--music", default=None, help="the spoken-word track (an audio file or a video with it)")
    ap.add_argument("--handle", default="@rony_matar")
    ap.add_argument("--no-lyrics", action="store_true")
    a = ap.parse_args()
    export(a.export, a.music, a.handle, not a.no_lyrics)
