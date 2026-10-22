"""
The Curse of Being Human  |  9:16, red and black, every frame is text, cut to a spoken-word track.
One continuous camera, no cuts: each scene turns into the next.

the curse of being human: red code types across the screen and the code is ultron (from a reference),
his eyes glowing; he leans in toward us, swaying, his head turning and nodding -> yet manufacture chaos: his eyes flare, and from them pixel flames
(from a reference) spread out over him ->
they fear death: he burns away in sparks to an x-ray skull with a cassette for a brain (from a reference) ->
yet waste their lives: the tape unspools, the camera follows it down to money on a string (from a reference) ->
they demand freedom: the string snaps, the money falls and the camera falls after it, down into a cell
(from a reference photo) -> then construct their: the hands rattle the bars ->
own prisons: pull back, the prison is inside an eye -> perhaps the greatest flaw in your design: the eye is
one of a wall of eyes darting about, blinking, then all turning to look at you -> is not your mortality: they look
away again -> it is your extraordinary ability: they close, the camera comes back to the eye with the prison in it
and it closes too; its lid line becomes the seam of a closed ring box (from a reference, etched with the same
skin texture so the cut is seamless), and pulling back, two hands hold the box -> to remain alive: the fingers come down on the lid and on the hit lift it, the box opening
like the eye did, an Esc key inside, its light pulsing with the heart -> without ever truly living: it is never
pressed; the hands and the box go dark, the key's light last.

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
    "ash = burn(bust)", "money.hang(string)", "bars.extend(hands)",
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

# ------------------------------------------------------------------ scene A camera (photo px)
# the bust photo: screen px per photo px, where it sits, the point between the eyes
# the head is already moving at the first frame: it drifts in from up and to the left, tilted, and
# settles into its pose by "Humans pursue peace"
# ultron (reference image), on a locked camera: screen px per image px, the image px at the screen's centre,
# the neck his head turns on, his eyes
ULT_K, ULT_C = 3.5, (240.0, 285.0)
ULT_NECK = (262.0, 245.0)
ULT_LEAN, ULT_HIP = (235.0, 170.0), (300.0, 620.0)   # he leans in toward us about his face, sways about his hips
ULT_EYES = [(243.0, 120.0), (191.0, 156.0)]
ULT_HEAD = [(165, 100), (178, 80), (200, 66), (236, 60), (258, 68), (278, 86), (300, 88), (300, 130), (170, 130)]
# "yet manufacture chaos": his eyes flare and the fire spreads out from them
T_SPREAD = 5.05
FIRE_O = (460.0, 445.0)                          # between his eyes, on screen
# the x-ray: screen px per image px, the head's centre on screen, the reels, where the drawn tape ends
SKULL_K = [(6.8, 1.2), (7.4, 1.25), (9.2, 1.5), (12.6, 1.5)]
SKULL_HY = [(6.8, 520), (7.4, 545), (9.2, 860), (12.6, 880)]
SKULL_E = (300.0, 240.0)
REELS = [((243.0, 120.0), 40.0), ((368.0, 146.0), 62.0)]
SKULL_DOME = (308.8, 179.5, 296.8, 203.0)        # the ellipse of the skull's outline, for the top the photo crops off
TAPE0 = (250.0, 415.0)
TAPE_L = [(9.2, 0), (9.45, 400), (9.7, 1200), (9.95, 2080), (10.15, 2260), (10.4, 2140), (10.7, 2190), (11.2, 2175), (12.6, 2175)]
FIRE_K = 2.0                                     # fire pattern, screen px per image px
# the money photo: the string's top, the point that ends up at the centre of the screen, scale and tilt
MONEY_KNOT, MONEY_C = (159.0, 0.0), (200.0, 350.0)   # money photo px: the string's top, the stack's middle
K_M0 = 1.9                                       # screen px per photo px
T_SNAP, MONEY_G = 11.83, 5200.0                  # "They demand freedom": the string snaps, the money drops

def dome(img, pad=40):
    """the x-ray's photo cuts the top of the skull off flat: carry its outline on up and give it back its dome"""
    h, w = img.shape
    cx, cy, rx, ry = SKULL_DOME
    yy, xx = np.mgrid[-pad:0, 0:w]
    inside = (1 - np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)) * ry     # px in from the outline
    top = np.convolve(img[0:4].mean(0), np.ones(9) / 9, "same")[None, :]
    cap = top * (1 + 0.2 * np.exp(-np.maximum(inside, 0) / 6)) * np.clip(inside / 8, 0, 1)
    return np.concatenate([cap.astype(np.float32), img]), pad

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
LOGZ = [(12.6, math.log(3.5)), (13.4, math.log(3.75)), (14.3, math.log(3.95)), (15.4, math.log(1.9)),
        (16.7, math.log(1.0)), (17.6, math.log(0.5)), (18.8, math.log(0.36)), (20.0, math.log(0.33)),
        (21.5, math.log(0.31)), (22.6, math.log(0.34)), (23.3, math.log(0.62)), (24.0, math.log(1.75)),
        (24.45, math.log(2.3)), (24.85, math.log(2.38))]
CX = [(12.6, 0.3), (14.4, 0.4), (16.0, 0.14), (17.6, 0.0), (19.0, 0.8), (20.5, 0.3), (21.8, -0.6),
      (22.8, -0.35), (23.6, -0.03), (24.2, 0.0), (24.85, 0.0)]
CY = [(12.6, 1.3), (12.9, 0.95), (13.3, 0.3), (13.8, 0.03), (14.4, 0.09), (16.0, 0.04), (17.6, 0.2), (19.0, 0.9), (20.5, -0.4), (21.8, -0.9),
      (22.8, -0.6), (23.6, 0.0), (24.0, 0.06), (24.4, -0.38), (24.85, -0.43)]
ROLLB = [(12.6, 0), (13.6, 0.025), (14.6, -0.02), (16.0, 0.04), (17.6, 0.1), (19.5, -0.06), (21.0, 0.08), (22.6, 0.02),
         (23.6, 0.0), (30.55, 0.0)]
# the eye with bars (reference photo): photo px per eye unit, the photo px at the eye's centre
PRI_U, PRI_E = 400.0, (690.0, 774.0)
PRI_BAR = 0.3                                    # its middle bar, in eye units
T_REVEAL = 12.72                                 # the camera reaches the cell: dust starts to show in its light
BEATS = [(26.5, 0.8), (27.45, 0.48)]             # heart hits: the box's light pulses
# the ring box (reference image px): lid hinge, how deep it lies closed, how tall it stands open, its sides
HINGE, LID_D, LID_H, LID_TOP = 510, 135, 280, 230
LID_X, BASE_X = (372, 612), (368, 630)
KEY_BOX, KEY_C = (395, 600, 505, 645), (497, 560)
T_OPEN = 26.45                                   # the lid opens on the hit
T_FADE = (29.55, 30.15, 30.5)                    # the end: the picture goes dark, the Esc key's light last
LID_T = 0.95                                     # how long the lid takes to open
FING_TIP = 252                                   # the two fingers on the lid: where their tips rest, and their outline
FING_POLY = [(262, 30), (520, 30), (578, 180), (578, 296), (372, 302), (282, 135), (264, 75)]
# the match cut: the last eye closes, its lid line becomes the seam of the closed box (bent like the lid at
# first, then straight), the camera pulls back to the hands holding it; it opens like the eye did
MCUT = (24.45, 24.85)
BOXCAM_X = [(24.45, 499), (24.85, 499), (25.6, 498), (26.3, 497), (27.4, 496), (28.4, 494), (30.55, 497)]
BOXCAM_Y = [(24.45, 645), (24.85, 646), (25.6, 700), (26.3, 744), (27.4, 600), (28.4, 525), (29.5, 545), (30.55, 565)]
BOXCAM_Z = [(24.45, math.log(6.9)), (24.85, math.log(6.95)), (25.6, math.log(4.9)), (26.3, math.log(3.8)), (27.4, math.log(2.45)),
            (28.4, math.log(1.45)), (29.5, math.log(1.75)), (30.55, math.log(2.2))]


class Film:
    def __init__(self, handle="@rony_matar", lyrics=True, seed=5):
        rng = np.random.default_rng(seed)
        self.handle, self.lyrics = handle, lyrics
        self.rand = rng.random((FROWS, FCOLS))
        here = os.path.dirname(os.path.abspath(__file__))
        self.ultron = np.asarray(Image.open(os.path.join(here, "assets", "ultron.png")).convert("RGB"), np.float32) / 255.0
        # the flat red field behind him (flood filled from the edges), so the code can be the field and he can be shaded
        u = self.ultron
        red = Image.fromarray(((u[..., 0] > 0.75) & (u[..., 1] < 0.3)).astype(np.uint8) * 255)
        fld = red.filter(ImageFilter.MinFilter(9))                       # cut the thin bridges into his head first
        for xy in [(0, 0), (0, 200), (0, 400), (200, 0), (420, 0)]:
            if fld.getpixel(xy) == 255:
                ImageDraw.floodfill(fld, xy, 128)
        fld = Image.fromarray(((np.asarray(fld) == 128) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
        fld = Image.fromarray(np.minimum(np.asarray(fld), np.asarray(red)))
        ImageDraw.Draw(fld).polygon(ULT_HEAD, fill=0)                     # his red crest is him, not the field
        self.ult_bg = np.asarray(fld.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0
        fig = Image.fromarray(255 - np.asarray(fld))
        self.ult_near = np.asarray(fig.filter(ImageFilter.GaussianBlur(7)), np.float32) / 255.0
        self.fire = np.asarray(Image.open(os.path.join(here, "assets", "fire.png")).convert("RGB"), np.float32) / 255.0
        self.skull, self.skull_pad = dome(np.asarray(Image.open(os.path.join(here, "assets", "skull_xray.png")).convert("L"), np.float32) / 255.0)
        self.money = np.asarray(Image.open(os.path.join(here, "assets", "money.png")).convert("RGBA"), np.float32) / 255.0
        self.prison = np.asarray(Image.open(os.path.join(here, "assets", "eye_bars.png")).convert("L"), np.float32) / 255.0
        self.build_code(rng)
        # the fire eats him from his eyes outward
        dfire = np.hypot(FGX - FIRE_O[0], (FGY - FIRE_O[1]) * 0.9)
        self.T_burn = 6.75 + 1.45 * np.clip(dfire / 1500, 0, 1) + 0.35 * (fbm(FGX / 150, FGY / 150, 3, 19) - 0.5)
        self.setup_tape()
        self.setup_embers(rng)
        self.setup_prison()
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
        tf = tt.copy()
        for r in range(FROWS):
            fin = np.isfinite(tf[r])
            if fin.any():
                k = np.maximum.accumulate(np.where(fin, np.arange(FCOLS), -1))
                tf[r] = np.where(k >= 0, tf[r][np.maximum(k, 0)], tf[r][np.argmax(fin)])
        self.code_tf = tf
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
    def bust_sample(self, img, PX, PY):
        h, w = img.shape
        ok = (PX >= 0) & (PX < w - 1) & (PY >= 0) & (PY < h - 1)
        x0 = np.clip(PX, 0, w - 1.001); y0 = np.clip(PY, 0, h - 1.001)
        i, j = x0.astype(int), y0.astype(int)
        fx, fy = x0 - i, y0 - j
        v = (img[j, i] * (1 - fx) * (1 - fy) + img[j, i + 1] * fx * (1 - fy) +
             img[j + 1, i] * (1 - fx) * fy + img[j + 1, i + 1] * fx * fy)
        return np.where(ok, v, 0.0)

    def setup_prison(self):
        # the string the money hung from lines up with the middle bar in the photo
        _, self.snap_hang = self.tape(T_SNAP)
        X, Y = self.tape(T_B)[0]
        vis = (Y > 0) & (Y < H)
        sx = float(X[vis].mean()) if vis.any() else self.snap_hang[0]
        Z0 = math.exp(ckey(T_B, LOGZ))
        cx0 = PRI_BAR - (sx - 540) / (EU * Z0)
        self.CXK = [(T_B, cx0), (13.6, PRI_BAR)] + [k for k in CX if k[0] >= 14.4]
        cy0 = ckey(T_B, CY)
        self.string_y0 = cy0 - (self.snap_hang[1] + self.pan_since_snap(T_B) - 960) / (EU * Z0)
        self.string_k = 1.0 / (EU * Z0)
        self.impacts = [(T_SNAP, 6.0)]
        rng = np.random.default_rng(23)
        n = 140
        self.dust = dict(x=rng.uniform(-0.3, 0.95, n), y=rng.uniform(-0.6, 0.6, n), vx=rng.normal(0.01, 0.02, n),
                         vy=rng.uniform(-0.05, 0.02, n), ph=rng.uniform(0, 6.3, n))

    def setup_tape(self):
        # the tape hangs straight; once the camera has panned, the money hangs in the middle of the frame
        k = ckey(11.5, SKULL_K)
        off = (MONEY_C[0] - MONEY_KNOT[0], MONEY_C[1] - MONEY_KNOT[1])
        hxf = 540 - off[0] * K_M0 - (TAPE0[0] - SKULL_E[0]) * k
        self.HXS = [(6.8, 540), (8.6, 545), (10.6, hxf), (12.6, hxf)]
        self.PAN = [(0, 0)]
        y0 = ckey(11.5, SKULL_HY) + (TAPE0[1] - SKULL_E[1]) * k
        P = 960 - (y0 + TAPE_L[-1][1] + off[1] * K_M0)
        # ... and after the snap the camera keeps going down after the falling money
        self.PAN = [(0, 0), (9.35, 0), (9.9, 0.12 * P), (10.6, 0.58 * P), (11.3, 0.93 * P), (T_SNAP, P),
                    (12.25, P - 330), (T_B, P - 780)]
        # the money only exists on the tape once its end is out of frame
        self.t_att = 12.0
        for tt in np.arange(9.2, 10.4, 0.01):
            _, hang = self.tape(tt)
            if hang is not None and hang[1] > 2010:
                self.t_att = tt; break

    def setup_embers(self, rng):
        """sparks and ash from burning ultron: start points on him, at the locked camera"""
        lit = self.ultron[..., 0] < 0.35
        ys, xs = np.nonzero(lit)
        pick = rng.choice(len(xs), 1400, replace=False)
        sx = 540 + (xs[pick] - ULT_C[0]) * ULT_K
        sy = 960 + (ys[pick] - ULT_C[1]) * ULT_K
        r = np.clip(((sy - FOY) / FCH).astype(int), 0, FROWS - 1); c = np.clip((sx / FCW).astype(int), 0, FCOLS - 1)
        self.emb = dict(x=sx, y=sy, t0=self.T_burn[r, c] + rng.uniform(-0.05, 0.15, len(sx)),
                        vx=rng.normal(0, 45, len(sx)), vy=-rng.uniform(70, 260, len(sx)),
                        life=rng.uniform(0.7, 1.9, len(sx)), ph=rng.uniform(0, 6.3, len(sx)))

    def embers(self, b, col, gl, solid, t):
        e = self.emb
        age = t - e["t0"]
        live = (age > 0) & (age < e["life"])
        if live.any():
            a = age[live]; f = 1 - a / e["life"][live]
            x = e["x"][live] + e["vx"][live] * a + 18 * np.sin(a * 3 + e["ph"][live])
            y = e["y"][live] + e["vy"][live] * a + 35 * a * a
            r = ((y - FOY) / FCH).astype(int); c = (x / FCW).astype(int)
            ok = (r >= 0) & (r < FROWS) & (c >= 0) & (c < FCOLS)
            r, c, f = r[ok], c[ok], f[ok]
            b[r, c] = np.maximum(b[r, c], 0.3 + 0.7 * f)
            col[r, c] = np.where((f > 0.55)[:, None], HOT, RED)
            gl[r, c] = np.where(f > 0.7, CI["*"], np.where(f > 0.35, CI["'"], CI["."]))
            solid[r, c] = True

    def skull_cam(self, t):
        return ckey(t, SKULL_K), ckey(t, self.HXS), ckey(t, SKULL_HY) + ckey(t, self.PAN), 0.0

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
            cy += 0.5 * MONEY_G * tau ** 2 + 120 * tau + self.pan_since_snap(t)
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

    def pan_since_snap(self, t):
        return ckey(t, self.PAN) - ckey(T_SNAP, self.PAN) if t > T_SNAP else 0.0

    def string_end(self, t):
        """after the snap the tape's end springs back up: how far, in screen px"""
        tau = t - T_SNAP
        if tau <= 0: return 0.0
        return 120 * (1 - math.exp(-tau / 0.09) * math.cos(tau * 19))

    def scene_a(self, t, SX, SY, fi):
        shape = (FROWS, FCOLS)
        b = np.zeros(shape); col = np.zeros(shape + (3,), np.float32)
        gl = np.zeros(shape, np.int64); solid = np.zeros(shape, bool)
        # the camera is locked on ultron
        hm = 1 - float(smoother(6.5, 7.6, t))
        burnt = t >= self.T_burn

        # ultron: the code types him in; his eyes glow, his head turns slowly, looking about, like a man's would
        if t < 8.9:
            PX = ULT_C[0] + (SX - 540) / ULT_K
            PY = ULT_C[1] + (SY - 960) / ULT_K
            # he leans in toward us the whole time, and his body sways a little, like someone standing there
            lean = 1 + 0.12 * float(smoother(0.0, 6.6, t))
            PX = ULT_LEAN[0] + (PX - ULT_LEAN[0]) / lean
            PY = ULT_LEAN[1] + (PY - ULT_LEAN[1]) / lean
            sway = hm * 0.022 * math.sin(t * 0.6 + 0.3)
            qx, qy = PX - ULT_HIP[0], PY - ULT_HIP[1]
            PX = ULT_HIP[0] + qx * math.cos(sway) - qy * math.sin(sway)
            PY = ULT_HIP[1] + qx * math.sin(sway) + qy * math.cos(sway)
            hw = np.clip((255 - PY) / 45, 0, 1) * np.clip((350 - PX) / 40, 0, 1)      # the head, not the body
            tilt = hm * (0.08 * math.sin(t * 0.8 + 1.0) + 0.03 * math.sin(t * 1.7 + 0.4))
            ang = -tilt * hw
            qx, qy = PX - ULT_NECK[0], PY - ULT_NECK[1]
            PX = ULT_NECK[0] + qx * np.cos(ang) - qy * np.sin(ang)
            PY = ULT_NECK[1] + qx * np.sin(ang) + qy * np.cos(ang)
            PX = PX - hm * hw * 6.0 * math.sin(t * 0.45 + 2.0)                     # a slow turn and nod
            PY = PY - hm * hw * 6.0 * math.sin(t * 0.8 + 0.5)
            PY = PY - hm * 2.5 * math.sin(t * 1.2) * (1 - hw)                     # breathing
            h, w = self.ultron.shape[:2]
            rgb = sample(self.ultron, np.clip(PX, 0, w - 1.01), np.clip(PY, 0, h - 1.01))
            fld = sample(self.ult_bg, np.clip(PX, 0, w - 1.01), np.clip(PY, 0, h - 1.01))
            near = sample(self.ult_near, np.clip(PX, 0, w - 1.01), np.clip(PY, 0, h - 1.01))
            fb = np.clip((rgb[..., 0] - 0.05) / 0.8, 0, 1) ** 0.85                  # him: shaded, darks pushed down
            # the field: a wall of red code that fades out softly around him, so he stands in front of it
            ub = fld * 0.62 * np.clip((0.5 - near) / 0.38, 0, 1) ** 1.2 + (1 - fld) * fb
            white = rgb.min(-1) > 0.55
            ub = np.where(white, 1.0, ub)
            ucol = np.where(white[..., None], PALE, np.where((fld > 0.5)[..., None], RED, redmap(np.minimum(fb * 1.15, 1))))
            # his eyes burn brighter and brighter, then flare: the fire starts there
            flare = float(smoother(4.55, 5.0, t)) * math.exp(-max(t - 5.0, 0.0) / 0.35)
            de = np.minimum.reduce([np.hypot(PX - ex, PY - ey) for ex, ey in ULT_EYES])
            core = np.exp(-de / (6.0 + 2.0 * flare)) * (1.1 + 0.2 * math.sin(t * 2.2) + 0.8 * flare)
            halo = flare * np.exp(-de / 9.0)
            streak = flare * np.maximum.reduce([np.exp(-np.abs(PY - ey) / 2.0 - np.abs(PX - ex) / (20 + 40 * flare))
                                                for ex, ey in ULT_EYES])          # a flare of light across each eye
            ub = np.maximum(ub, np.clip(np.maximum(core, streak), 0, 1))
            ub = np.where(halo > 0.12, np.maximum(ub, 0.25 + 0.6 * halo), ub)
            ucol = np.where((np.maximum(core, streak) > 0.25)[..., None], PALE,
                            np.where((halo > 0.12)[..., None], RED * 0.5 + HOT * 0.5, ucol))
            hot = np.maximum.reduce([core, halo, streak])
            # typed in with the code: a cell shows him once its line of code has reached it
            ci = (np.arange(FCOLS)[None, :] + np.floor(self.code_v * t).astype(int)[:, None]) % FCOLS
            seen = t >= np.take_along_axis(self.code_tf, ci, 1)
            code = np.take_along_axis(self.code_idx, ci, 1)
            on = seen & ~burnt
            fon = on & (fld > 0.5)
            gap = fon & (code == 0)                                                   # spaces in the code stay spaces
            b = np.where(on & ~gap, ub, b); col = np.where(on[..., None], ucol, col)
            gl = np.where(fon & (code > 0) & (hot < 0.25), code, gl)
            solid |= on & ~fon

        # the x-ray, revealed where the fire has burnt through; the reels spin, a scan line runs down it
        if t > 7.0:
            k, hx, hy, rl = self.skull_cam(t)
            c2, s2 = math.cos(rl), math.sin(rl)
            ex, ey = SX - hx, SY - hy
            PX = SKULL_E[0] + (ex * c2 + ey * s2) / k
            PY = SKULL_E[1] + (-ex * s2 + ey * c2) / k
            ink = self.bust_sample(self.skull, PX, PY + self.skull_pad)
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

        # fire: from his eyes the pixel flames spread out over him, scrolling up, burn him away, then die
        if T_SPREAD < t < 9.0:
            dF = np.hypot(SX - FIRE_O[0], (SY - FIRE_O[1]) * 0.9)
            R = 2100 * float(smoother(T_SPREAD, T_SPREAD + 1.6, t)) + 170 * (fbm(SX / 260, SY / 260 + t * 0.6, 2, 3) - 0.5)
            live = (dF < R) & (t < self.T_burn + 0.4)
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

        self.embers(b, col, gl, solid, t)

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
                ye = self.snap_hang[1] + self.pan_since_snap(t) - self.string_end(t)
                keep = Y < ye
                X, Y = X[keep], Y[keep]
                tau = t - T_SNAP
                if len(X) > 2:                                   # the frayed end curls as it recoils
                    curl = np.clip((Y - (ye - 60)) / 60, 0, 1) ** 2 * 25 * math.sin(tau * 23) * math.exp(-tau / 0.25)
                    X = X + curl
                fl = math.exp(-tau / 0.09)
                if fl > 0.05:                                    # the snap: a flash where it broke
                    d = np.hypot(SX - self.snap_hang[0], (SY - self.snap_hang[1] - self.pan_since_snap(t)) * 0.8)
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
        self.eye_close = 22.85 + 0.9 * (1 - dist / dist.max())           # outside in
        self.eye_close[0] = 24.0                                          # the hero eye last, with the camera on it
        self.eye_ph = rng.uniform(0, 6.3, len(eyes))
        # where each eye looks: a new spot every so often, reached in a quick jump, like real eyes do
        self.gaze = []
        for _ in eyes:
            ts = np.cumsum(rng.uniform(0.3, 1.0, 40)) + 16.0
            self.gaze.append((ts, rng.uniform(-1, 1, (40, 2)) * [0.3, 0.12]))
        self.eye_blinks = [16.9 + rng.uniform(0, 2.0) + np.cumsum(rng.uniform(1.3, 3.0, 6)) for _ in eyes]

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

    def gaze_at(self, i, t):
        """where eye i looks at t: it holds a spot, then jumps to the next one in a quick saccade"""
        ts, pts = self.gaze[i]
        k = int(np.searchsorted(ts, t))                  # the last jump was at ts[k - 1], to pts[k - 1]
        if k == 0: return pts[0]
        f = min((t - ts[k - 1]) / 0.07, 1.0)
        a = pts[max(k - 2, 0)]
        return a + (pts[k - 1] - a) * f * f * (3 - 2 * f)

    def hero_open(self, t):
        return 1 - float(smoother(self.eye_close[0], self.eye_close[0] + 0.35, t))

    def prison_photo(self, X, Y, t):
        """the eye with bars (reference photo), sampled at eye units; the hands rattle the bars on the hit"""
        PXp = PRI_E[0] + X * PRI_U
        PYp = PRI_E[1] - Y * PRI_U
        # the hands never let go: they keep working their grip, sliding a little on the bars
        for (hx, hy), ph in (((636.0, 775.0), 0.0), ((980.0, 792.0), 1.7)):
            bump = np.exp(-(((PXp - hx) / 75) ** 2 + ((PYp - hy) / 95) ** 2))
            gx_ = 2.5 * math.sin(t * 2.3 + ph) + 1.5 * math.sin(t * 5.1 + ph)
            gy_ = 4.0 * math.sin(t * 1.6 + ph * 0.7) + 2.0 * math.sin(t * 3.7 + ph)
            PXp = PXp - gx_ * bump; PYp = PYp - gy_ * bump
        a = t - 14.0
        if 0 <= a < 0.9:
            cell = (np.abs(X - 0.3) < 0.62) & (Y > -0.5) & (Y < 0.6)
            PXp = PXp + cell * 7 * math.exp(-a / 0.22) * math.sin(a * 38)
        lum = sample(self.prison, PXp, PYp)
        inside = (PXp >= 0) & (PXp < self.prison.shape[1] - 1) & (PYp >= 0) & (PYp < self.prison.shape[0] - 1)
        flick = 0.93 + 0.07 * math.sin(t * 13) * math.sin(t * 7.3)          # a bare bulb somewhere
        b = np.clip((lum - 0.025) / 0.7, 0, 1) ** 0.8 * flick
        return b, inside

    def prison_layer(self, X, Y, t, fi):
        """the string becomes the middle bar, the cell decodes in around it, the camera pulls back to the eye"""
        shape = X.shape
        b, inside = self.prison_photo(X, Y, t)
        # close up it's only the cell: the eye around it stays dark until the camera pulls back
        vig = 1 - float(smoother(14.4, 15.8, t))
        e = ((X - 0.29) / 0.82) ** 2 + ((Y - 0.04) / 0.75) ** 2
        b = b * (1 - vig * sstep(0.75, 1.15, e))
        # a bare bulb swings above the cell: its light slides back and forth over the bars and hands
        lx = PRI_BAR + 0.34 * math.sin(t * 2.4)
        lamp = 0.55 + 0.75 * np.exp(-(((X - lx) / 0.42) ** 2 + ((Y - 0.6) / 1.0) ** 2))
        cw = 1 - float(smoother(14.6, 15.6, t))
        b = np.clip(b * (1 + (lamp - 1) * cw), 0, 1)
        # reveal: outward from the bar the string turned into
        rv = np.full(shape, -1.0)                                   # it's just there, under the camera
        on = inside
        b = np.where(on, b, 0.0)
        col = redmap(b)
        gl = np.zeros(shape, np.int64)
        dec = inside & (t > rv - 0.08) & (t < rv)
        rng = np.random.default_rng(fi + 5)
        gl = np.where(dec, rng.choice(PRINTABLE, shape), gl)
        b = np.where(dec, 0.8, b); col = np.where(dec[..., None], HOT, col)
        # the string, swinging back from the snap, thickening into the bar
        if t < 13.6:
            Z = math.exp(ckey(t, LOGZ))
            ye = self.string_y0 + self.string_end(t) * self.string_k
            sw = 0.004 * math.sin(t * 9) * math.exp(-(t - T_SNAP) / 0.6)
            k = 0.0
            hw = max(0.0016, 0.5 * FCW / (EU * Z))
            st = (np.abs(X - PRI_BAR - sw * (1 - k)) < hw) & (Y > ye) & ~on
            b = np.where(st, 0.9 - 0.3 * k, b); col = np.where(st[..., None], HOT, col)
            gl = np.where(st & (k < 0.5), CI["|"], gl)
        # dust drifting through the light
        if cw > 0 and t > T_REVEAL:
            d = self.dust
            age = t - T_REVEAL
            px_ = (d["x"] + d["vx"] * age + 0.01 * np.sin(age * 1.7 + d["ph"]) - (-0.3)) % 1.25 + (-0.3)
            py_ = (d["y"] + d["vy"] * age + 0.6) % 1.2 - 0.6
            SXd, SYd = self.from_eye(px_, py_, self.cam_b(t))
            r = ((SYd - FOY) / FCH).astype(int); c = (SXd / FCW).astype(int)
            ok = (r >= 0) & (r < FROWS) & (c >= 0) & (c < FCOLS)
            r, c = r[ok], c[ok]
            lv = (0.35 + 0.35 * np.sin(t * 3 + d["ph"][ok])) * cw
            lit = (t >= rv[r, c]) & (lv > 0.08)
            r, c, lv = r[lit], c[lit], lv[lit]
            b[r, c] = np.maximum(b[r, c], lv); col[r, c] = PALE * 0.8; gl[r, c] = CI["."]
        solid = (b > 0.03) | dec
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
        if 14.4 <= t < MCUT[1]:
            b, folds = self.flesh(X, Y, t, Z)
            col = redmap(b); gl = np.zeros(shape, np.int64); solid = folds | (b > 0.03)
            # "flaw in your design": they all snap to look at you; "is not your mortality": they look away again
            look = float(smoother(17.9, 18.0, t)) * (1 - float(smoother(19.9, 20.05, t)))
            for i, (ex, ey, es, er) in enumerate(self.eyes):
                ph = self.eye_ph[i]
                dr = 1 - float(smoother(23.0, 23.8, t)) if i == 0 else 1.0       # the hero eye settles for the cut
                ex = ex + 0.06 * dr * math.sin(t * 0.9 + ph); ey = ey + 0.04 * dr * math.sin(t * 0.7 + 2 * ph)   # the skin breathes
                er = er + 0.04 * dr * math.sin(t * 0.6 + 3 * ph)
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
                    blink = max(math.exp(-((t - tb) / 0.06) ** 2) for tb in self.eye_blinks[i])
                    op *= (1 - blink) * (0.9 + 0.1 * math.sin(t * 1.1 + ph))           # they squint and widen
                    g = self.gaze_at(i, t)
                    gx = g[0] * (1 - look) + 0.012 * math.sin(t * 9 + ph)
                    gy = g[1] * (1 - look) + 0.008 * math.sin(t * 7 + ph)
                eb, ec, eg, es_, reg = self.paint_eye(xl, yl, op, gx, gy, t, self.rand[sl])
                if i == 0:                                        # inside the hero eye: the cell from the photo
                    inside_h = (np.abs(xl) < EX) & (yl > lids(xl, op)[1]) & (yl < lids(xl, op)[2])
                    if inside_h.any():
                        cb, _ = self.prison_photo(X[sl], Y[sl], t)
                        eb = np.where(inside_h, cb, eb); ec = np.where(inside_h[..., None], redmap(cb), ec)
                        eg = np.where(inside_h, 0, eg); es_ = np.where(inside_h, cb > 0.03, es_ & ~inside_h)
                b[sl] = np.where(reg, eb, b[sl]); col[sl] = np.where(reg[..., None], ec, col[sl])
                gl[sl] = np.where(reg, eg, gl[sl]); solid[sl] = np.where(reg, es_ | (eb > 0.03), solid[sl])
            col = np.where((b > 0.03)[..., None] & (col.max(-1, keepdims=True) == 0), redmap(b), col)
        else:
            b = np.zeros(shape); col = np.zeros(shape + (3,), np.float32)
            gl = np.zeros(shape, np.int64); solid = np.zeros(shape, bool)
        if t < 17.6:                                              # the eye with bars, from the photo
            pb, pc, pg, ps = self.prison_layer(X, Y, t, fi)
            # around the cell it's dark until the camera pulls back; then the photo's eye sits in the
            # drawn skin with a soft edge, among the other eyes, and slowly becomes the drawn eye
            vig = 1 - float(smoother(14.4, 15.8, t))
            e = ((X - 0.29) / 0.82) ** 2 + ((Y - 0.04) / 0.75) ** 2
            dark = vig * sstep(0.75, 1.15, e)
            b = b * (1 - dark)
            m = (1 - sstep(0.75, 1.2, np.hypot(X / 1.55, (Y - 0.08) / 0.9))) * (1 - float(smoother(16.8, 17.5, t)))
            if t < 14.4:
                m = np.ones(shape)
            mm = m[..., None]
            b = pb * m + b * (1 - m); col = pc * mm + col * (1 - mm)
            gl = np.where(m > 0.5, pg, gl); solid = np.where(m > 0.5, ps, solid)

        # the match cut: the closed eye's lid line becomes the seam of a closed ring box; the seam starts out
        # bent along the lid and straightens as the camera pulls back to the hands holding the box
        if t > MCUT[0]:
            k = 1 - float(smoother(MCUT[0], MCUT[1] + 0.4, t))
            bend = 0.0
            if k > 0:
                ylo = lids(cam[0] + (SX - 540) / (EU * Z), 0.0)[1]
                bend = -(ylo - cam[1]) * EU * Z * k
            rb, rc, rg, rsolid = self.box_scene(SX, SY, t, bend)
            a = float(smoother(MCUT[0], MCUT[1], t))
            b = b * (1 - a) + rb * a; col = col * (1 - a) + rc * a
            gl = np.where(a > 0.5, rg, gl); solid = np.where(a > 0.5, rsolid, solid)
        return b, col, gl, solid

    # ---------- the ending: a ring box (from a reference) opens, an Esc key inside, its light breathing; it goes dark
    def setup_box(self):
        here = os.path.dirname(os.path.abspath(__file__))
        img = np.asarray(Image.open(os.path.join(here, "assets", "esc_box.png")).convert("RGB"), np.float32) / 255.0
        self.box_img = img
        h, w = img.shape[:2]
        mx, mn = img.max(-1), img.min(-1)
        red = (mx - mn) > 0.2
        # the two fingers that hold the lid, whole, with their outlines
        yy, xx = np.mgrid[0:h, 0:w]
        pm = Image.new("L", (w, h), 0)
        ImageDraw.Draw(pm).polygon(FING_POLY, fill=255)
        strip = np.asarray(pm) > 127
        skin = strip & ~red & (img.mean(-1) > 0.26)
        m = Image.fromarray((skin * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(3))
        fing = (np.asarray(m) > 127) & strip & ~red
        self.fing = fing.astype(np.float32)
        # the plate: the open lid and the fingers taken out, and the dark backdrop of the picture (its grain,
        # mirrored from an empty corner) put where they were, so both can move
        plate = img.copy()
        lidreg = (xx > LID_X[0] - 6) & (xx < LID_X[1] + 6) & (yy > LID_TOP - 4) & (yy < HINGE + 3) & red
        hole = lidreg | fing
        tex = img[0:270, 690:960]
        th, tw = tex.shape[:2]
        ty = yy % (2 * th); ty = np.where(ty >= th, 2 * th - 1 - ty, ty)
        tx = xx % (2 * tw); tx = np.where(tx >= tw, 2 * tw - 1 - tx, tx)
        plate[hole] = tex[ty[hole], tx[hole]]
        self.box_plate = plate
        # the box is etched like the skin it comes out of: the skin's texture, lined up with the wall's at the cut
        tm = (MCUT[0] + MCUT[1]) / 2
        ce, cb = self.cam_b(tm), self.box_cam(tm)
        self.box_uv = (ce[0], ce[1], cb[0], cb[1], cb[2] / (EU * ce[2]))

    def box_cam(self, t):
        cx, cy = ckey(t, BOXCAM_X), ckey(t, BOXCAM_Y)
        z = math.exp(ckey(t, BOXCAM_Z)) * 1080 / self.box_img.shape[1]     # screen px per image px
        return cx, cy, z, 0.02 * math.sin(t * 0.8) * float(smoother(MCUT[1], MCUT[1] + 1.0, t))

    def lid_angle(self, t):
        x = (t - T_OPEN) / LID_T
        if x <= 0: return 0.0
        if x < 1: return math.pi / 2 * (1 - (1 - x) ** 2) ** 1.6       # eases off the hit, eases into place
        a = t - T_OPEN - LID_T
        return math.pi / 2 + 0.03 * math.exp(-a / 0.2) * math.sin(a * 14)

    def box_scene(self, SX, SY, t, bend=0.0):
        shape = SX.shape
        cx, cy, z, rl = self.box_cam(t)
        c, s = math.cos(rl), math.sin(rl)
        dx, dy = SX - 540, SY - bend - 960
        AX = cx + (dx * c + dy * s) / z
        AY = cy + (-dx * s + dy * c) / z
        on_key = (AX > KEY_BOX[0]) & (AX < KEY_BOX[1]) & (AY > KEY_BOX[2]) & (AY < KEY_BOX[3])
        rgb = sample(self.box_plate, AX, AY)
        # the lid: closed it lies over the opening (we see its top), opening it swings up on its hinge
        th = self.lid_angle(t)
        ye = HINGE + LID_D * math.cos(th) - LID_H * math.sin(th)
        # the box under the lid sits in its shadow until it's open
        o = math.sin(min(th, math.pi / 2))
        under = (AX > BASE_X[0]) & (AX < BASE_X[1]) & (AY > HINGE) & (AY < HINGE + LID_D + 20)
        rgb = np.where(under[..., None], rgb * (0.3 + 0.7 * o), rgb)
        lx0 = BASE_X[0] + (LID_X[0] - BASE_X[0]) * o                 # the lid's sides follow it round
        lx1 = BASE_X[1] + (LID_X[1] - BASE_X[1]) * o
        if ye > HINGE + 1:                                       # its outside: the same red card as the box
            m = (AX > lx0) & (AX < lx1) & (AY > HINGE) & (AY < ye)
            f = np.clip((AY - HINGE) / (ye - HINGE), 0, 1)
            face = sample(self.box_img, AX, 668 + 70 * f) * (1.05 - 0.25 * f)[..., None] * (1 - 0.35 * o)
            lip = m & (AY > ye - 16 * (1 - o))
            face = np.where(lip[..., None], face * 0.7, face)
            edge = m & ((np.abs(AX - lx0) < 2.5) | (np.abs(AX - lx1) < 2.5) | (np.abs(AY - ye) < 2.5))
            face = np.where(edge[..., None], 0.05, face)
            rgb = np.where(m[..., None], face, rgb)
        elif ye < HINGE - 1:                                     # its inside, squashed until it's fully up
            m = (AX > lx0) & (AX < lx1) & (AY > ye) & (AY < HINGE)
            IY = HINGE - (HINGE - AY) * (HINGE - LID_TOP) / (HINGE - ye)
            IX = LID_X[0] + (AX - lx0) * (LID_X[1] - LID_X[0]) / (lx1 - lx0)
            lid = sample(self.box_img, IX, IY) * (0.55 + 0.45 * o)
            rgb = np.where(m[..., None], lid, rgb)
        # the two fingers come down onto the closed lid, then ride its edge: they lift it and hold it open
        dfin = max(ye - FING_TIP, 0.0) + 7 * math.exp(-((t - T_OPEN + 0.12) / 0.1) ** 2)
        FY = AY - dfin + 430 * (1 - float(smoother(25.25, 26.15, t)))
        fa = sample(self.fing, AX, FY)
        if (fa > 0.5).any():
            fa = np.where(fa > 0.5, 1.0, 0.0) * (1 - (dfin > 1) * (1 - sstep(30, 120, FY)))   # lowered, they come out of the dark
            rgb = rgb * (1 - fa[..., None]) + sample(self.box_img, AX, FY) * fa[..., None]
        # light spills out of the opened box, pulses with the heart, then keeps breathing
        pulse = sum(math.exp(-(t - tb) / 0.22) for tb, _ in BEATS if t >= tb)
        breath = float(smoother(27.6, 28.4, t)) * (0.5 + 0.5 * math.sin((t - 28.4) * 3.3))
        d = np.hypot(AX - KEY_C[0], (AY - KEY_C[1]) * 1.3)
        glow = o * (0.5 + 0.5 * pulse + 0.3 * breath) * np.exp(-d / 85)
        mx = rgb.max(-1); mn = rgb.min(-1)
        b = np.clip((mx - 0.13) / 0.8, 0, 1) ** 1.05
        red = (mx - mn) > 0.22
        white = mn > 0.72
        col = np.where(red[..., None], rgb / (mx[..., None] + 0.02), redmap(b * 0.95))
        col = np.where(white[..., None], PALE, col)
        # etched like the skin it came out of: the same mottling and ink lines, matching the wall's at the cut
        ex0, ey0, bx0, by0, kk = self.box_uv
        U = ex0 + (AX - bx0) * kk
        V = ey0 - (cy + (-dx * s + (SY - 960) * c) / z - by0) * kk
        fl, ink = self.flesh(U, V, t, z / (kk * EU))
        e = 1 - 0.6 * float(smoother(MCUT[1], 26.4, t))                 # strong at the cut, lighter once it's a box
        b = b * (1 - 0.28 * e + 0.4 * e * np.clip((fl - 0.44) / 0.56, 0, 1)) * np.where(red, 1 - 0.18 * e, 1.0)
        b = np.where(ink & ~(white & (t > 25.6)), b * (1 - 0.88 * e), b)
        b = np.clip(b + 0.6 * glow * (on_key | (b < 0.2)), 0, 1)
        col = np.where(((glow > 0.12) & ~red & ~white)[..., None], HOT, col)
        # soften the edges of the picture into the dark
        h, w = self.box_img.shape[:2]
        edge = np.clip(np.minimum.reduce([AX, w - AX, AY, h - AY]) / 45, 0, 1)
        b = b * edge
        # the end: the hands and the box go dark, the Esc key and its light last
        lit = np.clip(np.maximum(on_key, glow / 0.35), 0, 1)
        b = b * (lit * (1 - float(smoother(T_FADE[1], T_FADE[2], t))) + (1 - lit) * (1 - float(smoother(T_FADE[0], T_FADE[1], t))))
        solid = b > 0.05
        return np.where(solid, b, 0.0), col.astype(np.float32), np.zeros(shape, np.int64), solid

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
        lock = float(sstep(7.2, 8.4, t))                              # still while the bust is on screen
        fx = (7 * math.sin(t * 0.7) + 4 * math.sin(t * 1.9 + 1)) * lock
        fy = (6 * math.sin(t * 0.9 + 2) + 3 * math.sin(t * 2.3)) * lock
        for th, amp in [(h, 14.0) for h in HITS if h > 7] + self.impacts:
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
        ccol = ccol * (1 + 2.5 * pulse) * (1 - float(smoother(T_FADE[0], T_FADE[1], t)))
        idx = np.where(empty, cidx, idx); col = np.where(empty[..., None], ccol, col)
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
