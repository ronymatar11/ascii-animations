"""
The Curse of Being Human  |  9:16, red and black, every frame is text, cut to a spoken-word track.
One continuous camera, no cuts: each scene turns into the next.

the curse of being human: a red rim light traces a classical bust (from a reference photo) ->
humans pursue peace: dots fill it in, the camera moves in on the head ->
yet manufacture chaos: pixel flames (from a reference) climb over it ->
they fear death: the fire burns away to an x-ray skull with a cassette for a brain (from a reference) ->
yet waste their lives: the tape unspools, the camera follows it down to a stack of money on a string ->
they demand freedom: the money drops, the string stays, more strings fall and harden into bars,
hands grab them -> own prisons: pull back, the prison is inside an eye ->
perhaps the greatest flaw in your design: the eye is one of a wall of eyes ->
is not your mortality: they all close -> it is your extraordinary ability: one opens in a red square ->
to remain alive / without ever truly living: it cries a line that becomes a heartbeat, then goes flat.

Export:   python curse_of_being_human.py --music voice.mp4 --export curse.mp4
"""
import argparse, math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from code_to_monalisa import W, H, CI, RAMP, PRINTABLE, sstep, smoother, fbm, find_font, ffmpeg_bin
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

def splat(idx, col, solid, X, Y, color, b=None, bright=0.9):
    """draw a polyline (screen px) into the fine grid as strokes"""
    if len(X) < 2: return
    c = np.round(X / FCW - 0.5).astype(int); r = np.round((Y - FOY) / FCH - 0.5).astype(int)
    g = stroke_glyph(np.arctan2(np.gradient(Y) / FCH, np.gradient(X) / FCW))
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
BUST_K = [(0, 1.82), (2.6, 2.1), (5.0, 2.65), (7.4, 2.9), (9.0, 3.1)]
HY = [(0, 670), (2.6, 720), (5.0, 860), (7.4, 900), (9.0, 920)]
ROLL = [(0, -0.03), (2.6, 0.02), (5.0, -0.02), (7.4, 0.02), (9.2, 0.0), (12.6, 0.0)]
BUST_E = (255.0, 290.0)
# the x-ray: screen px per image px, the head's center, the reels, where the drawn tape ends
SKULL_K = [(6.8, 2.25), (7.4, 2.1), (9.2, 1.55), (12.6, 1.5)]
SKULL_HY = [(6.8, 980), (9.2, 900), (12.6, 900)]
SKULL_E = (300.0, 240.0)
REELS = [((243.0, 120.0), 40.0), ((368.0, 146.0), 62.0)]
TAPE0 = (250.0, 415.0)
TAPE_L = [(9.2, 0), (9.45, 400), (9.7, 1200), (9.95, 2080), (10.15, 2260), (10.4, 2140), (10.7, 2190), (11.2, 2175), (12.6, 2175)]
FIRE_K = 2.0                                     # fire pattern, screen px per image px
T_CUT = 11.95                                    # the money drops
MB = (112.0, 210.0, 34.0)                        # money stack half size, px

def bill_texture():
    """a (made-up) banknote, drawn sideways and turned upright: brightness 0..1 per px"""
    w, h = int(2 * MB[1]), int(2 * MB[0])
    im = Image.new("L", (w, h), 120); d = ImageDraw.Draw(im)
    d.rectangle([4, 4, w - 5, h - 5], outline=230, width=10)
    for x in range(14, w - 14, 7):
        d.line([(x, 14), (x + 6, 20)], fill=175)
        d.line([(x, h - 20), (x + 6, h - 14)], fill=175)
    d.rectangle([22, 22, w - 23, h - 23], outline=200, width=2)
    cx, cy = w // 2, h // 2
    d.ellipse([cx - 62, cy - 78, cx + 62, cy + 78], fill=95, outline=235, width=6)
    d.ellipse([cx - 30, cy - 52, cx + 30, cy + 18], fill=170)                 # a portrait: head
    d.pieslice([cx - 52, cy + 4, cx + 52, cy + 110], 180, 360, fill=150)      # shoulders
    f = mono(54)
    for (x, y) in ((30, 26), (w - 140, 26), (30, h - 82), (w - 140, h - 82)):
        d.text((x, y), "100", font=f, fill=240)
    f2 = mono(18)
    d.text((cx, 34), "IN CODE WE TRUST", font=f2, fill=220, anchor="mm")
    d.text((cx, h - 34), "ONE HUNDRED", font=f2, fill=220, anchor="mm")
    return np.asarray(im.rotate(90, expand=True), np.float32) / 255.0

# ------------------------------------------------------------------ scene B camera (eye units)
EU = 330.0
LOGZ = [(12.6, math.log(6.5)), (14.6, math.log(5.9)), (15.6, math.log(2.6)), (16.7, math.log(1.0)),
        (17.6, math.log(0.5)), (18.8, math.log(0.36)), (20.0, math.log(0.33)), (21.5, math.log(0.31)),
        (22.5, math.log(0.38)), (23.6, math.log(0.98)), (24.6, math.log(1.06)), (25.6, math.log(1.0)),
        (26.8, math.log(0.72)), (30.55, math.log(0.66))]
CX = [(12.6, 0), (13.6, 0.13), (14.6, 0.14), (16.0, 0.05), (17.6, 0.0), (19.0, 0.8), (20.5, 0.3),
      (21.8, -0.6), (22.6, -0.3), (23.6, 0.0), (25.6, 0.0)]
CY = [(12.6, 0.08), (14.6, 0.06), (16.0, 0.0), (17.6, 0.2), (19.0, 0.9), (20.5, -0.4), (21.8, -0.9),
      (22.6, -0.6), (23.6, -0.85), (25.0, -0.9), (25.6, -0.8)]
ROLLB = [(12.6, 0), (14.6, 0.0), (16.0, 0.04), (17.6, 0.1), (19.5, -0.06), (21.0, 0.08), (22.6, 0.02),
         (23.6, 0.0), (25.6, 0.0), (26.9, -math.pi / 2), (30.55, -math.pi / 2 - 0.05)]
BAR_DX, BAR_R = 0.26, 0.016
SQ = (1.35, -3.2, 0.95)                          # the red square: half width, bottom, top (eye units)
T_TEAR, TEAR_A, TEAR_V = 25.0, 7.0, 2.6
BEATS = [(26.5, 0.8), (27.45, 0.48)]             # (R-spike time, amplitude)


class Film:
    def __init__(self, handle="@rony_matar", lyrics=True, seed=5):
        rng = np.random.default_rng(seed)
        self.handle, self.lyrics = handle, lyrics
        self.rand = rng.random((FROWS, FCOLS))
        self.bill = bill_texture()
        self.setup_bust()
        here = os.path.dirname(os.path.abspath(__file__))
        self.skull = np.asarray(Image.open(os.path.join(here, "assets", "skull_xray.png")).convert("L"), np.float32) / 255.0
        self.fire = np.asarray(Image.open(os.path.join(here, "assets", "fire.png")).convert("RGB"), np.float32) / 255.0
        self.build_code(rng)
        # the fire eats the face from the bottom up
        self.T_burn = 7.25 + 1.2 * np.clip((1750 - FGY) / 1500, 0, 1) + 0.35 * (fbm(FGX / 150, FGY / 150, 3, 19) - 0.5)
        self.setup_tape()
        self.setup_wall(rng)
        self.tear_beats = [(tb - 0.3 / TEAR_V, A) for tb, A in BEATS]
        # at the handoff the only thing on screen is the string: start the bars scene from its end
        _, hang = self.tape(T_B)
        ky = hang[1] - 30 * (1 - math.exp(-(T_B - T_CUT) / 0.08))
        cam = self.cam_b(T_B)
        self.knot_y = cam[1] + (960 - ky) / (EU * cam[2])

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
        # the tape hangs straight at screen x = 540 once the camera has panned; the money ends up centred
        k = ckey(11.5, SKULL_K)
        hxf = 540 - (TAPE0[0] - SKULL_E[0]) * k
        self.HXS = [(6.8, 540), (8.6, 545), (10.6, hxf), (12.6, hxf)]
        self.PAN = [(0, 0)]
        y0 = ckey(11.5, SKULL_HY) + (TAPE0[1] - SKULL_E[1]) * k
        P = 960 - (y0 + TAPE_L[-1][1] + 20 + MB[1])
        self.PAN = [(0, 0), (9.35, 0), (9.9, 0.12 * P), (10.6, 0.58 * P), (11.3, 0.93 * P), (12.6, P)]
        # the money only exists on the tape once its end is out of frame
        self.t_att = 12.0
        for tt in np.arange(9.2, 10.4, 0.01):
            _, hang = self.tape(tt)
            if hang is not None and hang[1] > 2010:
                self.t_att = tt; break

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

    def money_layer(self, b, col, gl, solid, t, SX, SY, hang):
        tw = t - 9.95
        swing = 0.1 * math.sin(2.4 * tw) * math.exp(-max(tw, 0) / 1.8) + 0.02 * math.sin(t * 1.1)
        twist = 0.55 * math.sin(t * 0.8) + 0.2
        fall = 0.0
        if t > T_CUT:
            fall = 0.5 * 7000 * (t - T_CUT) ** 2
            swing += 0.6 * (t - T_CUT)
        hx_, hy_ = hang
        cx_ = hx_ - math.sin(swing) * (20 + MB[1])
        cy_ = hy_ + math.cos(swing) * (20 + MB[1]) + fall
        sel = (np.abs(SX - cx_) < 320) & (np.abs(SY - cy_) < 320)
        if not sel.any():
            return
        dx, dy = SX[sel] - cx_, SY[sel] - cy_
        c, s = math.cos(-swing), math.sin(-swing)
        lx, ly = dx * c - dy * s, dx * s + dy * c
        Rm = rot(twist, 0.12)
        o = np.stack([lx, ly, np.full(len(lx), 300.0)], 1) @ Rm
        dv = np.array([0.0, 0.0, -1.0]) @ Rm
        bx, by, bz = MB
        def box(x, y, z):
            qx, qy, qz = np.abs(x) - bx + 6, np.abs(y) - by + 6, np.abs(z) - bz + 6
            return (np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2) +
                    np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - 6)
        tt, hit = march(box, o, dv, 100.0, 520.0, steps=32, eps=0.5)
        if not hit.any():
            return
        p = o[hit] + tt[hit, None] * dv
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        fx, fy, fz = np.abs(x) / bx, np.abs(y) / by, np.abs(z) / bz
        n = np.zeros_like(p)
        ax = np.argmax(np.stack([fx, fy, fz], 1), 1)
        n[np.arange(len(p)), ax] = np.sign(p[np.arange(len(p)), ax])
        nc = n @ Rm.T
        lam = np.clip(nc @ unit(-0.75, -0.35, 0.55), 0, 1)
        # the bill on the front and back, page edges on the sides, a rubber band around the middle
        face = ax == 2
        tx = np.clip((np.where(z > 0, x, -x) + bx).astype(int), 0, self.bill.shape[1] - 1)
        ty = np.clip((y + by).astype(int), 0, self.bill.shape[0] - 1)
        pat = np.where(face, self.bill[ty, tx], 0.38 + 0.35 * (np.sin(z * 1.7) > 0))
        pat = np.where(np.abs(y) < 11, 0.12, pat)                               # the rubber band
        bb = np.clip((0.4 + 0.9 * lam) * pat ** 1.4 * 1.35, 0, 1)
        idxs = np.nonzero(sel)
        hr, hc = idxs[0][hit], idxs[1][hit]
        b[hr, hc] = bb
        col[hr, hc] = redmap(bb * 1.05)
        gl[hr, hc] = 0
        solid[hr, hc] = True

    def scene_a(self, t, SX, SY, fi):
        shape = (FROWS, FCOLS)
        b = np.zeros(shape); col = np.zeros(shape + (3,), np.float32)
        gl = np.zeros(shape, np.int64); solid = np.zeros(shape, bool)
        roll = ckey(t, ROLL)
        cr, sr = math.cos(roll), math.sin(roll)
        dx, dy = SX - 540, SY - ckey(t, HY)
        burnt = t >= self.T_burn

        # the bust: a red rim light traces it, dots fill it in, then firelight
        if t < 8.9:
            k = ckey(t, BUST_K)
            PX = BUST_E[0] + (dx * cr + dy * sr) / k
            PY = BUST_E[1] + (-dx * sr + dy * cr) / k
            rim = self.bust_sample(self.bust_rim, PX, PY)
            inner = self.bust_sample(self.bust_in, PX, PY)
            glow = self.bust_sample(self.bust_glow, PX, PY)
            theta = np.abs(np.arctan2(PX - BUST_E[0], -(PY - 420)))
            rimI = sstep(-0.3, 0.3, math.pi * 1.1 * float(smoother(0.1, 2.3, t)) - theta)
            keyI = float(smoother(2.9, 3.8, t))
            fireI = float(smoother(5.0, 6.0, t))
            flick = 0.75 + 0.25 * np.sin(t * 23 + PY * 0.05) * np.sin(t * 17 + PX * 0.04)
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

        # the tape, then the money hanging off its end
        pts, hang = self.tape(t)
        if pts is not None:
            X, Y = pts
            if t > T_CUT:                                       # the end springs back up when the money drops
                k = Y > hang[1] - 30 * (1 - math.exp(-(t - T_CUT) / 0.08))
                X, Y = X[~k], Y[~k]
                hang = (hang[0], hang[1] - 30 * (1 - math.exp(-(t - T_CUT) / 0.08)))
            splat(gl, col, solid, X, Y, HOT)
            b[np.round((np.clip(Y, -1e4, 1e4) - FOY) / FCH - 0.5).astype(int).clip(0, FROWS - 1),
              np.round(np.clip(X, -1e4, 1e4) / FCW - 0.5).astype(int).clip(0, FCOLS - 1)] = 0.95
            if t >= self.t_att:
                self.money_layer(b, col, gl, solid, t, SX, SY, (pts[0][-1], pts[1][-1]))
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

    def cam_b(self, t):
        Z = math.exp(ckey(t, LOGZ))
        cx, cy = ckey(t, CX), ckey(t, CY)
        w = float(smoother(25.25, 26.5, t))
        if w > 0:                                   # follow the falling tear / the heartbeat line
            tip = self.tear_tip(t)
            lead = 300 / (EU * Z)
            cx, cy = cx + (self.tear_x - cx) * w, cy + (tip[1] + lead - cy) * w
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
        if t < 23.0:
            return 1 - float(smoother(self.eye_close[0], self.eye_close[0] + 0.2, t))
        return float(smoother(23.0, 23.6, t))

    def flesh(self, X, Y, t, Z=1.0):
        n = fbm(X * 1.1, Y * 1.1, 3, 23)
        ridge = np.abs(fbm(X * 0.9 + 3, Y * 1.6, 3, 29) - 0.5)
        b = 0.44 + 0.26 * n + 0.3 * sstep(0.66, 0.8, fbm(X * 0.7 + t * 0.03, Y * 0.7, 2, 31))
        folds = ridge < 0.02 * min(1.0, 0.7 / Z)
        return np.where(folds, 0.0, b), folds

    def cell(self, X, Y, t, fi):
        """inside the prison: black, strings that fall and harden into bars, a crossbar, two fists"""
        shape = X.shape
        b = np.zeros(shape); col = np.zeros(shape + (3,), np.float32)
        gl = np.zeros(shape, np.int64); solid = np.zeros(shape, bool)
        top = ckey(T_B, CY) + 0.6
        rb = 0.0015 + (BAR_R - 0.0015) * float(smoother(13.15, 13.65, t))
        y0 = self.knot_y
        for k in range(-5, 6):
            xk = k * BAR_DX
            if k == 0:
                yend = y0 + (-2.5 - y0) * float(smoother(T_B + 0.02, 13.1, t))
            else:
                tk = 12.75 + 0.06 * abs(k) + 0.03 * (k > 0)
                yend = top + (-2.5 - top) * float(np.clip((t - tk) / 0.42, 0, 1)) ** 1.6
                if t < tk: continue
            rr = max(rb, 0.5 * FCW / (EU * math.exp(ckey(t, LOGZ))))
            dxn = (X - xk) / rr
            on = (np.abs(dxn) < 1) & (Y > yend)
            if not on.any(): continue
            nz = np.sqrt(np.clip(1 - dxn ** 2, 0, 1))
            bb = 0.2 + 0.6 * np.clip(-0.55 * dxn + 0.8 * nz, 0, 1) + 0.35 * np.exp(-((dxn + 0.35) / 0.18) ** 2)
            bb = np.clip(bb, 0, 1)
            b = np.where(on, bb, b); col = np.where(on[..., None], redmap(bb), col)
            gl = np.where(on, np.where((np.abs(dxn + 0.35) < 0.2) | (rr < 0.003), CI["|"], 0), gl)
            solid |= on
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

    def paint_eye(self, xl, yl, op, gx, gy, t, poster, rnd):
        """an eye in its own units: lids, sclera, iris, pupil, highlight, lashes"""
        sx, y_low, y_up = lids(xl, op)
        inside = (np.abs(xl) < EX) & (yl > y_low) & (yl < y_up)
        b = np.zeros(xl.shape); col = np.zeros(xl.shape + (3,), np.float32)
        gl = np.zeros(xl.shape, np.int64); solid = np.zeros(xl.shape, bool)
        d = np.hypot(xl - gx, yl - gy)
        if poster:
            b = np.where(inside, 0.62, b); col = np.where(inside[..., None], RED, col)
            iris = inside & (d < 0.36)
            b = np.where(iris, 0.0, b); solid |= iris
            ring = np.abs(np.hypot(xl - gx - 0.07, yl - gy - 0.09) - 0.05) < 0.016
            b = np.where(iris & ring, 1.0, b); col = np.where((iris & ring)[..., None], PALE, col)
            gl = np.where(iris & ring, CI["o"], gl)
            band = (np.abs(np.minimum(yl - y_low, y_up - yl)) < 0.1) & (np.abs(xl) < EX + 0.08)
            stip = band & (rnd < 0.55)
            b = np.where(stip, 0.0, b); solid |= stip
        else:
            ball = np.clip(1 - (xl / 1.4) ** 2, 0, 1)
            shadow = sstep(0.0, 0.14, y_up - yl)
            scl = (0.45 + 0.35 * ball) * (0.55 + 0.45 * shadow)
            vein = np.abs(fbm(xl * 6 + 7, yl * 6, 3, 61) - 0.5) < 0.014
            b = np.where(inside, np.where(vein, 0.5, scl), b)
            col = np.where(inside[..., None], np.where(vein[..., None], RED, HOT), col)
            a = np.arctan2(yl - gy, xl - gx)
            fib = 0.5 + 0.5 * np.sin(a * 60 + 8 * fbm(np.cos(a) * 4, np.sin(a) * 4, 2, 21))
            iris = inside & (d < 0.33)
            ib = (0.25 + 0.3 * fib) * (1 - 0.7 * sstep(0.26, 0.33, d))
            b = np.where(iris, ib, b); col = np.where(iris[..., None], redmap(ib + 0.15), col)
            gl = np.where(iris & (fib > 0.7), stroke_glyph(np.arctan2(np.sin(a) / FCH, np.cos(a) / FCW)), gl)
            pup = inside & (d < 0.13)
            b = np.where(pup, 0.0, b); solid |= pup
            hl = inside & ((np.hypot(xl - gx + 0.1, yl - gy - 0.11) < 0.06) | (np.hypot(xl - gx - 0.08, yl - gy + 0.06) < 0.025))
            b = np.where(hl, 1.0, b); col = np.where(hl[..., None], PALE, col)
            gl = np.where(hl, CI["@"], gl)
        # black outline along both lids, lashes above
        line = (np.abs(xl) < EX + 0.04) & ((np.abs(yl - y_up) < 0.035) | (np.abs(yl - y_low) < 0.03) | (inside & (np.minimum(yl - y_low, y_up - yl) < 0.03)))
        lh = 0.2 * sx ** 0.6 * (0.45 + 0.55 * op)
        above = (yl - y_up > 0) & (yl - y_up < lh) & (np.abs(xl) < EX)
        ph = xl * 18 + (yl - y_up) * 7 * np.sign(xl)
        lash = above & ((ph % 1.0) < 0.32)
        below = (y_low - yl > 0) & (y_low - yl < 0.06 * sx) & (((xl * 22) % 1.0) < 0.3)
        dark = line | lash | below
        b = np.where(dark, 0.0, b); solid |= dark
        region = (np.abs(xl) < EX + 0.06) & (yl > y_low - 0.1) & (yl < y_up + lh + 0.04)
        return b, col, gl, solid, region | inside

    def scene_b(self, t, SX, SY, fi):
        shape = (FROWS, FCOLS)
        cam = self.cam_b(t)
        X, Y = self.to_eye(SX, SY, cam)
        Z = cam[2]
        b, folds = self.flesh(X, Y, t, Z)
        col = redmap(b); gl = np.zeros(shape, np.int64); solid = folds | (b > 0.03)
        poster_t = 21.6
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
            # folds of skin around every eye
            e = (xl / 1.5) ** 2 + (yl / 0.85) ** 2
            ring = (e > 1.05) & (e < 2.8) & (((e * 2.2 + 0.2 * fbm(xl * 2, yl * 2, 2, 71)) % 1.0) < 0.09 * min(1.0, 0.8 / (Z * es)))
            b[sl] = np.where(ring, 0.0, b[sl])
            if i == 0:
                op = self.hero_open(t)
                gx = 0.04 * math.sin(t * 0.6); gy = 0.02 * math.sin(t * 0.8)
            else:
                op = float(smoother(self.eye_open[i], self.eye_open[i] + 0.25, t)) * (1 - float(smoother(self.eye_close[i], self.eye_close[i] + 0.2, t)))
                ph = self.eye_ph[i]
                gx = 0.12 * math.sin(t * 0.7 + ph) + 0.05 * (math.sin(t * 3.1 + ph) > 0.7)
                gy = 0.05 * math.sin(t * 0.5 + 2 * ph)
            eb, ec, eg, es_, reg = self.paint_eye(xl, yl, op, gx, gy, t, i == 0 and t > poster_t, self.rand[sl])
            if i == 0 and t < poster_t:
                inside_h = (np.abs(xl) < EX) & (yl > lids(xl, op)[1]) & (yl < lids(xl, op)[2])
                if inside_h.any():
                    cb, cc, cg, csd = self.cell(X[sl], Y[sl], t, fi)
                    eb = np.where(inside_h, cb, eb); ec = np.where(inside_h[..., None], cc, ec)
                    eg = np.where(inside_h, cg, eg); es_ = np.where(inside_h, csd, es_ & ~inside_h)
            b[sl] = np.where(reg, eb, b[sl]); col[sl] = np.where(reg[..., None], ec, col[sl])
            gl[sl] = np.where(reg, eg, gl[sl]); solid[sl] = np.where(reg, es_ | (eb > 0.03), solid[sl])
        col = np.where((b > 0.03)[..., None] & (col.max(-1, keepdims=True) == 0), redmap(b), col)

        # the red square: everything else sinks into black, the square goes flat red
        if t > 22.5:
            hw, yb, yt = SQ
            wob = 0.03 * (fbm(X * 4, Y * 4, 2, 81) - 0.5)
            insq = (np.abs(X) < hw + wob) & (Y > yb + wob) & (Y < yt + wob)
            rc = np.hypot(X / 1.4, (Y + 0.3) / 1.4)
            sw = 22.55 + 0.25 * np.clip(rc / 4, 0, 1) + 0.6 * self.rand
            a = sstep(sw, sw + 0.35, t)
            hero = (np.abs(X) < EX + 0.1) & (Y > -0.6) & (Y < 0.8)
            flat = insq & ~hero
            stip = flat & (self.rand < 0.06 + 0.5 * np.exp(-((np.abs(X) - 0.0) ** 2 / 2.0 + (Y - 0.1) ** 2 / 0.25)))
            fb = np.where(stip, 0.0, 0.62)
            b = np.where(flat, b * (1 - a) + fb * a, np.where(insq, b, b * (1 - a)))
            col = np.where(flat[..., None], col * (1 - a[..., None]) + RED * a[..., None], col)
            gl = np.where(flat & (a > 0.5), 0, gl)
            solid = np.where(insq, solid, solid & (a < 0.5))
            dec = (t > sw - 0.06) & (t < sw) & insq
            rng = np.random.default_rng(fi + 11)
            gl = np.where(dec, rng.choice(PRINTABLE, shape), gl)

        # the tear, then the heartbeat line
        if t > T_TEAR - 0.35:
            self.draw_line(b, col, gl, solid, t, cam)
        return b, col, gl, solid

    # ---------- the tear and the line it becomes
    @property
    def tear_x(self):
        return 0.0

    def tear_s(self, t):
        tau = t - T_TEAR
        if tau <= 0: return 0.0
        t1 = TEAR_V / TEAR_A
        return 0.5 * TEAR_A * tau ** 2 if tau < t1 else 0.5 * TEAR_A * t1 ** 2 + TEAR_V * (tau - t1)

    def tear_tip(self, t):
        y0 = float(lids(np.array([self.tear_x]), 1.0)[1][0])
        return self.tear_x, y0 - self.tear_s(t)

    def ecg(self, s):
        h = np.zeros_like(s)
        for tb, A in self.tear_beats:
            sg = s - self.tear_s(tb)
            h += A * (0.07 * np.exp(-((sg - 0.08) / 0.035) ** 2) - 0.06 * np.exp(-((sg - 0.26) / 0.012) ** 2) +
                      np.exp(-((sg - 0.30) / 0.016) ** 2) - 0.25 * np.exp(-((sg - 0.335) / 0.014) ** 2) +
                      0.16 * np.exp(-((sg - 0.55) / 0.05) ** 2))
        return h

    def draw_line(self, b, col, gl, solid, t, cam):
        y0 = float(lids(np.array([self.tear_x]), 1.0)[1][0])
        s_tip = self.tear_s(t)
        bead = float(smoother(T_TEAR - 0.35, T_TEAR, t))
        if s_tip > 0:
            s = np.arange(0, s_tip, 0.0015)
            xs = self.tear_x + self.ecg(s)
            ys = y0 - s
            SXl, SYl = self.from_eye(xs, ys, cam)
            age = (s_tip - s) / TEAR_V
            c = PALE[None, :] * (0.7 + 0.3 * np.exp(-age / 0.6))[:, None]
            nx, ny = np.gradient(SYl), -np.gradient(SXl)
            nl = np.hypot(nx, ny) + 1e-9
            for off in (-0.6, 0.6):                     # two cells wide
                splat(gl, col, solid, SXl + nx / nl * FCW * off, SYl + ny / nl * FCH * off, c, b, 0.95)
            rr = np.round((SYl - FOY) / FCH - 0.5).astype(int); cc = np.round(SXl / FCW - 0.5).astype(int)
            ok = (rr >= 0) & (rr < FROWS) & (cc >= 0) & (cc < FCOLS)
            b[rr[ok], cc[ok]] = 0.9
        # the drop (or the bright head of the line)
        tx, ty = self.tear_x + float(self.ecg(np.array([s_tip]))[0]), y0 - s_tip - 0.03 * bead
        px, py = self.from_eye(np.array([tx]), np.array([ty]), cam)
        rpx = (6 + 10 * bead) * (1 if s_tip <= 0 else 1.4)
        d = np.hypot(FGX - px[0], (FGY - py[0]) * 0.8)
        glow = np.clip(1 - d / (rpx * 2.2), 0, 1)
        core = d < rpx
        on = glow > 0.05
        b[on] = np.maximum(b[on], glow[on]); col[on] = np.where(core[on][:, None], PALE, HOT * glow[on][:, None] + col[on] * (1 - glow[on][:, None]))
        gl[core] = CI["@"]; solid[on] = True

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
        for th in HITS:
            if 0 <= t - th < 0.5:
                k = math.exp(-(t - th) / 0.12)
                fx += 14 * k * math.sin((t - th) * 70); fy += 10 * k * math.cos((t - th) * 61)
        SX, SY = FGX + fx, FGY + fy
        if t < T_B:
            b, col, gl, solid = self.scene_a(t, SX, SY, fi)
        else:
            b, col, gl, solid = self.scene_b(t, SX, SY, fi)
        idx = np.where(gl > 0, gl, glyphs(np.clip(b, 0, 1), int(t * 12), (FROWS, FCOLS)))
        col = np.clip(col * np.clip(0.4 + 0.9 * b, 0, 1)[..., None], 0, 1)
        empty = ~solid & (b < 0.035) & (gl == 0)
        cidx, ccol = self.code_layer(t, fi)
        idx = np.where(empty, cidx, idx); col = np.where(empty[..., None], ccol, col)
        g = 1 - float(sstep(29.7, DUR - 0.05, t))
        img = render_fine(idx, col.astype(np.float32) * g)
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
