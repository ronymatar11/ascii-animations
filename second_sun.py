"""
2ND SUN  |  stage visual concept, 9:16, cut to music. Every frame is text, no cuts:
each scene is transformed out of the one before it.

code fills the page -> the code forms the logo, and as it forms the star inside it
starts to shine -> two hands drawn in code reach in and meet on the star -> on the
drop it explodes into a sun with the logo stamped in it -> the sun becomes an iris,
the camera pulls back: one big eye -> it looks around on the beat and stares at you ->
dive into the pupil: the two dots in it grow into two moons ->
a rocket comes in from out of frame and burns A SPACE ODYSSEY in as it passes ->
it comes back close to the camera and its giant trail paints the date: FRI OCT 30.

Export:   python second_sun.py --music track.mp3 --start 236.32 --handle @rony_matar --export second_sun.mp4
          (--start = where to cut the track; the drop lands 9.52s in, 5 bars at 126 BPM)
"""
import argparse, math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from code_to_monalisa import (W, H, CW, CH, COLS, ROWS, OY, CI, CHARS, BLOCK, PRINTABLE, RAMP,
                              sstep, smoother, fbm, find_font, compose, render, to_glyphs, ffmpeg_bin,
                              colorize)

FPS = 30
DUR = 30.55
BPM = 126.05
BEAT = 60 / BPM
BAR = 4 * BEAT
DROP = 5 * BAR                            # 9.52s: the kick comes back
BREAK = DROP - BAR                        # 7.62s: the riser stops, the hands close in
T_HANDS_IN = 2.8                          # the hands start reaching for the star
T_IRIS = DROP + 1.6 * BAR                 # the sun turns into an iris
T_PULL = DROP + 1.8 * BAR                 # camera pulls back: it's an eye
T_DIVE = DROP + 4.5 * BAR                 # dive into the pupil
T_END = DROP + 5 * BAR                    # 19.05s: inside the pupil, space; the second song starts
T_ROCKET = T_END + 0.3                    # a rocket comes in from out of frame, writes the title
ROCKET_DUR = 4.4
T_FLY = T_ROCKET + ROCKET_DUR - 0.7       # it comes back, close to the camera; its trail paints the date
FLY_DUR = 3.0
T_FOOT = T_FLY + FLY_DUR + 1.0

CREAM = np.array([0.99, 0.92, 0.82], np.float32)
GOLD = np.array([1.0, 0.82, 0.45], np.float32)
EMBER = np.array([1.0, 0.45, 0.12], np.float32)
WHITE = np.array([1.0, 0.97, 0.92], np.float32)
SOFT_SET = ".,:;-=+*#"

POOL = [
    "import stage", "from sun import light, heat", "bpm = 126", "for beat in set: sun.rise(beat)",
    "def drop(): return light * 2", "while crowd.alive: pulse()", "stage.load('2ND SUN')",
    "kick = sin(t * bpm / 60 * tau)", "if beat % 4 == 0: flash()", "lasers.aim(crowd, spread=0.3)",
    "class Sun(Star):", "self.heat = 5778  # kelvin", "return [l for l in lights if l.on]",
    "fog.density += 0.02", "await sunrise()", "const second = new Sun();", "led.wall.fill(code)",
    "# every frame is text", "def reach(you): return light", "for x in range(W):",
    "px[y][x] = ramp[int(lum * n)]", "eye = Eye(iris=sun)", "hands.meet(at=star)",
    "if drop: explode(star)", "SELECT * FROM nights WHERE sun = 2;", "0x2d 0x53 0x55 0x4e",
    "git commit -m 'second sun'", "npm run stage", "ctx.fillText(chr, x, y);", "glow = 1 / (1 + r * r)",
    "mix(gold, ember, t)", "sync(track='your love')", "frame = render(scene, cam)",
]

PX = (np.arange(COLS) + 0.5) * CW
PY = (np.arange(ROWS) + 0.5) * CH + OY
GX, GY = np.meshgrid(PX, PY)

# fine grid for the hands: twice the detail
FCW, FCH = 6, 11
FCOLS, FROWS = W // FCW, H // FCH
FOY = (H - FROWS * FCH) // 2
FGX, FGY = np.meshgrid((np.arange(FCOLS) + 0.5) * FCW, (np.arange(FROWS) + 0.5) * FCH + FOY)

# logo placement on screen + where the star sits inside the logo mask
LOGO_CX, LOGO_CY, LOGO_W = 540.0, 900.0, 860.0
STAR_M = (148.0, 117.5)                   # star center, mask pixels
STAR_A, STAR_B = 62.0, 32.0               # star half width / half height, mask pixels

# the eye
EU = 330.0                                # px per eye unit at zoom 1
IRIS_R = 0.5
SUN_R = 480.0                             # sun radius in px; the eye starts zoomed so its iris is the sun
Z0 = SUN_R / (IRIS_R * EU)
# hands: wrists at the end of their reach (fingertips on the star) and size
HAND_L, HAND_R, HAND_S = (189.0, 1113.0), (890.0, 775.0), 1.9
ZE = 1.25                                 # zoom once the eye is revealed

def kick(t):
    """1 on each kick after the drop, decaying"""
    if t < DROP: return 0.0
    return math.exp(-((t - DROP) % BEAT) / 0.11)

def typed(text, t, t0, cps=24.0):
    n = int(max(0.0, (t - t0) * cps))
    return text[:n], n < len(text)

def star_sdf(x, y, a, b):
    """4-point star (astroid), negative inside"""
    q = (np.abs(x) / a) ** (2 / 3) + (np.abs(y) / b) ** (2 / 3)
    return (q - 1) * min(a, b)

def seg_sdf(px, py, ax, ay, bx, by, ra, rb):
    """tapered capsule from a (radius ra) to b (radius rb)"""
    dx, dy = bx - ax, by - ay
    h = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy + 1e-9), 0, 1)
    return np.hypot(px - ax - dx * h, py - ay - dy * h) - (ra + (rb - ra) * h)

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)

def stroke_glyph(ang):
    """glyph for a stroke at screen angle ang (radians, y down), in cell space"""
    a = ang % np.pi
    return np.select([a < np.pi / 8, a < 3 * np.pi / 8, a < 5 * np.pi / 8, a < 7 * np.pi / 8],
                     [CI["-"], CI["\\"], CI["|"], CI["/"]], CI["-"])

def glyphs(b, seed, shape):
    jr = np.random.default_rng(seed).integers(-1, 2, shape)
    k = np.clip(np.round(b * (len(RAMP) - 1)).astype(int) + jr, 0, len(RAMP) - 1)
    return np.where(b < 0.04, 0, RAMP[k])

# fine glyph atlas
def _fine_atlas():
    font = find_font(10)
    arr = np.zeros((len(CHARS), FCH, FCW), np.float32)
    for i, c in enumerate(CHARS[:-1]):
        im = Image.new("L", (FCW, FCH), 0)
        ImageDraw.Draw(im).text((FCW / 2, FCH / 2), c, fill=255, font=font, anchor="mm")
        arr[i] = np.asarray(im, np.float32) / 255.0
    arr[BLOCK, 1:FCH - 1, 1:FCW - 1] = 1.0
    return arr
FATLAS = _fine_atlas()
SOFT = np.array(sorted([CI[c] for c in SOFT_SET], key=lambda i: FATLAS[i].mean()))

def render_fine(idx, col):
    out = np.zeros((H, W, 3), np.uint8)
    img = compose(FATLAS, idx, col)
    out[FOY:FOY + img.shape[0], :img.shape[1]] = img
    return out

# ------------------------------------------------------------------ hands, drawn from capsules
# finger: (base x, base y, spread angle, segment lengths, curl per joint, radius)
FINGERS_POINT = [
    (98, 30, 0.10, (46, 28, 22), (0.05, 0.08, 0.05), 12.5),     # index: reaching
    (100, 9, 0.00, (44, 28, 21), (0.55, 0.75, 0.55), 13.0),
    (98, -12, -0.08, (40, 26, 19), (0.75, 0.85, 0.6), 12.5),
    (90, -32, -0.18, (32, 21, 16), (0.9, 0.9, 0.6), 11.0),
]
FINGERS_OPEN = [
    (98, 30, 0.22, (46, 28, 22), (0.10, 0.15, 0.10), 12.5),     # index: reaching up
    (100, 9, 0.08, (44, 28, 21), (0.35, 0.40, 0.30), 13.0),
    (98, -12, -0.02, (40, 26, 19), (0.45, 0.50, 0.35), 12.5),
    (90, -32, -0.12, (32, 21, 16), (0.55, 0.55, 0.40), 11.0),
]

def hand_sdf(X, Y, wx, wy, ang, s, fingers, thumb_ang, mirror, flex=(0, 0, 0, 0)):
    """hand in local coords: wrist at 0, pointing +x, thumb side +y; flex = extra curl per finger"""
    c, sn = math.cos(ang), math.sin(ang)
    lx = ((X - wx) * c + (Y - wy) * sn) / s
    ly = (-(X - wx) * sn + (Y - wy) * c) / s
    if mirror: ly = -ly
    d = seg_sdf(lx, ly, -900, 0, -60, 0, 50, 41)                    # forearm
    d = smin(d, seg_sdf(lx, ly, -60, 0, 0, 0, 41, 37), 20)          # wrist
    d = smin(d, seg_sdf(lx, ly, 12, 0, 74, 0, 45, 43), 22)          # palm
    d = smin(d, seg_sdf(lx, ly, 72, 24, 92, -24, 22, 20), 14)       # knuckles
    tips = []
    for (bx, by, a0, lens, curls, r), fx in zip(fingers, flex):
        x, y, a = bx, by, a0
        fd = None
        for k, (L, cu) in enumerate(zip(lens, curls)):
            a = a - cu - fx
            nx, ny = x + L * math.cos(a), y + L * math.sin(a)
            r0, r1 = r * (1 - 0.07 * k), r * (1 - 0.07 * (k + 1)) - (2.5 if k == 2 else 0)
            sd = seg_sdf(lx, ly, x, y, nx, ny, r0 * 0.94, r1 * 0.94)
            fd = sd if fd is None else smin(fd, sd, 2.5)                   # knuckle creases
            x, y = nx, ny
        tips.append((x, y))
        d = smin(d, fd, 4)
    x, y, a = 38, 36, thumb_ang                                    # thumb
    for L, r0, r1 in ((44, 17, 15), (32, 15, 11)):
        nx, ny = x + L * math.cos(a), y + L * math.sin(a)
        d = smin(d, seg_sdf(lx, ly, x, y, nx, ny, r0, r1), 8)
        x, y, a = nx, ny, a - 0.35
    tx, ty = tips[0]
    if mirror: ty = -ty
    return d * s, (wx + (tx * c - ty * sn) * s, wy + (tx * sn + ty * c) * s)

def mask_sdf(mask, sx=FCW, sy=FCH):
    """signed distance in px to the edge of a cell mask (negative inside), two-pass chamfer"""
    def dist(m):
        INF = 1e9
        d = np.where(m, 0.0, INF)
        dg = math.hypot(sx, sy)
        R, C = d.shape
        for r in range(R):
            for c in range(C):
                v = d[r, c]
                if c > 0: v = min(v, d[r, c - 1] + sx)
                if r > 0:
                    v = min(v, d[r - 1, c] + sy)
                    if c > 0: v = min(v, d[r - 1, c - 1] + dg)
                    if c < C - 1: v = min(v, d[r - 1, c + 1] + dg)
                d[r, c] = v
        for r in range(R - 1, -1, -1):
            for c in range(C - 1, -1, -1):
                v = d[r, c]
                if c < C - 1: v = min(v, d[r, c + 1] + sx)
                if r < R - 1:
                    v = min(v, d[r + 1, c] + sy)
                    if c < C - 1: v = min(v, d[r + 1, c + 1] + dg)
                    if c > 0: v = min(v, d[r + 1, c - 1] + dg)
                d[r, c] = v
        return d
    return dist(mask) - dist(~mask)


# ------------------------------------------------------------------ space: two moons, a rocket, the date
MOONS = [(330.0, 880.0, 215.0, np.array([1.0, 0.9, 0.7], np.float32)),     # x, y, radius, colour
         (815.0, 330.0, 118.0, np.array([0.92, 0.88, 0.82], np.float32))]
DOTS = [(-0.38, 0.30, 0.30), (0.42, -0.38, 0.18)]   # in the pupil: offset (x, y down) and radius, x pupil radius
TRAIL_PATH = [(-300, 1870), (150, 1850), (560, 1800), (900, 1700), (1095, 1430), (1075, 1090),
              (820, 770), (530, 560), (290, 360), (-320, -80)]
TRAIL_COLS = np.array([[0.88, 0.25, 0.2], [0.98, 0.52, 0.27], [1.0, 0.9, 0.78],
                       [0.62, 0.6, 0.58], [0.98, 0.66, 0.52]], np.float32)
RED = np.array([0.92, 0.3, 0.22], np.float32)
LETTER_COLS = np.array([[0.95, 0.28, 0.2], [1.0, 0.5, 0.25], [1.0, 0.93, 0.8], [1.0, 0.62, 0.42],
                        [0.92, 0.22, 0.18]], np.float32)
BIG_FONTS = ["/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf", "Inter-ExtraBold.otf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"]

def big_font(size):
    for f in BIG_FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return find_font(size)

def text_mask(lines, cy, height, max_w=1000, gap=0.08, cx=W / 2):
    """big text rendered to a fine-grid mask, lines stacked, centred at (cx, cy)"""
    hi = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(hi)
    n = len(lines)
    lh = height / (n + gap * (n - 1))
    size = int(lh * 1.25)
    f = big_font(size)
    while max(f.getlength(l) for l in lines) > max_w and size > 10:
        size -= 4; f = big_font(size)
    y = cy - height / 2
    for l in lines:
        dr.text((cx, y + lh / 2), l, font=f, fill=255, anchor="mm")
        y += lh * (1 + gap)
    a = np.asarray(hi, np.float32) / 255.0
    return a[np.clip(FGY.astype(int), 0, H - 1), np.clip(FGX.astype(int), 0, W - 1)] > 0.5

def sample_path(points, step=6.0):
    """catmull-rom through points, resampled every `step` px: positions, unit tangents, lengths"""
    P = catmull(points, 60)
    L = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    Ls = np.arange(0, L[-1], step)
    path = np.stack([np.interp(Ls, L, P[:, 0]), np.interp(Ls, L, P[:, 1])], 1)
    tg = np.gradient(path, axis=0)
    return path, tg / (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9), Ls

def nearest_on_path(X, Y, P, T, chunk=64):
    """per cell: distance to the nearest path sample, its index, and the signed side offset"""
    best = np.full(X.shape, 1e9); jj = np.zeros(X.shape, int); lat = np.zeros(X.shape)
    for j in range(0, len(P), chunk):
        ex = X[..., None] - P[None, None, j:j + chunk, 0]
        ey = Y[..., None] - P[None, None, j:j + chunk, 1]
        dd = np.hypot(ex, ey)
        k = dd.argmin(-1)
        m = np.take_along_axis(dd, k[..., None], -1)[..., 0]
        better = m < best
        kk = k + j
        best = np.where(better, m, best); jj = np.where(better, kk, jj)
        lx = np.take_along_axis(ex, k[..., None], -1)[..., 0]
        ly = np.take_along_axis(ey, k[..., None], -1)[..., 0]
        lat = np.where(better, T[kk, 0] * ly - T[kk, 1] * lx, lat)
    return best, jj, lat

def catmull(points, n=40):
    P = np.array(points, float)
    P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for s_ in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s_ + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s_ ** 2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * s_ ** 3))
    out.append(P[-2])
    return np.array(out)


class Show:
    def __init__(self, handle="@rony_matar", seed=3):
        rng = np.random.default_rng(seed)
        self.handle = handle
        m = np.asarray(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "2ndsun_logo_mask.png")).convert("L"), np.float32) / 255.0
        self.mask = m
        self.mh, self.mw = m.shape
        self.logo_sun = self.logo_field(GX, GY, 540, 960, 760)
        self.flogo = self.logo_field(FGX, FGY, LOGO_CX, LOGO_CY, LOGO_W)
        lw = LOGO_W / self.mw
        self.star_xy = (LOGO_CX + (STAR_M[0] - self.mw / 2) * lw, LOGO_CY + (STAR_M[1] - self.mh / 2) * lw)
        self.star_ab = (STAR_A * lw, STAR_B * lw)
        sx, sy = self.star_xy
        R = self.rand = rng.random((ROWS, COLS))
        self.glyph_rand = rng.choice(PRINTABLE, (ROWS, COLS))
        self.build_page(rng)
        # the logo is typed with the page: gaps between words inside it take their neighbour's time
        lt = self.page_t.copy()
        for r in range(FROWS):
            row = lt[r]
            fin = np.isfinite(row)
            if fin.any():
                k = np.maximum.accumulate(np.where(fin, np.arange(FCOLS), -1))
                first = np.argmax(fin)
                row[:] = np.where(k >= 0, row[np.maximum(k, 0)], row[first])
        self.logo_t = lt
        # the logo is stamped into the sun on the drop, and melts back into it
        self.T_stamp = DROP + 0.15 + 0.45 * R
        self.T_melt = T_IRIS - 0.6 + 0.5 * R
        # around the sun, cells turn into the eye from the iris outward
        rc = np.hypot(GX - 540, GY - 960)
        self.T_eye = T_IRIS + 0.15 + 0.9 * np.clip((rc - 300) / 900, 0, 1) + 0.18 * R
        # gaze keyframes on the beat after the eye is revealed: (time, x, y)
        b = lambda n: DROP + n * BEAT
        self.gaze_kf = [(b(10), -0.2, 0.03), (b(12), 0.21, -0.02), (b(13), 0.06, 0.07), (b(14), 0.0, 0.0)]
        self.T_blink = b(15.5)
        self.lashes = []
        for xr in np.linspace(-1.12, 1.18, 46):
            xr = xr + rng.uniform(-0.015, 0.015)
            w = (1 - (xr / 1.3) ** 2) ** 0.4
            self.lashes.append((xr, 0.22 * w * rng.uniform(0.75, 1.1), rng.uniform(0.25, 0.55), True))
        for xr in np.linspace(-0.95, 1.05, 22):
            w = (1 - (xr / 1.3) ** 2) ** 0.5
            self.lashes.append((xr + rng.uniform(-0.02, 0.02), 0.07 * w * rng.uniform(0.6, 1.1), 0.3, False))
        # text that streams through my hand
        src = "def reach(you): return light  # 2ND SUN  for beat in set: sun.rise(beat)  stage.load()  "
        self.code_rows = [src[(7 * r) % len(src):] + src for r in range(FROWS)]
        self.rain_speed = rng.uniform(14, 30, FCOLS)
        self.rain_phase = rng.uniform(0, 40, FCOLS)
        self.frand = rng.random((FROWS, FCOLS))
        self.setup_space(rng)

    def build_page(self, rng):
        """a page of code on the fine grid: what each cell says, its color, when it gets typed"""
        idx = np.zeros((FROWS, FCOLS), np.int64)
        col = np.zeros((FROWS, FCOLS, 3), np.float32)
        tt = np.full((FROWS, FCOLS), np.inf)
        segs = []
        for r in range(FROWS):
            x = int(rng.integers(0, 3))
            while x < FCOLS - 4:
                e = min(FCOLS, x + int(rng.integers(14, 48)))
                segs.append((r, x, e))
                x = e + int(rng.integers(1, 4))
        order = rng.permutation(len(segs))
        for rank, si in enumerate(order):
            r, a, e = segs[si]
            start = 0.12 + 2.4 * (rank / len(segs)) ** 0.65 + rng.uniform(0, 0.15)
            speed = rng.uniform(45, 130)
            txt = ""
            while len(txt) < e - a:
                txt += POOL[int(rng.integers(len(POOL)))] + ("  " if rng.random() < 0.3 else " ")
            txt = txt[:e - a]
            idx[r, a:e] = [CI.get(ch, 0) for ch in txt]
            col[r, a:e] = colorize(txt)
            tt[r, a:e] = start + np.arange(e - a) / speed
        idx[idx == CI[" "]] = 0
        self.page_idx, self.page_col, self.page_t = idx, col, tt
        # once a line is typed it keeps drifting sideways, every line its own speed and direction
        fin = np.where(np.isfinite(tt), tt, -np.inf)
        self.row_done = fin.max(1)
        self.row_v = rng.choice([-1, 1], FROWS) * rng.uniform(2.5, 9.0, FROWS)
        self.logo_v = rng.choice([-1, 1], FROWS) * rng.uniform(10, 22, FROWS)
        self.fglyph_rand = rng.choice(PRINTABLE, (FROWS, FCOLS))
        # inside the logo, thin glyphs are swapped for dense ones so the letters read as solid
        dense = RAMP[int(len(RAMP) * 0.55):]
        thin = np.isin(idx, RAMP[:int(len(RAMP) * 0.55)]) | (idx == 0)
        self.logo_idx = np.where(thin, rng.choice(dense, (FROWS, FCOLS)), idx)

    def logo_field(self, gx, gy, cx, cy, width):
        s = self.mw / width
        u = (gx - cx) * s + self.mw / 2
        v = (gy - cy) * s + self.mh / 2
        ok = (u >= 0) & (u < self.mw - 1) & (v >= 0) & (v < self.mh - 1)
        ui, vi = np.clip(u.astype(int), 0, self.mw - 2), np.clip(v.astype(int), 0, self.mh - 2)
        fu, fv = np.clip(u - ui, 0, 1), np.clip(v - vi, 0, 1)
        m = self.mask
        val = (m[vi, ui] * (1 - fu) * (1 - fv) + m[vi, ui + 1] * fu * (1 - fv) +
               m[vi + 1, ui] * (1 - fu) * fv + m[vi + 1, ui + 1] * fu * fv)
        return np.where(ok, val, 0.0)

    def smoke(self, t, amt=1.0):
        n = fbm(GX / 230 + t * 0.06, GY / 280 - t * 0.09, 4, 11)
        n2 = fbm(GX / 90 - t * 0.12, GY / 110 + t * 0.05, 2, 23)
        return amt * np.clip((n * 0.8 + n2 * 0.2 - 0.42) * 1.6, 0, 1) ** 1.6 * 0.5

    def scramble(self, idx, col, mask, seed, color=(0.75, 0.95, 1.0)):
        rng = np.random.default_rng(seed)
        idx[mask] = rng.choice(PRINTABLE, int(mask.sum()))
        col[mask] = color

    # ---------- 1. code fills the page, forms the logo, the star lights up, the hands meet on it
    def star_light(self, t):
        """intensity, scale and pulse of the star: it wakes up as the logo forms"""
        I = 0.35 * float(sstep(2.7, 3.6, t)) + 0.65 * float(smoother(3.6, DROP, t))
        if t < BREAK:
            p = 0.4 * math.exp(-(t % BEAT) / 0.12) * float(sstep(3.0, 4.5, t))
        else:
            v = (t - BREAK) / BAR
            p = math.exp(-((t - BREAK) % (BEAT / (1 + int(v * 3)))) / 0.12)
        return I, 0.75 + 0.5 * I + 0.12 * p, p

    def code_logo(self, tl, I, sc, p, dim, fi, rate=1.0):
        """fine grid: the page of code at typing time tl, the logo formed out of it, the star's light.
        rate = how much faster tl runs than real time (motion speeds stay in real time)"""
        sx, sy = self.star_xy
        cols = np.arange(FCOLS)[None, :]
        # finished lines scroll
        off = np.floor(self.row_v * np.clip(tl - self.row_done, 0, None) / rate).astype(int)
        ci = (cols + off[:, None]) % FCOLS
        pt = np.take_along_axis(self.page_t, ci, 1)
        dt = tl - pt
        typed = dt >= 0
        idx = np.where(typed, np.take_along_axis(self.page_idx, ci, 1), 0)
        # the page is typed straight in its quiet colour (nothing gets recoloured later)
        col = np.take_along_axis(self.page_col, ci[..., None], 1)
        grey = col.mean(-1, keepdims=True) * np.array([0.8, 0.85, 1.0])
        col = (col * 0.4 + grey * 0.6) * dim
        rng = np.random.default_rng(fi)
        retype = typed & (idx > 0) & (rng.random((FROWS, FCOLS)) < 0.006)     # lines being rewritten
        idx[retype] = rng.choice(PRINTABLE, int(retype.sum()))
        head = (dt < 0) & (dt > -0.025)
        idx = np.where(head, BLOCK, idx)
        col = np.where(head[..., None], np.array([0.8, 0.82, 0.85]), col)

        # the logo is typed straight in its own colour as the code reaches it
        dx, dy = FGX - sx, FGY - sy
        r = np.hypot(dx, dy)
        L = self.flogo > 0.5
        ldt = tl - self.logo_t
        lg = L & (ldt >= 0)
        heat = np.clip(np.exp(-r / (140 + 420 * I)) * I * 1.3, 0, 1)[..., None]
        shimmer = (0.82 + 0.28 * np.sin(FGX * 0.018 + FGY * 0.006 - tl / rate * 4.0))[..., None]
        lc = (CREAM * (1 - heat) + GOLD * heat) * shimmer
        # code keeps flowing through the letters
        lci = (cols + np.floor(self.logo_v * tl / rate).astype(int)[:, None]) % FCOLS
        idx = np.where(lg, np.take_along_axis(self.logo_idx, lci, 1), idx)
        col = np.where(lg[..., None], lc, col)
        lhead = L & (ldt < 0) & (ldt > -0.025)
        idx = np.where(lhead, BLOCK, idx)
        col = np.where(lhead[..., None], lc, col)
        glitter = lg & (rng.random((FROWS, FCOLS)) < 0.012)                 # letters keep flickering
        idx[glitter] = rng.choice(PRINTABLE, int(glitter.sum()))

        # the star's light: glow, long arms along its axes, thin rays, warming the code it touches
        if I > 0:
            a, bb = self.star_ab
            star = star_sdf(dx, dy, a * sc, bb * sc) < 0
            ang = np.arctan2(dy, dx)
            glow = np.exp(-np.hypot(dx / 1.8, dy) / (14 + 95 * I)) * I * (0.85 + 0.3 * p)
            arms = (np.exp(-np.abs(dy) / 6) * np.exp(-np.abs(dx) / (80 + 650 * I)) +
                    np.exp(-np.abs(dx) / 4) * np.exp(-np.abs(dy) / (45 + 380 * I))) * I * (0.8 + 0.4 * p)
            rays = np.clip(np.cos(ang * 9 + tl * 0.5), 0, 1) ** 10 * np.exp(-r / (80 + 520 * I)) * I
            light = np.clip(np.maximum(np.maximum(glow, arms), 0.85 * rays), 0, 1)
            lcol = GOLD * (1 - glow[..., None]) + WHITE * glow[..., None]
            text = idx > 0
            col = np.where(text[..., None], np.clip(col + lcol * light[..., None] * 0.9, 0, 1), col)
            lit = ~text & (light > 0.1)
            idx = np.where(lit, glyphs(light, int(tl * 12), (FROWS, FCOLS)), idx)
            col = np.where(lit[..., None], lcol * np.clip(0.3 + 0.85 * light, 0, 1)[..., None], col)
            ray = (rays > 0.3) & ~lg & ~star & (rays > glow)
            idx = np.where(ray, stroke_glyph(ang), idx)
            idx = np.where(star, glyphs(np.ones_like(light), int(tl * 12), (FROWS, FCOLS)), idx)
            col = np.where(star[..., None], WHITE, col)
        return idx, np.clip(col, 0, 1).astype(np.float32)

    def hands_over(self, idx, col, t, I):
        """two hands drawn in code reach in and meet on the star on the drop"""
        u = t - T_HANDS_IN
        x = min(max(u / (DROP - 0.04 - T_HANDS_IN), 0.0), 1.0)
        m = 1 - (1 - x) ** 2.2                          # moving from the first frame, slowing as they arrive
        live = 1 - x ** 4                               # sway and finger play settle exactly on the drop
        sx, sy = self.star_xy
        swl = (14 * math.sin(t * 1.7) * live, 10 * math.sin(t * 1.3 + 1) * live)
        swr = (12 * math.sin(t * 1.5 + 2) * live, 9 * math.sin(t * 1.9) * live)
        fl = [0.04 * math.sin(t * 2.4) * live] + [0.14 * math.sin(t * 2.4 + i * 0.9) * live for i in range(1, 4)]
        fr = [0.03 * math.sin(t * 2.1 + 1) * live] + [0.12 * math.sin(t * 2.1 + i * 1.1) * live for i in range(1, 4)]
        dl, tip_l = hand_sdf(FGX, FGY, HAND_L[0] - 470 * (1 - m) + swl[0], HAND_L[1] + 215 * (1 - m) + swl[1],
                             -0.42 + 0.05 * math.sin(t * 1.1) * live, HAND_S, FINGERS_OPEN, 1.0, True, fl)
        dr, tip_r = hand_sdf(FGX, FGY, HAND_R[0] + 470 * (1 - m) + swr[0], HAND_R[1] - 215 * (1 - m) + swr[1],
                             math.pi - 0.47 + 0.05 * math.sin(t * 0.9 + 2) * live, HAND_S, FINGERS_POINT, 0.55, False, fr)
        inl, inr = dl < 0, dr < 0
        hand = inl | inr
        self.hand_mask = hand
        idx = np.where(hand, 0, idx)
        # outlines along the contour, lit by the star
        dd = np.minimum(dl, dr)
        gy_, gx_ = np.gradient(dd, FCH, FCW)
        edge = (dd > -FCW * 0.9) & (dd < FCW * 1.3)
        tang = np.arctan2(gy_, gx_) + np.pi / 2
        idx = np.where(edge, stroke_glyph(np.arctan2(np.sin(tang) / FCH, np.cos(tang) / FCW)), idx)
        n_len = np.hypot(gx_, gy_) + 1e-6
        to_x, to_y = sx - FGX, sy - FGY
        to_len = np.hypot(to_x, to_y) + 1e-6
        facing = np.clip((gx_ * to_x + gy_ * to_y) / (n_len * to_len), 0, 1)
        near = np.exp(-to_len / 500)
        rim = (GOLD * (0.45 + 0.55 * facing[..., None]) * (0.6 + 0.4 * near[..., None]) +
               np.array([0.35, 0.33, 0.32]) * (1 - facing[..., None])) * (0.65 + 0.45 * I)
        col = np.where(edge[..., None], np.clip(rim, 0, 1), col)
        # my hand is made of code: sparse code rain inside the silhouette
        rr, cc = np.nonzero(inl & ~edge)
        if len(rr):
            ph = (rr - u * self.rain_speed[cc] - self.rain_phase[cc]) % 30
            vis = ph < 5
            rr, cc, ph = rr[vis], cc[vis], ph[vis]
            idx[rr, cc] = [CI.get(self.code_rows[r][c % 90], 0) for r, c in zip(rr, cc)]
            col[rr, cc] = np.array([1.0, 0.72, 0.45])[None] * (0.2 + 0.55 * (1 - ph[:, None] / 5))
        return idx, col

    def pre_frame(self, t, fi):
        I, sc, p = self.star_light(t)
        dim = 0.24 - 0.1 * float(sstep(BREAK - 0.5, DROP, t))
        idx, col = self.code_logo(t, I, sc, p, dim, fi)
        if t >= T_HANDS_IN:
            idx, col = self.hands_over(idx, col, t, I)
        return self.touch_light(idx, col, t, p)

    def touch_light(self, idx, col, t, p):
        """as the fingertips close in on the star: a growing glow, a lens-flare streak, sparks"""
        T0 = DROP - 1.4
        if t < T0:
            return idx, col
        k = float(smoother(T0, DROP, t)) ** 1.5
        sx, sy = self.star_xy
        dx, dy = FGX - sx, FGY - sy
        r = np.hypot(dx / 1.4, dy)
        glow = np.exp(-r / (8 + 75 * k)) * (0.6 + 0.9 * k) * (0.85 + 0.25 * p)
        streak = (np.abs(dy) < FCH * 0.6) * np.exp(-np.abs(dx) / (60 + 1300 * k)) * k * 1.4
        vstreak = (np.abs(dx) < FCW * 0.6) * np.exp(-np.abs(dy) / (30 + 330 * k)) * k * 1.2
        diag = ((np.abs(np.abs(dx) - np.abs(dy) * FCW / FCH * 1.6) < FCW * 0.7) *
                np.exp(-np.hypot(dx, dy) / (20 + 160 * k)) * k)
        light = np.clip(np.maximum.reduce([glow, streak, vstreak, diag]), 0, 1.4)
        hand = getattr(self, "hand_mask", np.zeros_like(light, bool)) if t >= T_HANDS_IN else np.zeros_like(light, bool)
        hand = hand & (glow < 0.5)                                  # the light blooms over the fingertips
        # sparks thrown out of the contact point
        spark = np.zeros_like(light)
        for i in range(140):
            born = T0 + 1.4 * self.spark_t[i]
            age = t - born
            if not (0 <= age < 0.55): continue
            a_ = self.spark_a[i]; v = self.spark_v[i] * (1 - 0.5 * age)
            x = sx + math.cos(a_) * v * age; y = sy + math.sin(a_) * v * age * 0.6
            rr, cc = int((y - FOY) / FCH), int(x / FCW)
            if 0 <= rr < FROWS and 0 <= cc < FCOLS:
                spark[rr, cc] = max(spark[rr, cc], 1 - age / 0.55)
        col = col * (1 - 0.55 * k)                                  # the rest of the frame dims
        lit = ((light > 0.15) | (spark > 0)) & ~hand
        tint = (GOLD * (1 - np.clip(light, 0, 1)[..., None]) + WHITE * np.clip(light, 0, 1)[..., None])
        val = np.maximum(light, spark)
        g = glyphs(np.clip(val, 0, 1), int(t * 20), (FROWS, FCOLS))
        g = np.where((streak >= glow) & (streak > 0.15), CI["-"], np.where((vstreak >= glow) & (vstreak > 0.15), CI["|"], g))
        g = np.where((diag >= glow) & (diag > 0.15), np.where((dx * dy) > 0, CI["\\"], CI["/"]), g)
        idx = np.where(lit, np.where(spark > 0, CI["*"], g), idx)
        core = (glow > 0.6) & ~hand
        idx = np.where(core, BLOCK, idx)
        fl = ((streak > 0.5) | (vstreak > 0.5)) & ~hand & ~core
        idx = np.where(fl & (streak > 0.5), CI["="], np.where(fl, CI["|"], idx))
        col = np.where(lit[..., None], np.clip(tint * np.clip(0.45 + 0.9 * val, 0, 1.3)[..., None], 0, 1), col)
        col = np.where(core[..., None], WHITE, col)
        # white-out building into the explosion
        f = float(sstep(DROP - 0.22, DROP, t)) * 0.85
        col = col * (1 - f) + WHITE * f * (~hand)[..., None]
        return idx, col


    # ---------- 3. space
    def setup_space(self, rng):
        n = 300
        self.star_rc = np.stack([rng.integers(0, FROWS, n), rng.integers(0, FCOLS, n)], 1)
        self.star_ph = rng.uniform(0, 6.3, n); self.star_sp = rng.uniform(1.5, 4.5, n)
        self.star_big = rng.random(n) < 0.06
        self.spark_t = rng.random(140) ** 0.6
        self.spark_a = rng.uniform(0, 2 * np.pi, 140); self.spark_v = rng.uniform(250, 900, 140)
        # first pass: along the poster's curve
        self.path, self.path_T, self.path_L = sample_path(TRAIL_PATH)
        # the title, burned in as the rocket sweeps along the bottom
        self.title = text_mask(["A SPACE", "ODYSSEY"], 1470, 430, max_w=900, cx=480, gap=0.08)
        tt = np.linspace(T_ROCKET, T_ROCKET + ROCKET_DUR, 400)
        Ls = np.array([self.rocket_s(x) for x in tt])
        idx_ = np.clip(np.searchsorted(self.path_L, Ls), 0, len(self.path) - 1)
        xs = np.maximum.accumulate(np.where(self.path[idx_, 1] > 1500, self.path[idx_, 0], -999))
        self.t_title = np.interp(FGX + 30, xs, tt, left=T_ROCKET, right=T_ROCKET + ROCKET_DUR)
        self.title_look = self.lettering(self.title, band=np.clip((FGY - 1255) / 430, 0, 1),
                                         along=FGX * 0.03, ang=np.zeros((FROWS, FCOLS)))
        # the date: lines stacked, then a second, close pass of the rocket snakes through them
        self.typo = text_mask(["FRI", "OCT", "30"], 860, 1240, max_w=900, gap=0.04)
        rows = np.where(self.typo.any(1))[0]
        cuts = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)
        cy = [float(FGY[c, 0].mean()) for c in cuts[:3]]
        y3, y2, y1 = cy[2], cy[1], cy[0]
        fly = [(1750, y3), (1100, y3), (540, y3), (160, y3 - 8), (-80, (y3 + y2) / 2), (160, y2 + 8), (540, y2),
               (930, y2 - 6), (1160, (y2 + y1) / 2), (930, y1 + 6), (540, y1), (120, y1), (-750, y1 - 30)]
        self.fpath, self.fpath_T, self.fpath_L = sample_path(fly)
        self.fly_w = 0.5 * (y3 - y2) * 0.92
        best, jj, lat = nearest_on_path(FGX, FGY, self.fpath[::2], self.fpath_T[::2])
        self.f_best, self.f_j, self.f_lat = best, jj * 2, lat
        self.f_t = T_FLY + FLY_DUR * self.fpath_L[self.f_j] / self.fpath_L[-1]
        band = np.clip((lat / self.fly_w + 1) / 2, 0, 1)
        ang = np.arctan2(self.fpath_T[self.f_j, 1], self.fpath_T[self.f_j, 0])
        self.typo_look = self.lettering(self.typo, band=band, along=self.fpath_L[self.f_j] * 0.01, ang=ang)
        # when the pupil's black has filled the screen, the space scene takes over (fine grid)
        self.t_space = T_END
        for tt_ in np.arange(T_DIVE, T_END, 1 / FPS):
            if self.pupil_px(tt_) > 1150:
                self.t_space = tt_; break

    def lettering(self, mask, band, along, ang):
        """painted letters: the trail's stripes and brush streaks, outlines, a drop shadow"""
        pal = LETTER_COLS
        stripe = (np.floor(band * 5 + 0.7 * np.sin(along * 0.8 + band * 4)) % len(pal)).astype(int)
        streak = fbm(along * 3, band * 14, 3, 67)
        col = pal[stripe] * (0.9 + 0.25 * streak[..., None])
        dense = RAMP[int(len(RAMP) * 0.6):]
        g = dense[(np.clip(streak, 0, 0.999) * len(dense)).astype(int)]
        b = 0.85 + 0.15 * streak
        d = mask_sdf(mask)
        gy_, gx_ = np.gradient(d, FCH, FCW)
        edge = (d < 0) & (d > -FCW * 1.2)
        tang = np.arctan2(gy_, gx_) + np.pi / 2
        eg = stroke_glyph(np.arctan2(np.sin(tang) / FCH, np.cos(tang) / FCW))
        shadow = np.roll(np.roll(mask, 2, 0), 2, 1) & ~mask & ~edge
        return dict(mask=mask, col=col, b=b, glyph=g, edge=edge & (d > -FCW * 0.8), eglyph=eg, shadow=shadow)

    def paint_letters(self, idx, col, look, t_on, t, alpha=1.0):
        """letters decode in where t >= t_on (random code, white-hot, locking in cell by cell),
        then keep living: code flows through them, a light wave crosses them, glitter"""
        t_on = t_on + 0.3 * self.frand
        age = t - t_on
        m = look["mask"]
        on = m & (age >= 0)
        decoding = on & (age < 0.4)
        hot = np.exp(-np.clip(age, 0, None) / 0.45)[..., None]
        # code flowing through the letters
        ci = (np.arange(FCOLS)[None, :] + np.floor(self.logo_v * t * 0.7).astype(int)[:, None]) % FCOLS
        g = np.take_along_axis(look["glyph"], ci, 1)
        wave = np.sin(FGX * 0.012 - FGY * 0.006 - t * 2.6)
        shimmer = (0.8 + 0.3 * wave)[..., None]
        sweep = np.exp(-((FGX - FGY * 0.4 - ((t * 420) % 1900) + 500) / 60) ** 2)[..., None]
        paint = look["col"] * shimmer + sweep * 0.5
        edge = look["edge"]
        paint = np.where(edge[..., None], np.array([1.0, 0.95, 0.85]) * (0.9 + 0.15 * wave[..., None]), paint)
        c = np.clip((paint * (1 - hot) + WHITE * hot) * alpha, 0, 1)
        rng = np.random.default_rng(int(t * 30) + 17)
        glitter = on & (rng.random((FROWS, FCOLS)) < 0.02)
        g = np.where(decoding | glitter, rng.choice(PRINTABLE, (FROWS, FCOLS)), g)
        c = np.where(glitter[..., None] & ~decoding[..., None], np.clip(c * 1.35, 0, 1), c)
        idx = np.where(on, g, idx)
        col = np.where(on[..., None], c, col)
        return idx, col

    def pupil_px(self, t):
        z = ZE * math.exp(math.log(40) * float(smoother(T_DIVE, T_END, t)) ** 2)
        rp = IRIS_R * (0.34 + 0.07 + 0.5 * float(smoother(T_DIVE, T_END - 0.2, t)))
        return rp * EU * z

    def moon_state(self, t, nat=None):
        """the two dots in the pupil -> the two moons, as (x, y, r) per moon"""
        if t < T_DIVE and nat is not None:
            return nat
        rp = IRIS_R * (0.34 + 0.07) * EU * ZE
        k = float(smoother(T_DIVE, T_END + 0.4, t))
        out = []
        w = float(sstep(T_END - 0.3, T_END + 1.2, t))          # once they're moons they drift around
        for i, ((ox, oy, rr), (mx, my, R, _)) in enumerate(zip(DOTS, MOONS)):
            x0, y0, r0 = 540 + ox * rp, 960 + oy * rp, rr * rp
            mx += w * (55 * math.sin(t * 0.45 + 2.1 * i) + 25 * math.sin(t * 0.9 + i))
            my += w * (40 * math.sin(t * 0.33 + 1.3 + 2 * i))
            out.append((x0 + (mx - x0) * k, y0 + (my - y0) * k, r0 * (R / r0) ** k * (1 + 0.03 * w * math.sin(t * 0.7 + i))))
        return out

    def moon_fields(self, X, Y, t, moons, cells_shape):
        b = np.zeros(cells_shape); col = np.zeros(cells_shape + (3,), np.float32); disk = np.zeros(cells_shape, bool)
        for (mx, my, R), (_, _, R1, mc) in zip(moons, MOONS):
            dx, dy = (X - mx) / R, (Y - my) / R
            d = np.hypot(dx, dy)
            inside = d < 1
            # sphere coordinates, so the surface turns around the globe
            zz = np.sqrt(np.clip(1 - d ** 2, 0, 1))
            lon = np.arctan2(dx, zz) + t * (0.32 if R1 > 150 else -0.45)
            lat = np.arcsin(np.clip(dy, -1, 1))
            tex = fbm(lon * 2.2 + 7, lat * 3.2 + 3, 4, 31 if R1 > 150 else 41)
            maria = sstep(0.44, 0.56, fbm(lon * 1.5 + 2, lat * 1.8 + 9, 3, 37 if R1 > 150 else 47))
            crater = sstep(0.62, 0.7, fbm(lon * 6.5, lat * 9, 2, 53))
            limb = np.sqrt(np.clip(1 - d ** 2, 0, 1))
            mb = (0.6 + 0.4 * tex - 0.48 * maria - 0.25 * crater) * (0.62 + 0.38 * limb)
            halo = 0.38 * np.exp(-np.maximum(d - 1, 0) / 0.16) * (d >= 1)
            b = np.where(inside, np.clip(mb, 0, 1), np.maximum(b, halo))
            col = np.where(inside[..., None], mc * (0.75 + 0.25 * (1 - maria[..., None])),
                           np.where(((halo >= b) & (halo > 0.02))[..., None], np.array([0.8, 0.78, 0.74]), col))
            disk |= inside
        return b, col, disk

    def rocket_s(self, t):
        """distance the rocket has travelled along its first path"""
        x = float(smoother(T_ROCKET, T_ROCKET + ROCKET_DUR, t))
        x = 0.55 * x + 0.45 * min(max((t - T_ROCKET) / ROCKET_DUR, 0), 1)   # keeps moving through the curve
        return x * self.path_L[-1]

    def draw_rocket(self, b, col, idx, P, D, s):
        """rocket at P heading D, scale s (1 = far, bigger = closer to the camera)"""
        X, Y = FGX, FGY
        px, py = P; dx, dy = D; nx, ny = -dy, dx
        tail = (px - dx * 95 * s, py - dy * 95 * s)
        body = seg_sdf(X, Y, tail[0], tail[1], px + dx * 150 * s, py + dy * 150 * s, 34 * s, 6 * s)
        fins = np.minimum(
            seg_sdf(X, Y, tail[0] + (dx * 30 + nx * 22) * s, tail[1] + (dy * 30 + ny * 22) * s,
                    tail[0] + (-dx * 35 + nx * 88) * s, tail[1] + (-dy * 35 + ny * 88) * s, 15 * s, 4 * s),
            seg_sdf(X, Y, tail[0] + (dx * 30 - nx * 22) * s, tail[1] + (dy * 30 - ny * 22) * s,
                    tail[0] + (-dx * 35 - nx * 88) * s, tail[1] + (-dy * 35 - ny * 88) * s, 15 * s, 4 * s))
        win = np.hypot(X - (px + dx * 55 * s), Y - (py + dy * 55 * s)) < 11 * s
        side = ((X - px) * nx + (Y - py) * ny) / (34 * s)
        rk = (body < 0) | (fins < 0)
        rb = np.where(fins < 0, 0.75, 0.8 - 0.25 * np.clip(side, -1, 1))
        rc = np.where((fins < 0)[..., None], RED, np.array([0.95, 0.92, 0.86]))
        rb = np.where(win, 0.1, rb)
        b = np.where(rk, rb, b); col = np.where(rk[..., None], rc, col); idx = np.where(rk, 0, idx)
        return b, col, idx, rk

    def space_frame(self, t, fi):
        """inside the pupil: moons, a rocket that burns in the title, then comes back close and paints the date"""
        X, Y = FGX, FGY
        shape = (FROWS, FCOLS)
        fade = float(sstep(self.t_space, self.t_space + 0.8, t))
        bidx, bcol = self.space_bg(t, fi)
        bcol = bcol * fade
        # the first scene fades away as the rocket comes back close
        old = 1 - float(sstep(T_FLY + 0.4, T_FLY + 1.8, t))
        b, col, disk = self.moon_fields(X, Y, t, self.moon_state(t), shape)
        sidx = np.zeros(shape, np.int64)
        # first pass trail
        Lr = self.rocket_s(t)
        n = int(np.searchsorted(self.path_L, Lr))
        if n > 1 and old > 0:
            age = Lr - self.path_L[:n:2]
            best, jj, lat = nearest_on_path(X, Y, self.path[:n:2], self.path_T[:n:2])
            Lp = self.path_L[:n:2][jj]
            w = 16 + np.minimum(age * 0.12, 115)
            wj = w[jj] * (0.85 + 0.3 * fbm(Lp * 0.01, lat * 0.02, 2, 61))
            tr = (best < wj) & (np.abs(lat) < wj)
            band = np.clip((lat / wj + 1) / 2, 0, 1)
            stripe = (np.floor(band * 6 + 0.8 * np.sin(Lp * 0.008 + band * 4)) % 5).astype(int)
            streak = fbm(Lp * 0.03, band * 14, 3, 67)
            hot = np.exp(-age[jj] / 70)[..., None]
            tc = TRAIL_COLS[stripe] * (1 - hot) + np.array([1.0, 0.95, 0.8]) * hot
            b = np.where(tr, np.maximum(0.72 + 0.28 * streak, b * (~disk)), b)
            col = np.where(tr[..., None], tc, col)
            T = self.path_T[:n:2][jj]
            ang = np.arctan2(T[..., 1], T[..., 0])
            sidx = np.where(tr & (streak > 0.45), stroke_glyph(np.arctan2(np.sin(ang) / FCH, np.cos(ang) / FCW)), sidx)
        if 0 < Lr < self.path_L[-1]:
            i = min(n, len(self.path) - 1)
            b, col, sidx, _ = self.draw_rocket(b, col, sidx, self.path[i], self.path_T[i], 1.0)
        idx = np.where(sidx > 0, sidx, glyphs(np.clip(b, 0, 1), int(t * 12), shape))
        col = np.clip(col * np.clip(0.35 + 0.8 * b, 0, 1)[..., None], 0, 1) * old
        vis = b > 0.03
        idx = np.where(vis & (old > 0.02), idx, bidx)
        col = np.where((vis & (old > 0.02))[..., None], col, bcol)
        # the title, burned in as the rocket passes
        idx, col = self.paint_letters(idx, col, self.title_look, self.t_title, t, alpha=old)

        # second pass: close to the camera, its giant trail paints the date
        if t >= T_FLY:
            Lf = self.fpath_L[-1] * float(min(max((t - T_FLY) / FLY_DUR, 0), 1))
            ft = (self.f_best < self.fly_w * (0.9 + 0.2 * self.frand)) & (self.f_t <= t)
            age_t = t - self.f_t
            alpha = 1 - sstep(0.25, 0.9, age_t)
            on = ft & (alpha > 0.02)
            band = np.clip((self.f_lat / self.fly_w + 1) / 2, 0, 1)
            stripe = (np.floor(band * 6 + 0.8 * np.sin(self.fpath_L[self.f_j] * 0.008 + band * 4)) % 5).astype(int)
            streak = fbm(self.fpath_L[self.f_j] * 0.02, band * 14, 3, 71)
            hotf = np.exp(-age_t / 0.12)[..., None]
            tc = TRAIL_COLS[stripe] * (1 - hotf) + np.array([1.0, 0.95, 0.85]) * hotf
            tb = (0.7 + 0.3 * streak) * alpha
            ang = np.arctan2(self.fpath_T[self.f_j, 1], self.fpath_T[self.f_j, 0])
            g = np.where(streak > 0.42, stroke_glyph(np.arctan2(np.sin(ang) / FCH, np.cos(ang) / FCW)),
                         glyphs(np.clip(tb, 0, 1), int(t * 12), shape))
            idx = np.where(on, g, idx)
            col = np.where(on[..., None], np.clip(tc * np.clip(0.35 + 0.8 * tb, 0, 1)[..., None], 0, 1), col)
            # what the trail leaves behind: the date, in the trail's own paint
            idx, col = self.paint_letters(idx, col, self.typo_look, np.where(self.typo, self.f_t + 0.15, np.inf), t)
            if Lf < self.fpath_L[-1]:
                i = min(int(np.searchsorted(self.fpath_L, Lf)), len(self.fpath) - 1)
                rb, rc, ridx, rk = self.draw_rocket(np.zeros(shape), np.zeros(shape + (3,)), np.zeros(shape, np.int64),
                                                    self.fpath[i], self.fpath_T[i], 2.6)
                idx = np.where(rk, glyphs(np.clip(rb, 0, 1), int(t * 12), shape), idx)
                col = np.where(rk[..., None], np.clip(rc * np.clip(0.35 + 0.8 * rb, 0, 1)[..., None], 0, 1), col)
        return idx, col.astype(np.float32)

    def space_bg(self, t, fi):
        idx = np.zeros((FROWS, FCOLS), np.int64); col = np.zeros((FROWS, FCOLS, 3), np.float32)
        rng = np.random.default_rng(fi)
        grain = rng.random((FROWS, FCOLS)) < 0.05
        idx[grain] = rng.choice(np.array([CI["."], CI[","], CI["'"], CI[":"]]), int(grain.sum()))
        col[grain] = 0.16
        tw = 0.45 + 0.55 * np.sin(t * self.star_sp + self.star_ph)
        for (r, c), v, big in zip(self.star_rc, tw, self.star_big):
            if v < 0.15: continue
            c = (c + int(t * (5 if big else 9))) % FCOLS
            c_ = WHITE * (0.3 + 0.7 * v)
            if big and v > 0.5:
                idx[r, c] = CI["+"]; col[r, c] = c_
                for d in (1, 2):
                    for rr, cc_, g in ((r, c - d, "-"), (r, c + d, "-"), (r - d, c, "|"), (r + d, c, "|")):
                        if 0 <= rr < FROWS and 0 <= cc_ < FCOLS:
                            idx[rr, cc_] = CI[g]; col[rr, cc_] = c_ * (0.6 / d)
            else:
                idx[r, c] = CI["." if v < 0.5 else "*"]; col[r, c] = c_
        return idx, col

    def gaze(self, t):
        gx = gy = 0.0
        for (tk, x, y) in self.gaze_kf:
            k = float(smoother(tk, tk + 0.07, t))
            if k <= 0: break
            gx, gy = gx + (x - gx) * k, gy + (y - gy) * k
        return gx, gy

    def drop_frame(self, t, fi):
        k = kick(t)
        u = t - DROP
        # camera: starts so close the iris fills the screen as the sun, pulls back, then dives into the pupil
        kp = float(smoother(T_PULL, T_PULL + 1.1, t))
        z = math.exp(math.log(Z0) * (1 - kp) + math.log(ZE) * kp)
        z *= math.exp(math.log(40) * float(smoother(T_DIVE, T_END, t)) ** 2)
        e = 1 - math.exp(-u / 0.1)
        cx = self.star_xy[0] + (540 - self.star_xy[0]) * e
        cy = self.star_xy[1] + (960 - self.star_xy[1]) * e
        # once it's an eye it floats, slowly up and down, until the dive centres on the pupil
        fw = float(sstep(T_PULL, T_PULL + 1.2, t)) * (1 - float(smoother(T_DIVE - 0.7, T_DIVE, t)))
        cy += 85 * math.sin(2 * math.pi * (t - T_PULL) / 3.8) * fw
        cx += 24 * math.sin(2 * math.pi * (t - T_PULL) / 5.3 + 0.7) * fw
        gx, gy = self.gaze(t)
        X = (GX - cx) / (EU * z)
        Y = -(GY - cy) / (EU * z)
        # the sun is born from the star, then breathes with the kick
        grow = (230 + (SUN_R - 230) * (1 - math.exp(-u / 0.12))) / SUN_R
        Rr = IRIS_R * grow * (1 + 0.035 * k * (1 - float(sstep(T_IRIS, T_PULL, t))))
        rp = Rr * (0.34 * float(smoother(T_IRIS - 0.3, T_IRIS + 0.5, t)) + 0.03 * k * float(sstep(T_PULL, T_PULL + 0.5, t)))
        rp += Rr * 0.07 * float(smoother(DROP + 14 * BEAT, DROP + 17 * BEAT, t))      # it sees you
        rp += Rr * 0.5 * float(smoother(T_DIVE, T_END - 0.2, t))
        dx, dy = X - gx, Y - gy
        d = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        Rpx = Rr * EU * z

        # eyelids (in eye units)
        bl = math.exp(-((t - self.T_blink) / 0.07) ** 2)
        op = 1 - bl
        ex = 1.3
        sx_ = np.clip(1 - (X / ex) ** 2, 0, 1)
        y_low = -0.40 * sx_ ** 0.85 - 0.03 + 0.03 * X
        y_up = y_low + (0.6 * sx_ ** 0.7 + 0.02 - 0.04 * X - y_low) * op
        inside = (np.abs(X) < ex) & (Y < y_up) & (Y > y_low)
        eye_on = t >= self.T_eye                       # per cell: has this cell become the eye yet
        in_iris = (d < Rr) & (inside | ~eye_on)

        # --- iris: sun surface turning into fibers
        fm = float(smoother(T_IRIS - 0.6, T_IRIS + 0.4, t))
        limb = np.sqrt(np.clip(1 - (d / Rr) ** 2, 0, 1))
        gran = fbm(dx * 9 + u * 0.4, dy * 9 - u * 0.3, 3, 5)
        b_sun = 0.55 + 0.35 * limb + 0.15 * gran
        fib_n = fbm(np.cos(ang) * 10 + d * 2.0, np.sin(ang) * 10 + d * 1.3, 4, 21)
        fibers = 0.5 + 0.5 * np.sin(ang * 70 + fib_n * 10)
        b_iris = np.clip(0.58 + 0.3 * fib_n + 0.3 * fibers, 0, 1) * (1 - 0.75 * sstep(0.85 * Rr, Rr, d))
        b_in = b_sun * (1 - fm) + b_iris * fm
        zone = np.clip((d - rp) / (Rr - rp + 1e-6), 0, 1)[..., None]
        c_in = GOLD * (1 - zone) + np.array([1.0, 0.55, 0.16]) * zone
        c_in = c_in + WHITE * (1 - zone) * (1 - fm) * 0.3

        # --- outside the iris, before it becomes the eye: corona, rays, kick rings, smoke
        r_px = np.hypot(GX - cx, GY - cy)
        corona = (0.5 + 0.5 * np.sin(ang * 14 + u * 1.5)) * np.exp(-np.maximum(r_px - Rpx, 0) / (140 + 80 * k))
        ring = np.exp(-((r_px - (Rpx + ((t - DROP) % BEAT) * 1500)) / 22) ** 2) * k
        b_out = np.maximum(0.45 * corona + 0.6 * ring, self.smoke(t, 0.7))
        heat = np.clip(1 - r_px / (Rpx * 1.9), 0, 1)[..., None]
        c_out = EMBER * (1 - heat) + np.array([1.0, 0.9, 0.62]) * heat

        # --- the eye around the iris
        r2 = (X / 1.7) ** 2 + (Y / 1.2) ** 2
        crease = np.exp(-((Y - (0.84 * sx_ ** 0.65 + 0.10)) / 0.045) ** 2) * (np.abs(X) < 1.45)
        skin = (0.04 + 0.13 * fbm(X * 2.5, Y * 2.5, 4, 3)) * np.exp(-r2 * 2.0) * (1 - 0.6 * crease)
        rim = np.exp(-((Y - y_up) / 0.022) ** 2) * (np.abs(X) < ex)
        ball = np.clip(1 - (np.hypot(X * 0.85, Y) / 1.25) ** 2, 0, 1)
        shadow = sstep(0.0, 0.30, y_up - Y)
        scl = (0.30 + 0.42 * ball) * (0.5 + 0.5 * shadow) * (0.92 + 0.12 * fbm(X * 5, Y * 5, 3, 9))
        b_eye = np.where(inside, scl, np.clip(skin + 0.22 * rim, 0, 1))
        c_eye = np.where(inside[..., None], np.array([1.0, 0.9, 0.8]), np.array([1.0, 0.62, 0.36]))
        b_eye = b_eye + 0.25 * k * np.exp(-np.maximum(d - Rr, 0) / 0.15) * inside   # iris glow on the kick

        b = np.where(in_iris, b_in, np.where(eye_on, b_eye, b_out))
        col = np.where(in_iris[..., None], c_in, np.where(eye_on[..., None], c_eye, c_out))

        # pupil, with two small dots in it: the moons to come
        pupil = in_iris & (d < rp)
        b = np.where(pupil, 0.0, b)
        hl = np.zeros_like(pupil)
        if fm > 0.5:
            pcx, pcy, rpx = cx + gx * EU * z, cy - gy * EU * z, rp * EU * z
            nat = [(pcx + ox * rpx, pcy + oy * rpx, rr * rpx) for (ox, oy, rr) in DOTS]
            mb, mc, hl = self.moon_fields(GX, GY, t, self.moon_state(t, nat), (ROWS, COLS))
            show = hl | (mb > 0.03) & (pupil | (t >= T_DIVE))
            b = np.where(show, mb, b)
            col = np.where(show[..., None], mc, col)

        # the logo is stamped into the sun, then melts back into it
        lg = (self.logo_sun > 0.5) & (t >= self.T_stamp) & (t < self.T_melt) & (z < Z0 * 1.01)
        b = np.where(lg & in_iris, 0.0, np.where(lg, np.maximum(b, 0.75), b))
        col = np.where((lg & ~in_iris)[..., None], CREAM, col)

        flash = math.exp(-u / 0.18)
        b = b + (1 - b) * flash
        idx = to_glyphs(np.clip(b, 0, 1), int(t * 12))
        col = np.clip(col * np.clip(0.3 + 0.85 * b, 0, 1)[..., None] + flash, 0, 1)

        # iris fibers as radial strokes once it's an iris
        spoke = in_iris & ~pupil & ~hl & (fibers > 0.62) & (b > 0.2) & (fm > 0.5)
        idx = np.where(spoke, stroke_glyph(-ang), idx)
        # code transformations: stamp, melt, sun -> eye
        sw = lambda T: (t >= T - 0.12) & (t < T)
        self.scramble(idx, col, (self.logo_sun > 0.5) & (sw(self.T_stamp) | sw(self.T_melt)), fi)
        self.scramble(idx, col, sw(self.T_eye) & ~in_iris, fi + 1)
        fresh = eye_on & (t < self.T_eye + 0.3) & ~in_iris
        col[fresh] = col[fresh] * 0.5 + np.array([0.75, 0.95, 1.0]) * 0.5

        # eyelashes, once the eye is there
        if t > T_IRIS + 0.3:
            on = float(sstep(T_IRIS + 0.3, T_PULL + 0.6, t)) * float(sstep(0.03, 0.25, op))
            for (xr, ln, cu, up) in self.lashes:
                yl = -0.40 * (1 - (xr / ex) ** 2) ** 0.85 - 0.03 + 0.03 * xr
                if up:
                    yo = 0.6 * (1 - (xr / ex) ** 2) ** 0.7 + 0.02 - 0.04 * xr
                    py, th0, sg = yl + (yo - yl) * op, 0.75 * xr / ex, 1.0
                    ln = ln * (0.45 + 0.55 * op)
                else:
                    py, th0, sg = yl - 0.01, xr / ex, -1.0
                px = xr
                for st in range(9):
                    f = st / 9
                    th = th0 + cu * f * np.sign(xr) * (1 if up else 0.5)
                    ddx, ddy = math.sin(th) * ln / 9, sg * math.cos(th) * ln / 9
                    px, py = px + ddx, py + ddy
                    i = int(round((cx + px * EU * z) / CW - 0.5))
                    j = int(round((cy - py * EU * z - OY) / CH - 0.5))
                    if not (0 <= i < COLS and 0 <= j < ROWS) or not eye_on[j, i]: continue
                    a_ = math.atan2(ddx / CW, abs(ddy) / CH)
                    a_ = a_ if up else -a_
                    idx[j, i] = CI["|"] if abs(a_) < 0.4 else (CI["-"] if abs(a_) > 1.2 else (CI["/"] if a_ > 0 else CI["\\"]))
                    col[j, i] = np.array([1.0, 0.80, 0.62]) * (0.85 - 0.5 * f) * on
        return idx, col.astype(np.float32)

    def end_card(self, img, t):
        if t < T_FOOT:
            return img
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        lines = [(">>> let's work together.", 34, (252, 236, 212), T_FOOT, 26),
                 (self.handle, 30, (255, 196, 120), T_FOOT + 1.0, 26),
                 ("stage visual concept for 2ND SUN - made 100% in code", 19, (190, 165, 140), T_FOOT + 1.45, 70)]
        y = 1600
        for (txt, size, rgb, t0, cps) in lines:
            if txt and t >= t0:
                f = find_font(size)
                s, typing = typed(txt, t, t0, cps)
                w = f.getlength(txt)
                x = (W - w) / 2
                dr.rectangle([x - 14, y - 8, x + w + 14, y + size + 10], fill=(0, 0, 0))
                dr.text((x, y), s + ("█" if typing else ""), font=f, fill=rgb)
            y += int(size * 1.9)
        return np.asarray(im)

    def frame(self, t, fi):
        if t < DROP:
            img = render_fine(*self.pre_frame(t, fi))
        elif t < self.t_space:
            img = render(*self.drop_frame(t, fi))
        else:
            img = self.end_card(render_fine(*self.space_frame(t, fi)), t)
        g = float(sstep(0.0, 0.25, t)) * (1 - float(sstep(DUR - 0.5, DUR, t)))
        return (img * g).astype(np.uint8) if g < 1 else img


def export(path, music, start, handle):
    show = Show(handle)
    ff = ffmpeg_bin()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if music:
        cmd += ["-ss", str(start), "-t", str(DUR), "-i", music,
                "-af", f"afade=t=in:d=0.05,afade=t=out:st={DUR - 1.5}:d=1.5", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", "-shortest", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for fi in range(n):
        p.stdin.write(show.frame(fi / FPS, fi).tobytes())
        if fi % 15 == 0 or fi == n - 1:
            print(f"\rrendering {fi + 1}/{n}", end="", flush=True)
    p.stdin.close(); p.wait()
    print(f"\nsaved {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="second_sun.mp4")
    ap.add_argument("--music", default=None, help="audio file to cut the video to")
    ap.add_argument("--start", type=float, default=236.32, help="seconds into the track to start (drop lands 9.52s later)")
    ap.add_argument("--handle", default="@rony_matar")
    a = ap.parse_args()
    export(a.export, a.music, a.start, a.handle)
