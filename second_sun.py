"""
2ND SUN  |  stage visual concept, 9:16, cut to music. Every frame is text, no cuts:
each scene is transformed out of the one before it.

code fills the page -> the code forms the logo, and as it forms the star inside it
starts to shine -> two hands drawn in code reach in and meet on the star -> on the
drop it explodes into a sun with the logo stamped in it -> the sun becomes an iris,
the camera pulls back: one big eye -> it looks around on the beat and stares at you ->
dive into the pupil -> inside, an old phone drops in spinning and types "you need me" ->
the phone melts into the logo and the star lights up again -> let's work together.

Export:   python second_sun.py --music track.mp3 --start 236.32 --handle @you --export second_sun.mp4
          (--start = where to cut the track; the drop lands 9.52s in, 5 bars at 126 BPM)
"""
import argparse, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from code_to_monalisa import (W, H, CW, CH, COLS, ROWS, OY, CI, CHARS, BLOCK, PRINTABLE, RAMP,
                              sstep, smoother, fbm, find_font, compose, render, to_glyphs, ffmpeg_bin,
                              colorize)

FPS = 30
DUR = 28.6
BPM = 126.05
BEAT = 60 / BPM
BAR = 4 * BEAT
DROP = 5 * BAR                            # 9.52s: the kick comes back
BREAK = DROP - BAR                        # 7.62s: the riser stops, the hands close in
T_HANDS_IN = 2.8                          # the hands start reaching for the star
T_IRIS = DROP + 1.6 * BAR                 # the sun turns into an iris
T_PULL = DROP + 1.8 * BAR                 # camera pulls back: it's an eye
T_DIVE = DROP + 4.5 * BAR                 # dive into the pupil
T_END = DROP + 5 * BAR                    # 19.05s: inside the pupil, an old phone drops in
T_PHONE = T_END
T_OUTRO = T_PHONE + 2 * BAR + 2 * BEAT    # 23.81s: the phone melts into the logo
MORPH = 1.1                               # the phone melts into the logo
T_CARD = T_OUTRO + 2.1                    # let's work together

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

# ------------------------------------------------------------------ the old phone, ray-traced into characters
def rbox(px, py, pz, bx, by, bz, r):
    qx, qy, qz = np.abs(px) - bx, np.abs(py) - by, np.abs(pz) - bz
    out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r

def phone_sdf(x, y, z):
    d = rbox(x, y, z, 112, 310, 34, 18)                                   # body
    d = smin(d, rbox(x, y + 300, z, 118, 42, 40, 22), 30)                 # flared mouthpiece
    d = np.minimum(d, rbox(x - 72, y - 430, z, 6, 110, 6, 6))             # antenna
    d = np.minimum(d, np.sqrt((x - 72) ** 2 + (y - 548) ** 2 + z ** 2) - 15)  # antenna tip
    return d

LCD = (-92, 92, 182, 290)                 # x0, x1, y0, y1 on the front face
KEYS = [(kx, ky, lab) for ky, row in zip((105, 35, -35, -105, -175), ("123", "456", "789", "*0#", "<>!"))
        for kx, lab in zip((-74, 0, 74), row)]


class Phone:
    """drops in from above the frame spinning, bounces, buzzes, types on its screen"""
    def pose(self, u):
        # already coming in at the first frame, glides down, eases past and settles back once
        y = PHONE_FROM + (PHONE_Y - PHONE_FROM) * float(smoother(0.0, 1.6, u))
        y += 32 * math.sin(math.pi * min(max((u - 1.2) / 0.7, 0.0), 1.0))
        th = 2 * math.pi * float(smoother(0.0, 1.8, u))                     # one slow turn, ends facing us
        ph = 0.22 * math.sin(u * 3.0) * math.exp(-u * 1.8)
        return y, th, ph

    def fields(self, t):
        u = t - T_PHONE
        cy, th, ph = self.pose(u)
        cx = 540.0 + 4 * kick(t) * math.sin(t * 90) * float(sstep(1.6, 1.8, u)) * (1 - float(sstep(T_OUTRO - T_PHONE - 0.3, T_OUTRO - T_PHONE, u)))   # buzz on the kick
        b = np.zeros((FROWS, FCOLS)); col = np.zeros((FROWS, FCOLS, 3), np.float32)
        lcd = np.zeros((FROWS, FCOLS), bool)
        wx, wy = (FGX - cx) / PHONE_S, -(FGY - cy) / PHONE_S
        m = (np.abs(wx) < 380) & (wy > -470) & (wy < 690)
        rx, ry = wx[m], wy[m]
        # world = tilt(ph) . spin(th) . phone; rays come from the viewer along -z
        c1, s1, c2, s2 = math.cos(ph), math.sin(ph), math.cos(-th), math.sin(-th)
        ox, oy = rx * c1 + ry * s1, -rx * s1 + ry * c1
        o = np.stack([ox * c2 + 900 * s2, oy, -ox * s2 + 900 * c2], -1)
        dvec = np.array([-s2, 0, -c2])
        tt = np.zeros(len(rx)); hit = np.zeros(len(rx), bool)
        for _ in range(48):
            p = o + tt[:, None] * dvec
            d = phone_sdf(p[:, 0], p[:, 1], p[:, 2])
            hit |= d < 0.6
            tt = np.where(hit, tt, tt + np.maximum(d, 0.5))
            if (tt > 1800).all(): break
        hit &= tt < 1800
        if hit.any():
            p = o[hit] + tt[hit][:, None] * dvec
            e = 1.5
            n = np.stack([phone_sdf(p[:, 0] + e, p[:, 1], p[:, 2]) - phone_sdf(p[:, 0] - e, p[:, 1], p[:, 2]),
                          phone_sdf(p[:, 0], p[:, 1] + e, p[:, 2]) - phone_sdf(p[:, 0], p[:, 1] - e, p[:, 2]),
                          phone_sdf(p[:, 0], p[:, 1], p[:, 2] + e) - phone_sdf(p[:, 0], p[:, 1], p[:, 2] - e)], -1)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            # light in phone space (fixed in the world, so rotate it with the phone)
            L = np.array([-0.45, 0.55, 0.7]); L /= np.linalg.norm(L)
            Lp = np.array([L[0] * c2 + L[2] * s2, L[1], -L[0] * s2 + L[2] * c2])
            lam = np.clip(n @ Lp, 0, 1)
            bb = 0.16 + 0.7 * lam
            cc = np.tile(np.array([0.82, 0.8, 0.78], np.float32), (len(bb), 1))
            front = (n[:, 2] > 0.75) & (p[:, 2] > 25)
            x, y = p[:, 0], p[:, 1]
            # earpiece slots
            ear = front & (np.abs(x) < 55) & (y > 222 + 80) & (y < 262 + 80) & ((y.astype(int) // 7) % 2 == 0)
            bb = np.where(ear, 0.05, bb)
            # keypad
            on_key = np.zeros(len(bb), bool)
            for kx, ky, lab in KEYS:
                k = front & (np.abs(x - kx) < 28) & (np.abs(y - ky) < 20)
                on_key |= k
            bb = np.where(on_key, np.clip(bb + 0.22, 0, 1), np.where(front & (y < 135) & (y > -205), bb * 0.6, bb))
            cc = np.where(on_key[:, None], np.array([1.0, 0.93, 0.82]), cc)
            # screen: dark until it wakes, then green
            scr = front & (x > LCD[0]) & (x < LCD[1]) & (y > LCD[2]) & (y < LCD[3])
            wake = float(sstep(1.6, 1.75, u)) * (0.85 + 0.15 * (math.sin(t * 40) > -0.6))
            bb = np.where(scr, 0.05 + 0.33 * wake, bb)
            cc = np.where(scr[:, None], np.array([0.45, 1.0, 0.55]), cc)
            bi = np.zeros(m.sum()); bi[hit] = bb
            ci = np.zeros((m.sum(), 3), np.float32); ci[hit] = cc
            li = np.zeros(m.sum(), bool); li[hit] = scr
            b[m], col[m], lcd[m] = bi, ci, li
        # key labels: where each key's center lands on screen, while the front faces us
        labels = []
        if math.cos(th) > 0.8:
            for kx, ky, lab in KEYS:
                xs = kx * math.cos(th) + 36 * math.sin(th)
                X, Y = xs * c1 - ky * s1, xs * s1 + ky * c1
                labels.append((cx + X * PHONE_S, cy - Y * PHONE_S, lab))
        return b, col, lcd, labels

PHONE_Y = 930.0
PHONE_FROM = 150.0                        # where the fall starts: the bottom of the phone already in frame
PHONE_S = 1.35                            # phone units -> px


class Show:
    def __init__(self, handle, seed=3):
        rng = np.random.default_rng(seed)
        self.handle = handle
        m = np.asarray(Image.open("assets/2ndsun_logo_mask.png").convert("L"), np.float32) / 255.0
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
        self.phone = Phone()
        self.frand = rng.random((FROWS, FCOLS))
        # shapes for the phone -> logo morph: the phone's front silhouette and the logo, as distances
        lx, ly = (FGX - 540) / PHONE_S, -(FGY - PHONE_Y) / PHONE_S
        self.d_phone = phone_sdf(lx, ly, np.zeros_like(lx)) * PHONE_S
        self.d_logo = mask_sdf(self.flogo > 0.5)

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
        return idx, col

    def phone_frame(self, t, fi, phone=True):
        """inside the pupil: dim code drifting in the dark, the phone falling through it"""
        if phone:
            b, pc, lcd, labels = self.phone.fields(t)
        else:
            b, pc, lcd, labels = np.zeros((FROWS, FCOLS)), np.zeros((FROWS, FCOLS, 3)), np.zeros((FROWS, FCOLS), bool), []
        rr = np.arange(FROWS)[:, None]
        ph = (rr - t * self.rain_speed[None, :] * 0.6 - self.rain_phase[None, :] * 3) % 45
        rain = (ph < 4) & (self.frand < 0.5)
        idx = np.where(rain, self.page_idx, 0)
        col = np.broadcast_to(np.array([0.45, 0.55, 0.75], np.float32), (FROWS, FCOLS, 3)) * \
            (0.12 + 0.18 * (1 - ph / 4)).clip(0, 0.3)[..., None]
        on = self.phone_on = b > 0.02
        idx = np.where(on, glyphs(np.clip(b, 0, 1), int(t * 12), (FROWS, FCOLS)), idx)
        col = np.where(on[..., None], pc * np.clip(0.35 + 0.8 * b, 0, 1)[..., None], col)
        idx = np.where(lcd & (b < 0.2), 0, idx)          # the screen stays dark under the text
        for (x, y, lab) in labels:
            r, c = int(round((y - FOY) / FCH - 0.5)), int(round(x / FCW - 0.5))
            if 0 <= r < FROWS and 0 <= c < FCOLS and lab in CI:
                idx[r, c] = CI[lab]; col[r, c] = (0.15, 0.12, 0.1)
        return idx, col.astype(np.float32)

    def phone_text(self, img, t):
        """the green screen once the phone has landed: a new message, typed"""
        u = t - T_PHONE
        if u < 1.75:
            return img
        y0, _, _ = self.phone.pose(u)
        x0 = 540 + LCD[0] * PHONE_S
        top = y0 - LCD[3] * PHONE_S
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        f = 1 - float(sstep(T_OUTRO, T_OUTRO + 0.35, t))              # fades as the phone melts
        green = tuple(int(c * f) for c in (120, 255, 140))
        dr.text((x0 + 14, top + 14), "1 NEW MESSAGE", font=find_font(19), fill=tuple(int(c * f) for c in (90, 200, 110)))
        s, typing = typed("you need me", t, T_PHONE + 1.95, 7.5)
        cur = "\u2588" if typing or (t * 2.5) % 2 < 1 else ""
        dr.text((x0 + 14, top + 64), s + cur, font=find_font(34), fill=green)
        return np.asarray(im)

    def end_frame(self, t, fi):
        """the phone melts into the logo, the star inside it lights up again"""
        k = kick(t) * (1 - float(sstep(DUR - 3, DUR - 1, t)))
        a = float(smoother(T_OUTRO, T_OUTRO + MORPH, t))
        I = (0.55 + 0.08 * math.sin(t * 2.2)) * float(sstep(T_OUTRO + MORPH - 0.3, T_OUTRO + MORPH + 0.7, t))
        lidx, lcol = self.code_logo(60 + t - T_OUTRO, I, 1.0 + 0.12 * k, k, 0.0, fi)   # logo + light, no page
        pidx, pcol = self.phone_frame(t, fi, phone=a < 1)
        if a >= 1:
            show = lidx > 0
            light = show & (lcol.max(-1) > 0.03)
            return np.where(light, lidx, pidx), np.where(light[..., None], lcol, pcol)
        inside = (self.d_phone * (1 - a) + self.d_logo * a) < 0
        phone_cell = inside & self.phone_on
        logo_char = np.where(lidx > 0, lidx, self.logo_idx)
        take_logo = self.frand < a
        idx = np.where(inside, np.where(take_logo | ~phone_cell, logo_char, pidx), pidx)
        base = np.where(phone_cell[..., None], pcol, np.array([0.55, 0.53, 0.5]) * 0.8)
        lc = np.where((lcol.max(-1) > 0.05)[..., None], lcol, CREAM * 0.9)
        col = np.where(inside[..., None], base * (1 - a) + lc * a, pcol)
        # whatever of the phone is outside the shape is gone; its light comes in with the logo
        idx = np.where(~inside & self.phone_on, 0, idx)
        glow = ~inside & (self.flogo <= 0.5) & (lidx > 0) & (lcol.max(-1) > 0.03)
        idx = np.where(glow & (self.frand < a), lidx, idx)
        col = np.where((glow & (self.frand < a))[..., None], lcol, col)
        return idx, col.astype(np.float32)

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
        fw = float(sstep(T_PULL, T_PULL + 1.2, t)) * (1 - float(smoother(T_DIVE - 0.5, T_DIVE + 0.2, t)))
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

        # pupil, with their star as the catch light
        pupil = in_iris & (d < rp)
        b = np.where(pupil, 0.0, b)
        hs = (0.4 + 0.15 * k) * Rr
        hd = star_sdf(dx + 0.38 * Rr, dy - 0.36 * Rr, hs * 0.5, hs * 0.38)
        hl = in_iris & (hd < 0) & (fm > 0.5) & (t < T_DIVE + 0.1)
        b = np.where(hl, 1.0, b)
        col = np.where(hl[..., None], WHITE, col)

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
        if t < T_CARD:
            return img
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        lines = [(">>> let's work together.", 46, (252, 236, 212), T_CARD, 26),
                 (self.handle, 38, (255, 196, 120), T_CARD + 1.15, 26),
                 ("stage visual concept for 2ND SUN - made 100% in code", 21, (190, 165, 140), T_CARD + 1.65, 70)]
        y = 1490
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
        elif t < T_END:
            img = render(*self.drop_frame(t, fi))
        elif t < T_OUTRO:
            img = self.phone_text(render_fine(*self.phone_frame(t, fi)), t)
        else:
            img = render_fine(*self.end_frame(t, fi))
            if t < T_OUTRO + 0.4:
                img = self.phone_text(img, t)
            img = self.end_card(img, t)
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
    ap.add_argument("--handle", default="@yourhandle")
    a = ap.parse_args()
    export(a.export, a.music, a.start, a.handle)
