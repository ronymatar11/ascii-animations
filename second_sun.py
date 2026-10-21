"""
2ND SUN  |  stage visual concept, 9:16, cut to music. Every frame is text, no cuts:
each scene is transformed out of the one before it.

the star inside the logo starts to shine -> the riser pulls the logo's characters
into it -> break: only the star -> drop: the star bursts into a sun with the logo
stamped in it -> the sun becomes an iris, the camera pulls back: one big eye ->
it looks around on the beat -> dive into the pupil -> out of the dark, two hands
drawn in code reach for each other, the star sparks between them -> let's work together.

Export:   python second_sun.py --music track.mp3 --start 236.32 --handle @you --export second_sun.mp4
          (--start = where to cut the track; the drop lands 9.52s in, 5 bars at 126 BPM)
"""
import argparse, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from code_to_monalisa import (W, H, CW, CH, COLS, ROWS, OY, CI, CHARS, BLOCK, PRINTABLE, RAMP,
                              sstep, smoother, fbm, find_font, compose, render, to_glyphs, ffmpeg_bin)

FPS = 30
DUR = 25.0
BPM = 126.05
BEAT = 60 / BPM
BAR = 4 * BEAT
DROP = 5 * BAR                            # 9.52s: the kick comes back
BREAK = DROP - BAR                        # 7.62s: only the star is left
T_IRIS = DROP + 1.6 * BAR                 # the sun turns into an iris
T_PULL = DROP + 1.8 * BAR                 # camera pulls back: it's an eye
T_DIVE = DROP + 3.5 * BAR                 # dive into the pupil
T_HANDS = DROP + 4 * BAR                  # 17.14s: hands out of the dark
T_SPARK = T_HANDS + 2 * BAR               # fingertips meet on a downbeat

CREAM = np.array([0.99, 0.92, 0.82], np.float32)
GOLD = np.array([1.0, 0.82, 0.45], np.float32)
EMBER = np.array([1.0, 0.45, 0.12], np.float32)
WHITE = np.array([1.0, 0.97, 0.92], np.float32)
SOFT_SET = ".,:;-=+*#"

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

def hand_sdf(X, Y, wx, wy, ang, s, fingers, thumb_ang, mirror):
    """hand in local coords: wrist at 0, pointing +x, thumb side +y"""
    c, sn = math.cos(ang), math.sin(ang)
    lx = ((X - wx) * c + (Y - wy) * sn) / s
    ly = (-(X - wx) * sn + (Y - wy) * c) / s
    if mirror: ly = -ly
    d = seg_sdf(lx, ly, -900, 0, -60, 0, 50, 41)                    # forearm
    d = smin(d, seg_sdf(lx, ly, -60, 0, 0, 0, 41, 37), 20)          # wrist
    d = smin(d, seg_sdf(lx, ly, 12, 0, 74, 0, 45, 43), 22)          # palm
    d = smin(d, seg_sdf(lx, ly, 72, 24, 92, -24, 22, 20), 14)       # knuckles
    tips = []
    for (bx, by, a0, lens, curls, r) in fingers:
        x, y, a = bx, by, a0
        fd = None
        for k, (L, cu) in enumerate(zip(lens, curls)):
            a = a - cu
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


class Show:
    def __init__(self, handle, seed=3):
        rng = np.random.default_rng(seed)
        self.handle = handle
        m = np.asarray(Image.open("assets/2ndsun_logo_mask.png").convert("L"), np.float32) / 255.0
        self.mask = m
        self.mh, self.mw = m.shape
        self.logo = self.logo_field(LOGO_CX, LOGO_CY, LOGO_W)
        self.logo_sun = self.logo_field(540, 960, 760)
        lw = LOGO_W / self.mw
        self.star_xy = (LOGO_CX + (STAR_M[0] - self.mw / 2) * lw, LOGO_CY + (STAR_M[1] - self.mh / 2) * lw)
        self.star_ab = (STAR_A * lw, STAR_B * lw)
        sx, sy = self.star_xy
        rs = np.hypot(GX - sx, (GY - sy) * 1.3)
        R = self.rand = rng.random((ROWS, COLS))
        self.glyph_rand = rng.choice(PRINTABLE, (ROWS, COLS))
        # the riser pulls the logo into the star, nearest characters first
        self.T_absorb = 5.4 + 1.9 * np.clip(rs / 560, 0, 1) + 0.2 * R
        # the logo is stamped into the sun on the drop, and melts back into it
        self.T_stamp = DROP + 0.15 + 0.45 * R
        self.T_melt = T_IRIS - 0.6 + 0.5 * R
        # around the sun, cells turn into the eye from the iris outward
        rc = np.hypot(GX - 540, GY - 960)
        self.T_eye = T_IRIS + 0.15 + 0.9 * np.clip((rc - 300) / 900, 0, 1) + 0.18 * R
        # gaze keyframes on the beat after the eye is revealed: (time, x, y)
        b = lambda n: DROP + n * BEAT
        self.gaze_kf = [(b(10), -0.2, 0.03), (b(12), 0.21, -0.02), (b(13), 0.06, 0.07), (b(14), 0.0, 0.0)]
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

    def logo_field(self, cx, cy, width):
        s = self.mw / width
        u = (GX - cx) * s + self.mw / 2
        v = (GY - cy) * s + self.mh / 2
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

    # ---------- 1. the star inside the logo starts to shine, swallows the logo, break
    def star_light(self, t):
        """intensity, scale and pulse of the star from the first frame to the drop"""
        I = 0.3 * float(sstep(0.1, 0.9, t)) + 0.7 * float(smoother(0.9, BREAK + 0.6, t))
        if t < BREAK:
            p = 0.4 * math.exp(-(t % BEAT) / 0.12) * float(sstep(1.0, 3.0, t))
        else:
            v = (t - BREAK) / BAR
            p = math.exp(-((t - BREAK) % (BEAT / (1 + int(v * 3)))) / 0.12)
        sc = 0.75 + 0.55 * I + 0.12 * p
        return I, sc, p

    def pre_drop(self, t, fi):
        I, sc, p = self.star_light(t)
        sx, sy = self.star_xy
        dx, dy = GX - sx, GY - sy
        r = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        a, bb = self.star_ab
        d = star_sdf(dx, dy, a * sc, bb * sc)
        star = d < 0
        # light: soft glow, long arms along the star's axes, thin code rays all around
        glow = np.exp(-np.hypot(dx / 1.8, dy) / (18 + 110 * I)) * I * (0.85 + 0.3 * p)
        arms = (np.exp(-np.abs(dy) / 7) * np.exp(-np.abs(dx) / (90 + 650 * I)) +
                np.exp(-np.abs(dx) / 5) * np.exp(-np.abs(dy) / (50 + 380 * I))) * I * (0.8 + 0.4 * p)
        rays = np.clip(np.cos(ang * 9 + t * 0.5), 0, 1) ** 10 * np.exp(-r / (80 + 520 * I)) * I
        light = np.maximum(np.maximum(glow, arms), 0.85 * rays)
        sm = self.smoke(t, float(sstep(0.0, 0.8, t)) * (1 - float(sstep(6.4, BREAK, t))))
        b = np.maximum(sm, light)
        col = np.where((light > sm)[..., None], GOLD * (1 - glow[..., None]) + WHITE * glow[..., None],
                       np.array([0.85, 0.82, 0.78], np.float32))
        # the logo, lit by the star, until the riser pulls it in
        L = self.logo > 0.5
        here = L & (t < self.T_absorb)
        heat = np.clip(np.exp(-r / (120 + 380 * I)) * I * 1.4, 0, 1)
        b = np.where(here, 0.55 + 0.35 * self.logo + 0.25 * heat, b)
        lc = CREAM * (1 - heat[..., None]) + GOLD * heat[..., None]
        col = np.where(here[..., None], lc, col)
        b = np.where(star, 1.0, b)
        col = np.where(star[..., None], WHITE, col)
        idx = to_glyphs(np.clip(b, 0, 1), int(t * 12))
        col = col * np.clip(0.3 + 0.85 * b, 0, 1)[..., None]
        # rays are drawn as strokes pointing at the star
        ray_c = (rays > 0.3) & ~here & ~star & (rays > glow)
        idx = np.where(ray_c, stroke_glyph(ang), idx)
        # characters flying into the star
        fly = L & (t >= self.T_absorb) & (t < self.T_absorb + 0.45)
        rr, cc = np.nonzero(fly)
        if len(rr):
            f = ((t - self.T_absorb[rr, cc]) / 0.45) ** 2
            tr = (rr + ((sy - OY) / CH - 0.5 - rr) * f).round().astype(int)
            tc = (cc + (sx / CW - 0.5 - cc) * f).round().astype(int)
            ok = (tr >= 0) & (tr < ROWS) & (tc >= 0) & (tc < COLS)
            idx[tr[ok], tc[ok]] = self.glyph_rand[rr[ok], cc[ok]]
            col[tr[ok], tc[ok]] = (GOLD * 0.6 + WHITE * 0.4) * (0.6 + 0.4 * f[ok, None])
        return idx, col.astype(np.float32)

    # ---------- 2. sun -> iris -> eye -> pupil
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
        z *= math.exp(math.log(40) * float(smoother(T_DIVE, T_HANDS, t)) ** 2)
        e = 1 - math.exp(-u / 0.1)
        cx = self.star_xy[0] + (540 - self.star_xy[0]) * e
        cy = self.star_xy[1] + (960 - self.star_xy[1]) * e
        gx, gy = self.gaze(t)
        X = (GX - cx) / (EU * z)
        Y = -(GY - cy) / (EU * z)
        # the sun is born from the star, then breathes with the kick
        grow = (230 + (SUN_R - 230) * (1 - math.exp(-u / 0.12))) / SUN_R
        Rr = IRIS_R * grow * (1 + 0.035 * k * (1 - float(sstep(T_IRIS, T_PULL, t))))
        rp = Rr * (0.34 * float(smoother(T_IRIS - 0.3, T_IRIS + 0.5, t)) + 0.03 * k * float(sstep(T_PULL, T_PULL + 0.5, t)))
        rp += Rr * 0.5 * float(smoother(T_DIVE, T_HANDS - 0.2, t))
        dx, dy = X - gx, Y - gy
        d = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        Rpx = Rr * EU * z

        # eyelids (in eye units)
        bl = math.exp(-((t - (DROP + 11.5 * BEAT)) / 0.07) ** 2)
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

    # ---------- 3. out of the dark: hands (fine grid)
    def hands(self, t):
        u = t - T_HANDS
        k = kick(t) * (1 - float(sstep(DUR - 3, DUR - 1, t)))
        X, Y = FGX, FGY
        dawn = float(smoother(0.0, 1.2, u))                 # light rises out of the pupil's black
        glow = np.exp(-(((X - 560) / 800) ** 2 + ((Y - 1500) / 1100) ** 2))
        b = (0.10 + 0.42 * glow + 0.05 * k + 0.04 * fbm(X / 120, Y / 120 + u * 0.2, 2, 7)) * dawn
        lit = (0.55 + 0.75 * glow + 0.15 * k)[..., None]
        col = (np.array([0.80, 0.22, 0.06]) * (1 - glow[..., None]) +
               np.array([1.0, 0.62, 0.24]) * glow[..., None]) * lit

        m = float(smoother(0.0, T_SPARK - T_HANDS, u))
        drift = 6 * math.sin(u * 1.3)
        dl, tip_l = hand_sdf(X, Y, 165 - 420 * (1 - m), 1218 + 190 * (1 - m) + drift, -0.42, 2.3, FINGERS_OPEN, 1.0, True)
        dr, tip_r = hand_sdf(X, Y, 1026 + 420 * (1 - m), 812 - 190 * (1 - m) - drift, math.pi - 0.47, 2.3, FINGERS_POINT, 0.55, False)
        dd = np.minimum(dl, dr)
        inl, inr = dl < 0, dr < 0
        hand = inl | inr
        b = np.where(hand, 0.0, b)

        # the spark: their star, born where the fingertips meet
        sx, sy = (tip_l[0] + tip_r[0]) / 2, (tip_l[1] + tip_r[1]) / 2
        v = t - T_SPARK
        star = np.zeros_like(hand)
        sc = 0.0
        if v > 0:
            sc = 1.6 * (1 - math.exp(-v / 0.08)) * (1 + 0.35 * math.exp(-v / 0.25)) * (1 + 0.1 * k)
            d = star_sdf(X - sx, Y - sy, 30 * sc, 25 * sc)
            star = d < 0
            halo = np.exp(-np.hypot((X - sx) / 1.8, Y - sy) / (28 + 12 * k)) * (0.7 + 0.3 * math.exp(-v / 0.5)) * sc
            wave = np.exp(-((np.hypot(X - sx, Y - sy) - v * 900) / 40) ** 2) * math.exp(-v / 0.8)
            add = np.where(hand, 0, halo + 0.5 * wave)
            b = np.where(star, 1.0, np.maximum(b, add))
            col = np.where(star[..., None] | (add > 0.35)[..., None], WHITE, col)

        # quiet ramp for the backlight, full ramp only for the brightest light
        jr = np.random.default_rng(int(t * 10)).integers(-1, 2, (FROWS, FCOLS))
        idx = SOFT[np.clip((np.clip(b, 0, 1) * 1.6 * (len(SOFT) - 1)).astype(int) + jr, 0, len(SOFT) - 1)]
        idx = np.where(b < 0.03, 0, idx)
        idx = np.where((b > 0.6) | star, glyphs(np.clip(b, 0, 1), int(t * 10), (FROWS, FCOLS)), idx)
        col = np.clip(col * np.clip(0.55 + 0.6 * b, 0, 1)[..., None], 0, 1)

        # outlines drawn along the contour, lit from the glow behind
        gy_, gx_ = np.gradient(dd, FCH, FCW)
        edge = (dd > -FCW * 0.6) & (dd < FCW * 0.9)
        tang = np.arctan2(gy_, gx_) + np.pi / 2
        idx = np.where(edge, stroke_glyph(np.arctan2(np.sin(tang) / FCH, np.cos(tang) / FCW)), idx)
        n_len = np.hypot(gx_, gy_) + 1e-6
        facing = np.clip(0.5 + 0.5 * (gy_ / n_len), 0, 1)   # normals facing down catch the glow
        rimc = (np.array([1.0, 0.72, 0.42]) * (0.35 + 0.65 * facing[..., None]) +
                np.array([0.25, 0.22, 0.2]) * (1 - facing[..., None])) * (0.4 + 0.6 * dawn)
        col = np.where(edge[..., None], rimc, col)

        # my hand is made of code: sparse code rain inside the silhouette
        rr, cc = np.nonzero(inl & ~edge & ~star)
        if len(rr):
            ph = (rr - u * self.rain_speed[cc] - self.rain_phase[cc]) % 30
            vis = ph < 5
            rr, cc, ph = rr[vis], cc[vis], ph[vis]
            idx[rr, cc] = [CI.get(self.code_rows[r][c % 90], 0) for r, c in zip(rr, cc)]
            col[rr, cc] = np.array([1.0, 0.72, 0.45])[None] * (0.2 + 0.55 * (1 - ph[:, None] / 5)) * dawn

        # sparkle arms so it reads as their star
        if v > 0:
            r0 = int(round((sy - FOY) / FCH - 0.5)); c0 = int(round(sx / FCW - 0.5))
            for dc in range(-int(40 * sc), int(40 * sc) + 1):
                f = 1 - abs(dc) / (40 * sc + 1)
                if 0 <= c0 + dc < FCOLS and abs(dc) > 5 and not hand[r0, c0 + dc]:
                    idx[r0, c0 + dc] = CI["-"]; col[r0, c0 + dc] = WHITE * f
            for dr_ in range(-int(13 * sc), int(13 * sc) + 1):
                f = 1 - abs(dr_) / (13 * sc + 1)
                if 0 <= r0 + dr_ < FROWS and abs(dr_) > 2 and not hand[r0 + dr_, c0]:
                    idx[r0 + dr_, c0] = CI["|"]; col[r0 + dr_, c0] = WHITE * f
        return idx, col.astype(np.float32)

    def end_card(self, img, t):
        if t < T_SPARK + 0.35:
            return img
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        lines = [(">>> let's work together.", 46, (252, 236, 212), T_SPARK + 0.35, 26),
                 (self.handle, 38, (255, 196, 120), T_SPARK + 1.5, 26),
                 ("stage visual concept for 2ND SUN - made 100% in code", 21, (190, 165, 140), T_SPARK + 2.0, 70)]
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
            img = render(*self.pre_drop(t, fi))
        elif t < T_HANDS:
            img = render(*self.drop_frame(t, fi))
        else:
            img = self.end_card(render_fine(*self.hands(t)), t)
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
