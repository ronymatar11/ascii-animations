"""
2ND SUN  |  stage visual concept, 9:16, cut to music. Every frame is text.

stage.load("2ND SUN") -> code falls into the logo on the beat -> riser glitch ->
break: only the star in the logo glows -> drop: the star becomes a sun ->
tunnel -> strobe -> two hands drawn in code reach for each other, the star
sparks between the fingertips -> let's work together.

Export:   python second_sun.py --music track.mp3 --start 236.32 --handle @you --export second_sun.mp4
          (--start = where to cut the track; the drop lands 9.52s in, 5 bars at 126 BPM)
"""
import argparse, math, subprocess
import numpy as np
from PIL import Image

from code_to_monalisa import (W, H, CW, CH, COLS, ROWS, OY, CI, BLOCK, PRINTABLE,
                              sstep, smoother, fbm, render, to_glyphs, ffmpeg_bin)

FPS = 30
DUR = 25.0
BPM = 126.05
BEAT = 60 / BPM
BAR = 4 * BEAT
DROP = 5 * BAR                            # 9.52s: the kick comes back
BREAK = DROP - BAR                        # 7.62s: everything goes dark but the star
T_SUN, T_TUNNEL, T_STROBE, T_HANDS = DROP, DROP + 2 * BAR, DROP + 3 * BAR, DROP + 4 * BAR
T_SPARK = T_HANDS + 2 * BAR               # fingertips meet on a downbeat

CREAM = np.array([0.99, 0.92, 0.82], np.float32)
SOFT = np.array([CI[c] for c in ".,:;-=+*#"])     # quiet ramp for the warm backlight
PX = (np.arange(COLS) + 0.5) * CW
PY = (np.arange(ROWS) + 0.5) * CH + OY
GX, GY = np.meshgrid(PX, PY)

# logo placement on screen + where the star sits inside the logo mask
LOGO_CX, LOGO_CY, LOGO_W = 540.0, 900.0, 860.0
STAR_M = (148.0, 117.5)                   # star center, mask pixels
STAR_A, STAR_B = 62.0, 32.0               # star half width / half height, mask pixels

def kick(t):
    """1 on each kick after the drop, decaying"""
    if t < DROP: return 0.0
    return math.exp(-((t - DROP) % BEAT) / 0.11)

def beat_n(t, t0):
    return int(math.floor((t - t0) / BEAT))

def put_text(idx, col, r, c0, text, color):
    for i, ch in enumerate(text):
        c = c0 + i
        if 0 <= r < ROWS and 0 <= c < COLS and ch in CI:
            idx[r, c] = CI[ch]; col[r, c] = color

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
    d = seg_sdf(lx, ly, -900, 0, 0, 0, 52, 40)                     # forearm
    d = smin(d, seg_sdf(lx, ly, 15, 0, 78, 0, 46, 42), 25)          # palm
    d = smin(d, seg_sdf(lx, ly, 70, 22, 92, -22, 26, 24), 18)       # knuckles
    tips = []
    for (bx, by, a0, lens, curls, r) in fingers:
        x, y, a = bx, by, a0
        fd = None
        for k, (L, cu) in enumerate(zip(lens, curls)):
            a = a - cu
            nx, ny = x + L * math.cos(a), y + L * math.sin(a)
            r0, r1 = r * (1 - 0.08 * k), r * (1 - 0.08 * (k + 1))
            sd = seg_sdf(lx, ly, x, y, nx, ny, r0, r1)
            fd = sd if fd is None else np.minimum(fd, sd)
            x, y = nx, ny
        tips.append((x, y))
        d = smin(d, fd, 6)
    x, y, a = 38, 36, thumb_ang                                    # thumb
    for L, r0, r1 in ((44, 17, 15), (32, 15, 12)):
        nx, ny = x + L * math.cos(a), y + L * math.sin(a)
        d = smin(d, seg_sdf(lx, ly, x, y, nx, ny, r0, r1), 10)
        x, y, a = nx, ny, a - 0.35
    # index tip back to world
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
        # build: cells land left to right, quantized to beats
        u = np.clip((GX - (LOGO_CX - LOGO_W / 2)) / LOGO_W, 0, 1)
        grp = np.clip(np.floor(u * 8 + rng.random((ROWS, COLS)) * 3), 0, 9)
        self.T_land = 1.9 + BEAT * (grp + 0.5) + rng.uniform(0, 0.12, (ROWS, COLS))
        self.rand = rng.random((ROWS, COLS))
        self.glyph_rand = rng.choice(PRINTABLE, (ROWS, COLS))
        # text that fills the left hand
        src = "def reach(you): return light  # 2ND SUN  for beat in set: sun.rise(beat)  stage.load()  "
        self.code_rows = [src[(7 * r) % len(src):] + src for r in range(ROWS)]
        self.rain_speed = rng.uniform(10, 22, COLS)
        self.rain_phase = rng.uniform(0, 30, COLS)

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

    # ---------- scenes: each returns idx, col
    def terminal(self, t):
        idx = np.zeros((ROWS, COLS), np.int64); col = np.zeros((ROWS, COLS, 3), np.float32)
        r, c0 = ROWS // 2, 22
        put_text(idx, col, r, c0, ">>> ", np.array([0.55, 0.55, 0.55]))
        txt, typing = typed('stage.load("2ND SUN")', t, 0.25, 17)
        for i, ch in enumerate(txt):
            cc = np.array([0.95, 0.62, 0.45]) if 10 <= i <= 20 else CREAM
            put_text(idx, col, r, c0 + 4 + i, ch, cc)
        if typing or (t * 2.2) % 2 < 1:
            idx[r, c0 + 4 + len(txt)] = BLOCK; col[r, c0 + 4 + len(txt)] = CREAM * 0.8
        if t > 1.65:
            put_text(idx, col, r + 2, c0, "loading visuals " + "." * min(3, int((t - 1.65) * 8)), CREAM * 0.6)
        return idx, col

    def build(self, t, fi):
        sm = self.smoke(t, float(sstep(1.9, 3.0, t)) * (1 + 0.8 * float(sstep(6.0, BREAK, t))))
        b = sm.copy()
        col = np.broadcast_to(np.array([0.85, 0.82, 0.78], np.float32), (ROWS, COLS, 3)).copy()
        L = self.logo > 0.5
        landed = L & (t >= self.T_land)
        settle = sstep(self.T_land, self.T_land + 0.25, t)
        b = np.where(landed, 0.55 + 0.4 * self.logo, b)
        idx = to_glyphs(b, int(t * 10))
        col = np.where(landed[..., None], CREAM * (1 + 0.6 * (1 - settle[..., None])), col * (0.4 + 0.9 * b[..., None]))
        fresh = landed & (t < self.T_land + 0.15)
        idx = np.where(fresh, self.glyph_rand, idx)
        # characters falling into place
        fall = L & (t < self.T_land) & (t > self.T_land - 0.4)
        rr, cc = np.nonzero(fall)
        if len(rr):
            dr = ((self.T_land[rr, cc] - t) * 70).astype(int)
            r2 = rr - dr
            ok = r2 >= 0
            idx[r2[ok], cc[ok]] = self.glyph_rand[rr[ok], cc[ok]]
            col[r2[ok], cc[ok]] = np.array([0.75, 0.95, 1.0]) * (1 - dr[ok, None] / 30).clip(0.3, 1)
        # riser: the logo starts to tear
        g = float(sstep(5.7, BREAK, t))
        if g > 0:
            rng = np.random.default_rng(fi)
            for r in range(ROWS):
                if rng.random() < 0.15 + 0.5 * g:
                    sh = int(rng.normal(0, 1 + 7 * g))
                    idx[r] = np.roll(idx[r], sh); col[r] = np.roll(col[r], sh, axis=0)
        return idx, col

    def star_glow(self, scale, cx=None, cy=None):
        cx = self.star_xy[0] if cx is None else cx
        cy = self.star_xy[1] if cy is None else cy
        a, b = self.star_ab
        d = star_sdf(GX - cx, GY - cy, a * scale, b * scale)
        return d

    def break_(self, t):
        u = (t - BREAK) / BAR
        p = math.exp(-((t - BREAK) % (BEAT / (1 + int(u * 3)))) / 0.12)    # pulses get faster
        d = self.star_glow(0.9 + 0.5 * u + 0.08 * p)
        b = np.where(d < 0, 0.75 + 0.25 * p, (0.55 + 0.4 * p) * np.exp(-np.maximum(d, 0) / (25 + 50 * u)))
        b = np.maximum(b, 0.07 * self.logo * (1 - u))
        col = np.where((d < 0)[..., None], np.array([1.0, 0.97, 0.9]), np.array([1.0, 0.72, 0.38]))
        idx = to_glyphs(np.clip(b, 0, 1), int(t * 12))
        col = col * np.clip(0.35 + 0.8 * b, 0, 1)[..., None]
        return idx, col.astype(np.float32)

    def sun(self, t):
        k = kick(t); u = t - T_SUN
        cx, cy = 540.0, 960.0
        R = 480 * (1 - math.exp(-u / 0.12)) * (1 + 0.04 * k) + 1
        dx, dy = GX - cx, GY - cy
        d = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        gran = fbm(GX / 45 + u * 0.4, GY / 45 - u * 0.3, 3, 5)
        disk = d < R
        limb = np.sqrt(np.clip(1 - (d / R) ** 2, 0, 1))
        rays = (0.5 + 0.5 * np.sin(ang * 14 + u * 1.5)) * np.exp(-np.maximum(d - R, 0) / (140 + 80 * k))
        ring_r = R + ((t - DROP) % BEAT) * 1500
        ring = np.exp(-((d - ring_r) / 22) ** 2) * k
        b = np.where(disk, 0.55 + 0.35 * limb + 0.15 * gran, 0.45 * rays + 0.6 * ring)
        b = np.maximum(b, self.smoke(t, 0.7))
        heat = np.clip(1 - d / (R * 1.9), 0, 1)
        col = (np.array([1.0, 0.42, 0.12]) * (1 - heat[..., None]) + np.array([1.0, 0.9, 0.62]) * heat[..., None])
        col = np.where(disk[..., None], np.array([1.0, 0.86, 0.5]) * (0.75 + 0.3 * limb[..., None]), col)
        # logo cut out of the sun
        lg = self.logo_sun > 0.5
        b = np.where(lg & disk, 0.0, b)
        b = np.where(lg & ~disk, np.maximum(b, 0.75), b)
        col = np.where((lg & ~disk)[..., None], CREAM, col)
        flash = math.exp(-u / 0.18)
        b = b + (1 - b) * flash
        idx = to_glyphs(np.clip(b, 0, 1), int(t * 12))
        col = np.clip(col * np.clip(0.3 + 0.85 * b, 0, 1)[..., None] + flash, 0, 1)
        return idx, col.astype(np.float32)

    def tunnel(self, t):
        k = kick(t); u = t - T_TUNNEL
        dx, dy = GX - 540, (GY - 960) * 1.0
        r = np.hypot(dx, dy) + 1e-3
        z = 260.0 / r + u * 4.5 + 0.35 * beat_n(t, DROP)  # rush forward, jump on kicks
        ang = np.arctan2(dy, dx) / (2 * np.pi) + u * 0.08
        rings = 0.5 + 0.5 * np.cos(z * 2 * np.pi)
        spokes = 0.5 + 0.5 * np.cos(ang * 2 * np.pi * 12 + z * 1.5)
        depth = np.clip(r / 700, 0, 1)
        b = (0.15 + 0.85 * rings ** 3 * (0.6 + 0.4 * spokes)) * (0.25 + 0.75 * depth) * (0.85 + 0.3 * k)
        d = star_sdf(dx, dy, 120 * (1 + 0.2 * k), 62 * (1 + 0.2 * k))
        b = np.where(d < 0, 1.0, np.maximum(b, 0.8 * np.exp(-np.maximum(d, 0) / 40)))
        word = "2NDSUN"
        wi = (np.floor(ang * 48) + np.floor(z * 2)).astype(int) % len(word)
        idx = to_glyphs(np.clip(b, 0, 1), int(t * 12))
        letters = np.array([CI[c] for c in word])[wi]
        idx = np.where((rings > 0.75) & (b > 0.3) & (d > 0), letters, idx)
        warm = (np.floor(z) % 2)[..., None]
        col = CREAM * (1 - warm) + np.array([1.0, 0.55, 0.22]) * warm
        col = np.where((d < 0)[..., None], np.array([1.0, 1.0, 0.95]), col)
        col = col * np.clip(0.3 + 0.85 * b, 0, 1)[..., None]
        return idx, col.astype(np.float32)

    def strobe(self, t, fi):
        k = kick(t); n = beat_n(t, T_STROBE)
        lg = self.logo_field(540, 960, 1000 * (1 + 0.03 * k)) > 0.5
        inv = n % 2 == 1
        on = lg ^ inv
        b = np.where(on, 0.9 * (0.7 + 0.3 * k), self.smoke(t, 0.5))
        idx = to_glyphs(b, int(t * 15))
        col = np.where(on[..., None], CREAM if not inv else np.array([1.0, 0.6, 0.28]), np.array([0.8, 0.78, 0.74]))
        col = col * np.clip(0.3 + 0.85 * b, 0, 1)[..., None]
        rng = np.random.default_rng(fi)
        g = 0.2 + 0.8 * k
        for r in range(ROWS):
            if rng.random() < 0.18 * g:
                sh = int(rng.normal(0, 10 * g))
                idx[r] = np.roll(idx[r], sh); col[r] = np.roll(col[r], sh, axis=0)
        # last beat bleeds into warm light for the hands
        w = float(sstep(T_HANDS - BEAT, T_HANDS, t))
        col = col * (1 - w) + np.array([1.0, 0.5, 0.18]) * w * 0.6
        return idx, col.astype(np.float32)

    def hands(self, t):
        u = t - T_HANDS
        k = kick(t) * (1 - float(sstep(DUR - 3, DUR - 1, t)))
        # warm backlight like a sunset
        glow = np.exp(-(((GX - 560) / 800) ** 2 + ((GY - 1500) / 1100) ** 2))
        # light glyphs, the glow lives in the color so the black hands read clearly
        b = 0.10 + 0.40 * glow + 0.05 * k + 0.04 * fbm(GX / 120, GY / 120 + u * 0.2, 2, 7)
        lit = (0.55 + 0.75 * glow + 0.15 * k)[..., None]
        col = (np.array([0.80, 0.22, 0.06]) * (1 - glow[..., None]) +
               np.array([1.0, 0.62, 0.24]) * glow[..., None]) * lit
        # the hands slide in and almost touch at the spark
        m = float(smoother(0.0, T_SPARK - T_HANDS, u))
        drift = 6 * math.sin(u * 1.3)
        dl, tip_l = hand_sdf(GX, GY, 165 - 420 * (1 - m), 1218 + 190 * (1 - m) + drift, -0.42, 2.3, FINGERS_OPEN, 1.0, True)
        dr, tip_r = hand_sdf(GX, GY, 1026 + 420 * (1 - m), 812 - 190 * (1 - m) - drift, math.pi - 0.47, 2.3, FINGERS_POINT, 0.55, False)
        inl, inr = dl < 0, dr < 0
        rim = np.exp(-np.maximum(np.minimum(dl, dr), 0) / 12) * (np.minimum(dl, dr) > 0)
        b = np.where(inl | inr, 0.0, b + 0.75 * rim)
        col = np.where((rim > 0.3)[..., None], np.array([1.0, 0.88, 0.66]), col)
        # the spark: their star, born where the fingers meet
        sx, sy = (tip_l[0] + tip_r[0]) / 2, (tip_l[1] + tip_r[1]) / 2
        v = t - T_SPARK
        star = np.zeros_like(b, bool)
        if v > 0:
            sc = 1.6 * (1 - math.exp(-v / 0.08)) * (1 + 0.35 * math.exp(-v / 0.25)) * (1 + 0.1 * k)
            d = star_sdf(GX - sx, GY - sy, 36 * sc, 30 * sc)
            star = d < 0
            rd = np.hypot((GX - sx) / 1.8, GY - sy)
            halo = np.exp(-rd / (28 + 12 * k)) * (0.7 + 0.3 * math.exp(-v / 0.5)) * sc
            wave = np.exp(-((np.hypot(GX - sx, GY - sy) - v * 900) / 40) ** 2) * math.exp(-v / 0.8)
            dark = inl | inr
            halo, wave = np.where(dark, 0, halo), np.where(dark, 0, wave)
            b = np.where(star, 1.0, np.maximum(b, halo + 0.5 * wave))
            col = np.where(star[..., None] | (halo > 0.35)[..., None], np.array([1.0, 0.97, 0.88]), col)
        jr = np.random.default_rng(int(t * 10)).integers(-1, 2, (ROWS, COLS))
        idx = SOFT[np.clip((np.clip(b, 0, 1) * 1.6 * (len(SOFT) - 1)).astype(int) + jr * (b < 0.45), 0, len(SOFT) - 1)]
        idx = np.where(b < 0.03, 0, idx)
        bright = np.clip(b, 0, 1) > 0.6
        idx = np.where(bright | star, to_glyphs(np.clip(b, 0, 1), int(t * 10)), idx)
        col = np.clip(col * np.clip(0.55 + 0.6 * b, 0, 1)[..., None], 0, 1)
        # sparkle arms so it reads as their star
        if v > 0:
            r0 = int(round((sy - OY) / CH - 0.5)); c0 = int(round(sx / CW - 0.5))
            for dc in range(-int(26 * sc), int(26 * sc) + 1):
                f = 1 - abs(dc) / (26 * sc + 1)
                if 0 <= c0 + dc < COLS and abs(dc) > 3 and not (inl | inr)[r0, c0 + dc]:
                    idx[r0, c0 + dc] = CI["-"]; col[r0, c0 + dc] = np.array([1.0, 0.95, 0.85]) * f
            for dr_ in range(-int(9 * sc), int(9 * sc) + 1):
                f = 1 - abs(dr_) / (9 * sc + 1)
                if 0 <= r0 + dr_ < ROWS and abs(dr_) > 1 and not (inl | inr)[r0 + dr_, c0]:
                    idx[r0 + dr_, c0] = CI["|"]; col[r0 + dr_, c0] = np.array([1.0, 0.95, 0.85]) * f
        # the left hand (me) is made of code: sparse code rain inside the silhouette
        rr, cc = np.nonzero(inl & ~star)
        if len(rr):
            ph = (rr - u * self.rain_speed[cc] - self.rain_phase[cc]) % 22
            vis = ph < 3.5
            rr, cc, ph = rr[vis], cc[vis], ph[vis]
            idx[rr, cc] = [CI.get(self.code_rows[r][c % 60], 0) for r, c in zip(rr, cc)]
            col[rr, cc] = np.array([1.0, 0.72, 0.45])[None] * (0.2 + 0.55 * (1 - ph[:, None] / 3.5))
        # end card
        rows = int(ROWS * 0.80)
        if t > T_SPARK + 0.35:
            lines = [(">>> let's work together.", CREAM, T_SPARK + 0.35),
                     (self.handle, np.array([1.0, 0.75, 0.45]), T_SPARK + 1.6),
                     ("stage visual concept for 2ND SUN - made 100% in code", CREAM * 0.55, T_SPARK + 2.1)]
            for li, (txt, cc_, t0) in enumerate(lines):
                if not txt or t < t0: continue
                s, typing = typed(txt, t, t0, 28 if li < 2 else 60)
                r = rows + li * 2
                c0 = (COLS - len(txt)) // 2
                idx[r, c0 - 2:c0 + len(txt) + 2] = 0
                put_text(idx, col, r, c0, s, cc_)
                if typing: idx[r, c0 + len(s)] = BLOCK; col[r, c0 + len(s)] = CREAM
        return idx, col.astype(np.float32)

    def frame(self, t, fi):
        if t < 1.9:
            idx, col = self.terminal(t)
        elif t < BREAK:
            idx, col = self.build(t, fi)
        elif t < T_SUN:
            idx, col = self.break_(t)
        elif t < T_TUNNEL:
            idx, col = self.sun(t)
        elif t < T_STROBE:
            idx, col = self.tunnel(t)
        elif t < T_HANDS:
            idx, col = self.strobe(t, fi)
        else:
            idx, col = self.hands(t)
        img = render(idx, col)
        g = 1 - float(sstep(DUR - 0.5, DUR, t))
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
