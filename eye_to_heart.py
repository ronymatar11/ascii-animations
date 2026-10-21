"""
Eye -> Heart  |  ASCII animation in Python
An ASCII eye slowly opens, looks around, blinks, then the camera dives into the
pupil and finds a real (anatomical) heart beating in the dark.

Install:   pip install numpy pillow pygame
Live:      python eye_ascii.py
Export:    python eye_ascii.py --export eye.mp4            (needs ffmpeg, or: pip install imageio-ffmpeg)
Vertical:  python eye_ascii.py --export eye_vertical.mp4 --size 1080x1920
Detail:    --cell 10  (smaller letters = more detail)

Keys (live): ESC quit, R restart, SPACE pause
"""
import argparse, shutil, subprocess, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DUR = 15.0
BPM = 72
CHARSET = "`.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"

# ---------------------------------------------------------------- helpers
def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def sstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)

def smoother(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)

def _hash(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFFFF) / 4294967295.0

def vnoise(x, y, seed=0):
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash(ix, iy, seed), _hash(ix + 1, iy, seed)
    c, d = _hash(ix, iy + 1, seed), _hash(ix + 1, iy + 1, seed)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy

def fbm(x, y, octaves=4, seed=0):
    v, amp, tot = 0.0, 0.5, 0.0
    for i in range(octaves):
        v = v + amp * vnoise(x, y, seed + i * 17)
        tot += amp
        x, y, amp = x * 2.03, y * 2.03, amp * 0.5
    return v / tot

def find_font(size):
    for p in ["C:/Windows/Fonts/consola.ttf", "consola.ttf",
              "/System/Library/Fonts/Menlo.ttc", "Menlo.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVuSansMono.ttf",
              "C:/Windows/Fonts/cour.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()

# ---- signed distance pieces for the heart
def sd_ellipse(px, py, cx, cy, rx, ry, ang):
    c, s = np.cos(ang), np.sin(ang)
    x, y = (px - cx) * c + (py - cy) * s, -(px - cx) * s + (py - cy) * c
    return (np.sqrt((x / rx) ** 2 + (y / ry) ** 2) - 1.0) * min(rx, ry)

def sd_tube(px, py, pts, r0, r1=None):
    """distance to a polyline with radius tapering r0 -> r1"""
    r1 = r0 if r1 is None else r1
    best = np.full(px.shape, 1e9)
    n = len(pts) - 1
    for i in range(n):
        ax, ay = pts[i]; bx, by = pts[i + 1]
        vx, vy = bx - ax, by - ay
        h = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        r = r0 + (r1 - r0) * (i + h) / n
        best = np.minimum(best, np.hypot(px - ax - vx * h, py - ay - vy * h) - r)
    return best

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)

def heartbeat(t):
    ph = (t * BPM / 60.0) % 1.0
    lub = np.exp(-((ph - 0.06) / 0.05) ** 2)
    dub = 0.55 * np.exp(-((ph - 0.30) / 0.045) ** 2)
    return lub + dub, ph

# ---------------------------------------------------------------- renderer
class AsciiEye:
    def __init__(self, W, H, cell_h=12, seed=7):
        self.W, self.H = W, H
        font = find_font(cell_h - 1)
        cw = max(5, int(round(font.getlength("M"))))
        self.cw, self.ch = cw, cell_h
        self.cols, self.rows = W // cw, H // cell_h
        self.ox = (W - self.cols * cw) // 2
        self.oy = (H - self.rows * cell_h) // 2

        # glyph atlas: ramp sorted by ink density + special stroke glyphs
        chars = sorted(set(CHARSET + " +-|/\\"))
        masks = {}
        for c in chars:
            im = Image.new("L", (cw, cell_h), 0)
            ImageDraw.Draw(im).text((cw / 2, cell_h / 2), c, fill=255, font=font, anchor="mm")
            masks[c] = np.asarray(im, np.float32) / 255.0
        ramp = sorted([c for c in chars if c in CHARSET], key=lambda c: masks[c].mean())
        extra = ["+", "-", "|", "/", "\\"]
        self.atlas = np.stack([masks[" "]] + [masks[c] for c in ramp] + [masks[c] for c in extra])
        self.n_ramp = len(ramp)
        self.PLUS, self.DASH, self.PIPE, self.SL, self.BSL = range(self.n_ramp + 1, self.n_ramp + 6)

        unit = min(H / 2, W / (2 * 1.5)) * 1.1
        jx = (np.arange(self.cols) + 0.5) * cw + self.ox
        jy = (np.arange(self.rows) + 0.5) * cell_h + self.oy
        self.X, self.Y = np.meshgrid((jx - W / 2) / unit, -(jy - H / 2) / unit)
        self.unit = unit

        rng = np.random.default_rng(seed)
        self.rng = rng
        n = 520
        self.star_q = rng.uniform(-1.0, 1.0, (n, 2))
        self.star_b = rng.uniform(0.35, 1.0, n) ** 2
        self.star_ph = rng.uniform(0, 6.28, n)
        self.star_sp = rng.uniform(1.5, 5.0, n)
        self.star_big = rng.random(n) < 0.15
        self.jitter = np.zeros((self.rows, self.cols), np.int64)
        self.dither = rng.random((self.rows, self.cols))
        # (root x, length, curl, upper?)
        self.lashes = []
        for xr in np.linspace(-1.12, 1.18, 46):
            xr = xr + rng.uniform(-0.015, 0.015)
            w = (1 - (xr / 1.3) ** 2) ** 0.4
            self.lashes.append((xr, 0.22 * w * rng.uniform(0.75, 1.1), rng.uniform(0.25, 0.55), True))
        for xr in np.linspace(-0.95, 1.05, 22):
            w = (1 - (xr / 1.3) ** 2) ** 0.5
            self.lashes.append((xr + rng.uniform(-0.02, 0.02), 0.07 * w * rng.uniform(0.6, 1.1), 0.3, False))

    # ---------- timeline
    def params(self, t):
        # slow, soft opening: lids part, pause a beat half open, then fully open
        o1 = smoother(0.3, 1.6, t) * 0.45
        o2 = smoother(1.7, 2.9, t) * 0.55
        openness = o1 + o2
        # gaze: drift left, then right, back to center
        gx = -0.18 * smoother(3.0, 3.6, t) + 0.36 * smoother(3.9, 4.5, t) - 0.18 * smoother(4.8, 5.4, t)
        gy = 0.03 * smoother(3.9, 4.5, t) - 0.03 * smoother(4.8, 5.4, t)
        # soft blink
        bl = np.exp(-((t - 5.6) / 0.13) ** 2)
        openness *= 1 - bl
        rp = 0.11 + 0.04 * smoother(5.9, 6.9, t)  # pupil dilates before the dive
        zf = 2.7 / rp
        k = smoother(6.6, 10.4, t)
        zoom = np.exp(np.log(zf) * k) * np.exp(0.01 * max(0.0, t - 10.4))
        return gx, gy, openness, rp, zoom

    # ---------- the heart (in pupil-local coords)
    def heart(self, Qx, Qy, t, size):
        pulse, ph = heartbeat(t)
        s = size * (1 + 0.05 * pulse)
        shake = 0.006 * pulse
        u = (Qx - shake) / (s * 1.25)
        v = Qy / s + 0.16
        sq = 0.06 * pulse          # ventricles squeeze on the beat
        at = 0.06 * (1 - pulse)    # atria swell between beats

        # muscle: ventricles + atria + auricle
        vent = sd_ellipse(u, v, 0.0, -0.12, 0.62 * (1 - sq), 0.55 * (1 - sq * 0.6), -0.3)
        apex = sd_tube(u, v, [(0.0, -0.15), (0.18, -0.60), (0.30, -0.98)], 0.50 * (1 - sq), 0.06)
        ra = sd_ellipse(u, v, -0.50, 0.18, 0.30 * (1 + at), 0.36 * (1 + at), 0.2)
        la = sd_ellipse(u, v, 0.52, 0.22, 0.22 * (1 + at), 0.17 * (1 + at), 0.5)
        muscle = smin(smin(vent, apex, 0.25), smin(ra, la, 0.15), 0.18)

        # vessels
        aorta = sd_tube(u, v, [(0.02, 0.30), (0.00, 0.75), (0.12, 1.02), (0.38, 1.08), (0.58, 0.92), (0.62, 0.60)], 0.15, 0.12)
        br1 = sd_tube(u, v, [(0.10, 1.00), (0.04, 1.30)], 0.055, 0.04)
        br2 = sd_tube(u, v, [(0.27, 1.08), (0.27, 1.38)], 0.05, 0.035)
        br3 = sd_tube(u, v, [(0.44, 1.04), (0.52, 1.30)], 0.05, 0.035)
        art = smin(smin(aorta, br1, 0.06), smin(br2, br3, 0.06), 0.06)
        pulm = sd_tube(u, v, [(0.22, 0.25), (0.12, 0.60), (-0.10, 0.82), (-0.28, 0.86)], 0.13, 0.10)
        pulm = smin(pulm, sd_tube(u, v, [(0.06, 0.70), (0.30, 0.78), (0.48, 0.70)], 0.08, 0.06), 0.05)
        svc = sd_tube(u, v, [(-0.52, 0.40), (-0.50, 0.80), (-0.48, 1.15)], 0.11, 0.10)

        d = smin(smin(muscle, art, 0.08), smin(pulm, svc, 0.08), 0.08)
        mask = d < 0

        # fake 3D: height from distance, normals from its gradient
        thick = 0.42
        hgt = np.sqrt(np.clip(-d, 0, None) * (2 * thick + np.clip(d, -thick, 0)))
        gy_, gx_ = np.gradient(hgt)
        cell_u = (Qx[0, 1] - Qx[0, 0]) / s if Qx.shape[1] > 1 else 1
        cell_v = (Qy[0, 0] - Qy[1, 0]) / s if Qy.shape[0] > 1 else 1
        nx, ny = -gx_ / max(cell_u, 1e-6), gy_ / max(cell_v, 1e-6)
        nz = np.ones_like(nx)
        nl = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nl, ny / nl, nz / nl
        L = np.array([-0.5, 0.6, 0.62]); L /= np.linalg.norm(L)
        lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
        hx, hy, hz = L[0], L[1], L[2] + 1
        hn = np.sqrt(hx * hx + hy * hy + hz * hz)
        spec = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0, 1) ** 40

        # surface detail: striations, coronary arteries, fat in the grooves
        strio = fbm(u * 6 + v * 3, v * 9 - u * 2, 4, 51)
        cor = np.abs(fbm(u * 1.6 + 3, v * 1.6, 2, 61) - 0.5)
        coronary = (cor < 0.02) & (muscle < 0) & (muscle < art) & (v < 0.25)
        groove = np.exp(-(((u - 0.12 + v * 0.35) / 0.05) ** 2)) * (muscle < 0) * (v < 0.2)

        b = (0.10 + 0.95 * lam ** 1.3 + 0.10 * strio) * (0.90 + 0.30 * pulse)
        b = b + 0.55 * spec
        b = np.where(coronary, b * 0.55 + 0.25, b)

        base = np.stack([np.full_like(b, 0.88), 0.16 + 0.10 * strio, 0.20 + 0.06 * strio], -1)
        col = base
        col = np.where((art < muscle)[..., None] & (art < pulm)[..., None], np.array([0.95, 0.42, 0.45]), col)
        col = np.where((pulm < muscle)[..., None] & (pulm < art)[..., None], np.array([0.85, 0.35, 0.50]), col)
        col = np.where((svc < muscle)[..., None] & (svc < art)[..., None] & (svc < pulm)[..., None],
                       np.array([0.55, 0.38, 0.75]), col)
        col = np.where(coronary[..., None], np.array([1.0, 0.55, 0.45]), col)
        col = col * (1 - groove[..., None] * 0.5) + groove[..., None] * 0.5 * np.array([1.0, 0.82, 0.45])
        col = col + spec[..., None] * np.array([0.6, 0.6, 0.6])
        return mask, np.clip(b, 0, 1), np.clip(col, 0, 1), d * s, pulse

    # ---------- one frame -> RGB uint8 image
    def frame(self, t, fi):
        gx, gy, op, rp, zoom = self.params(t)
        pc = np.array([gx, gy])
        X = self.X / zoom + pc[0] * (1 - 1 / zoom)
        Y = self.Y / zoom + pc[1] * (1 - 1 / zoom)

        # -- lids (almond, upper lid peaks a bit toward the inner corner)
        ex = 1.3
        sx = np.clip(1 - (X / ex) ** 2, 0, 1)
        y_low = -0.40 * sx ** 0.85 - 0.03 + 0.03 * X
        y_up_open = 0.58 * sx ** 0.7 + 0.02 - 0.04 * X
        y_up = y_low + (y_up_open - y_low) * op
        inside = (np.abs(X) < ex) & (Y < y_up) & (Y > y_low)

        # -- skin: soft, low-contrast, gentle folds
        r2 = (X / 1.7) ** 2 + (Y / 1.2) ** 2
        tex = fbm(X * 2.5, Y * 2.5, 4, 3)
        crease_y = 0.84 * sx ** 0.65 + 0.10
        crease = np.exp(-((Y - crease_y) / 0.045) ** 2) * (np.abs(X) < 1.45)
        lidskin = smooth(0.0, 0.25, Y - y_up) * (1 - smooth(crease_y - 0.1, crease_y, Y))
        lower_fold = np.exp(-((Y - (-0.62 * sx ** 0.8 - 0.07)) / 0.07) ** 2) * (np.abs(X) < 1.45)
        skin = (0.03 + 0.12 * tex + 0.03 * lidskin + 0.04 * lower_fold) * np.exp(-r2 * 2.4)
        skin = skin * (1 - 0.6 * crease)
        rim = np.exp(-((Y - y_up) / 0.022) ** 2) * (np.abs(X) < ex)
        b = np.where(inside, 0, np.clip(skin + 0.22 * rim, 0, 1))
        col = np.zeros(X.shape + (3,))
        col[:] = (1.0, 0.62, 0.36)

        # -- sclera: round, shaded, pink inner corner, wet waterline
        dx, dy = X - pc[0], Y - pc[1]
        d = np.hypot(dx, dy)
        ball = np.clip(1 - (np.hypot(X * 0.85, Y) / 1.25) ** 2, 0, 1)
        shadow = smooth(0.0, 0.30, y_up - Y)
        scl = (0.30 + 0.42 * ball) * (0.5 + 0.5 * shadow) * (0.92 + 0.12 * fbm(X * 5, Y * 5, 3, 9))
        b = np.where(inside, scl, b)
        scol = np.stack([np.full_like(X, 1.0), 0.88 - 0.1 * (1 - ball), 0.80 - 0.15 * (1 - ball)], -1)
        car = np.exp(-(((X + 1.18) / 0.10) ** 2 + ((Y + 0.02) / 0.09) ** 2))
        scol = scol * (1 - car[..., None]) + car[..., None] * np.array([1.0, 0.45, 0.45])
        col = np.where(inside[..., None], scol, col)
        water = np.exp(-((Y - y_low - 0.018) / 0.012) ** 2) * inside * (np.abs(X) < ex * 0.9)
        b = b + 0.35 * water

        # -- iris: gold collar, copper outer, fibers, crypts, dark limbal ring
        R = 0.33
        a = np.arctan2(dy, dx)
        fib_n = fbm(np.cos(a) * 10 + d * 2.0, np.sin(a) * 10 + d * 1.3, 5, 21)
        fibers = 0.5 + 0.5 * np.sin(a * 90 + fib_n * 10)
        crypt = smooth(0.60, 0.70, fbm(np.cos(a) * 5 + d * 6, np.sin(a) * 5 + d * 6, 3, 77))
        zone = smooth(rp * 1.6, rp * 2.3, d)                     # 0 inner gold, 1 outer copper
        ib = 0.40 + 0.35 * fib_n + 0.28 * fibers - 0.30 * crypt * zone
        ib = ib + 0.30 * np.exp(-((d - rp * 1.9) / 0.025) ** 2)  # collarette ridge
        ib = ib * (1 - 0.85 * smooth(0.80 * R, R, d))            # limbal ring
        ib = ib * (0.45 + 0.55 * shadow)                         # lid shadow
        iris = inside & (d < R * 1.02)
        b = np.where(iris, np.clip(ib, 0, 1), b)
        gold = np.array([1.0, 0.78, 0.38]); copper = np.array([0.95, 0.48, 0.22])
        icol = gold * (1 - zone[..., None]) + copper * zone[..., None]
        icol = icol * (0.85 + 0.3 * fib_n[..., None])
        col = np.where(iris[..., None], icol, col)

        # -- pupil + what's inside it
        pupil = inside & (d < rp)
        b = np.where(pupil, 0.0, b)
        Qx, Qy = dx / rp, dy / rp
        space_a = sstep(6.8, 8.2, t)

        hsize = 0.235 * smoother(7.6, 10.4, t)
        hmask = np.zeros_like(pupil)
        if hsize > 0.01:
            hmask, hb, hcol, hd, pulse = self.heart(Qx, Qy, t, hsize)
            hmask = hmask & pupil
            b = np.where(hmask, hb * space_a, b)
            col = np.where(hmask[..., None], hcol, col)
            # faint red glow around the heart, breathing with the beat
            glow = np.exp(-np.clip(hd, 0, None) / 0.05) * (0.10 + 0.10 * pulse) * space_a
            gm = pupil & ~hmask & (glow > 0.075)
            b = np.where(gm, glow, b)
            col = np.where(gm[..., None], np.array([1.0, 0.25, 0.30]), col)

        # -- wet highlights on the cornea (window + small dot)
        hl1 = np.exp(-(((X - pc[0] + 0.10) / 0.06) ** 4 + ((Y - pc[1] - 0.10) / 0.045) ** 4))
        hl2 = np.exp(-(((X - pc[0] - 0.12) ** 2 + (Y - pc[1] + 0.12) ** 2) / 0.0005))
        hl = np.maximum(hl1, 0.7 * hl2) * (1 - sstep(6.3, 7.0, t)) * op
        b = np.where(inside, np.maximum(b, hl), b)
        col = np.where((inside & (hl > 0.25))[..., None], np.array([1.0, 0.97, 0.94]), col)

        # global fade
        g = sstep(0.0, 0.8, t) * (1 - sstep(14.2, 15.0, t))
        b = np.clip(b, 0, 1) * g

        # -- brightness -> chars (gentle flicker)
        if fi % 4 == 0:
            self.jitter = self.rng.integers(-1, 2, (self.rows, self.cols))
            self.dither = self.rng.random((self.rows, self.cols))
        idx = np.clip(np.round(b * 1.1 * (self.n_ramp - 1)).astype(np.int64) + 1 + self.jitter, 1, self.n_ramp)
        idx = np.where(b < 0.06 + 0.06 * self.dither, 0, idx)
        rgb = col * np.clip(0.30 + 0.85 * b, 0, 1)[..., None]

        # -- eyelashes: real curved strokes, drawn with / | \\ - glyphs
        if zoom < 5 and op > 0.03:
            lash_on = smooth(0.03, 0.25, op) * g
            for (xr, ln, cu, up) in self.lashes:
                yu_r = -0.40 * (1 - (xr / ex) ** 2) ** 0.85 - 0.03 + 0.03 * xr
                if up:
                    yo = 0.58 * (1 - (xr / ex) ** 2) ** 0.7 + 0.02 - 0.04 * xr
                    yr = yu_r + (yo - yu_r) * op
                    th0, sgn = 0.75 * xr / ex, 1.0
                    ln = ln * (0.45 + 0.55 * op)
                else:
                    yr, th0, sgn = yu_r - 0.01, 1.0 * xr / ex, -1.0
                px, py = xr, yr
                steps = 9
                for k in range(steps):
                    f = k / steps
                    th = th0 + cu * f * np.sign(xr) * (1 if up else 0.5)
                    ddx, ddy = np.sin(th) * ln / steps, sgn * np.cos(th) * ln / steps
                    px, py = px + ddx, py + ddy
                    sxp = (px - pc[0] * (1 - 1 / zoom)) * zoom
                    syp = (py - pc[1] * (1 - 1 / zoom)) * zoom
                    i = int(round((sxp * self.unit + self.W / 2 - self.ox) / self.cw - 0.5))
                    j = int(round((-syp * self.unit + self.H / 2 - self.oy) / self.ch - 0.5))
                    if not (0 <= i < self.cols and 0 <= j < self.rows):
                        continue
                    ang = np.arctan2(ddx * self.unit / self.cw, abs(ddy) * self.unit / self.ch)
                    ang = ang if up else -ang
                    gl = self.PIPE if abs(ang) < 0.4 else (self.DASH if abs(ang) > 1.2 else (self.SL if ang > 0 else self.BSL))
                    br = (0.85 - 0.5 * f) * lash_on
                    idx[j, i] = gl
                    rgb[j, i] = np.array([1.0, 0.80, 0.62]) * br

        # -- stars inside the pupil
        if space_a > 0:
            px_unit = rp * zoom
            sx_ = pc[0] + self.star_q[:, 0] * px_unit
            sy_ = pc[1] + self.star_q[:, 1] * px_unit
            ci = np.round((sx_ * self.unit + self.W / 2 - self.ox) / self.cw - 0.5).astype(int)
            cj = np.round((-sy_ * self.unit + self.H / 2 - self.oy) / self.ch - 0.5).astype(int)
            tw = self.star_b * (0.55 + 0.45 * np.sin(t * self.star_sp + self.star_ph)) * space_a * g
            for k in range(len(tw)):
                i, j = ci[k], cj[k]
                if not (0 <= i < self.cols and 0 <= j < self.rows):
                    continue
                if not pupil[j, i] or hmask[j, i] or tw[k] < 0.08:
                    continue
                c = np.array([0.95, 0.95, 1.0]) * (0.35 + 0.65 * tw[k])
                if self.star_big[k] and tw[k] > 0.3:
                    idx[j, i], rgb[j, i] = self.PLUS, c
                    arm = 1 + int(tw[k] * 3)
                    for s_ in range(1, arm + 1):
                        f = c * (1 - s_ / (arm + 1))
                        for ii in (i - s_, i + s_):
                            if 0 <= ii < self.cols and not hmask[j, ii]:
                                idx[j, ii], rgb[j, ii] = self.DASH, f
                        if s_ <= max(1, arm // 2):
                            for jj in (j - s_, j + s_):
                                if 0 <= jj < self.rows and not hmask[jj, i]:
                                    idx[jj, i], rgb[jj, i] = self.PIPE, f
                else:
                    idx[j, i] = max(1, int(tw[k] * 6))
                    rgb[j, i] = c

        # -- compose glyphs
        glyphs = self.atlas[idx]
        img = glyphs[..., None] * rgb[:, :, None, None, :]
        img = img.transpose(0, 2, 1, 3, 4).reshape(self.rows * self.ch, self.cols * self.cw, 3)
        out = np.zeros((self.H, self.W, 3), np.uint8)
        out[self.oy:self.oy + img.shape[0], self.ox:self.ox + img.shape[1]] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return out

# ---------------------------------------------------------------- run modes
def export(path, W, H, fps, cell):
    eye = AsciiEye(W, H, cell)
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            sys.exit("ffmpeg not found. Install it or run: pip install imageio-ffmpeg")
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                          "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium", path],
                         stdin=subprocess.PIPE)
    n = int(DUR * fps)
    for fi in range(n):
        p.stdin.write(eye.frame(fi / fps, fi).tobytes())
        if fi % 15 == 0 or fi == n - 1:
            print(f"\rrendering {fi + 1}/{n}", end="", flush=True)
    p.stdin.close(); p.wait()
    print(f"\nsaved {path}")

def live(W, H, cell):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Eye -> Heart")
    eye = AsciiEye(W, H, cell)
    clock = pygame.time.Clock()
    t0, fi, paused, tp = time.time(), 0, False, 0.0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                t0 = time.time()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
                if paused: tp = time.time() - t0
                else: t0 = time.time() - tp
        t = tp if paused else (time.time() - t0) % DUR
        img = eye.frame(t, fi)
        screen.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
        pygame.display.flip()
        fi += 1
        clock.tick(30)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help="render to an mp4 instead of opening a window")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cell", type=int, default=12, help="character height in px (smaller = more detail)")
    a = ap.parse_args()
    W, H = map(int, a.size.lower().split("x"))
    if a.export:
        export(a.export, W, H, a.fps, a.cell)
    else:
        live(W, H, a.cell)
