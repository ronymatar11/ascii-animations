"""
Code -> Mona Lisa  |  9:16 ASCII animation in Python

pulse -> hello world -> code everywhere -> code talks -> an eye forms ->
it looks at us -> camera pulls back: the Mona Lisa on a museum wall.

Install:  pip install numpy pillow          (+ ffmpeg, or: pip install imageio-ffmpeg)
Export:   python code_to_monalisa.py --export monalisa.mp4
Options:  --image my_monalisa.jpg   use your own (bigger) Mona Lisa image
          --eye 202 202             eye position in that image (pixels), if you change image
          --no-audio
"""
import argparse, math, shutil, subprocess, sys, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

W, H = 1080, 1920
CW, CH = 10, 18
COLS, ROWS = W // CW, H // CH            # 108 x 106 character grid
OY = (H - ROWS * CH) // 2
FPS = 30
DUR = 23.0
CX, CY = COLS / 2, ROWS / 2               # grid center (54, 53)
ROW0 = ROWS // 2 - 1                      # hello world row
CP = COLS // 2 - 12                       # hello world column
CHARS = [chr(i) for i in range(32, 127)] + ["\u2588"]
CI = {c: i for i, c in enumerate(CHARS)}
BLOCK = CI["\u2588"]
RAMP_SET = "`.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"

# ------------------------------------------------------------------ utils
def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def smoother(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
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
    for p in ["C:/Windows/Fonts/consola.ttf", "consola.ttf", "/System/Library/Fonts/Menlo.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVuSansMono.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()

# ------------------------------------------------------------------ glyphs + grid renderer
class Atlas:
    def __init__(self):
        self.cache = {}

    def get(self, k):
        if k not in self.cache:
            font = find_font(int(round(16 * k)))
            cw, ch = CW * k, CH * k
            arr = np.zeros((len(CHARS), ch, cw), np.float32)
            for i, c in enumerate(CHARS[:-1]):
                im = Image.new("L", (cw, ch), 0)
                ImageDraw.Draw(im).text((cw / 2, ch / 2), c, fill=255, font=font, anchor="mm")
                arr[i] = np.asarray(im, np.float32) / 255.0
            arr[BLOCK, int(ch * 0.12):int(ch * 0.92), int(cw * 0.1):int(cw * 0.9)] = 1.0
            self.cache[k] = arr
        return self.cache[k]

ATLAS = Atlas()
_a1 = ATLAS.get(1)
RAMP = np.array(sorted([CI[c] for c in RAMP_SET], key=lambda i: _a1[i].mean()))
PRINTABLE = np.array([CI[c] for c in CHARS[1:-1]])

def compose(atlas, idx, col):
    g = atlas[idx]
    img = g[..., None] * col[:, :, None, None, :]
    r, c, gh, gw, _ = img.shape
    return (np.clip(img.transpose(0, 2, 1, 3, 4).reshape(r * gh, c * gw, 3), 0, 1) * 255).astype(np.uint8)

def render(idx, col, z=1.0, cx=CX, cy=CY):
    if abs(z - 1) < 1e-3 and abs(cx - CX) < 1e-3 and abs(cy - CY) < 1e-3:
        out = np.zeros((H, W, 3), np.uint8)
        img = compose(ATLAS.get(1), idx, col)
        out[OY:OY + img.shape[0], :img.shape[1]] = img
        return out
    k = int(math.ceil(z - 1e-6))
    hw, hh = W / 2 / (CW * z), H / 2 / (CH * z)
    c0, c1 = max(0, int(cx - hw) - 1), min(COLS, int(math.ceil(cx + hw)) + 1)
    r0, r1 = max(0, int(cy - hh) - 1), min(ROWS, int(math.ceil(cy + hh)) + 1)
    blk = Image.fromarray(compose(ATLAS.get(k), idx[r0:r1, c0:c1], col[r0:r1, c0:c1]))
    a = k / z
    ox = (-W / 2 / (CW * z) + cx - c0) * CW * k
    oy = (-(H / 2 - OY * 0) / (CH * z) + cy - r0) * CH * k
    out = blk.transform((W, H), Image.AFFINE, (a, 0, ox, 0, a, oy), resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    return np.asarray(out)

# ------------------------------------------------------------------ code text
POOL = [
    "import numpy as np", "from eye import Iris, Pupil", "def see(self): return self.light",
    "while alive: heart.beat()", "for y in range(H):", "for x in range(W):",
    "px[y][x] = ramp[int(lum * len(ramp))]", "class Eye(Model):", "self.iris = Iris(r=0.33)",
    "return np.clip(v, 0, 1)", "# TODO: learn how to see", "if gaze.locked(): look_at(you)",
    "const pulse = Math.sin(t * 72 / 60);", "async function dream() { await wake(); }",
    "let canvas = new Canvas(1080, 1920);", "SELECT * FROM memories WHERE seen = 1;",
    "0x7f3a 0x00ff 0xdead 0xbeef", "for (let i = 0; i < n; i++) {", "model.fit(X, y, epochs=1503)",
    "artist = Artist('Leonardo')", "smile = 0.5 * mystery", "sfumato = blur(edges, sigma=1.2)",
    "light = normalize(L - P)", "out.append(chr(65 + k))", "if not hope: raise SystemExit",
    "print('hello world')", "x = (x * 1103515245 + 12345) % 2**31", "def __init__(self, *a):",
    "yield from self.thoughts", "assert soul is not None", "ctx.drawImage(face, 0, 0);",
    "return [p for p in pixels if p.alive]", "neurons[i] += w[i][j] * a[j]", "try: connect(node)",
    "except Timeout: retry()", "mem = bytearray(1 << 20)", "color = (0.82, 0.64, 0.31)",
    "while True: listen()", "frame = render(scene, cam)", "sig = 1 / (1 + exp(-z))",
    "git commit -m 'she smiles'", "npm run dream", "eyes = detect(face)[:2]", "# who is watching who?",
    "loss.backward(); opt.step()", "fn main() { println!(\"hi\"); }", "if (x > y) swap(x, y);",
    "return f'{name} was here'", "grid = [[0] * 108 for _ in rows]", "theta += 0.0174533",
]
KW = {"import", "from", "def", "return", "while", "for", "in", "if", "class", "not", "is", "None",
      "const", "async", "function", "await", "let", "new", "try", "except", "yield", "raise",
      "True", "SELECT", "FROM", "WHERE", "fn", "as", "lambda"}
C_DEF, C_KW, C_STR = (0.80, 0.82, 0.86), (0.78, 0.50, 0.92), (0.95, 0.62, 0.45)
C_NUM, C_FN, C_COM = (0.70, 0.88, 0.62), (0.52, 0.80, 1.00), (0.45, 0.66, 0.40)

def colorize(text):
    cols = np.tile(np.array(C_DEF, np.float32), (len(text), 1))
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "#" or text.startswith("//", i):
            cols[i:] = C_COM; break
        if ch in "'\"":
            j = text.find(ch, i + 1); j = n - 1 if j < 0 else j
            cols[i:j + 1] = C_STR; i = j + 1; continue
        if ch.isdigit():
            j = i
            while j < n and (text[j].isalnum() or text[j] == "."): j += 1
            cols[i:j] = C_NUM; i = j; continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"): j += 1
            w = text[i:j]
            if w in KW: cols[i:j] = C_KW
            elif j < n and text[j] == "(": cols[i:j] = C_FN
            i = j; continue
        i += 1
    return cols

HELLO = 'print("hello world")'
PROMPT = ">>> "

# ------------------------------------------------------------------ timeline: phase A (pulse, hello, code)
BEATS = [1.05, 1.9, 2.75]
ECG_W, ECG_SPEED = 32, 29.0

def ecg_val(tau):
    v = 0.0
    for b in BEATS:
        d = tau - b
        v += 0.12 * math.exp(-((d + 0.18) / 0.03) ** 2) - 0.15 * math.exp(-((d + 0.03) / 0.012) ** 2)
        v += 1.0 * math.exp(-(d / 0.013) ** 2) - 0.28 * math.exp(-((d - 0.03) / 0.012) ** 2)
        v += 0.25 * math.exp(-((d - 0.22) / 0.05) ** 2)
    return v

def zoomA(t):
    return math.exp(math.log(3.4) * (1 - float(smoother(6.3, 10.2, t))))

class Story:
    def __init__(self, img_path, eye, seed=11):
        rng = np.random.default_rng(seed)
        self.rng = rng
        # typing times for hello world
        iv = rng.uniform(0.06, 0.17, len(HELLO))
        self.type_t = 3.75 + np.cumsum(iv)
        self.enter_t = self.type_t[-1] + 0.35

        # code segments
        tz = np.linspace(6.0, 10.3, 800)
        inv = np.array([1 / zoomA(x) for x in tz])
        segs = []
        for r in range(ROWS):
            spans = [(0, CP - 2), (CP + 27, COLS)] if r in (ROW0, ROW0 + 1, ROW0 + 2) else [(0, COLS)]
            for (a, b) in spans:
                x = a
                while x < b - 3:
                    L = int(rng.integers(16, 46)); e = min(b, x + L)
                    if b - e < 6: e = b
                    segs.append([r, x, e])
                    x = e + int(rng.integers(1, 4))
        need = []
        for r, a, b in segs:
            dc = max(0.0, max(a - CX, CX - b))
            need.append(max(dc / (COLS / 2), abs(r + 0.5 - CY) / (ROWS / 2)))
        need = np.array(need)
        order = np.argsort(need + rng.uniform(0, 0.06, len(need)))
        N = len(segs)
        self.segs = []
        for rank, si in enumerate(order):
            r, a, b = segs[si]
            frac = rank / N
            k = np.searchsorted(-inv, -max(need[si], 1e-3) * 1.0)
            k = np.argmax(inv >= need[si]) if (inv >= need[si]).any() else len(tz) - 1
            t_vis = tz[k]
            start = max(6.25 + 3.3 * math.sqrt(frac), t_vis - 0.4) + rng.uniform(0, 0.25)
            speed = 18 + 55 * frac + rng.uniform(0, 15)
            L = b - a
            txt = " " * (int(rng.choice([0, 0, 4, 8])) if a == 0 else 0)
            while len(txt) < L:
                txt += POOL[int(rng.integers(len(POOL)))] + ("  " if rng.random() < 0.3 else " ")
            txt = txt[:L]
            self.segs.append((r, a, np.array([CI.get(c, 0) for c in txt]), colorize(txt), start, speed))
        self.code_done = 10.8

        # phase B packets
        pk = []
        t = 10.9
        while t < 13.5:
            lam = 3 + 90 * float(sstep(10.9, 13.0, t))
            t += rng.exponential(1 / lam)
            ar, ac = rng.uniform(2, ROWS - 2), rng.uniform(2, COLS - 2)
            if t < 12.0:
                ang = rng.uniform(0, 6.283); dist = rng.uniform(8, 40)
                br, bc = ar + math.sin(ang) * dist * 0.55, ac + math.cos(ang) * dist
            else:
                spread = 30 - 26 * float(sstep(12.0, 13.4, t))
                br, bc = CY + rng.normal(0, spread * 0.55), CX + rng.normal(0, spread)
            br, bc = min(max(br, 0), ROWS - 1), min(max(bc, 0), COLS - 1)
            dist = math.hypot(bc - ac, (br - ar) * 1.8) + 1e-3
            pk.append((t, ar, ac, br, bc, dist, rng.uniform(40, 80), rng.uniform(-0.25, 0.25)))
        self.packets = pk
        rr, cc = np.mgrid[0:ROWS, 0:COLS]
        rn = np.hypot((cc + 0.5 - CX) / (COLS / 2), (rr + 0.5 - CY) / (ROWS / 2)) / 1.414
        self.Ts = 12.0 + 1.9 * (0.72 * rn + 0.28 * rng.random((ROWS, COLS)))

        # painting
        self.setup_painting(img_path, eye)
        self.full_idx, self.full_col = self.phaseA(self.code_done + 5)
        self.target = self.phaseC(14.0)

    # ---------- phase A
    def phaseA(self, t):
        idx = np.zeros((ROWS, COLS), np.int64)
        col = np.zeros((ROWS, COLS, 3), np.float32)
        # ECG pulse
        if 0.5 <= t < 3.4:
            fade = 1 - float(sstep(3.0, 3.35, t))
            c0 = int(CX - ECG_W / 2)
            pos = (t - 0.5) * ECG_SPEED
            ho = pos % ECG_W
            prev = None
            base = ROW0 + 1
            for co in range(ECG_W):
                back = (ho - co) % ECG_W
                if back > ECG_W - 3:
                    prev = None; continue
                tc = t - back / ECG_SPEED
                if tc < 0.5:
                    prev = None; continue
                rowf = base - ecg_val(tc) * 7
                rr = int(round(rowf))
                f = math.exp(-(t - tc) / 0.9) * fade
                c = np.array([0.35, 1.0, 0.55]) * f if back > 0.6 else np.array([0.9, 1.0, 0.9]) * fade
                cc = c0 + co
                if prev is None or prev == rr:
                    idx[rr, cc] = CI["-"]; col[rr, cc] = c
                else:
                    lo, hi = min(prev, rr), max(prev, rr)
                    for q in range(lo, hi + 1):
                        idx[q, cc] = CI["|"]; col[q, cc] = c
                    idx[rr, cc] = CI["/"] if rr < prev else CI["\\"]
                prev = rr
        # hello world
        blink = (t * 1.8) % 2 < 1
        if t >= 3.3:
            pc = np.array([0.55, 0.58, 0.62])
            for i, ch in enumerate(PROMPT):
                idx[ROW0, CP + i] = CI[ch]; col[ROW0, CP + i] = pc
            n = int(np.searchsorted(self.type_t, t, side="right"))
            hc = colorize(HELLO)
            for i in range(n):
                age = t - self.type_t[i]
                idx[ROW0, CP + 4 + i] = CI[HELLO[i]]
                col[ROW0, CP + 4 + i] = hc[i] + (1 - hc[i]) * math.exp(-age / 0.15)
            if t < self.enter_t and (blink or (n > 0 and t - self.type_t[n - 1] < 0.4)):
                idx[ROW0, CP + 4 + n] = BLOCK; col[ROW0, CP + 4 + n] = (0.85, 0.85, 0.85)
            if t >= self.enter_t + 0.08:
                for i, ch in enumerate("hello world"):
                    idx[ROW0 + 1, CP + i] = CI[ch]; col[ROW0 + 1, CP + i] = (0.95, 0.95, 0.95)
                for i, ch in enumerate(PROMPT):
                    idx[ROW0 + 2, CP + i] = CI[ch]; col[ROW0 + 2, CP + i] = pc
                if blink:
                    idx[ROW0 + 2, CP + 4] = BLOCK; col[ROW0 + 2, CP + 4] = (0.85, 0.85, 0.85)
        # code writers
        if t >= 6.2:
            for (r, a, txt, tc, start, speed) in self.segs:
                if t < start: continue
                L = len(txt)
                n = L if t >= self.code_done else min(L, int((t - start) * speed))
                if n > 0:
                    idx[r, a:a + n] = txt[:n]
                    age = t - (start + np.arange(n) / speed)
                    fl = np.exp(-np.clip(age, 0, None) / 0.12)[:, None]
                    col[r, a:a + n] = tc[:n] + (1 - tc[:n]) * fl
                if n < L and t < self.code_done:
                    idx[r, a + n] = BLOCK; col[r, a + n] = (0.75, 0.78, 0.8)
        return idx, col

    # ---------- phase B: code talks, then becomes the eye
    def phaseB(self, t, fi):
        idx = self.full_idx.copy()
        col = self.full_col.copy() * (1 - 0.45 * float(sstep(10.8, 11.6, t)))
        rng = np.random.default_rng(fi * 7 + 1)
        p = 0.07 * float(sstep(11.2, 12.8, t))
        if p > 0:
            m = rng.random((ROWS, COLS)) < p
            idx[m] = rng.choice(PRINTABLE, int(m.sum()))
            col[m] = (0.55, 0.85, 1.0)
        for (tl, ar, ac, br, bc, dist, sp, bend) in self.packets:
            if t < tl: continue
            dur = dist / sp
            sh = (t - tl) / dur
            if sh > 1.0 + 0.35 / dur:
                continue
            for k in range(8):
                s = sh - k * 0.8 / dist
                if s < 0 or s > 1: continue
                off = math.sin(math.pi * s) * bend * dist
                r = ar + (br - ar) * s + off * (bc - ac) / dist * 0.55
                c = ac + (bc - ac) * s - off * (br - ar) * 1.8 / dist
                ri, ci = int(round(r)), int(round(c))
                if 0 <= ri < ROWS and 0 <= ci < COLS:
                    f = 1 - k / 8
                    col[ri, ci] = np.array([0.55, 0.95, 1.0]) * f + col[ri, ci] * (1 - f)
                    if k == 0: idx[ri, ci] = CI["@"]; col[ri, ci] = (1, 1, 1)
            ta = t - (tl + dur)
            if 0 <= ta < 0.35:
                R = ta / 0.35 * 3.5
                f = 1 - ta / 0.35
                for dr in range(-3, 4):
                    for dc in range(-6, 7):
                        if abs(math.hypot(dc, dr * 1.8) - R) < 0.6:
                            ri, ci = int(round(br)) + dr, int(round(bc)) + dc
                            if 0 <= ri < ROWS and 0 <= ci < COLS:
                                col[ri, ci] = np.maximum(col[ri, ci], np.array([0.6, 0.95, 1.0]) * f)
        # convergence into the eye
        ti, tcol = self.target
        dec = (t > self.Ts - 0.3) & (t < self.Ts)
        idx[dec] = rng.choice(PRINTABLE, int(dec.sum()))
        col[dec] = (0.85, 0.95, 1.0)
        done = t >= self.Ts
        a = sstep(self.Ts, self.Ts + 0.5, t)[..., None]
        idx[done] = ti[done]
        mixed = np.array([0.95, 0.97, 1.0]) * (1 - a) + tcol * a
        col[done] = mixed[done]
        return idx, col

    # ---------- painting / frame / wall
    def setup_painting(self, path, eye):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        tw = min(w, int(h * 0.78))
        x0 = (w - tw) // 2
        im = im.crop((x0, 0, x0 + tw, h))
        self.ex, self.ey = eye[0] - x0, eye[1]
        self.pw, self.ph = tw, h
        up = 4 if max(tw, h) < 1200 else 1
        big = im.resize((tw * up, h * up), Image.LANCZOS)
        big = big.filter(ImageFilter.UnsharpMask(radius=2 * up, percent=90, threshold=2))
        big = ImageEnhance.Contrast(big).enhance(1.12)
        self.up = up
        self.levels = [np.asarray(big, np.float32) / 255.0]
        while self.levels[-1].shape[1] > 40:
            a = self.levels[-1]
            hh, ww = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
            a = a[:hh, :ww]
            self.levels.append(0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]))
        # world layout (units = source pixels); painting at (0..pw, 0..ph)
        self.FB = 0.085 * tw
        s = tw / 400.0
        self.w0 = 40 * s
        self.w1 = 580 * s
        self.c1 = (tw / 2, h * 0.585)
        self.label_y = h + self.FB + 0.12 * h

    def _sample(self, lvl, x, y):
        a = self.levels[lvl]
        sc = self.up / (2 ** lvl)
        u = np.clip(x * sc - 0.5, 0, a.shape[1] - 1.001)
        v = np.clip(y * sc - 0.5, 0, a.shape[0] - 1.001)
        i, j = u.astype(int), v.astype(int)
        fu, fv = (u - i)[..., None], (v - j)[..., None]
        return (a[j, i] * (1 - fu) * (1 - fv) + a[j, i + 1] * fu * (1 - fv) +
                a[j + 1, i] * (1 - fu) * fv + a[j + 1, i + 1] * fu * fv)

    def view(self, t):
        w0 = self.w0
        w_look = w0 * (1 - 0.1 * float(smoother(14.0, 16.6, t)))
        k = float(smoother(16.6, 21.2, t))
        w = math.exp(math.log(w_look) * (1 - k) + math.log(self.w1) * k) * (1 + 0.012 * max(0, t - 21.2))
        f = (w - w_look) / (self.w1 - w_look) if k > 0 else 0.0
        cx = self.ex + (self.c1[0] - self.ex) * f
        cy = self.ey + (self.c1[1] - self.ey) * f
        g = 1 - float(smoother(15.0, 15.45, t))
        gaze = (-2.6 * self.pw / 400 * g, 0.3 * self.pw / 400 * g)
        return w, cx, cy, gaze

    def phaseC(self, t):
        w, cx, cy, gaze = self.view(t)
        upc = w / W
        jx = (np.arange(COLS) + 0.5 - CX) * CW * upc + cx
        jy = (np.arange(ROWS) + 0.5 - CY) * CH * upc + cy
        X, Y = np.meshgrid(jx, jy)
        pw, ph, FB = self.pw, self.ph, self.FB

        # gaze warp around the iris
        s = pw / 400
        dx, dy = (X - self.ex) / (7.5 * s), (Y - self.ey) / (4.2 * s)
        fall = np.clip(1 - (dx * dx + dy * dy), 0, 1) ** 2
        Xs, Ys = X - gaze[0] * fall, Y - gaze[1] * fall

        # painting (trilinear sampling)
        fp = max(CW * upc * self.up, 1.0)
        lf = min(math.log2(fp), len(self.levels) - 1.001)
        l0 = int(lf); fr = lf - l0
        rgb = self._sample(l0, Xs, Ys) * (1 - fr) + self._sample(l0 + 1, Xs, Ys) * fr
        lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        lumn = np.clip((lum - 0.05) / 0.72, 0, 1) ** 0.85
        pcol = np.clip(rgb * 1.75, 0, 1) * (0.40 + 0.75 * lumn)[..., None]

        # frame
        e = np.maximum(np.maximum(-X, X - pw), np.maximum(-Y, Y - ph))
        u = np.clip(e / FB, 0, 1)
        along = np.where(np.abs(-Y) > np.abs(X - pw), X, Y) if False else (X + Y)
        top = (Y < 0) & (Y <= np.minimum(X, pw - X))
        bot = (Y > ph) & (Y - ph >= np.maximum(-X, X - pw))
        left = (X < 0) & ~top & ~bot
        light = np.where(top, 0.25, np.where(bot, -0.2, np.where(left, 0.08, -0.08)))
        orn = 0.18 * np.sin(np.where(top | bot, X, Y) * 0.32 / s) * np.sin(np.pi * u)
        prof = 0.48 + 0.25 * np.cos(u * 2 * np.pi * 1.5) + orn + light
        prof = np.where(u < 0.07, 0.12, prof)
        prof = np.where(u > 0.93, 0.85 + light * 0.5, prof)
        fcol = np.array([1.0, 0.76, 0.38], np.float32)

        # wall + spotlight + frame shadow
        spot = 0.35 + 0.95 * np.exp(-(((X - pw / 2) / (0.95 * pw)) ** 2 + ((Y - 0.3 * ph) / (1.05 * ph)) ** 2))
        wl = (0.07 + 0.08 * fbm(X / (40 * s), Y / (40 * s), 4, 5) + 0.03 * fbm(X / (6 * s), Y / (6 * s), 2, 8)) * spot
        es = np.maximum(np.maximum(-(X - 8 * s) - FB, (X - 8 * s) - pw - FB), np.maximum(-(Y - 16 * s) - FB, (Y - 16 * s) - ph - FB))
        wl = wl * (1 - 0.65 * (1 - np.clip(es / (22 * s), 0, 1)))
        wcol = np.array([0.55, 0.64, 0.85], np.float32)

        inside = e <= 0
        inframe = (e > 0) & (e <= FB)
        b = np.where(inside, lumn, np.where(inframe, np.clip(prof, 0, 1) * 0.85, wl))
        col = np.where(inside[..., None], pcol,
                       np.where(inframe[..., None], fcol * (0.35 + 0.75 * np.clip(prof, 0, 1))[..., None],
                                wcol * (0.4 + 1.6 * wl)[..., None]))

        jr = np.random.default_rng(int(t * 8)).integers(-1, 2, (ROWS, COLS))
        k = np.clip(np.round(b * (len(RAMP) - 1)).astype(int) + jr, 0, len(RAMP) - 1)
        idx = RAMP[k]
        idx = np.where(b < 0.04, 0, idx)

        # museum label
        lf_ = float(sstep(20.4, 21.4, t))
        if lf_ > 0:
            lines = ["MONA LISA", "Leonardo da Vinci, c. 1503"]
            ry = (self.label_y - cy) / (CH * upc) + CY
            r0 = int(round(ry))
            for li, txt in enumerate(lines):
                r = r0 + li * 2
                c0 = int(round((pw / 2 - cx) / (CW * upc) + CX - len(txt) / 2))
                if 0 <= r < ROWS:
                    idx[r, max(0, c0 - 2):min(COLS, c0 + len(txt) + 2)] = 0
                    for i, ch in enumerate(txt):
                        if 0 <= c0 + i < COLS:
                            idx[r, c0 + i] = CI[ch]
                            col[r, c0 + i] = np.array([1.0, 0.93, 0.82]) * lf_ * (1.0 if li == 0 else 0.7)
        return idx, col.astype(np.float32)

    # ---------- any frame
    def frame(self, t, fi):
        if t < 10.8:
            idx, col = self.phaseA(t)
            img = render(idx, col, zoomA(t), CX, CY)
        elif t < 14.0:
            idx, col = self.phaseB(t, fi)
            img = render(idx, col)
        else:
            idx, col = self.phaseC(t)
            img = render(idx, col)
        g = float(sstep(0.0, 0.4, t)) * (1 - float(sstep(22.4, 23.0, t)))
        return (img * g).astype(np.uint8) if g < 1 else img

# ------------------------------------------------------------------ sound design
SR = 48000

def make_audio(story, path):
    n = int(SR * DUR)
    out = np.zeros(n, np.float64)
    rng = np.random.default_rng(5)

    def add(t0, sig, gain=1.0):
        i = int(t0 * SR)
        if i >= n or i + len(sig) <= 0: return
        j = min(n, i + len(sig))
        out[max(i, 0):j] += gain * sig[max(0, -i):j - i]

    def env(d, att=0.003, dec=None):
        tt = np.arange(int(d * SR)) / SR
        e = np.minimum(tt / att, 1)
        return tt, e * (np.exp(-tt / dec) if dec else 1)

    def beep(f, d=0.16):
        tt, e = env(d, 0.004, 0.05)
        return np.sin(2 * np.pi * f * tt) * e

    def thump(d=0.3):
        tt, e = env(d, 0.004, 0.07)
        return np.sin(2 * np.pi * (50 + 40 * np.exp(-tt / 0.03)) * tt) * e

    def click(dur=0.012, bright=1.0):
        tt, e = env(dur, 0.0005, dur / 4)
        s = rng.normal(0, 1, len(tt))
        s = np.diff(s, prepend=0) * bright + s * (1 - bright)
        return s * e

    # pulse
    for b in BEATS:
        add(b, beep(1000), 0.22); add(b - 0.02, thump(), 0.5); add(b + 0.28, thump(0.25), 0.28)
    # hello world typing (clearly audible) + enter
    for tt in story.type_t:
        add(tt, click(0.014, 0.8), 0.32 * rng.uniform(0.8, 1.1))
    add(story.enter_t, click(0.03, 0.4), 0.45)
    # code writers: lots of tiny clicks, quieter as it gets denser
    times = []
    for (r, a, txt, tc, start, speed) in story.segs:
        L = len(txt)
        end = min(start + L / speed, story.code_done)
        k = int((end - start) * speed)
        if k > 0: times.append(start + np.arange(k) / speed)
    times = np.concatenate(times)
    times = times[times < story.code_done]
    hist, edges = np.histogram(times, bins=np.arange(6.0, 11.0, 0.1))
    sel = times[rng.random(len(times)) < 0.35]
    for tt in sel:
        dens = hist[min(max(int((tt - 6.0) / 0.1), 0), len(hist) - 1)] * 10 * 0.35
        add(tt, click(0.008, 0.9), 0.22 / math.sqrt(1 + dens / 15) * rng.uniform(0.5, 1.0))
    # rising drone under the code
    tt = np.arange(int(SR * 6.5)) / SR
    f = 55 * (1 + 0.5 * tt / 6.5)
    ph = 2 * np.pi * np.cumsum(f) / SR
    dr = sum(np.sin(ph * h + h) / h for h in range(1, 7)) * (np.minimum(tt / 4.5, 1) ** 2) * np.exp(-np.maximum(tt - 5.6, 0) / 0.3)
    add(8.0, dr, 0.10)
    # data blips when code talks
    for p in story.packets[::2]:
        add(p[0], beep(rng.uniform(1300, 3200), 0.05), 0.045)
    # eye forms: soft pad swell
    tt = np.arange(int(SR * 5.0)) / SR
    pad = sum(np.sin(2 * np.pi * fq * tt + i) for i, fq in enumerate([220, 261.6, 329.6, 440]))
    pad *= np.sin(np.pi * np.clip(tt / 5.0, 0, 1)) ** 2
    add(12.6, pad, 0.035)
    # gaze lock
    add(15.05, click(0.02, 0.3), 0.35); add(15.05, beep(1800, 0.12), 0.06)
    # pull back whoosh
    d = 5.0
    nz = rng.normal(0, 1, int(SR * d))
    tt = np.arange(len(nz)) / SR
    cut = 0.25 * np.exp(-tt / 1.6) + 0.01
    y = np.zeros_like(nz); acc = 0.0
    for i in range(len(nz)):
        acc += cut[i] * (nz[i] - acc); y[i] = acc
    add(16.5, y * np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 1.5, 0.9)
    # room tone + one last heartbeat
    rt = rng.normal(0, 1, int(SR * 3.0)); rt = np.convolve(rt, np.ones(60) / 60, "same")
    add(20.0, rt * np.minimum(np.arange(len(rt)) / SR / 1.0, 1), 0.25)
    add(21.9, thump(), 0.4); add(22.18, thump(0.25), 0.22)

    out = out / max(1e-6, np.abs(out).max()) * 0.85
    st = np.stack([out, out], 1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((st * 32767).astype(np.int16).tobytes())

# ------------------------------------------------------------------ export
def ffmpeg_bin():
    ff = shutil.which("ffmpeg")
    if ff: return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found. Install it or run: pip install imageio-ffmpeg")

def export(path, img, eye, audio=True):
    story = Story(img, eye)
    ff = ffmpeg_bin()
    vid = path if not audio else path + ".video.mp4"
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                          "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", vid], stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for fi in range(n):
        p.stdin.write(story.frame(fi / FPS, fi).tobytes())
        if fi % 15 == 0 or fi == n - 1:
            print(f"\rrendering {fi + 1}/{n}", end="", flush=True)
    p.stdin.close(); p.wait()
    if audio:
        wav = path + ".wav"
        make_audio(story, wav)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", vid, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", path], check=True)
        import os
        os.remove(vid); os.remove(wav)
    print(f"\nsaved {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="monalisa.mp4")
    ap.add_argument("--image", default="monalisa.png")
    ap.add_argument("--eye", nargs=2, type=float, default=[202.5, 202.5])
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()
    export(a.export, a.image, a.eye, not a.no_audio)
