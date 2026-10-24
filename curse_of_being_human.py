"""
The Curse of Being Human  |  9:16, red and black, every frame is text, cut to a spoken-word track.
One continuous camera, no cuts: each scene turns into the next.

the curse of being human: red code types across the screen and the code is ultron (from a reference),
written in the code itself, which streams across him with pulses of light running along it, his eyes
glowing (he blinks once); he speaks the words: his jaw swings down on its hinge with the voice and his mouth
opens along his own curved lip line, black inside; he leans in toward us, swaying, his head turning and
nodding -> yet manufacture chaos: his eyes flare and catch, and a burning edge spreads out over him from
them; behind it is a painting (from a reference) of a dark figure in a field, its two white eyes exactly where
his were; the camera turns level and pulls back: the figure stands in the wheat, which burns like flames and sways
in gusts of wind, the clouds drift; the figure sways from its feet and breathes, its head tilts, the smear round it
flutters, it closes its eyes and opens them, twice ->
they fear death: the figure burns through, from its eyes, to the x-ray skull underneath (from a reference), its
eye socket where the figure's eyes were, a cassette for a brain; the camera pushes in on it ->
yet waste their lives: the tape unspools, the camera follows it down to money on a string (from a reference) ->
they demand freedom: the string snaps, the money falls and the camera falls after it; far below in the dark a
lit cell (from a reference photo) comes up, the camera falls into it, dust rushing past -> then construct their:
it lands on the bars on the hit, the hands rattle them ->
own prisons: pull back, the prison is inside an eye -> perhaps the greatest flaw in your design: the eye is
one of a wall of eyes darting about, blinking, then all turning to look at you -> is not your mortality: they look
away again -> it is your extraordinary ability: they close, the eye with the prison last; the camera keeps
pulling back without a cut, the skin darkening to stone in spreading patches, and the wall of eyes turns out to be
the face of a stone (from a reference drawing): its
outline closes in from the edges, the skin and the closed eyes shrink into its pencil texture; a butterfly rests on
it, opening and closing its wings -> to remain alive: it takes off and flies up, and on the hit the string
tying it to the stone snaps tight -> without ever truly living: it beats its wings and pulls with everything, the
string yanks it back, the stone barely rocks; it never gets free, and the video ends on it still trying.

The field lingers through a 2 s pause put in the track after "chaos"; under the voice a sound design synthesised
here (no samples): a drone, keys, fire and wind, the tape, the snap, the fall, the bars, the stone, wings, the rope.

Export:   python curse_of_being_human.py --music voice.mp4 --export curse.mp4
"""
import argparse, math, os, subprocess, tempfile, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from code_to_monalisa import W, H, CI, CHARS, RAMP, BLOCK, PRINTABLE, sstep, smoother, fbm, find_font, ffmpeg_bin, compose
from second_sun import (FCW, FCH, FCOLS, FROWS, FOY, FGX, FGY, FATLAS, SOFT, render_fine, glyphs,
                        stroke_glyph, smin, typed)

FPS = 30
DUR = 30.55                               # the story's length: the spoken track's own
# the figure in the field gets more time: a pause is put in the track in the silence after "chaos" (filled with the
# track's own background), and the film slows through it so the field lingers; all times below are the track's
PAUSE_AT, PAUSE = 7.1, 2.0
DUR_V = DUR + PAUSE                       # the video's length

def _warp():
    """video time -> story time: speed 1, easing down to a crawl through the pause, back to 1, losing PAUSE s"""
    tv = np.arange(0, DUR_V + 1, 1 / 1200)
    def lost(e):
        w = np.clip((tv - 6.75) / 0.4, 0, 1); w = w * w * (3 - 2 * w)
        d = np.clip((tv - e) / 0.7, 0, 1); w = w * (1 - d * d * (3 - 2 * d))
        sp = 1 - 0.95 * w
        return np.concatenate([[0], np.cumsum((sp[1:] + sp[:-1]) / 2) / 1200])
    lo, hi = 7.2, 9.5
    for _ in range(40):
        e = (lo + hi) / 2
        if tv[-1] - lost(e)[-1] < PAUSE: lo = e
        else: hi = e
    return tv, lost(e)
_TV, _TS = _warp()

def story(tv):
    return float(np.interp(tv, _TV, _TS))

def video(ts):
    return float(np.interp(ts, _TS, _TV))
HITS = [3.0, 14.0, 17.0, 26.5]            # the heaviest hits in the track
LYRICS = [
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
STALK = np.array(sorted([CI[c] for c in ":;!|lI1H#"], key=lambda i: FATLAS[i].mean()))     # upright strokes: wheat, flames

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


def glow(img, gain=1.15, bloom=0.85):
    """the light of the characters spills around them: a soft blurred copy added on top"""
    im = Image.fromarray(img)
    h, w = img.shape[:2]
    small = im.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2.5))
    halo = np.asarray(small.resize((w, h), Image.BILINEAR), np.float32)
    return np.clip(img.astype(np.float32) * gain + halo * bloom, 0, 255).astype(np.uint8)

_ATLASES = {}
def render_scaled(idx, col, k):
    """the same grid of characters drawn k times bigger (the font itself, not the picture, scaled), for close-ups"""
    if k not in _ATLASES:
        font = find_font(10 * k)
        fw, fh = FCW * k, FCH * k
        arr = np.zeros((len(FATLAS), fh, fw), np.float32)
        for i, ch in enumerate(CHARS[:-1]):
            im = Image.new("L", (fw, fh), 0)
            ImageDraw.Draw(im).text((fw / 2, fh / 2), ch, fill=255, font=font, anchor="mm")
            arr[i] = np.asarray(im, np.float32) / 255.0
        arr[BLOCK, k:fh - k, k:fw - k] = 1.0
        _ATLASES[k] = arr
    out = np.zeros((H * k, W * k, 3), np.uint8)
    img = compose(_ATLASES[k], idx, col)
    out[FOY * k:FOY * k + img.shape[0], :img.shape[1]] = img
    return out

def brighten(col, gain, gamma):
    """lift a colour's brightness, keeping its hue: the dim parts lift the most"""
    m = col.max(-1, keepdims=True)
    return (col * np.minimum(1.0, gain * np.maximum(m, 1e-4) ** gamma) / np.maximum(m, 1e-4)).astype(np.float32)

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
# his mouth: the dark slit between his lips, traced from the reference (it slants up toward his ear with his
# tilted face, and curves), the jaw below it (an ellipse along the mouth: centre, half sizes), the hinge by his ear
ULT_LIP = np.array([(219, 210), (225, 206), (235, 196), (245, 188), (255, 183), (265, 177), (275, 171), (281, 167)], float)
ULT_JAW, ULT_JAW_R, ULT_HINGE = (256.0, 196.0), (38.0, 24.0), (291.0, 160.0)
ULT_MDIR = (0.819, -0.573)                       # along the mouth, front corner to back corner
ULT_LEAN, ULT_HIP = (235.0, 170.0), (300.0, 620.0)   # he leans in toward us about his face, sways about his hips
ULT_EYES = [(243.0, 120.0), (191.0, 156.0)]
ULT_HEAD = [(165, 100), (178, 80), (200, 66), (236, 60), (258, 68), (278, 86), (300, 88), (300, 130), (170, 130)]
# "yet manufacture chaos": his eyes flare and a burning edge spreads out from them; behind it is the painting of
# a dark figure in a field (from a reference), its two white eyes exactly where his were
T_SPREAD = 5.05
FIRE_O = [(546.0, 359.0), (401.0, 447.0)]        # his eyes, on screen, when they catch
FIELD_EYES = [(646.6, 564.1), (729.0, 553.7)]    # the figure's eyes in the painting
FIELD_EM = (687.8, 558.9)                        # between them
FIELD_FEET, FIELD_NECK = (700.0, 1600.0), (690.0, 760.0)   # the figure sways from its feet, its head tilts on its neck
FIELD_BLINKS = [6.15, 7.75, 8.75]                # it closes its eyes and opens them (video time)
# the painting's camera: where its eyes are on screen, screen px per painting px, its tilt. it starts lined up with
# ultron's eyes (his face is tilted), turns level and pulls back to the whole figure in the field
FIELD_EX = [(5.0, 473.5), (5.7, 485.0), (6.5, 505.0), (7.4, 509.0), (8.5, 511.0)]
FIELD_EY = [(5.0, 403.0), (5.7, 440.0), (6.5, 495.0), (7.4, 508.0), (8.5, 511.0)]
FIELD_K = [(5.0, math.log(2.04)), (5.7, math.log(1.55)), (6.5, math.log(1.08)), (7.4, math.log(0.95)), (8.5, math.log(0.93))]
FIELD_R = [(5.0, -0.419), (5.7, -0.2), (6.5, -0.03), (7.4, 0.0), (8.5, 0.0)]
# the x-ray: screen px per image px, the head's centre on screen, the reels, where the drawn tape ends
# at first it sits in the figure's head, its eye socket where the figure's eyes were; then the camera pushes in on it
SKULL_K = [(7.0, 0.62), (7.8, 0.66), (8.6, 1.3), (9.2, 1.5), (12.6, 1.5)]
SKULL_HY = [(7.0, 424), (7.8, 430), (8.6, 640), (9.2, 860), (12.6, 880)]
SKULL_E = (300.0, 240.0)
REELS = [((243.0, 120.0), 40.0), ((368.0, 146.0), 62.0)]
SKULL_DOME = (308.8, 179.5, 296.8, 203.0)        # the ellipse of the skull's outline, for the top the photo crops off
TAPE0 = (250.0, 415.0)
TAPE_L = [(9.2, 0), (9.45, 400), (9.7, 1200), (9.95, 2080), (10.15, 2260), (10.4, 2140), (10.7, 2190), (11.2, 2175), (12.6, 2175)]
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

def fill_rows(img, hole, blur=6, seed=7):
    """paint out the hole in a picture: each row filled across from either side of it, softened, grain put back"""
    plate = img.copy()
    cols = np.arange(img.shape[1])
    for y in np.nonzero(hole.any(1))[0]:
        k = ~hole[y]
        if k.sum() < 2: continue
        for c in range(img.shape[2]):
            plate[y, hole[y], c] = np.interp(cols[hole[y]], cols[k], img[y, k, c])
    soft = np.asarray(Image.fromarray((np.clip(plate, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)),
                      np.float32) / 255.0
    grain = np.random.default_rng(seed).normal(0, 0.045, plate.shape[:2]).astype(np.float32)[..., None]
    return np.where(hole[..., None], np.clip(soft + grain, 0, 1), img)

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
LOGZ = [(12.6, math.log(0.7)), (13.3, math.log(1.4)), (13.85, math.log(3.0)), (14.05, math.log(3.85)),
        (14.35, math.log(3.95)), (15.4, math.log(1.9)),
        (16.7, math.log(1.0)), (17.6, math.log(0.5)), (18.8, math.log(0.36)), (20.0, math.log(0.33)),
        (21.5, math.log(0.31)), (22.6, math.log(0.34)), (23.2, math.log(0.48)), (24.0, math.log(0.3)),
        (24.8, math.log(0.12)), (25.7, math.log(0.087)), (26.4, math.log(0.065)), (27.4, math.log(0.056)),
        (28.4, math.log(0.0553)), (29.5, math.log(0.059)), (30.55, math.log(0.0636))]
CX = [(12.6, 0.3), (14.4, 0.4), (16.0, 0.14), (17.6, 0.0), (19.0, 0.8), (20.5, 0.3), (21.8, -0.6),
      (22.8, -0.35), (23.6, 0.0), (24.6, 0.6), (25.7, 2.1), (26.4, 8.9), (27.4, 13.7), (28.4, 13.9), (29.5, 14.9),
      (30.55, 16.1)]
CY = [(12.6, 0.3), (13.3, 0.13), (13.8, 0.04), (14.4, 0.09), (16.0, 0.04), (17.6, 0.2), (19.0, 0.9), (20.5, -0.4), (21.8, -0.9),
      (22.8, -0.6), (23.6, 0.0), (24.6, 0.8), (25.7, 2.8), (26.4, 11.6), (27.4, 14.8), (28.4, 15.2), (29.5, 16.8),
      (30.55, 18.8)]
ROLLB = [(12.6, -0.09), (13.6, 0.025), (14.6, -0.02), (16.0, 0.04), (17.6, 0.1), (19.5, -0.06), (21.0, 0.08), (22.6, 0.02),
         (23.6, 0.0), (30.55, 0.0)]
# the eye with bars (reference photo): photo px per eye unit, the photo px at the eye's centre
PRI_U, PRI_E = 400.0, (690.0, 774.0)
PRI_BAR = 0.3                                    # its middle bar, in eye units
T_REVEAL = 12.72                                 # the camera reaches the cell: dust starts to show in its light
BEATS = [(26.5, 0.8), (27.45, 0.48)]             # heart hits: the butterfly pulls hardest
# the butterfly tied to a stone (drawing px, from a reference): the knot on the stone, where the string ties to its
# body, its head, the box round it, the rope band round the stone, the foot the stone rocks on
BF_KNOT, BF_TIE, BF_HEAD = (580.0, 1120.0), (1198.0, 548.0), (1288.0, 415.0)
BF_BOX = (895, 240, 1345, 575)
STONE_C, STONE_FOOT = (415.0, 1370.0), (410.0, 1600.0)
BF_TUGS = [26.5, 27.45, 28.35, 29.3, 30.2]       # it pulls with everything, and the string yanks it back
BF_REST = (590.0, 1128.0)                        # at first it rests on the stone, by the knot
T_FLY = (25.5, 26.5)                             # it takes off, and on the hit the string snaps tight
# no cut: the wall of eyes is the face of the stone. one camera keeps pulling back, the stone's outline closes in,
# the skin and closed eyes shrink into its pencil texture: stone px per eye unit, and when the wall gives way
KS = 25.0
T_STONE = (23.6, 24.2, 25.1)                     # the stone starts to show, the skin starts to turn to pencil, gone
T_PETRIFY = (22.9, 24.9)                         # meanwhile the skin darkens to stone, in patches spreading over it


class Film:
    def __init__(self, handle="", lyrics=False, seed=5, voice=None):
        rng = np.random.default_rng(seed)
        self.handle, self.lyrics = handle, lyrics
        self.voice = voice
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
        self.field = np.asarray(Image.open(os.path.join(here, "assets", "shadow_field.png")).convert("RGB"), np.float32) / 255.0
        self.setup_field()
        self.skull, self.skull_pad = dome(np.asarray(Image.open(os.path.join(here, "assets", "skull_xray.png")).convert("L"), np.float32) / 255.0)
        self.money = np.asarray(Image.open(os.path.join(here, "assets", "money.png")).convert("RGBA"), np.float32) / 255.0
        self.prison = np.asarray(Image.open(os.path.join(here, "assets", "eye_bars.png")).convert("L"), np.float32) / 255.0
        self.build_code(rng)
        # the fire eats him from his eyes outward
        # "they fear death": the figure burns through to the skull, from its eyes outward
        ex, ey, _, _ = self.field_cam(7.4)
        self.T_burn = 7.35 + 1.05 * np.clip(np.hypot(FGX - ex, FGY - ey) / 1500, 0, 1) + 0.3 * (fbm(FGX / 150, FGY / 150, 3, 19) - 0.5)
        self.setup_tape()
        self.setup_embers(rng)
        self.setup_prison()
        self.setup_wall(rng)
        self.setup_stone()

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
        _, self.snap_hang = self.tape(T_SNAP)
        # falling after the money, the camera falls into the cell, aimed at its middle
        self.CXK = [(T_B, 0.33), (13.6, PRI_BAR)] + [k for k in CX if k[0] >= 14.4]
        self.impacts = [(T_SNAP, 6.0)]
        rng = np.random.default_rng(23)
        n = 140
        self.dust = dict(x=rng.uniform(-0.3, 0.95, n), y=rng.uniform(-0.6, 0.6, n), vx=rng.normal(0.01, 0.02, n),
                         vy=rng.uniform(-0.05, 0.02, n), ph=rng.uniform(0, 6.3, n), z=rng.random(n) ** 2)

    def setup_tape(self):
        # the tape hangs straight; once the camera has panned, the money hangs in the middle of the frame
        k = ckey(11.5, SKULL_K)
        off = (MONEY_C[0] - MONEY_KNOT[0], MONEY_C[1] - MONEY_KNOT[1])
        hxf = 540 - off[0] * K_M0 - (TAPE0[0] - SKULL_E[0]) * k
        self.HXS = [(7.0, 338), (7.8, 342), (8.8, 545), (10.6, hxf), (12.6, hxf)]
        self.PAN = [(0, 0)]
        y0 = ckey(11.5, SKULL_HY) + (TAPE0[1] - SKULL_E[1]) * k
        P = 960 - (y0 + TAPE_L[-1][1] + off[1] * K_M0)
        # ... and after the snap the camera keeps going down after the falling money
        self.PAN = [(0, 0), (9.35, 0), (9.9, 0.12 * P), (10.6, 0.58 * P), (11.3, 0.93 * P), (T_SNAP, P),
                    (12.25, P - 330), (T_B, P - 780), (13.1, P - 1420)]
        # the money only exists on the tape once its end is out of frame
        self.t_att = 12.0
        for tt in np.arange(9.2, 10.4, 0.01):
            _, hang = self.tape(tt)
            if hang is not None and hang[1] > 2010:
                self.t_att = tt; break

    def voice_at(self, t):
        """how loud the voice is at t, 0..1 (from the track; silent without one)"""
        if self.voice is None: return 0.0
        return float(np.interp(t * FPS, np.arange(len(self.voice)), self.voice))

    def fire_dist(self, SX, SY):
        """how far the fire has to go from his eyes to get here: it climbs faster than it falls"""
        d = [np.hypot(SX - fx, np.where(SY < fy, (SY - fy) * 0.65, SY - fy)) for fx, fy in FIRE_O]
        return np.minimum.reduce(d)

    def field_cam(self, t):
        return ckey(t, FIELD_EX), ckey(t, FIELD_EY), math.exp(ckey(t, FIELD_K)), ckey(t, FIELD_R)

    def field_to_screen(self, PX, PY, t):
        ex, ey, k, r = self.field_cam(t)
        qx, qy = (PX - FIELD_EM[0]) * k, (PY - FIELD_EM[1]) * k
        return ex + qx * math.cos(r) - qy * math.sin(r), ey + qx * math.sin(r) + qy * math.cos(r)

    def setup_field(self):
        """the painting without the figure (sky and wheat filled in behind it), so the figure can move over it"""
        img = self.field
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        dark = (img @ np.array([0.3, 0.59, 0.11], np.float32)) < 0.27
        where = ((yy > 330) & (yy < 840) & (xx > 60) & (xx < 1380)) | (np.hypot((xx - 700) / 330, (yy - 1150) / 560) < 1)
        m = Image.fromarray(((dark & where) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
        hole = np.asarray(m) > 127
        for fx, fy in FIELD_EYES:
            hole |= np.hypot(xx - fx, yy - fy) < 16
        plate = fill_rows(img, hole)                                   # the sky and the wheat carried in behind it
        self.field_plate = plate

    def field_scene(self, SX, SY, t):
        """the painting of the dark figure in the field: the wheat sways in gusts of wind like flames, the clouds
        drift, the figure sways from its feet and breathes, its head tilts, the smear round it flutters, it blinks"""
        ex, ey, k, r = self.field_cam(t)
        dx, dy = (SX - ex) / k, (SY - ey) / k
        c, s_ = math.cos(r), math.sin(r)
        PX = FIELD_EM[0] + dx * c + dy * s_
        PY = FIELD_EM[1] - dx * s_ + dy * c
        img = self.field
        tl = self.tv                                                    # it lives in real time, even in the pause
        tt = tl - 5.0
        # the figure: sways from its feet, breathes, its head tilts on its neck (all of it smoothly, around it)
        wfig = 1 - sstep(0.85, 1.3, np.hypot((PX - 700) / 560, (PY - 1050) / 700))
        ang = 0.035 * math.sin(tt * 1.3 + 0.5) + 0.018 * math.sin(tt * 0.55 + 2.0)
        qx, qy = PX - FIELD_FEET[0], PY - FIELD_FEET[1]
        bx = FIELD_FEET[0] + qx * math.cos(ang) + qy * math.sin(ang)
        by = FIELD_FEET[1] - qx * math.sin(ang) + qy * math.cos(ang)
        by = FIELD_FEET[1] + (by - FIELD_FEET[1]) * (1 - 0.018 * math.sin(tt * 2.1))          # breathing
        wh = np.clip((860 - by) / 160, 0, 1)
        hang = (0.08 * math.sin(tt * 0.9 + 1.2) + 0.03 * math.sin(tt * 2.3)) * wh
        hx, hy = bx - FIELD_NECK[0], by - FIELD_NECK[1]
        hx2 = FIELD_NECK[0] + hx * np.cos(hang) + hy * np.sin(hang)
        hy2 = FIELD_NECK[1] - hx * np.sin(hang) + hy * np.cos(hang)
        FXp = PX + (hx2 - PX) * wfig; FYp = PY + (hy2 - PY) * wfig
        # the smear round its head flutters in the wind
        flutter = 7 * np.sin(tt * 5 + FYp / 22 + FXp / 90) * np.clip((800 - FYp) / 200, 0, 1)
        fg = sample(img, FXp - flutter, FYp + 2.5 * np.sin(tt * 3 + FXp / 50) * np.clip((800 - FYp) / 200, 0, 1))
        hole0 = (fg @ np.array([0.3, 0.59, 0.11], np.float32))
        fig = np.clip((0.3 - hole0) / 0.2, 0, 1) * (wfig > 0.05)
        # the field: gusts roll across it, the near wheat bends the most; the clouds drift
        gust = 0.6 + 0.4 * math.sin(tt * 0.9)
        wind = gust * (20 * np.sin(tt * 2.2 - PX / 170 + PY / 240) + 9 * np.sin(tt * 3.7 - PX / 60 + PY / 90))
        wind = wind * np.clip((PY - 820) / 220, 0, 1) * (0.6 + 0.4 * np.clip((PY - 1000) / 900, 0, 1))
        drift = 16 * tt * np.clip((800 - PY) / 200, 0, 1)
        bg = sample(self.field_plate, PX - wind - drift, PY)
        rgb = bg * (1 - fig[..., None]) + fg * fig[..., None]
        # it blinks: its eyes go dark and light again
        bl = max(math.exp(-((tl - tb) / 0.1) ** 4) for tb in FIELD_BLINKS)         # shut a moment, then open
        de = np.minimum.reduce([np.hypot(FXp - fx, FYp - fy) for fx, fy in FIELD_EYES])
        if bl > 0.01:
            shut = np.clip((14 - de) / 6, 0, 1)[..., None] * bl
            rgb = rgb * (1 - shut) + 0.04 * shut
        PX, PY = FXp, FYp                                               # the eyes move with its head
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        lum = 0.3 * R + 0.59 * G + 0.11 * B
        warm = np.clip((R - B) / 0.4, 0, 1)                            # the wheat: it burns
        white = np.clip((rgb.min(-1) - 0.6) / 0.25, 0, 1) * (1 - warm)  # the clouds
        flick = 0.88 + 0.14 * np.sin(tt * 11 + PX / 40 - PY / 60)
        # the sky goes dark, the clouds pale, the wheat bright: its yellow strokes the brightest, like flames
        wheat = np.clip((G - 0.3) / 0.45, 0, 1) * flick * np.clip(lum / 0.35, 0, 1)
        sky = 0.12 + 0.75 * white
        b = np.clip(warm * (0.42 + 0.58 * wheat) + (1 - warm) * sky, 0, 1) * np.clip((lum - 0.06) / 0.12, 0, 1)
        col = warm[..., None] * (RED * (1 - wheat[..., None]) + HOT * wheat[..., None]) + \
              (1 - warm[..., None]) * (RED * 0.6 * (1 - white[..., None]) + PALE * 0.8 * white[..., None])
        gl = np.where((warm > 0.5) & (b > 0.06), STALK[np.clip((b * len(STALK)).astype(int), 0, len(STALK) - 1)], 0)
        eyes = np.exp(-de / 9.0) * (1.3 + 0.2 * math.sin(t * 2.2)) * (1 - bl)
        hot = eyes > 0.3
        b = np.maximum(b, np.clip(eyes, 0, 1)); col = np.where(hot[..., None], PALE, col); gl = np.where(hot, 0, gl)
        inside = (PX >= 0) & (PX < img.shape[1] - 1) & (PY >= 0) & (PY < img.shape[0] - 1)
        return np.where(inside, b, 0.0), col.astype(np.float32), np.where(inside, gl, 0)

    def setup_embers(self, rng):
        """sparks and ash from the burning figure: start points on it, where the camera has it when it burns"""
        lit = self.field.max(-1) < 0.25
        ys, xs = np.nonzero(lit)
        pick = rng.choice(len(xs), 1400, replace=False)
        sx, sy = self.field_to_screen(xs[pick].astype(float), ys[pick].astype(float), 7.5)
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
        hm = 1 - float(smoother(6.4, 7.3, t))                         # he goes still before he burns through
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
            # he speaks: his jaw swings down on its hinge with the voice, and his mouth opens along his own lip
            # line; inside it is black
            v = self.voice_at(t) * hm
            mouth = np.zeros(PX.shape, bool)
            if v > 0.02:
                lip = np.interp(PX, ULT_LIP[:, 0], ULT_LIP[:, 1])
                du, dw = PX - ULT_JAW[0], PY - ULT_JAW[1]
                u = du * ULT_MDIR[0] + dw * ULT_MDIR[1]; w = -du * ULT_MDIR[1] + dw * ULT_MDIR[0]
                e = np.hypot(u / ULT_JAW_R[0], w / ULT_JAW_R[1])
                th = 0.1 * v * (1 - sstep(0.5, 1.0, e)) * (PY > lip) * (PX > ULT_LIP[0, 0])    # fades into cheek and neck
                qx, qy = PX - ULT_HINGE[0], PY - ULT_HINGE[1]
                jx = ULT_HINGE[0] + qx * np.cos(th) - qy * np.sin(th)
                jy = ULT_HINGE[1] + qx * np.sin(th) + qy * np.cos(th)
                ljaw = np.interp(jx, ULT_LIP[:, 0], ULT_LIP[:, 1])
                mouth = (th > 0) & (jy < ljaw)                                   # between the upper lip and the jaw
                PX = np.where(mouth, PX, jx); PY = np.where(mouth, PY, jy)
            h, w = self.ultron.shape[:2]
            rgb = sample(self.ultron, np.clip(PX, 0, w - 1.01), np.clip(PY, 0, h - 1.01))
            fld = sample(self.ult_bg, np.clip(PX, 0, w - 1.01), np.clip(PY, 0, h - 1.01))
            near = sample(self.ult_near, np.clip(PX, 0, w - 1.01), np.clip(PY, 0, h - 1.01))
            fb = np.clip((rgb[..., 0] - 0.05) / 0.8, 0, 1) ** 0.85                  # him: shaded, darks pushed down
            # he is written in the code too, brighter than the field; the field fades out softly around him
            fb = np.clip(fb * 1.3, 0, 1) ** 0.9
            ub = fld * 0.4 * np.clip((0.5 - near) / 0.38, 0, 1) ** 1.2 + (1 - fld) * fb
            white = rgb.min(-1) > 0.55
            ub = np.where(white, 1.0, ub)
            ucol = np.where(white[..., None], PALE, np.where((fld > 0.5)[..., None], RED, redmap(np.minimum(fb * 1.15, 1))))
            # his eyes burn brighter and brighter, then flare: the fire starts there
            flare = float(smoother(4.55, 5.0, t)) * math.exp(-max(t - 5.0, 0.0) / 0.35)
            de = np.minimum.reduce([np.hypot(PX - ex, PY - ey) for ex, ey in ULT_EYES])
            # he blinks, once: a lid comes down over the light and lifts again
            bl = math.exp(-((t - 3.7) / 0.07) ** 2)
            if bl > 0.01:
                lid = np.zeros(PX.shape, bool)
                for ex, ey in ULT_EYES:
                    lid |= (np.hypot((PX - ex) / 1.4, PY - ey) < 10) & (PY < ey - 10 + 20 * bl)
                ub = np.where(lid, 0.05, ub); white = white & ~lid
                ucol = np.where(lid[..., None], INK, ucol)
            core = np.exp(-de / (6.0 + 2.0 * flare)) * (1.1 + 0.2 * math.sin(t * 2.2) + 0.8 * flare) * (1 - bl)
            halo = flare * np.exp(-de / 9.0)
            streak = flare * np.maximum.reduce([np.exp(-np.abs(PY - ey) / 2.0 - np.abs(PX - ex) / (20 + 40 * flare))
                                                for ex, ey in ULT_EYES])          # a flare of light across each eye
            ub = np.maximum(ub, np.clip(np.maximum(core, streak), 0, 1))
            ub = np.where(halo > 0.12, np.maximum(ub, 0.25 + 0.6 * halo), ub)
            ucol = np.where((np.maximum(core, streak) > 0.25)[..., None], PALE,
                            np.where((halo > 0.12)[..., None], RED * 0.5 + HOT * 0.5, ucol))
            hot = np.maximum.reduce([core, halo, streak])
            fld = np.where(mouth, 0.0, fld)
            # typed in with the code: a cell shows him once its line of code has reached it
            ci = (np.arange(FCOLS)[None, :] + np.floor(self.code_v * t).astype(int)[:, None]) % FCOLS
            seen = t >= np.take_along_axis(self.code_tf, ci, 1)
            code = np.take_along_axis(self.code_idx, ci, 1)
            on = seen & ~burnt
            fon = on & (fld > 0.5)
            gap = fon & (code == 0)                                                   # spaces in the code stay spaces
            b = np.where(on & ~gap, ub, b); col = np.where(on[..., None], ucol, col)
            # every cell of him is a letter of the scrolling code; only his darkest shadows and the gaps
            # between words fall back to plain shading
            # on him the code runs: its lines stream past four times faster than the field's, pulses of light
            # run along them like data, and letters keep flipping as it rewrites itself
            cj = (np.arange(FCOLS)[None, :] + np.floor(self.code_v * t * 4 + 7).astype(int)[:, None]) % FCOLS
            run = np.take_along_axis(self.code_idx, cj, 1)
            fill = np.where(run > 0, run, np.roll(run, 5, axis=0))                 # his gaps take the next line's letters
            fill = np.where(fill > 0, fill, np.roll(run, 11, axis=1))
            rng = np.random.default_rng(fi + 17)
            fill = np.where(rng.random((FROWS, FCOLS)) < 0.06, rng.choice(PRINTABLE, (FROWS, FCOLS)), fill)
            fig = on & ~fon & (fill > 0) & (hot < 0.25) & (fb > 0.12)
            rr = np.arange(FROWS)[:, None]; cc = np.arange(FCOLS)[None, :]
            pulse = np.maximum(0, np.sin(cc * 0.11 * np.sign(self.code_v)[:, None] - t * 9 + rr * 1.7)) ** 10
            b = np.where(fig, np.clip(b * (1 + 0.9 * pulse), 0, 1), b)
            col = np.where((fig & (pulse > 0.5))[..., None], col * 0.4 + HOT * 0.6, col)
            gl = np.where(fon & (code > 0) & (hot < 0.25), code, gl)
            gl = np.where(fig, fill, gl)
            solid |= on & ~fon
            dark = on & mouth                                                         # the open mouth: black
            b = np.where(dark, 0.0, b); gl = np.where(dark, 0, gl); solid |= dark

        # the x-ray, revealed where the fire has burnt through; the reels spin, a scan line runs down it
        if t > 7.1:
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

        # chaos: it catches in his eyes and a burning edge spreads out over him from there, slowly at first; the
        # heat runs ahead of it; behind the edge is the figure in the burning field, its eyes where his were
        if T_SPREAD - 0.1 < t < 9.0:
            dF = self.fire_dist(SX, SY)
            nz = fbm(SX / 220, SY / 220 + t * 0.8, 2, 3) - 0.5
            nz2 = fbm(SX / 70, SY / 70 + t * 2.4, 2, 5) - 0.5                  # a ragged, licking edge
            R = 2300 * float(smoother(T_SPREAD, T_SPREAD + 2.3, t)) ** 1.5 + 45 * float(sstep(T_SPREAD - 0.1, T_SPREAD + 0.25, t))
            R = R + (60 + 0.15 * R) * nz + (30 + 0.1 * R) * nz2
            heat = np.clip(1 - (dF - R) / 280, 0, 1) * (dF > R) * float(t < 7.8)
            hb = (b > 0.04) & (heat > 0)
            b = np.where(hb, np.clip(b * (1 + 0.6 * heat), 0, 1), b)
            col = np.where(hb[..., None], col * (1 - 0.45 * heat[..., None]) + HOT * 0.45 * heat[..., None], col)
            a = np.clip((R - dF) / 150, 0, 1)
            live = (a > 0.02) & (t < self.T_burn + 0.4)
            if live.any():
                fb, fc, fg = self.field_scene(SX, SY, t)
                fade = 1 - sstep(self.T_burn - 0.05, self.T_burn + 0.4, t)     # then it burns through to the skull
                mix = np.where(live, a * fade, 0.0)
                b = b * (1 - mix) + fb * mix
                col = col * (1 - mix[..., None]) + fc * mix[..., None]
                gl = np.where(mix > 0.5, fg, gl)
                solid |= mix > 0.5                                         # the figure is a black silhouette
            # the burning edge: sparks of code
            edge = np.clip(1 - np.abs(R - dF) / 40, 0, 1) * (R > 30) * (t < 7.8) * np.clip(0.55 + 2.2 * nz2, 0, 1)
            eg = edge > 0.3
            if eg.any():
                rng = np.random.default_rng(fi + 13)
                gl = np.where(eg, rng.choice(PRINTABLE, (FROWS, FCOLS)), gl)
                b = np.where(eg, np.maximum(b, 0.55 + 0.45 * edge), b)
                col = np.where(eg[..., None], np.where((edge > 0.75)[..., None], PALE, HOT), col); solid |= eg
            ember = (t > self.T_burn - 0.05) & (t < self.T_burn + 0.02)          # sparks where it burns through
            if ember.any():
                rng = np.random.default_rng(fi + 3)
                ember &= rng.random((FROWS, FCOLS)) < 0.3
                gl = np.where(ember, rng.choice(PRINTABLE, (FROWS, FCOLS)), gl)
                b = np.where(ember, 1.0, b)
                col = np.where(ember[..., None], np.where((self.rand > 0.15)[..., None], HOT, PALE), col)

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
        self.eye_close[0] = 23.85                                         # the hero eye last
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
        """the camera falls into the cell out of the dark, lands on the bars, then pulls back to the eye"""
        shape = X.shape
        b, inside = self.prison_photo(X, Y, t)
        b = b * float(smoother(T_B, T_B + 0.55, t))                  # it comes up out of the dark
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
        # dust drifting through the light, some of it far nearer the camera than the cell: falling in,
        # the near dust rushes past much faster than the cell grows
        if cw > 0 and t > T_REVEAL:
            d = self.dust
            age = t - T_REVEAL
            px_ = (d["x"] + d["vx"] * age + 0.01 * np.sin(age * 1.7 + d["ph"]) - (-0.3)) % 1.25 + (-0.3)
            py_ = (d["y"] + d["vy"] * age + 0.6) % 1.2 - 0.6
            cx, cy, Zc, rl = self.cam_b(t)
            near = 1 + 2.2 * d["z"]
            SXd, SYd = self.from_eye(cx + (px_ - cx) * near, cy + (py_ - cy) * near, (cx, cy, Zc, rl))
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
        if 14.4 <= t < T_STONE[2]:
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

        if t < T_B + 0.4:                                         # the money falls on out of frame
            self.money_layer(b, col, gl, solid, t, SX, SY, self.snap_hang)

        # as the eyes close the skin turns to stone: dark patches spread over it (fixed on the surface) until it's as
        # dark as the rock, so there is no jump from red to black
        if T_PETRIFY[0] < t < T_STONE[2]:
            nz = fbm(X * 0.22 + 3.0, Y * 0.22, 3, 41)
            pet = np.clip((t - T_PETRIFY[0]) / (T_PETRIFY[1] - T_PETRIFY[0]) * 1.3 + 1.1 * (nz - 0.5) - 0.15, 0, 1)
            pet = pet * pet * (3 - 2 * pet)
            b = b * (1 - 0.72 * pet); col = col * (1 - 0.3 * pet[..., None])

        # the wall of eyes was the face of the stone all along: as the camera pulls back its outline closes in from
        # the edges of the frame, and the skin and the closed eyes on it turn into its pencil texture
        if t > T_STONE[0]:
            rb, rc, rg, rsolid = self.stone_scene(SX, SY, t)
            if t < T_STONE[2]:
                sx, sy, sz, _ = self.stone_cam(t)
                face = sample(self.stone_mask, sx + (SX - 540) / sz, sy + (SY - 960) / sz)
                w = face * (1 - float(smoother(T_STONE[1], T_STONE[2], t)))
                b = rb * (1 - w) + b * w; col = rc * (1 - w[..., None]) + col * w[..., None]
                gl = np.where(w > 0.5, gl, rg); solid = np.where(w > 0.5, solid, rsolid)
            else:
                b, col, gl, solid = rb, rc, rg, rsolid
        return b, col, gl, solid

    # ---------- the ending: a butterfly (from a reference drawing) tied to a stone; it tries to fly, it can't
    def setup_stone(self):
        here = os.path.dirname(os.path.abspath(__file__))
        img = np.asarray(Image.open(os.path.join(here, "assets", "butterfly_stone.png")).convert("RGB"), np.float32) / 255.0
        self.stone_img = img
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        lum = img.mean(-1)
        x0, y0, x1, y1 = BF_BOX
        inbox = (xx > x0) & (xx < x1) & (yy > y0) & (yy < y1)
        a = np.clip((0.72 - lum) / 0.12, 0, 1) * inbox
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
                       .filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255.0
        self.bf_alpha = a
        # the plate: the butterfly and its string painted out, so both can move
        kx, ky = BF_TIE[0] - BF_KNOT[0], BF_TIE[1] - BF_KNOT[1]
        u = np.clip(((xx - BF_KNOT[0]) * kx + (yy - BF_KNOT[1]) * ky) / (kx * kx + ky * ky), 0, 1)
        dline = np.hypot(xx - BF_KNOT[0] - u * kx, yy - BF_KNOT[1] - u * ky)
        lum_b = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(18)), np.float32) / 255.0
        hole = (a > 0.05) | ((dline < 7) & (u > 0.02)) | ((dline < 30) & (u > 0.02) & (u < 1.02) & (lum < lum_b - 0.04))
        hole = np.asarray(Image.fromarray((hole * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 127
        self.stone_plate = fill_rows(img, hole, blur=5, seed=9)
        # the stone's face: where the wall of eyes shows until it turns to pencil
        dark = (lum < 0.6) & (yy > 1110) & (yy < 1610) & (xx > 40) & (xx < 800)
        m = Image.fromarray((dark * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(21)).filter(ImageFilter.MinFilter(21))
        self.stone_mask = np.asarray(m.filter(ImageFilter.GaussianBlur(6)), np.float32) / 255.0

    def stone_cam(self, t):
        """the same camera as the wall of eyes, in the drawing's px: the hero eye sits at the stone's centre"""
        cx, cy, Z, rl = self.cam_b(t)
        return STONE_C[0] + cx * KS, STONE_C[1] - cy * KS, EU * Z / KS, 0.0

    def butterfly_pose(self, t):
        """the butterfly's tie point, its tilt, how far its wings are open, the stone's rock, and the string's slack"""
        kx, ky = BF_TIE[0] - BF_KNOT[0], BF_TIE[1] - BF_KNOT[1]
        L0, th0 = math.hypot(kx, ky), math.atan2(ky, kx)
        tug = max(math.exp(-((t - tb) / 0.16) ** 2) for tb in BF_TUGS)
        yank = sum(math.exp(-(t - tb) / 0.12) * math.sin((t - tb) * 30) for tb in BF_TUGS if t > tb)
        tired = float(smoother(28.7, 29.9, t))                                  # in the end it nearly gives up
        slack = (0.045 + 0.03 * math.sin(t * 2.3)) * (1 - tug) * (1 - 0.5 * tired) + 0.02 * tired
        r = L0 * (1 - slack + 0.012 * tug)
        th = th0 + 0.05 * math.sin(t * 1.3) + 0.025 * math.sin(t * 3.4) - 0.035 * tug + 0.01 * yank + 0.06 * tired
        px, py = BF_KNOT[0] + r * math.cos(th), BF_KNOT[1] + r * math.sin(th)
        rot = 0.12 * math.sin(t * 1.7) - 0.15 * tug + 0.05 * yank
        speed = 7.0 * (1 + 0.6 * tug) * (1 - 0.35 * tired)
        spread = 0.18 + 0.82 * abs(math.cos(math.pi * speed * t))
        rock = 0.012 * tug + 0.004 * yank
        sag = slack * L0
        # before that: it rests on the stone, slowly opening and closing its wings, then takes off and flies up
        # until the string runs out
        p = float(smoother(T_FLY[0], T_FLY[1], t))
        if p < 1:
            q = p * p * (3 - 2 * p)
            px = BF_REST[0] + (px - BF_REST[0]) * q + 20 * math.sin(t * 6) * 4 * p * (1 - p)
            py = BF_REST[1] + (py - BF_REST[1]) * q + 30 * math.sin(t * 11) * 4 * p * (1 - p)
            rot = rot * q - 0.15 * (1 - q)
            rest_sp = 0.55 + 0.45 * math.cos(t * 1.6)
            k8 = min(1.0, p * 8)
            spread = (0.18 + 0.82 * abs(math.cos(math.pi * 9.0 * t))) * k8 + rest_sp * (1 - k8)
            sag = min(L0 - math.hypot(px - BF_KNOT[0], py - BF_KNOT[1]), 260) * 0.6
        return (px, py), rot, spread, rock, sag

    def stone_scene(self, SX, SY, t):
        shape = SX.shape
        cx, cy, z, rl = self.stone_cam(t)
        c, s = math.cos(rl), math.sin(rl)
        dx, dy = SX - 540, SY - 960
        AX = cx + (dx * c + dy * s) / z
        AY = cy + (-dx * s + dy * c) / z
        (bx, by), rot, spread, rock, sag = self.butterfly_pose(t)
        # the stone rocks on its foot when the string yanks
        ws = 1 - sstep(0.9, 1.3, np.hypot((AX - STONE_C[0]) / 430, (AY - STONE_C[1]) / 320))
        qx, qy = AX - STONE_FOOT[0], AY - STONE_FOOT[1]
        RX = AX + ws * (qx * math.cos(rock) + qy * math.sin(rock) - qx)
        RY = AY + ws * (-qx * math.sin(rock) + qy * math.cos(rock) - qy)
        h, w = self.stone_img.shape[:2]
        def refl(v, n):
            v = np.abs(v); return np.where(v > n - 2, 2 * (n - 2) - v, v)
        rgb = sample(self.stone_plate, refl(RX, w), refl(RY, h))           # the paper runs on past its edges
        # the butterfly: moved and tilted to its pose, its wings folding about its body as it beats them
        ux, uy = AX - bx, AY - by
        cr, sr = math.cos(rot), math.sin(rot)
        LX = BF_TIE[0] + ux * cr + uy * sr
        LY = BF_TIE[1] - ux * sr + uy * cr
        ax_, ay_ = BF_HEAD[0] - BF_TIE[0], BF_HEAD[1] - BF_TIE[1]
        al = math.hypot(ax_, ay_); nx, ny = ay_ / al, -ax_ / al          # across the body, toward the wings
        dperp = (LX - BF_TIE[0]) * nx + (LY - BF_TIE[1]) * ny
        wing = dperp > 0
        LX = np.where(wing, LX - nx * dperp * (1 - 1 / spread), LX)
        LY = np.where(wing, LY - ny * dperp * (1 - 1 / spread), LY)
        ba = sample(self.bf_alpha, LX, LY)[..., None]
        rgb = rgb * (1 - ba) + sample(self.stone_img, LX, LY) * ba
        lum = rgb.mean(-1)
        b = np.clip((lum - 0.3) / 0.5, 0, 1) ** 1.7 * 0.9                # the paper red, the pencil dark
        # no page: the paper falls off into the dark around the stone and the butterfly, lit from above right
        b = b * (0.85 + 0.25 * np.clip((1000 - AY) / 700 + (AX - 600) / 1400, 0, 1))
        col = redmap(b)
        gl = np.zeros(shape, np.int64)
        # the drawing is solid where it's lit: the stone, the butterfly and the pencil are dark shapes in it, not
        # holes the code shows through
        solid = np.ones(shape, bool)
        b = np.where(solid, b, 0.0)
        # the string, from the knot on the rocking stone up to its body: taut when it pulls, sagging when it rests
        kx_, ky_ = BF_KNOT[0] - STONE_FOOT[0], BF_KNOT[1] - STONE_FOOT[1]
        kn = (STONE_FOOT[0] + kx_ * math.cos(rock) - ky_ * math.sin(rock), STONE_FOOT[1] + kx_ * math.sin(rock) + ky_ * math.cos(rock))
        u = np.linspace(0, 1, 400)
        PXs = kn[0] + (bx - kn[0]) * u
        PYs = kn[1] + (by - kn[1]) * u + 2.2 * sag * u * (1 - u)
        qx, qy = PXs - cx, PYs - cy
        Xs = 540 + z * (c * qx - s * qy); Ys = 960 + z * (s * qx + c * qy)
        if t > T_FLY[0]:                                          # resting, it lies coiled at the knot
            splat(gl, col, solid, Xs, Ys, np.zeros(3, np.float32), b, 0.0)        # a black line on the red paper
        return b, col.astype(np.float32), gl, solid

    # ---------- compose
    def lyric_overlay(self, img, t):
        if not self.lyrics and not self.handle:
            return img
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        items = LYRICS if self.lyrics else []
        items = items + [(29.6, DUR + 1, ">>> " + self.handle)] if self.handle else items
        for k, (t0, t1, txt) in enumerate(items):
            if not (t0 <= t < t1 + 0.15): continue
            f = mono(38)
            s, typing = typed(txt, t, t0, 26)
            fade = 1 - float(sstep(t1, t1 + 0.15, t))
            shown = s + ("█" if typing or (txt.startswith(">>>") and (t * 2.4) % 2 < 1) else "")
            w = f.getlength(txt)
            y = 1180                                                         # well clear of the app's caption and buttons
            c = (255, 205, 195) if not txt.startswith(">>>") else (255, 95, 75)
            dr.text(((W - w) / 2, y), shown, font=f, fill=tuple(int(v * fade) for v in c),
                    stroke_width=4, stroke_fill=(0, 0, 0))
        return np.asarray(im)

    def frame(self, tv, fi, scale=1):
        """the frame at video time tv: the story runs at story(tv), the code behind it in real time"""
        self.tv = tv
        t = story(tv)
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
        cidx, ccol = self.code_layer(tv, fi)
        pulse = sum(math.exp(-(t - tb) / 0.2) for tb, _ in BEATS if t >= tb)     # the heartbeat lights the room
        ccol = ccol * (1 + 2.5 * pulse)
        # brighter: the picture lifts a lot, the code behind it a little, so it stays behind
        col = np.where(empty[..., None], brighten(ccol, 1.6, 1.0), brighten(col, 1.35, 0.8))
        idx = np.where(empty, cidx, idx)
        if scale > 1:
            return glow(render_scaled(idx, col, scale))
        img = glow(render_fine(idx, col))
        return self.lyric_overlay(img, t)


def voice_envelope(music):
    """how loud the voice is per video frame, 0..1, for ultron's jaw: the voice band only, quick to open,
    slower to close, so it follows the syllables"""
    if not music: return None
    sr = 16000
    raw = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", music, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    n = int(len(x) / sr * FPS)
    hop = sr / FPS
    win = int(hop) * 2
    f = np.fft.rfftfreq(win, 1 / sr)
    band = (f > 250) & (f < 3500)
    e = np.zeros(n)
    for i in range(n):
        seg = x[int(i * hop):int(i * hop) + win]
        if len(seg) < win: seg = np.pad(seg, (0, win - len(seg)))
        e[i] = np.sqrt((np.abs(np.fft.rfft(seg * np.hanning(win)))[band] ** 2).mean())
    lo, hi = np.percentile(e, 25), np.percentile(e, 96)
    env = np.clip((e - lo) / (hi - lo + 1e-9), 0, 1) ** 0.8
    out = np.zeros_like(env); a = 0.0
    for i, v in enumerate(env):
        a = v if v > a else a + (v - a) * 0.5
        out[i] = a
    return out

# ------------------------------------------------------------------ sound design: all synthesised here, no samples
def _band(n, lo, hi, rng, sr):
    """n samples of noise between lo and hi Hz"""
    f = np.fft.rfftfreq(n, 1 / sr)
    spec = np.fft.rfft(rng.standard_normal(n)) * ((f > lo) & (f < hi))
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-9)

def sound_design(sr=44100, seed=4):
    """the background sounds, in video time: a low drone, keys typing, the fire and the wind in the field, the tape
    whirring, the string snapping, the fall, the bars clanking, the stone forming, wings beating, the rope going taut"""
    rng = np.random.default_rng(seed)
    n = int(DUR_V * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    def put(start, x):
        i = int(start * sr)
        if i >= n: return
        x = x[:n - i]; out[i:i + len(x)] += x
    def env(n_, a, d):
        e = np.ones(n_); ka, kd = int(a * sr), int(d * sr)
        if ka: e[:ka] = np.linspace(0, 1, ka)
        if kd: e[-kd:] *= np.linspace(1, 0, kd)
        return e
    V = video
    # a low drone under it all
    out += 0.05 * (np.sin(2 * np.pi * 42 * t) + 0.5 * np.sin(2 * np.pi * 84.4 * t)) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.13 * t)) \
        * np.clip(t / 2.5, 0, 1) * np.clip((DUR_V - t) / 0.5, 0, 1)
    # keys typing while the code fills the screen
    for tk in np.sort(rng.uniform(0.05, 2.6, 55)):
        k = int(0.006 * sr); put(tk, 0.06 * rng.uniform(0.5, 1) * _band(k, 2000, 6000, rng, sr) * np.exp(-np.arange(k) / (0.0015 * sr)))
    # his eyes flare: a rising hiss
    a, b = V(4.5), V(5.1); k = int((b - a) * sr)
    put(a, 0.12 * _band(k, 600, 4000, rng, sr) * np.linspace(0, 1, k) ** 2 * env(k, 0, 0.05))
    # the field: the fire's roar and crackle, the wind in the wheat
    a, b = V(5.05), V(8.6); k = int((b - a) * sr); tt = np.arange(k) / sr
    put(a, 0.1 * _band(k, 30, 380, rng, sr) * env(k, 0.6, 0.8) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.7 * tt)))
    for tk in np.sort(rng.uniform(a + 0.2, b - 0.3, int((b - a) * 28))):
        kk = int(rng.uniform(0.004, 0.02) * sr)
        put(tk, rng.uniform(0.03, 0.16) * _band(kk, 900, 6000, rng, sr) * np.exp(-np.arange(kk) / (0.003 * sr)))
    a, b = V(5.2), V(8.3); k = int((b - a) * sr); tt = np.arange(k) / sr
    put(a, 0.07 * _band(k, 200, 1300, rng, sr) * env(k, 0.8, 0.8) * (0.45 + 0.55 * np.sin(2 * np.pi * 0.35 * tt + 1) ** 2))
    # it burns through to the skull: a whoosh
    k = int(0.7 * sr); put(V(7.3), 0.16 * _band(k, 150, 2500, rng, sr) * np.sin(np.linspace(0, np.pi, k)) ** 2)
    # the tape running out: a whirr that speeds up and slows
    a, b = V(9.2), V(10.6); k = int((b - a) * sr); tt = np.arange(k) / sr
    fr = 90 + 170 * np.sin(np.pi * np.clip(tt / (b - a), 0, 1)) ** 1.5
    saw = 2 * ((np.cumsum(fr) / sr) % 1) - 1
    put(a, (0.035 * saw + 0.03 * _band(k, 2000, 7000, rng, sr)) * env(k, 0.1, 0.4))
    # the string snaps
    k = int(0.004 * sr); put(V(T_SNAP), 0.5 * _band(k, 500, 9000, rng, sr))
    k = int(0.5 * sr); tt = np.arange(k) / sr
    put(V(T_SNAP), 0.22 * np.sin(2 * np.pi * (220 * tt + 3 * np.sin(2 * np.pi * 9 * tt))) * np.exp(-tt / 0.12))
    # the fall
    a, b = V(11.9), V(13.0); k = int((b - a) * sr)
    put(a, 0.1 * _band(k, 300, 3000, rng, sr) * np.sin(np.linspace(0, np.pi, k)) ** 1.5)
    # the bars clank on the hit, and rattle
    for tc, amp in ((14.0, 0.2), (14.22, 0.09), (14.41, 0.05)):
        k = int(1.2 * sr); tt = np.arange(k) / sr
        x = sum(np.sin(2 * np.pi * f * tt + rng.uniform(0, 6)) * np.exp(-tt / d) / (1 + i)
                for i, (f, d) in enumerate([(523, 0.9), (1187, 0.6), (1873, 0.45), (2690, 0.3), (3611, 0.2)]))
        put(V(tc), amp * x / 2)
    # the skin turns to stone: a low grinding rumble
    a, b = V(23.0), V(25.3); k = int((b - a) * sr)
    put(a, 0.13 * _band(k, 25, 140, rng, sr) * np.sin(np.linspace(0, np.pi, k)) ** 1.2)
    # the butterfly's wings: soft flutter, faster when it pulls; the rope thrums when it goes taut
    a = V(T_FLY[0]); k = n - int(a * sr); tt = np.arange(k) / sr + a
    tug = np.max([np.exp(-((tt - V(tb)) / 0.2) ** 2) for tb in BF_TUGS], 0)
    rate = 9 * (1 + 0.5 * tug)
    flap = np.abs(np.sin(np.pi * np.cumsum(rate) / sr)) ** 4
    put(a, 0.07 * (0.6 + 0.8 * tug) * flap * _band(k, 150, 1100, rng, sr) * env(k, 0.3, 0.0))
    for tb in BF_TUGS:
        k = int(0.7 * sr); tt = np.arange(k) / sr
        x = 0.2 * np.sin(2 * np.pi * 68 * tt) * np.exp(-tt / 0.25) + 0.05 * _band(k, 400, 1800, rng, sr) * np.exp(-tt / 0.08)
        put(V(tb) - 0.03, x)
    return (out / max(1.0, np.abs(out).max() / 0.6)).astype(np.float32)

def write_wav(path, x, sr=44100):
    x = x if x.ndim == 2 else x[:, None]
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())

def stretch(x, dur, sr, win=0.3, seed=2):
    """the sound of x held for dur seconds: overlapping windows read slowly across it with their phases scattered
    (a paul stretch), so it becomes one continuous sustained sound of the same music, no loop and no jump"""
    n, w = int(dur * sr), int(win * sr) // 2 * 2
    hop = w // 4
    win_f = np.hanning(w)
    out = np.zeros((n + w, x.shape[1]))
    rng = np.random.default_rng(seed)
    for k, o in enumerate(range(0, n, hop)):
        i = int((len(x) - w) * o / max(n, 1))
        ph = np.exp(2j * np.pi * rng.random(w // 2 + 1))
        for c in range(x.shape[1]):
            spec = np.fft.rfft(x[i:i + w, c] * win_f)
            out[o:o + w, c] += np.fft.irfft(np.abs(spec) * ph, w) * win_f
    out = out[:n]
    ref = np.sqrt((x ** 2).mean()); got = np.sqrt((out ** 2).mean()) + 1e-9
    return out * ref / got

def paused_track(music, sr=44100):
    """the spoken track with PAUSE seconds put in at PAUSE_AT, held on the music sounding right there"""
    raw = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", music, "-vn", "-ac", "2", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32).reshape(-1, 2) / 32768
    a, D, f = int(PAUSE_AT * sr), int(PAUSE * sr), int(0.15 * sr)
    hold = stretch(x[int((PAUSE_AT - 0.05) * sr):int((PAUSE_AT + 0.6) * sr)], PAUSE + 0.15, sr)
    out = np.zeros((len(x) + D, 2))
    up = np.sin(np.linspace(0, np.pi / 2, f))[:, None]; down = np.cos(np.linspace(0, np.pi / 2, f))[:, None]
    head = x[:a + f].copy(); head[a:] *= down
    out[:a + f] += head
    hold[:f] *= up; hold[-f:] *= down
    out[a:a + len(hold)] += hold
    tail = x[a:].copy(); tail[:f] *= up
    out[a + D:a + D + len(tail)] += tail
    return out.astype(np.float32)

def export(path, music, handle, lyrics, sfx=True):
    film = Film(handle, lyrics, voice=voice_envelope(music))
    ff = ffmpeg_bin()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    tmp = tmpv = None
    if music:
        # the track with the pause put in (held on its own music), the sound design under it
        tmpv = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        write_wav(tmpv, paused_track(music))
        cmd += ["-i", tmpv]
        fc = "[1:a]anull[voice]"
        if sfx:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            write_wav(tmp, sound_design())
            cmd += ["-i", tmp]
            fc += ";[2:a]volume=0.55[fx];[voice][fx]amix=inputs=2:normalize=0:duration=first[mix]"
        else:
            fc += ";[voice]anull[mix]"
        fc += f";[mix]afade=t=out:st={DUR_V - 0.6}:d=0.6[aout]"
        cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-t", str(DUR_V)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "medium", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(DUR_V * FPS)
    for fi in range(n):
        p.stdin.write(film.frame(fi / FPS, fi).tobytes())
        if fi % 15 == 0 or fi == n - 1:
            print(f"\rrendering {fi + 1}/{n}", end="", flush=True)
    p.stdin.close(); p.wait()
    for f_ in (tmp, tmpv if music else None):
        if f_: os.unlink(f_)
    print(f"\nsaved {path}")

# ------------------------------------------------------------------ the carousel: a close-up of the code in each scene
# name, start (video time), centre of the close-up on the 1080x1920 frame, how much of it (1 = a 540x675 window)
CLIPS = [("1_ultron", 2.2, (430, 520, 1.55)), ("2_field", 7.2, (500, 560, 1.0)), ("3_skull", 10.0, (520, 560, 1.15)),
         ("4_money", 12.9, (310, 1380, 1.0)), ("5_cell", 15.6, (760, 1080, 1.35)), ("6_eye", 17.6, (560, 1000, 1.25)),
         ("7_wall", 19.6, (760, 800, 1.0)), ("8_stone", 28.4, (700, 700, 1.25))]
CLIP_LEN = 2.0

def export_carousel(outdir, music, sfx=True):
    """each scene, 2 s, as a 4:5 close-up at full quality: the frame rendered with characters twice the size, so
    the close-up is the code itself, crisp, not a picture blown up; with the sound under it"""
    os.makedirs(outdir, exist_ok=True)
    film = Film(voice=voice_envelope(music))
    ff = ffmpeg_bin()
    wav = None
    if music:
        mix = paused_track(music)
        if sfx:
            fx = sound_design(); n = min(len(mix), len(fx))
            mix = mix[:n] + 0.55 * fx[:n, None]
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        write_wav(wav, mix)
    for name, t0, (cx, cy, k) in CLIPS:
        path = os.path.join(outdir, f"curse_{name}.mp4")
        w, h = int(1080 * k), int(1350 * k)
        x0 = min(max(int(2 * cx - w / 2), 0), 2 * W - w); y0 = min(max(int(2 * cy - h / 2), 0), 2 * H - h)
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1350", "-r", str(FPS), "-i", "-"]
        if wav:
            cmd += ["-ss", str(t0), "-t", str(CLIP_LEN), "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-af", f"afade=t=in:d=0.08,afade=t=out:st={CLIP_LEN - 0.12}:d=0.12", "-c:a", "aac", "-b:a", "256k"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-preset", "slow", "-shortest", path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for j in range(int(CLIP_LEN * FPS)):
            tv = t0 + j / FPS
            img = Image.fromarray(film.frame(tv, int(round(tv * FPS)), scale=2)).crop((x0, y0, x0 + w, y0 + h))
            if k != 1:
                img = img.resize((1080, 1350), Image.LANCZOS)
            p.stdin.write(np.asarray(img).tobytes())
        p.stdin.close(); p.wait()
        print("saved", path)
    if wav: os.unlink(wav)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="curse_of_being_human.mp4")
    ap.add_argument("--music", default=None, help="the spoken-word track (an audio file or a video with it)")
    ap.add_argument("--handle", default="", help="a handle typed at the very end (none by default)")
    ap.add_argument("--lyrics", action="store_true", help="type the lyrics out as captions (off by default)")
    ap.add_argument("--no-sfx", action="store_true", help="the spoken track only, without the sound design")
    ap.add_argument("--carousel", default=None, help="a folder: write a 2 s close-up of the code in each scene there")
    a = ap.parse_args()
    if a.carousel:
        export_carousel(a.carousel, a.music, not a.no_sfx)
    else:
        export(a.export, a.music, a.handle, a.lyrics, not a.no_sfx)
