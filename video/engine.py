"""Khan-Academy-style chalkboard animation engine.

Dark board, hand-drawn wobbly strokes that draw themselves on over time,
handwritten text write-ins, math formulas revealed left-to-right, arrows,
highlights.  A Scene is a list of timed Items; frames are rendered with
matplotlib and piped as raw RGBA to ffmpeg.

Canvas coordinate system: x in [0, 16], y in [0, 9]  (16:9 board).
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Bbox

# ----------------------------------------------------------------------------
# fonts & palette
# ----------------------------------------------------------------------------
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
for _f in ("PatrickHand.ttf", "Kalam-Regular.ttf", "Kalam-Bold.ttf", "Gloria.ttf"):
    _p = os.path.join(FONT_DIR, _f)
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)

HAND = "Patrick Hand"
HAND2 = "Kalam"

BG = "#0d1117"          # near-black board with a whisper of blue
INK = "#e8e8e2"          # chalk white
C = {
    "white":  "#e8e8e2",
    "yellow": "#ffd166",
    "blue":   "#5bc8f5",
    "green":  "#9ee493",
    "pink":   "#ff7eb6",
    "orange": "#ffab5e",
    "purple": "#c3a6ff",
    "red":    "#ff6b6b",
    "teal":   "#63e6be",
    "grey":   "#8b949e",
    "dim":    "#4d5566",
}

plt.rcParams.update({
    "font.family": HAND,
    "mathtext.fontset": "cm",
    "text.color": INK,
})


# ----------------------------------------------------------------------------
# easing
# ----------------------------------------------------------------------------
def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def smoothstep(p):
    return p * p * (3 - 2 * p)


def ease_out(p):
    return 1 - (1 - p) ** 3


def overshoot(p, s=1.70158):
    p -= 1
    return p * p * ((s + 1) * p + s) + 1


# ----------------------------------------------------------------------------
# wobble: make a polyline look hand-drawn
# ----------------------------------------------------------------------------
def _smooth_noise(n, rng, scale=1.0, octaves=18):
    """Smooth random offsets: sum of a few random sines."""
    t = np.linspace(0, 1, n)
    out = np.zeros(n)
    for k in range(1, 4):
        f = rng.uniform(1.5, 4.5) * k * octaves / 18.0
        ph = rng.uniform(0, 2 * np.pi)
        out += rng.uniform(0.4, 1.0) / k * np.sin(2 * np.pi * f * t + ph)
    return out * scale / 2.2


def wobble(pts, amp=0.028, seed=0, densify=3.0):
    """pts: (N,2) canvas coords -> denser, jittered hand-drawn path."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = seg.sum()
    n = max(24, int(L * 14 * densify / 3.0))
    cum = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, cum[-1], n)
    x = np.interp(u, cum, pts[:, 0])
    y = np.interp(u, cum, pts[:, 1])
    rng = np.random.default_rng(seed)
    # offsets perpendicular-ish: just add smooth noise to x and y independently
    x = x + _smooth_noise(n, rng, amp)
    y = y + _smooth_noise(n, rng, amp)
    return np.column_stack([x, y])


# ----------------------------------------------------------------------------
# Items
# ----------------------------------------------------------------------------
class Item:
    """Base timed drawable. Draw-on between t0 and t0+dur.
    Optional fade-out starting at t_kill (alpha->0 over kill_dur)."""

    def __init__(self, t0=0.0, dur=0.6):
        self.t0 = float(t0)
        self.dur = max(float(dur), 1e-3)
        self.t_kill = None
        self.kill_dur = 0.4
        self._artists = []

    def until(self, t_kill, kill_dur=0.4):
        self.t_kill = float(t_kill)
        self.kill_dur = kill_dur
        return self

    # -- lifecycle ----------------------------------------------------------
    def reset(self):
        self._artists = []

    def ensure(self, ax):
        raise NotImplementedError

    def set_progress(self, p):
        raise NotImplementedError

    def master_alpha(self, t):
        a = 1.0
        if self.t_kill is not None and t >= self.t_kill:
            a = 1.0 - clamp01((t - self.t_kill) / max(self.kill_dur, 1e-3))
        return a

    def update(self, t, ax):
        if not self._artists:
            self.ensure(ax)
        p = clamp01((t - self.t0) / self.dur)
        vis = t >= self.t0
        ma = self.master_alpha(t)
        if ma <= 0:
            vis = False
        for a in self._artists:
            a.set_visible(vis)
        if vis:
            self.set_progress(p)
            self.apply_master_alpha(ma)

    def apply_master_alpha(self, ma):
        if ma >= 1.0:
            return
        for a in self._artists:
            try:
                cur = a.get_alpha()
                a.set_alpha((cur if cur is not None else 1.0) * ma)
            except Exception:
                pass


class Stroke(Item):
    """Progressive hand-drawn polyline with a soft chalk glow."""

    def __init__(self, pts, color=INK, lw=3.2, t0=0.0, dur=0.8, amp=0.028,
                 seed=None, alpha=1.0, glow=True, zorder=3, dashed=False):
        super().__init__(t0, dur)
        seed = np.random.randint(0, 10 ** 6) if seed is None else seed
        self.P = wobble(pts, amp=amp, seed=seed)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.color, self.lw, self.alpha0 = color, lw, alpha
        self.glow, self.zorder, self.dashed = glow, zorder, dashed

    def ensure(self, ax):
        ls = (0, (4.5, 3.2)) if self.dashed else "-"
        if self.glow:
            gl, = ax.plot([], [], color=self.color, lw=self.lw * 3.0,
                          alpha=0.10 * self.alpha0, solid_capstyle="round",
                          zorder=self.zorder - 0.1, ls=ls)
            self._artists.append(gl)
        ln, = ax.plot([], [], color=self.color, lw=self.lw,
                      alpha=self.alpha0, solid_capstyle="round",
                      solid_joinstyle="round", zorder=self.zorder, ls=ls)
        self._artists.append(ln)

    def set_progress(self, p):
        p = smoothstep(p)
        L = self.cum[-1] * p
        n = int(np.searchsorted(self.cum, L)) + 1
        n = min(n, len(self.P))
        for a in self._artists:
            a.set_data(self.P[:n, 0], self.P[:n, 1])
        # restore alphas (master alpha multiplies later)
        if self.glow:
            self._artists[0].set_alpha(0.10 * self.alpha0)
            self._artists[1].set_alpha(self.alpha0)
        else:
            self._artists[0].set_alpha(self.alpha0)


class Write(Item):
    """Handwritten text that appears character by character."""

    def __init__(self, xy, s, size=30, color=INK, t0=0.0, dur=None, ha="left",
                 va="baseline", font=HAND, weight="normal", alpha=1.0, zorder=5,
                 rotation=0):
        if dur is None:
            dur = max(0.35, min(2.2, 0.028 * len(s)))
        super().__init__(t0, dur)
        self.xy, self.s, self.size, self.color = xy, s, size, color
        self.ha, self.va, self.font, self.weight = ha, va, font, weight
        self.alpha0, self.zorder, self.rotation = alpha, zorder, rotation

    def ensure(self, ax):
        t = ax.text(self.xy[0], self.xy[1], "", fontsize=self.size,
                    color=self.color, family=self.font, ha=self.ha, va=self.va,
                    weight=self.weight, alpha=self.alpha0, zorder=self.zorder,
                    rotation=self.rotation)
        self._artists.append(t)

    def set_progress(self, p):
        n = int(round(len(self.s) * ease_out(p)))
        self._artists[0].set_text(self.s[:n])
        self._artists[0].set_alpha(self.alpha0)


class Tex(Item):
    """Math text revealed left-to-right (write-on illusion via clip box)."""

    def __init__(self, xy, s, size=30, color=INK, t0=0.0, dur=1.0, ha="left",
                 va="center", alpha=1.0, zorder=5):
        super().__init__(t0, dur)
        self.xy, self.s, self.size, self.color = xy, s, size, color
        self.ha, self.va, self.alpha0, self.zorder = ha, va, alpha, zorder
        self._bbox = None

    def reset(self):
        self._artists = []
        self._bbox = None

    def ensure(self, ax):
        t = ax.text(self.xy[0], self.xy[1], self.s, fontsize=self.size,
                    color=self.color, ha=self.ha, va=self.va,
                    alpha=self.alpha0, zorder=self.zorder)
        self._artists.append(t)
        self._ax = ax

    def set_progress(self, p):
        t = self._artists[0]
        if self._bbox is None:
            fig = self._ax.figure
            try:
                rend = fig.canvas.get_renderer()
            except AttributeError:
                try:
                    rend = fig._get_renderer()
                except AttributeError:
                    fig.canvas.draw()
                    rend = fig.canvas.renderer
            self._bbox = t.get_window_extent(renderer=rend)
        bb = self._bbox
        w = bb.width * ease_out(p) + 2
        clip = Bbox([[bb.x0, bb.y0 - 6], [bb.x0 + w, bb.y1 + 6]])
        t.set_clip_box(clip)
        t.set_clip_on(True)
        t.set_alpha(self.alpha0)


class Dot(Item):
    def __init__(self, xy, color=INK, size=9, t0=0.0, dur=0.35, zorder=6,
                 alpha=1.0, marker="o"):
        super().__init__(t0, dur)
        self.xy, self.color, self.size = xy, color, size
        self.zorder, self.alpha0, self.marker = zorder, alpha, marker

    def ensure(self, ax):
        ln, = ax.plot([self.xy[0]], [self.xy[1]], self.marker,
                      color=self.color, ms=self.size, zorder=self.zorder,
                      alpha=0.0, markeredgecolor="none")
        self._artists.append(ln)

    def set_progress(self, p):
        s = overshoot(clamp01(p))
        self._artists[0].set_markersize(self.size * max(s, 1e-3))
        self._artists[0].set_alpha(self.alpha0 * clamp01(p * 3))


class Fill(Item):
    """Translucent filled polygon (highlight region) that fades in."""

    def __init__(self, pts, color=C["yellow"], alpha=0.14, t0=0.0, dur=0.6,
                 zorder=1.5, closed=True):
        super().__init__(t0, dur)
        self.pts, self.color, self.alpha0 = np.asarray(pts, float), color, alpha
        self.zorder, self.closed = zorder, closed

    def ensure(self, ax):
        poly = Polygon(self.pts, closed=self.closed, facecolor=self.color,
                       edgecolor="none", alpha=0.0, zorder=self.zorder)
        ax.add_patch(poly)
        self._artists.append(poly)

    def set_progress(self, p):
        self._artists[0].set_alpha(self.alpha0 * smoothstep(p))


class ArrowItem(Item):
    """Hand-drawn arrow: progressive shaft, then head pops."""

    def __init__(self, p0, p1, color=INK, lw=3.0, t0=0.0, dur=0.6, amp=0.02,
                 seed=None, zorder=4, head=0.16, alpha=1.0, curve=0.0):
        super().__init__(t0, dur)
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        mid = (p0 + p1) / 2
        d = p1 - p0
        nrm = np.array([-d[1], d[0]])
        n = nrm / (np.linalg.norm(nrm) + 1e-9)
        ctrl = mid + n * curve
        ts = np.linspace(0, 1, 24)[:, None]
        pts = (1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * ctrl + ts ** 2 * p1
        seed = np.random.randint(0, 10 ** 6) if seed is None else seed
        self.P = wobble(pts, amp=amp, seed=seed)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.color, self.lw, self.zorder = color, lw, zorder
        self.head, self.alpha0 = head, alpha

    def ensure(self, ax):
        ln, = ax.plot([], [], color=self.color, lw=self.lw, alpha=self.alpha0,
                      solid_capstyle="round", zorder=self.zorder)
        hd = Polygon([[0, 0]] * 3, closed=True, facecolor=self.color,
                     edgecolor="none", alpha=0.0, zorder=self.zorder)
        ax.add_patch(hd)
        self._artists += [ln, hd]

    def set_progress(self, p):
        p = smoothstep(p)
        L = self.cum[-1] * p
        n = min(int(np.searchsorted(self.cum, L)) + 1, len(self.P))
        self._artists[0].set_data(self.P[:n, 0], self.P[:n, 1])
        self._artists[0].set_alpha(self.alpha0)
        if p > 0.92 and n >= 2:
            tip = self.P[n - 1]
            d = self.P[n - 1] - self.P[max(0, n - 8)]
            d = d / (np.linalg.norm(d) + 1e-9)
            nvec = np.array([-d[1], d[0]])
            h = self.head
            tri = [tip + d * h * 0.4, tip - d * h + nvec * h * 0.55,
                   tip - d * h - nvec * h * 0.55]
            self._artists[1].set_xy(tri)
            self._artists[1].set_alpha(self.alpha0)
        else:
            self._artists[1].set_alpha(0.0)


class BoxItem(Item):
    """Hand-drawn rectangle outline (optionally rounded feel via wobble)."""

    def __init__(self, x, y, w, h, color=INK, lw=3.0, t0=0.0, dur=0.7,
                 seed=None, zorder=3, alpha=1.0, fill=None, fill_alpha=0.10):
        super().__init__(t0, dur)
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        seed = np.random.randint(0, 10 ** 6) if seed is None else seed
        self.stroke = Stroke(pts, color=color, lw=lw, t0=t0, dur=dur,
                             seed=seed, zorder=zorder, alpha=alpha)
        self.fill_col, self.fill_alpha = fill, fill_alpha
        self.rect = (x, y, w, h)
        self.zorder = zorder

    def reset(self):
        self._artists = []
        self.stroke.reset()

    def ensure(self, ax):
        if self.fill_col:
            x, y, w, h = self.rect
            r = Rectangle((x, y), w, h, facecolor=self.fill_col,
                          edgecolor="none", alpha=0.0, zorder=self.zorder - 0.5)
            ax.add_patch(r)
            self._artists.append(r)
        self.stroke.ensure(ax)
        self._artists += self.stroke._artists

    def set_progress(self, p):
        self.stroke.set_progress(p)
        if self.fill_col:
            self._artists[0].set_alpha(self.fill_alpha * smoothstep(p))


# ----------------------------------------------------------------------------
# Scene & rendering
# ----------------------------------------------------------------------------
class Scene:
    def __init__(self, name, duration, fps=24, size=(1920, 1080)):
        self.name = name
        self.duration = duration
        self.fps = fps
        self.size = size
        self.items = []

    def add(self, *items):
        for it in items:
            self.items.append(it)
        return items[-1] if items else None

    # convenience: sequential add returning end time
    def play(self, item):
        self.add(item)
        return item.t0 + item.dur

    def make_fig(self):
        w, h = self.size
        dpi = 120
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(BG)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.axis("off")
        return fig, ax

    def render(self, out_path, audio_path=None, tail=0.0, quiet=True):
        fig, ax = self.make_fig()
        for it in self.items:
            it.reset()
        n_frames = int(np.ceil((self.duration + tail) * self.fps))
        w, h = self.size
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-"]
        if audio_path:
            cmd += ["-i", audio_path, "-af", "apad", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "21",
                "-pix_fmt", "yuv420p"]
        if audio_path:
            cmd += ["-c:a", "aac", "-b:a", "160k"]
        cmd += [out_path]
        stderr = subprocess.DEVNULL if quiet else None
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr)
        canvas = fig.canvas
        for i in range(n_frames):
            t = i / self.fps
            for it in self.items:
                it.update(t, ax)
            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())
            if buf.shape[1] != w or buf.shape[0] != h:
                # safety: resize by nearest if dpi rounding bites
                from math import floor
                buf = buf[:h, :w]
            proc.stdin.write(buf.tobytes())
        proc.stdin.close()
        proc.wait()
        plt.close(fig)
        return out_path

    def snapshot(self, t, out_png):
        """Render a single frame at time t (for QC)."""
        fig, ax = self.make_fig()
        for it in self.items:
            it.reset()
            it.update(t, ax)
        fig.savefig(out_png, facecolor=BG)
        plt.close(fig)
        return out_png


# ----------------------------------------------------------------------------
# Graph helper: maps data coords into a canvas rectangle
# ----------------------------------------------------------------------------
class Graph:
    def __init__(self, rect, xlim=(0, 1), ylim=(0, 1)):
        self.x0, self.y0, self.w, self.h = rect
        self.xlim, self.ylim = xlim, ylim

    def X(self, x):
        a, b = self.xlim
        return self.x0 + (np.asarray(x, float) - a) / (b - a) * self.w

    def Y(self, y):
        a, b = self.ylim
        return self.y0 + (np.asarray(y, float) - a) / (b - a) * self.h

    def XY(self, x, y):
        return np.column_stack([self.X(x), self.Y(y)])

    def axes_items(self, t0, color=C["grey"], lw=2.4, x_arrow=True,
                   y_arrow=True, dur=0.7, pad=0.15):
        items = []
        x1 = self.x0 + self.w + (0.35 if x_arrow else 0)
        y1 = self.y0 + self.h + (0.35 if y_arrow else 0)
        items.append(ArrowItem((self.x0 - pad, self.y0), (x1, self.y0),
                               color=color, lw=lw, t0=t0, dur=dur, head=0.13))
        items.append(ArrowItem((self.x0, self.y0 - pad), (self.x0, y1),
                               color=color, lw=lw, t0=t0 + 0.15, dur=dur,
                               head=0.13))
        return items


# ----------------------------------------------------------------------------
# real B-spline / NURBS math for authentic diagrams
# ----------------------------------------------------------------------------
def bspline_basis(x, knots, degree):
    """Cox-de Boor. x: (n,), knots: (m,). Returns (n_splines, n)."""
    knots = np.asarray(knots, float)
    x = np.asarray(x, float)
    # clamp so the right endpoint falls inside the last non-empty interval
    x = np.where(x >= knots[-1], knots[-1] - 1e-9, x)
    m = len(knots)
    nb = m - 1
    B = np.zeros((nb, len(x)))
    for i in range(nb):
        B[i] = ((x >= knots[i]) & (x < knots[i + 1])).astype(float)
    for d in range(1, degree + 1):
        nb_d = m - d - 1
        Bn = np.zeros((nb_d, len(x)))
        for i in range(nb_d):
            den1 = knots[i + d] - knots[i]
            den2 = knots[i + d + 1] - knots[i + 1]
            t1 = ((x - knots[i]) / den1) * B[i] if den1 > 0 else 0
            t2 = ((knots[i + d + 1] - x) / den2) * B[i + 1] if den2 > 0 else 0
            Bn[i] = t1 + t2
        B = Bn
    return B


def open_uniform_knots(n_ctrl, degree, lo=0.0, hi=1.0):
    inner = np.linspace(lo, hi, n_ctrl - degree + 1)
    return np.concatenate([[lo] * degree, inner, [hi] * degree])


def fit_bspline(x, y, n_ctrl, degree=3, weights=None):
    """Least-squares B-spline / NURBS-weighted fit. Returns (yhat, B, c)."""
    kn = open_uniform_knots(n_ctrl, degree, x.min(), x.max() + 1e-9)
    B = bspline_basis(x, kn, degree)
    if weights is not None:
        Wb = B * weights[:, None]
        B = Wb / (Wb.sum(axis=0, keepdims=True) + 1e-12)
    A = B.T
    reg = 1e-8 * np.eye(B.shape[0])
    c = np.linalg.solve(B @ A + reg, B @ y)
    return A @ c, B, c
