"""
Khan-academy-style chalkboard renderer.

Design coordinates are 1920x1080; frames are drawn at SS x supersample and
downscaled for anti-aliasing. Every visual is an El(ement) with a paint
closure; elements belong to scenes and fade with them.
"""
import math
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080
SS = 2
FPS = 30

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "assets", "fonts")

PAL = dict(
    bg=(16, 18, 23),
    panel=(26, 29, 36),
    ink=(240, 236, 224),
    dim=(148, 154, 164),
    faint=(96, 102, 112),
    dead=(70, 75, 83),
    blue=(88, 196, 221),
    green=(104, 211, 145),
    red=(255, 118, 118),
    gold=(255, 205, 100),
    purple=(186, 164, 224),
    cyan=(126, 208, 250),
    pink=(238, 150, 190),
    orange=(255, 160, 90),
)


def mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def dim(color, a):
    """Blend color toward the board background (a=1 -> full color)."""
    return mix(PAL["bg"], color, a)


def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def ease(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------- fonts

_font_cache = {}


def _font_path(kind):
    paths = {
        "hand": os.path.join(FONT_DIR, "Kalam-Regular.ttf"),
        "handb": os.path.join(FONT_DIR, "Kalam-Bold.ttf"),
        "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "monob": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    }
    p = paths[kind]
    if not os.path.exists(p):
        p = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return p


def font(kind, size):
    key = (kind, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(_font_path(kind), int(size * SS))
    return _font_cache[key]


def text_w(s, kind, size):
    f = font(kind, size)
    return f.getbbox(s)[2] / SS


# ---------------------------------------------------------------- sketch prims

def _noise(seed, k):
    return math.sin(seed * 12.9898 + k * 78.233) * 43758.5453 % 2.0 - 1.0


def _wobble_path(pts, seed, amp=2.0, step=16.0):
    """Subdivide a polyline and jitter points perpendicular to travel."""
    out = []
    k = 0
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(seg / step))
        for j in range(n):
            t = j / n
            x, y = lerp(x0, x1, t), lerp(y0, y1, t)
            dx, dy = (x1 - x0) / (seg + 1e-9), (y1 - y0) / (seg + 1e-9)
            off = _noise(seed, k) * amp
            out.append((x - dy * off, y + dx * off))
            k += 1
    out.append(pts[-1])
    return out


def _partial(path, prog):
    if prog >= 1.0:
        return path
    lens = [0.0]
    for i in range(len(path) - 1):
        lens.append(lens[-1] + math.hypot(path[i + 1][0] - path[i][0],
                                          path[i + 1][1] - path[i][1]))
    total = lens[-1]
    if total <= 0:
        return path[:1]
    target = total * clamp01(prog)
    out = [path[0]]
    for i in range(1, len(path)):
        if lens[i] <= target:
            out.append(path[i])
        else:
            seg = lens[i] - lens[i - 1]
            t = (target - lens[i - 1]) / (seg + 1e-9)
            out.append((lerp(path[i - 1][0], path[i][0], t),
                        lerp(path[i - 1][1], path[i][1], t)))
            break
    return out


def S(v):
    return v * SS


def spoly(d, pts, color, w=4, seed=1.0, prog=1.0, amp=2.0):
    if prog <= 0 or len(pts) < 2:
        return
    path = _partial(_wobble_path(pts, seed, amp), prog)
    if len(path) < 2:
        return
    d.line([(S(x), S(y)) for x, y in path], fill=color, width=int(w * SS),
           joint="curve")


def sline(d, p0, p1, color, w=4, seed=1.0, prog=1.0, amp=2.0):
    spoly(d, [p0, p1], color, w, seed, prog, amp)


def srect(d, xy, wh, color, w=4, seed=1.0, prog=1.0, fill=None, amp=2.0):
    x, y = xy
    ww, hh = wh
    if fill is not None:
        d.rounded_rectangle([S(x + 2), S(y + 2), S(x + ww - 2), S(y + hh - 2)],
                            radius=S(6), fill=fill)
    pts = [(x, y), (x + ww, y), (x + ww, y + hh), (x, y + hh), (x, y)]
    spoly(d, pts, color, w, seed, prog, amp)


def sarrow(d, p0, p1, color, w=4, seed=1.0, prog=1.0, head=12):
    sline(d, p0, p1, color, w, seed, prog)
    if prog >= 0.82:
        a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        h = head * clamp01((prog - 0.82) / 0.18)
        for da in (2.65, -2.65):
            q = (p1[0] + h * math.cos(a + da), p1[1] + h * math.sin(a + da))
            d.line([(S(p1[0]), S(p1[1])), (S(q[0]), S(q[1]))],
                   fill=color, width=int(w * SS))


def scircle(d, c, r, color, w=4, seed=1.0, prog=1.0, fill=None, amp=1.6):
    if fill is not None:
        d.ellipse([S(c[0] - r), S(c[1] - r), S(c[0] + r), S(c[1] + r)], fill=fill)
    pts = [(c[0] + r * math.cos(2 * math.pi * i / 28 - math.pi / 2),
            c[1] + r * math.sin(2 * math.pi * i / 28 - math.pi / 2))
           for i in range(30)]
    spoly(d, pts, color, w, seed, prog, amp)


def draw_text(d, pos, s, kind, size, color, anchor="la"):
    d.text((S(pos[0]), S(pos[1])), s, font=font(kind, size), fill=color,
           anchor=anchor)


# ---------------------------------------------------------------- anim

class Anim:
    """Piecewise keyframed scalar/tuple value."""

    def __init__(self, v0):
        self.segs = []          # (t0, t1, v_from, v_to)
        self.v0 = v0

    def to(self, t, v, dur=0.45):
        vf = self.value_at(t)
        self.segs.append((t, t + dur, vf, v))
        return self

    def set(self, t, v):
        return self.to(t, v, dur=1e-4)

    def value_at(self, t):
        v = self.v0
        for (t0, t1, vf, vt) in self.segs:
            if t >= t1:
                v = vt
            elif t >= t0:
                k = ease((t - t0) / (t1 - t0 + 1e-9))
                if isinstance(vf, tuple):
                    v = tuple(lerp(a, b, k) for a, b in zip(vf, vt))
                else:
                    v = lerp(vf, vt, k)
                return v
        return v

    __call__ = value_at


class StateTrack:
    """Piecewise-constant state with crossfade info: returns (prev, cur, blend)."""

    def __init__(self, s0, fade=0.28):
        self.keys = [(-1e9, s0)]
        self.fade = fade

    def set(self, t, s):
        self.keys.append((t, s))
        self.keys.sort(key=lambda k: k[0])

    def at(self, t):
        prev_s, cur_s, t_cur = self.keys[0][1], self.keys[0][1], -1e9
        for (kt, ks) in self.keys:
            if kt <= t:
                prev_s, cur_s, t_cur = cur_s, ks, kt
            else:
                break
        blend = clamp01((t - t_cur) / self.fade)
        return prev_s, cur_s, blend


# ---------------------------------------------------------------- elements

class El:
    __slots__ = ("t_in", "t_out", "paint", "fade_in", "fade_out", "z", "persist")

    def __init__(self, t_in, paint, fade_in=0.35, t_out=None, fade_out=0.35,
                 z=0, persist=False):
        self.t_in = t_in
        self.paint = paint
        self.fade_in = fade_in
        self.t_out = t_out
        self.fade_out = fade_out
        self.z = z
        self.persist = persist

    def alpha(self, t):
        if t < self.t_in:
            return 0.0
        a = 1.0 if self.fade_in <= 0 else ease((t - self.t_in) / self.fade_in)
        if self.t_out is not None:
            if t > self.t_out:
                a = 0.0
            else:
                a *= ease((self.t_out - t) / max(self.fade_out, 1e-6))
        return a


class Scene:
    def __init__(self, name, t0):
        self.name = name
        self.t0 = t0
        self.t1 = None
        self.els = []

    def active(self, t):
        return self.t0 <= t < (self.t1 or 1e9)

    def scene_alpha(self, t):
        if self.t1 is None:
            return 1.0
        return clamp01((self.t1 - t) / 0.45)


class Timeline:
    def __init__(self):
        self.scenes = []
        self.duration = 0.0

    def draw(self, t):
        img = Image.new("RGB", (W * SS, H * SS), PAL["bg"])
        d = ImageDraw.Draw(img)
        for sc in self.scenes:
            if not sc.active(t):
                continue
            sa = sc.scene_alpha(t)
            for el in sorted(sc.els, key=lambda e: e.z):
                a = el.alpha(t) * sa
                if a <= 0.004:
                    continue
                el.paint(d, t, a)
        return img.resize((W, H), Image.LANCZOS)


# ---------------------------------------------------------------- scene ctx

class SceneCtx:
    """Factory helpers; every add() returns the El (or a ref object)."""

    def __init__(self, scene):
        self.sc = scene
        self._seed = 0.0

    def seed(self):
        self._seed += 1.37
        return self._seed

    def el(self, t_in, paint, **kw):
        e = El(t_in, paint, **kw)
        self.sc.els.append(e)
        return e

    def clear_at(self, t, keep=()):
        """Fade out everything created so far (except persistent/kept)."""
        for e in self.sc.els:
            if e.persist or e in keep:
                continue
            if e.t_out is None and e.t_in < t:
                e.t_out = t + 0.35
        return self

    # ---- text ----------------------------------------------------------
    def text(self, t, pos, s, size=34, color=None, kind="hand", anchor="la",
             write=True, cps=40, **kw):
        color = color or PAL["ink"]
        n = len(s)

        def paint(d, tt, a):
            if write:
                k = int((tt - t) * cps)
                sub = s[:k]
                if not sub:
                    return
            else:
                sub = s
            dy = (1 - a) * 8
            draw_text(d, (pos[0], pos[1] + dy), sub, kind, size, dim(color, a),
                      anchor)
        return self.el(t, paint, fade_in=(0.0 if write else 0.3), **kw)

    def note(self, t, pos, s, size=26, color=None, anchor="la", **kw):
        return self.text(t, pos, s, size=size, color=color or PAL["dim"],
                         write=False, anchor=anchor, **kw)

    def title(self, t, s, y=140, size=72, color=None, sub=None, **kw):
        e = self.text(t, (W / 2, y), s, size=size, color=color or PAL["ink"],
                      kind="handb", anchor="mm", cps=28, **kw)
        sd = self.seed()

        def underline(d, tt, a):
            wdt = text_w(s, "handb", size) * 0.55
            p = ease_out((tt - (t + 0.7)) / 0.6)
            if p <= 0:
                return
            sline(d, (W / 2 - wdt, y + size * 0.75),
                  (W / 2 - wdt + 2 * wdt * p, y + size * 0.75),
                  dim(PAL["gold"], a), w=5, seed=sd)
        self.el(t + 0.7, underline, fade_in=0.0, **kw)
        if sub:
            self.text(t + 0.9, (W / 2, y + size * 1.35), sub, size=34,
                      color=PAL["dim"], anchor="mm", write=False, **kw)
        return e

    def section(self, t, s, color=None):
        color = color or PAL["gold"]
        e = self.text(t, (90, 62), s, size=40, color=color, kind="handb",
                      cps=34)
        e.persist = True
        sd = self.seed()

        def underline(d, tt, a):
            wdt = text_w(s, "handb", 40)
            p = ease_out((tt - (t + 0.45)) / 0.5)
            if p <= 0:
                return
            sline(d, (92, 118), (92 + wdt * p, 118), dim(color, a * 0.8),
                  w=4, seed=sd)
        u = self.el(t + 0.45, underline, fade_in=0.0)
        u.persist = True
        return e

    def chip(self, t, pos, s, color=None, size=26, kind="handb",
             pad=14, filled=True, **kw):
        color = color or PAL["blue"]
        wdt = text_w(s, kind, size)
        sd = self.seed()
        x, y = pos

        def paint(d, tt, a):
            dy = (1 - a) * 10
            fill = dim(color, 0.16 * a) if filled else None
            srect(d, (x, y + dy), (wdt + 2 * pad, size + 2 * pad - 6),
                  dim(color, a), w=3, seed=sd, fill=fill)
            draw_text(d, (x + pad + wdt / 2, y + dy + (size + 2 * pad - 6) / 2),
                      s, kind, size, dim(color, a), anchor="mm")
        return self.el(t, paint, fade_in=0.3, **kw)

    # ---- lines / arrows --------------------------------------------------
    def arrow(self, t, p0, p1, color=None, w=5, dur=0.5, head=13, **kw):
        color = color or PAL["ink"]
        sd = self.seed()

        def paint(d, tt, a):
            p = ease_out((tt - t) / dur)
            sarrow(d, p0, p1, dim(color, a), w=w, seed=sd, prog=p, head=head)
        return self.el(t, paint, fade_in=0.0, **kw)

    def line(self, t, p0, p1, color=None, w=4, dur=0.5, **kw):
        color = color or PAL["ink"]
        sd = self.seed()

        def paint(d, tt, a):
            p = ease_out((tt - t) / dur)
            sline(d, p0, p1, dim(color, a), w=w, seed=sd, prog=p)
        return self.el(t, paint, fade_in=0.0, **kw)

    def polyline(self, t, pts, color=None, w=5, dur=1.0, **kw):
        color = color or PAL["cyan"]
        sd = self.seed()

        def paint(d, tt, a):
            p = ease_out((tt - t) / dur)
            spoly(d, pts, dim(color, a), w=w, seed=sd, prog=p)
        return self.el(t, paint, fade_in=0.0, **kw)

    def circle(self, t, c, r, color=None, w=4, dur=0.4, fill=None, **kw):
        color = color or PAL["gold"]
        sd = self.seed()

        def paint(d, tt, a):
            p = ease_out((tt - t) / dur)
            f = dim(fill, a) if fill else None
            scircle(d, c, r, dim(color, a), w=w, seed=sd, prog=p, fill=f)
        return self.el(t, paint, fade_in=0.0, **kw)

    # ---- array of cells --------------------------------------------------
    def array(self, t, pos, values, cw=92, ch=80, idx_labels=True,
              val_size=34, stagger=0.06, dashed=(), val_kind="handb", **kw):
        return ArrayRef(self, t, pos, values, cw, ch, idx_labels, val_size,
                        stagger, dashed, val_kind, **kw)

    # ---- code panel ------------------------------------------------------
    def code(self, t, pos, lines, size=27, line_stagger=0.55, width=None, **kw):
        return CodeRef(self, t, pos, lines, size, line_stagger, width, **kw)

    # ---- grid table ------------------------------------------------------
    def grid(self, t, pos, colw, header, rows, rh=62, header_h=56,
             text_size=25, row_colors=None, **kw):
        return GridRef(self, t, pos, colw, header, rows, rh, header_h,
                       text_size, row_colors, **kw)

    # ---- misc ------------------------------------------------------------
    def card(self, t, pos, wdt, title, entries, color=None, title_size=27,
             entry_size=25, stagger=0.5, **kw):
        """Instantiation card: title + [(label, value, vcolor)] rows."""
        color = color or PAL["blue"]
        sd = self.seed()
        rh = entry_size + 22
        hgt = 56 + rh * len(entries) + 14
        x, y = pos

        def box(d, tt, a):
            p = ease_out((tt - t) / 0.5)
            srect(d, (x, y), (wdt, hgt), dim(color, a), w=3.5, seed=sd,
                  prog=p, fill=dim(color, 0.06 * a))
            if tt > t + 0.15:
                aa = a * ease((tt - t - 0.15) / 0.3)
                draw_text(d, (x + 18, y + 14), title, "handb", title_size,
                          dim(color, aa))
        e = self.el(t, box, fade_in=0.0, **kw)
        for i, (lab, val, vc) in enumerate(entries):
            ty = y + 58 + rh * i
            ti = t + 0.35 + stagger * i

            def paint(d, tt, a, lab=lab, val=val, vc=vc, ty=ty, ti=ti):
                draw_text(d, (x + 20, ty), lab, "handb", entry_size - 3,
                          dim(PAL["dim"], a))
                draw_text(d, (x + 150, ty), val, "hand", entry_size,
                          dim(vc or PAL["ink"], a))
            self.el(ti, paint, fade_in=0.3, **kw)
        return e


# ---------------------------------------------------------------- refs

STATE_COLORS = {
    # state -> (stroke, fill_alpha, text)
    "base": ("ink", 0.0, "ink"),
    "dead": ("dead", 0.05, "dead"),
    "T": ("green", 0.16, "green"),
    "F": ("red", 0.14, "red"),
    "hot": ("gold", 0.22, "gold"),
    "found": ("green", 0.30, "green"),
    "cyanH": ("cyan", 0.20, "cyan"),
}


class ArrayRef:
    def __init__(self, ctx, t, pos, values, cw, ch, idx_labels, val_size,
                 stagger, dashed, val_kind, **kw):
        self.ctx = ctx
        self.x, self.y = pos
        self.cw, self.ch = cw, ch
        self.n = len(values)
        self.values = [str(v) for v in values]
        self.t0 = t
        self.states = [StateTrack("base") for _ in range(self.n)]
        self.seeds = [ctx.seed() for _ in range(self.n)]
        self.dashed = set(dashed)
        self.kw = kw

        def paint(d, tt, a):
            for i in range(self.n):
                ti = t + stagger * i
                p = ease_out((tt - ti) / 0.4)
                if p <= 0:
                    continue
                cx = self.x + i * cw
                ps, cs, blend = self.states[i].at(tt)
                s0, f0, tx0 = STATE_COLORS[ps]
                s1, f1, tx1 = STATE_COLORS[cs]
                stroke = mix(PAL[s0], PAL[s1], blend)
                fa = lerp(f0, f1, blend)
                txc = mix(PAL[tx0], PAL[tx1], blend)
                fill = dim(stroke, fa * a) if fa > 0.01 else None
                if i in self.dashed:
                    # dashed border: draw 8 short segments
                    per = [(cx, self.y), (cx + cw, self.y),
                           (cx + cw, self.y + ch), (cx, self.y + ch), (cx, self.y)]
                    if fill:
                        d.rounded_rectangle([S(cx + 2), S(self.y + 2),
                                             S(cx + cw - 2), S(self.y + ch - 2)],
                                            radius=S(6), fill=fill)
                    for si in range(len(per) - 1):
                        x0, y0 = per[si]
                        x1, y1 = per[si + 1]
                        for k in range(4):
                            u0, u1 = k / 4, k / 4 + 0.14
                            sline(d, (lerp(x0, x1, u0), lerp(y0, y1, u0)),
                                  (lerp(x0, x1, u1), lerp(y0, y1, u1)),
                                  dim(stroke, a * p), w=3,
                                  seed=self.seeds[i] + k)
                else:
                    srect(d, (cx, self.y), (cw, ch), dim(stroke, a), w=3.5,
                          seed=self.seeds[i], prog=p, fill=fill)
                if p > 0.5:
                    ta = a * clamp01((p - 0.5) / 0.4)
                    draw_text(d, (cx + cw / 2, self.y + ch / 2),
                              self.values[i], val_kind, val_size,
                              dim(txc, ta), anchor="mm")
        self.el = ctx.el(t, paint, fade_in=0.0, **kw)

        if idx_labels:
            def paint_idx(d, tt, a):
                for i in range(self.n):
                    ti = t + 0.25 + stagger * i
                    if tt < ti:
                        continue
                    aa = a * ease((tt - ti) / 0.3)
                    _, cs, _ = self.states[i].at(tt)
                    c = PAL["faint"] if cs != "dead" else PAL["dead"]
                    draw_text(d, (self.x + i * cw + cw / 2, self.y + ch + 22),
                              str(i), "hand", 21, dim(c, aa), anchor="mm")
            self.idx_el = ctx.el(t, paint_idx, fade_in=0.0, **kw)

    def cx(self, i):
        return self.x + i * self.cw + self.cw / 2

    def state(self, t, i, s):
        self.states[i].set(t, s)
        return self

    def state_range(self, t, i0, i1, s, stagger=0.05):
        for k, i in enumerate(range(i0, i1 + 1)):
            self.states[i].set(t + stagger * k, s)
        return self

    def pointer(self, t, name, i, color, side="bottom", lane=0):
        """Animated arrow+label pointing at cell centers; .move(t,i), .off(t)."""
        ai = Anim(float(i))
        ctx = self.ctx
        sd = ctx.seed()
        xoff = {"lo": -13, "hi": 13}.get(name, 0)

        def paint(d, tt, a):
            fi = ai(tt)
            x = self.x + fi * self.cw + self.cw / 2 + xoff
            if side == "bottom":
                y0 = self.y + self.ch + 44 + lane * 58
                sarrow(d, (x, y0 + 40), (x, y0 + 6), dim(color, a), w=4.5,
                       seed=sd, head=11)
                draw_text(d, (x, y0 + 62), name, "handb", 27, dim(color, a),
                          anchor="mm")
            else:
                y0 = self.y - 10 - lane * 58
                sarrow(d, (x, y0 - 40), (x, y0), dim(color, a), w=4.5,
                       seed=sd, head=11)
                draw_text(d, (x, y0 - 62), name, "handb", 27, dim(color, a),
                          anchor="mm")
        e = ctx.el(t, paint, fade_in=0.3, **self.kw)

        class Ptr:
            def move(_s, tt, ii, dur=0.5):
                ai.to(tt, float(ii), dur)
                return _s

            def off(_s, tt):
                e.t_out = tt
                return _s
        return Ptr()

    def probe(self, t, i, resolve, dur=1.0, label="mid", letter=None,
              letter_size=30):
        """Highlight cell i as mid probe, then resolve to 'T'/'F' state."""
        self.state(t, i, "hot")
        ctx = self.ctx

        if label:
            def paint(d, tt, a):
                fade = 1.0
                if tt > t + dur - 0.25:
                    fade = clamp01((t + dur - tt) / 0.25)
                aa = a * fade
                if aa <= 0.01:
                    return
                draw_text(d, (self.cx(i), self.y - 26), label, "handb", 25,
                          dim(PAL["gold"], aa), anchor="mm")
            ctx.el(t, paint, fade_in=0.2, t_out=t + dur + 0.05, fade_out=0.01,
                   **self.kw)
        self.state(t + dur, i, resolve)
        if letter:
            col = PAL["green"] if resolve == "T" else PAL["red"]
            cxx, cyy = self.cx(i), self.y + self.ch / 2

            def paint_l(d, tt, a):
                draw_text(d, (cxx, cyy), letter, "handb", letter_size,
                          dim(col, a), anchor="mm")
            ctx.el(t + dur, paint_l, fade_in=0.25, **self.kw)
        return self


class CodeRef:
    KW = {"while", "if", "else", "return", "def", "for", "in", "and", "or",
          "not"}

    def __init__(self, ctx, t, pos, lines, size, line_stagger, width, **kw):
        self.ctx = ctx
        self.x, self.y = pos
        self.size = size
        self.lh = size * 1.62
        self.t0 = t
        self.lines = lines
        self.hl = Anim(-1.0)
        wdt = width or (max(text_w(l, "mono", size) for l in lines) + 56)
        hgt = self.lh * len(lines) + 40
        sd = ctx.seed()
        self.w, self.h = wdt, hgt

        def paint(d, tt, a):
            p = ease_out((tt - t) / 0.5)
            d.rounded_rectangle([S(self.x), S(self.y), S(self.x + wdt),
                                 S(self.y + hgt)], radius=S(10),
                                fill=dim(PAL["panel"], a))
            srect(d, (self.x, self.y), (wdt, hgt), dim(PAL["faint"], a * 0.9),
                  w=3, seed=sd, prog=p)
            # highlight bar
            hv = self.hl(tt)
            if hv >= -0.5:
                yy = self.y + 20 + hv * self.lh
                d.rounded_rectangle(
                    [S(self.x + 10), S(yy - 4), S(self.x + wdt - 10),
                     S(yy + self.lh - 10)], radius=S(6),
                    fill=dim(PAL["gold"], 0.14 * a))
            for li, line in enumerate(lines):
                lt = t + 0.2 + line_stagger * li
                if tt < lt:
                    break
                k = int((tt - lt) * 34)
                sub = line[:k]
                if not sub:
                    continue
                yy = self.y + 20 + li * self.lh
                self._draw_line(d, sub, self.x + 24, yy, a)
        ctx.el(t, paint, fade_in=0.25, **kw)

    def _draw_line(self, d, line, x, y, a):
        # split off comment
        ci = line.find("#")
        code_part = line if ci < 0 else line[:ci]
        cmt = "" if ci < 0 else line[ci:]
        cx = x
        for tok in code_part.split(" "):
            if tok:
                if tok in self.KW:
                    c = PAL["purple"]
                elif tok in ("ask(mid):", "ask(mid)"):
                    c = PAL["gold"]
                elif any(ch.isdigit() for ch in tok) and len(tok) <= 3:
                    c = PAL["cyan"]
                else:
                    c = PAL["ink"]
                draw_text(d, (cx, y), tok, "mono", self.size, dim(c, a))
            cx += text_w(tok + " ", "mono", self.size)
        if cmt:
            draw_text(d, (cx, y), cmt, "mono", self.size,
                      dim(PAL["faint"], a))

    def line_y(self, i):
        return self.y + 20 + i * self.lh

    def highlight(self, t, i):
        self.hl.to(t, float(i), 0.35)
        return self


class GridRef:
    def __init__(self, ctx, t, pos, colw, header, rows, rh, header_h,
                 text_size, row_colors, cell_kind="hand", **kw):
        self.ctx = ctx
        self.x, self.y = pos
        self.colw = colw
        self.rows = rows
        self.rh = rh
        self.hh = header_h
        self.ts = text_size
        self.row_colors = row_colors or [PAL["ink"]] * len(rows)
        self.reveal = [None] * len(rows)   # time each row appears
        self.hl_row = Anim(-1.0)
        self.t0 = t
        wdt = sum(colw)
        sd = ctx.seed()
        self.w = wdt
        self.h = header_h + rh * len(rows)

        def paint(d, tt, a):
            p = ease_out((tt - t) / 0.6)
            X, Y = self.x, self.y
            # header
            d.rounded_rectangle([S(X), S(Y), S(X + wdt), S(Y + header_h)],
                                radius=S(4), fill=dim(PAL["gold"], 0.10 * a))
            cx = X
            for ci, htxt in enumerate(header):
                if p > 0.2:
                    draw_text(d, (cx + colw[ci] / 2, Y + header_h / 2), htxt,
                              "handb", text_size + 1,
                              dim(PAL["gold"], a * clamp01((p - 0.2) / 0.4)),
                              anchor="mm")
                cx += colw[ci]
            srect(d, (X, Y), (wdt, header_h + rh * len(rows)),
                  dim(PAL["dim"], a * 0.85), w=3, seed=sd, prog=p)
            # vertical rules
            cx = X
            for ci in range(len(colw) - 1):
                cx += colw[ci]
                sline(d, (cx, Y), (cx, Y + header_h + rh * len(rows)),
                      dim(PAL["faint"], a * 0.5), w=2, seed=sd + ci)
            # highlight
            hv = self.hl_row(tt)
            if hv > -0.5:
                yy = Y + header_h + hv * rh
                d.rounded_rectangle([S(X + 4), S(yy + 3), S(X + wdt - 4),
                                     S(yy + rh - 3)], radius=S(4),
                                    fill=dim(PAL["gold"], 0.12 * a))
            # rows
            for ri, row in enumerate(rows):
                rt = self.reveal[ri]
                yy = Y + header_h + ri * rh
                sline(d, (X, yy), (X + wdt, yy), dim(PAL["faint"], a * 0.5),
                      w=2, seed=sd + 40 + ri)
                if rt is None or tt < rt:
                    continue
                ra = a * ease((tt - rt) / 0.45)
                rc = self.row_colors[ri]
                cx = X
                for ci, cell in enumerate(row):
                    c = rc if ci == 0 else PAL["ink"]
                    kindd = "handb" if ci == 0 else cell_kind
                    draw_text(d, (cx + 16, yy + rh / 2), cell, kindd,
                              text_size, dim(c, ra), anchor="lm")
                    cx += self.colw[ci]
        ctx.el(t, paint, fade_in=0.3, **kw)

    def show_row(self, t, i, hl=True):
        self.reveal[i] = t
        if hl:
            self.hl_row.set(t, float(i))
        return self


# ---------------------------------------------------------------- rendering

def render_range(tl, f0, f1, path, quiet=True):
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "21",
           "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL,
                         stderr=(subprocess.DEVNULL if quiet else None))
    for f in range(f0, f1):
        img = tl.draw(f / FPS)
        p.stdin.write(img.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
