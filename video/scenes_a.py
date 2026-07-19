"""Scenes 1-8 for the HNMD explainer (helpers + builders)."""
import numpy as np
from engine import (Scene, Graph, Stroke, Write, Tex, Dot, Fill, ArrowItem,
                    BoxItem, C, INK, HAND, HAND2, bspline_basis,
                    open_uniform_knots, fit_bspline)

TRUE_C, PRED_C = C["blue"], C["pink"]
MSE_C, TQ_C, HN_C = C["grey"], C["orange"], C["green"]
LEVELC = [C["yellow"], C["green"], C["blue"], C["purple"], C["pink"]]


# ---------------------------------------------------------------- helpers ---
def header(sc, s, t0=0.1, color=C["yellow"], size=42):
    sc.add(Write((8, 8.28), s, size=size, color=color, ha="center", t0=t0))


def smooth_noise(n, seed, k=8):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    out = np.zeros(n)
    for i in range(1, k):
        out += rng.normal(0, 1) / i * np.sin(2 * np.pi * i * t +
                                             rng.uniform(0, 2 * np.pi))
    return out / 1.8


def ett_series(n=480, seed=5, x_end=192.0):
    """ETT-flavoured synthetic series over [0, x_end] 'hours'."""
    x = np.linspace(0, x_end, n)
    y = (0.85 * np.sin(2 * np.pi * x / 24 - 1.2)
         + 0.5 * np.sin(2 * np.pi * x / 96 + 0.6)
         + 0.22 * np.sin(2 * np.pi * x / 8 + 2.0)
         + 0.32 * smooth_noise(n, seed))
    return x, y


def net_doodle(sc, x0, y0, w, h, t0, color=INK, seed=11):
    """Tiny 3-layer neural net."""
    cols = [4, 3, 2]
    xs = np.linspace(x0 + 0.25, x0 + w - 0.25, len(cols))
    nodes = []
    for ci, (cx, k) in enumerate(zip(xs, cols)):
        ys = np.linspace(y0 + 0.3, y0 + h - 0.3, k)
        nodes.append([(cx, yy) for yy in ys])
    tt = t0
    for a_col, b_col in zip(nodes[:-1], nodes[1:]):
        for a in a_col:
            for b in b_col:
                sc.add(Stroke([a, b], color=C["dim"], lw=1.6, t0=tt,
                              dur=0.35, amp=0.012, glow=False, zorder=2))
        tt += 0.18
    for col in nodes:
        for p in col:
            sc.add(Dot(p, color=color, size=8, t0=tt, dur=0.3))
        tt += 0.1
    return tt


def spectrum_bars(sc, g, mags, color, t0, width=0.55, stagger=0.05, lw=3.0,
                  alpha=1.0, dx=0.0):
    """Hand-drawn vertical bars in graph g at integer x positions."""
    for i, m in enumerate(mags):
        x = i + dx
        sc.add(Stroke([(g.X(x), g.Y(0)), (g.X(x), g.Y(m))], color=color,
                      lw=lw, t0=t0 + i * stagger, dur=0.3, amp=0.015,
                      alpha=alpha))


# ------------------------------------------------------------------- s01 ---
def s01(sc, T):
    t = T.t
    # neural net predicting the future
    tn = net_doodle(sc, 1.0, 4.9, 2.9, 2.5, t("trained a neural network"))
    sc.add(ArrowItem((4.15, 6.15), (4.95, 6.15), color=C["grey"], t0=tn,
                     dur=0.4))
    g = Graph((5.1, 4.9, 2.9, 2.4), xlim=(0, 10), ylim=(-1.5, 1.5))
    x = np.linspace(0, 6, 120)
    y = np.sin(x * 1.4) + 0.3 * np.sin(x * 4)
    sc.add(Stroke(g.XY(x, y), color=TRUE_C, lw=3.0,
                  t0=t("predict the future"), dur=0.8, seed=2))
    x2 = np.linspace(6, 10, 80)
    y2 = np.sin(x2 * 1.4) + 0.3 * np.sin(x2 * 4)
    sc.add(Stroke(g.XY(x2, y2), color=PRED_C, lw=3.0, dashed=True,
                  t0=t("predict the future", 0.5), dur=0.8, seed=3))
    sc.add(Write((6.55, 4.35), "predict the future", size=24, color=C["grey"],
                 ha="center", t0=t("predict the future", 0.7)))

    # two example signals
    g2 = Graph((1.2, 2.9, 2.6, 1.1), xlim=(0, 1), ylim=(-1.2, 1.2))
    xs = np.linspace(0, 1, 160)
    sc.add(Stroke(g2.XY(xs, np.sin(xs * 12) * (0.6 + 0.4 * np.sin(xs * 3))),
                  color=C["teal"], lw=2.6, t0=t("temperature of a transformer"),
                  dur=0.9, seed=4))
    sc.add(Write((2.5, 2.45), "transformer temp", size=20, color=C["grey"],
                 ha="center", t0=t("temperature of a transformer", 0.3)))
    g3 = Graph((4.6, 2.9, 2.6, 1.1), xlim=(0, 1), ylim=(0, 1.3))
    yt = 0.35 + 0.5 * np.exp(-((xs % 0.34) - 0.17) ** 2 / 0.004)
    sc.add(Stroke(g3.XY(xs, yt), color=C["orange"], lw=2.6,
                  t0=t("traffic on a freeway"), dur=0.9, seed=5))
    sc.add(Write((5.9, 2.45), "freeway traffic", size=20, color=C["grey"],
                 ha="center", t0=t("traffic on a freeway", 0.3)))

    # the question
    sc.add(Write((11.9, 7.6), "how do you make it better?", size=30,
                 color=INK, ha="center", t0=t("really nice question")))
    sc.add(BoxItem(8.9, 5.9, 6.0, 1.05, color=C["grey"], t0=t("isn't a better architecture"),
                   dur=0.6))
    sc.add(Write((11.9, 6.25), "a better architecture?", size=27,
                 color=C["grey"], ha="center",
                 t0=t("isn't a better architecture", 0.1)))
    sc.add(BoxItem(8.9, 4.45, 6.0, 1.05, color=HN_C, t0=t("better definition"),
                   dur=0.6))
    sc.add(Write((11.9, 4.8), "a better loss function?", size=27, color=HN_C,
                 ha="center", t0=t("better definition", 0.1)))
    # ellipse highlight
    th = np.linspace(0, 2 * np.pi, 60)
    ex = 11.9 + 3.6 * np.cos(th)
    ey = 4.97 + 0.85 * np.sin(th)
    sc.add(Stroke(np.column_stack([ex, ey]), color=C["yellow"], lw=2.6,
                  t0=t("counts as a good forecast"), dur=0.8, amp=0.05))

    # title
    sc.add(Write((8, 1.95), "Hierarchical NURBS-Inspired Multi-Domain Loss",
                 size=38, color=INK, ha="center",
                 t0=t("Hierarchical NURBS-Inspired"), dur=2.6))
    sc.add(Write((8, 1.0), "\"HNMD\"", size=40, color=C["yellow"], ha="center",
                 t0=t("call it H-N-M-D")))
    sc.add(Stroke([(5.4, 0.62), (10.6, 0.58)], color=C["yellow"], lw=2.6,
                  t0=t("from scratch"), dur=0.5, amp=0.03))


# ------------------------------------------------------------------- s02 ---
def s02(sc, T):
    t = T.t
    header(sc, "Time-series forecasting", t0=t("What is a time series?"))
    g = Graph((1.3, 2.7, 10.4, 4.6), xlim=(-6, 200), ylim=(-2.1, 2.4))
    for it in g.axes_items(t("Here, I'll draw one")):
        sc.add(it)
    sc.add(Write((9.95, 2.28), "time (hours)", size=22, color=C["grey"],
                 t0=t("Here, I'll draw one", 0.4)))
    sc.add(Write((0.95, 7.0), "temp", size=22, color=C["grey"],
                 t0=t("Here, I'll draw one", 0.6), rotation=90))

    x, y = ett_series()
    m_look = x <= 96
    # dots on look-back
    xi = np.linspace(2, 94, 22)
    yi = np.interp(xi, x, y)
    t_dots = t("Each dot is one measurement")
    for i, (a, b) in enumerate(zip(xi, yi)):
        sc.add(Dot((g.X(a), g.Y(b)), color=TRUE_C, size=7,
                   t0=t_dots + i * 0.055))
    sc.add(Stroke(g.XY(x[m_look], y[m_look]), color=TRUE_C, lw=3.2,
                  t0=t("if we connect them"), dur=1.6, seed=21))

    # look-back region
    t_lb = t("window of the recent past")
    sc.add(Fill([(g.X(0), g.y0), (g.X(96), g.y0), (g.X(96), g.y0 + g.h),
                 (g.X(0), g.y0 + g.h)], color=HN_C, alpha=0.12, t0=t_lb))
    sc.add(Write((g.X(48), 2.15), "look-back window (96 steps)", size=23,
                 color=HN_C, ha="center", t0=t("look-back window")))
    # horizon region
    t_hz = t("draw the next chunk")
    sc.add(Fill([(g.X(96), g.y0), (g.X(192), g.y0), (g.X(192), g.y0 + g.h),
                 (g.X(96), g.y0 + g.h)], color=C["orange"], alpha=0.10,
                t0=t_hz))
    sc.add(Write((g.X(144), 2.15), "horizon (96 ... 720)", size=23,
                 color=C["orange"], ha="center", t0=t("the horizon")))
    sc.add(Write((g.X(141), 6.6), "?", size=46, color=C["orange"],
                 ha="center", t0=t_hz + 0.4))
    sc.add(Stroke(g.XY(x[~m_look], y[~m_look]), color=C["orange"], lw=3.0,
                  dashed=True, t0=t("seven hundred and twenty"), dur=1.4,
                  seed=22))

    # channels
    t_ch = t("many channels recorded together")
    sc.add(Write((13.9, 7.35), "channels", size=24, color=INK, ha="center",
                 t0=t_ch))
    chans = [("temp", TRUE_C, 6.2), ("load", C["orange"], 5.15),
             ("humidity", HN_C, 4.1)]
    xs = np.linspace(0, 1, 140)
    for i, (nm, col, yy) in enumerate(chans):
        gg = Graph((12.35, yy, 2.4, 0.8), xlim=(0, 1), ylim=(-1.4, 1.4))
        yv = np.sin(xs * (7 + 3 * i) + i * 2) * 0.8 + 0.4 * np.sin(xs * 2 + i)
        sc.add(Stroke(gg.XY(xs, yv), color=col, lw=2.3, t0=t_ch + 0.3 + i * 0.35,
                      dur=0.7, seed=30 + i))
        sc.add(Write((14.85, yy + 0.15), nm, size=18, color=col,
                     t0=t_ch + 0.5 + i * 0.35))
    # matrix
    t_mx = t("input is really a little matrix")
    mx, my, mw, mh = 12.35, 1.15, 2.6, 1.9
    sc.add(BoxItem(mx, my, mw, mh, color=INK, t0=t_mx, dur=0.6))
    for i in range(1, 4):
        sc.add(Stroke([(mx, my + mh * i / 4), (mx + mw, my + mh * i / 4)],
                      color=C["dim"], lw=1.5, t0=t_mx + 0.25, dur=0.4,
                      glow=False))
    for j in range(1, 6):
        sc.add(Stroke([(mx + mw * j / 6, my), (mx + mw * j / 6, my + mh)],
                      color=C["dim"], lw=1.5, t0=t_mx + 0.4, dur=0.4,
                      glow=False))
    sc.add(Write((mx + mw / 2, my - 0.42), "time ->", size=18, color=C["grey"],
                 ha="center", t0=t_mx + 0.6))
    sc.add(Write((mx - 0.25, my + mh / 2), "channels", size=18,
                 color=C["grey"], ha="center", va="center", rotation=90,
                 t0=t_mx + 0.7))
    # the game
    sc.add(ArrowItem((g.X(48), 3.1), (g.X(60), 3.9), color=HN_C,
                     t0=t("reads the green region"), dur=0.5, curve=-0.3))
    sc.add(ArrowItem((g.X(120), 6.4), (g.X(132), 5.6), color=C["orange"],
                     t0=t("produce the orange region"), dur=0.5, curve=-0.3))


# ------------------------------------------------------------------- s03 ---
def s03(sc, T):
    t = T.t
    header(sc, "How do we train it?", t0=t("how do we train"))
    # input matrix icon
    t_in = t("kept deliberately simple")
    sc.add(BoxItem(0.9, 5.5, 1.5, 1.3, color=HN_C, t0=t_in, dur=0.5))
    for i in range(1, 3):
        sc.add(Stroke([(0.9, 5.5 + 1.3 * i / 3), (2.4, 5.5 + 1.3 * i / 3)],
                      color=C["dim"], lw=1.4, t0=t_in + 0.2, dur=0.3,
                      glow=False))
    sc.add(Write((1.65, 5.1), "past (96)", size=19, color=HN_C, ha="center",
                 t0=t_in + 0.3))
    # MLP box
    t_mlp = t("plain three-layer M-L-P")
    sc.add(BoxItem(3.3, 4.85, 3.3, 2.6, color=INK, t0=t_mlp, dur=0.7))
    for i, xx in enumerate(np.linspace(3.85, 6.05, 3)):
        sc.add(BoxItem(xx, 5.25, 0.55, 1.8, color=C["purple"],
                       t0=t_mlp + 0.3 + i * 0.2, dur=0.5,
                       fill=C["purple"], fill_alpha=0.12))
    sc.add(Write((4.95, 4.4), "3-layer MLP", size=24, color=C["purple"],
                 ha="center", t0=t_mlp + 0.9))
    sc.add(ArrowItem((2.5, 6.15), (3.2, 6.15), color=C["grey"],
                     t0=t("takes the look-back window in"), dur=0.4))
    # output
    t_out = t("entire forecast in one shot")
    sc.add(ArrowItem((6.7, 6.15), (7.4, 6.15), color=C["grey"], t0=t_out,
                     dur=0.4))
    go = Graph((7.55, 5.5, 1.8, 1.4), xlim=(0, 1), ylim=(-1.3, 1.3))
    xs = np.linspace(0, 1, 90)
    sc.add(Stroke(go.XY(xs, np.sin(xs * 9) * 0.8 + 0.3 * np.sin(xs * 3)),
                  color=PRED_C, lw=2.6, t0=t_out + 0.2, dur=0.7, seed=8))
    sc.add(Write((8.45, 5.1), "forecast (96)", size=19, color=PRED_C,
                 ha="center", t0=t_out + 0.5))

    # loss diagram right
    t_two = t("takes two things")
    g = Graph((10.3, 4.75, 5.0, 2.7), xlim=(0, 1), ylim=(-1.6, 1.6))
    ys_true = np.sin(xs * 7.5) + 0.35 * np.sin(xs * 2.4 + 1)
    ys_pred = 0.85 * np.sin(xs * 7.5 + 0.45) + 0.3 * np.sin(xs * 2.4 + 1.3)
    sc.add(Stroke(g.XY(xs, ys_true), color=TRUE_C, lw=2.8, t0=t_two, dur=0.9,
                  seed=9))
    sc.add(Stroke(g.XY(xs, ys_pred), color=PRED_C, lw=2.8, t0=t_two + 0.4,
                  dur=0.9, seed=10))
    sc.add(Write((10.5, 7.5), "truth", size=20, color=TRUE_C, t0=t_two + 0.7))
    sc.add(Write((11.9, 7.5), "vs", size=20, color=C["grey"], t0=t_two + 0.8))
    sc.add(Write((12.5, 7.5), "prediction", size=20, color=PRED_C,
                 t0=t_two + 0.9))
    t_gap = t("measure the vertical gap")
    for i, xx in enumerate(np.linspace(0.06, 0.94, 12)):
        ya, yb = np.interp(xx, xs, ys_true), np.interp(xx, xs, ys_pred)
        sc.add(Stroke([(g.X(xx), g.Y(ya)), (g.X(xx), g.Y(yb))],
                      color=C["yellow"], lw=2.2, t0=t_gap + i * 0.06, dur=0.25,
                      amp=0.01, glow=False))
    sc.add(Tex((12.8, 3.85), r"$\mathrm{MSE}=\frac{1}{T}\sum_t\,(\hat{y}_t-y_t)^2$",
               size=30, color=C["yellow"], ha="center",
               t0=t("Square it"), dur=1.2))
    sc.add(Write((12.8, 3.1), "small = good,  big = bad", size=21,
                 color=C["grey"], ha="center", t0=t("Small number")))
    # gradient descent note
    t_gd = t("gradient descent on a loss function")
    sc.add(ArrowItem((4.95, 3.1), (4.95, 4.2), color=C["yellow"], t0=t_gd,
                     dur=0.5))
    sc.add(Write((4.95, 2.6), "gradients nudge the weights", size=20,
                 color=C["grey"], ha="center", t0=t_gd + 0.3))
    # the big quote
    t_q = t("model becomes whatever")
    sc.add(BoxItem(3.1, 0.75, 9.8, 1.5, color=C["yellow"], t0=t_q - 0.3,
                   dur=0.7, fill=C["yellow"], fill_alpha=0.06))
    sc.add(Write((8.0, 1.3), "the loss function is the model's definition of truth",
                 size=27, color=C["yellow"], ha="center", t0=t_q,
                 dur=2.4))


# ------------------------------------------------------------------- s04 ---
def s04(sc, T):
    t = T.t
    header(sc, "What MSE can't see", t0=t("wrong with mean squared error"))
    g = Graph((1.3, 3.5, 9.9, 3.9), xlim=(0, 4), ylim=(-1.7, 1.7))
    for it in g.axes_items(t("Here's a ground truth signal") - 0.3):
        sc.add(it)
    xs = np.linspace(0, 4, 400)
    y_true = np.sin(2 * np.pi * xs)
    sc.add(Stroke(g.XY(xs, y_true), color=TRUE_C, lw=3.2,
                  t0=t("strong daily rhythm"), dur=1.6, seed=40))
    sc.add(Write((g.X(0.22), g.Y(1.34)), "truth", size=21, color=TRUE_C,
                 t0=t("strong daily rhythm", 1.2)))
    sc.add(Write((g.X(2.0), 2.95), "days", size=20, color=C["grey"],
                 ha="center", t0=t("strong daily rhythm", 0.6)))
    # candidate A
    t_a = t("Candidate A")
    sc.add(Stroke(g.XY(xs, 0 * xs), color=MSE_C, lw=3.0, t0=t_a, dur=1.0,
                  seed=41))
    sc.add(Write((g.X(0.52), g.Y(-0.48)), "A: flat", size=21, color=MSE_C,
                 t0=t_a + 0.6))
    t_ae = t("actually pretty decent")
    for xx in np.linspace(0.55, 1.45, 5):
        sc.add(Stroke([(g.X(xx), g.Y(0)), (g.X(xx), g.Y(np.sin(2 * np.pi * xx)))],
                      color=C["yellow"], lw=2.0, t0=t_ae - 1.2 + xx * 0.3,
                      dur=0.3, amp=0.01, glow=False))
    sc.add(Write((12.0, 6.7), "MSE(A) = 0.50", size=26, color=MSE_C,
                 t0=t_ae))
    sc.add(Write((12.0, 6.15), "never terribly wrong", size=20,
                 color=C["grey"], t0=t_ae + 0.5))
    # candidate B
    t_b = t("Candidate B nails the shape")
    y_b = np.sin(2 * np.pi * (xs - 0.25))
    sc.add(Stroke(g.XY(xs, y_b), color=PRED_C, lw=3.0, t0=t_b, dur=1.4,
                  seed=42))
    sc.add(Write((g.X(0.05), g.Y(-1.52)), "B: shifted", size=21, color=PRED_C,
                 t0=t_b + 0.9))
    t_bp = t("pays full price")
    for xx in np.linspace(2.25, 3.25, 5):
        sc.add(Stroke([(g.X(xx), g.Y(np.sin(2 * np.pi * xx))),
                       (g.X(xx), g.Y(np.sin(2 * np.pi * (xx - 0.25))))],
                      color=C["red"], lw=2.0, t0=t_bp - 0.8 + (xx - 2.25) * 0.4,
                      dur=0.3, amp=0.01, glow=False))
    sc.add(Write((12.0, 5.35), "MSE(B) = 1.00", size=26, color=PRED_C,
                 t0=t_bp))
    sc.add(Write((12.0, 4.8), "right shape, wrong phase", size=20,
                 color=C["grey"], t0=t_bp + 0.5))
    # verdict
    t_v = t("prefer the flat, useless forecast")
    sc.add(BoxItem(11.6, 3.68, 4.1, 0.95, color=C["yellow"], t0=t_v - 0.3,
                   dur=0.5))
    sc.add(Write((13.65, 4.0), "MSE prefers A ?!", size=26, color=C["yellow"],
                 ha="center", t0=t_v))
    # drift doodle
    t_d = t("blurry, over-smoothed averages")
    gd = Graph((1.5, 1.15, 4.6, 1.5), xlim=(0, 1), ylim=(-1.3, 1.3))
    xs2 = np.linspace(0, 1, 150)
    sc.add(Stroke(gd.XY(xs2, np.sin(xs2 * 14) * np.exp(-2.2 * xs2)),
                  color=TRUE_C, lw=2.5, t0=t_d - 1.0, dur=1.2, seed=43))
    sc.add(Write((3.8, 0.75), "training with MSE flattens the forecast",
                 size=20, color=C["grey"], ha="center", t0=t_d))
    # prior art + new road
    t_p = t("shape-aware losses")
    sc.add(Write((7.3, 2.35), "prior fixes: DILATE, Tilde-Q", size=22,
                 color=TQ_C, t0=t_p))
    sc.add(Write((7.3, 1.8), "(build in invariance to shifts)", size=19,
                 color=C["grey"], t0=t_p + 0.6))
    t_r = t("different road")
    sc.add(BoxItem(11.35, 1.0, 4.5, 1.7, color=HN_C, t0=t_r, dur=0.6,
                   fill=HN_C, fill_alpha=0.07))
    sc.add(Write((13.6, 2.15), "HNMD:", size=23, color=HN_C, ha="center",
                 t0=t_r + 0.3))
    sc.add(Write((13.6, 1.6), "extract the structure,", size=21, color=INK,
                 ha="center", t0=t("extracts the structure")))
    sc.add(Write((13.6, 1.15), "then grade the structure", size=21, color=INK,
                 ha="center", t0=t("grades the structure")))


# ------------------------------------------------------------------- s05 ---
def s05(sc, T):
    t = T.t
    header(sc, "The recipe: three ingredients", t0=t("here's the plan"))
    xs = np.linspace(0, 1, 140)
    boxes = [
        (0.9, "1. splines", C["blue"], t("Ingredient one")),
        (5.8, "2. hierarchy", C["purple"], t("Ingredient two")),
        (10.7, "3. report card", C["pink"], t("ingredient three")),
    ]
    for x0, label, col, tt in boxes:
        sc.add(BoxItem(x0, 3.3, 4.4, 3.6, color=col, t0=tt, dur=0.7))
        sc.add(Write((x0 + 2.2, 6.35), label, size=28, color=col, ha="center",
                     t0=tt + 0.3))
    # doodle 1: dots + spline bumps
    t1 = t("built out of splines")
    g1 = Graph((1.3, 4.0, 3.6, 1.9), xlim=(0, 1), ylim=(-0.4, 1.5))
    yv = 0.5 + 0.45 * np.sin(xs * 6) + 0.2 * np.sin(xs * 13)
    for i, xx in enumerate(np.linspace(0.05, 0.95, 7)):
        sc.add(Dot((g1.X(xx), g1.Y(np.interp(xx, xs, yv))), color=INK, size=6,
                   t0=t1 + i * 0.05))
    kn = open_uniform_knots(6, 3)
    B = bspline_basis(xs, kn, 3)
    for i in range(B.shape[0]):
        sc.add(Stroke(g1.XY(xs, B[i] * 0.9 - 0.35), color=C["blue"], lw=1.9,
                      t0=t1 + 0.4 + i * 0.07, dur=0.5, glow=False))
    sc.add(Write((3.1, 3.6), "series -> structure", size=20, color=C["grey"],
                 ha="center", t0=t1 + 0.8))
    # doodle 2: coarse->fine stack
    t2 = t("splits into layers")
    for i, (col, f, a) in enumerate([(C["yellow"], 1.5, 0.75),
                                     (C["green"], 5, 0.5),
                                     (C["blue"], 13, 0.3)]):
        gg = Graph((6.2, 5.55 - i * 0.75, 3.6, 0.62), xlim=(0, 1),
                   ylim=(-1, 1))
        sc.add(Stroke(gg.XY(xs, a * np.sin(2 * np.pi * f * xs)), color=col,
                      lw=2.2, t0=t2 + i * 0.3, dur=0.7, glow=False,
                      seed=50 + i))
    sc.add(Write((8.0, 3.6), "big slow -> fine fast", size=20,
                 color=C["grey"], ha="center", t0=t2 + 1.0))
    # doodle 3: three lenses
    t3 = t("three different views")
    sc.add(Write((11.3, 5.55), "value", size=21, color=INK,
                 t0=t("Where the curve is")))
    g3a = Graph((12.9, 5.35, 1.9, 0.6), xlim=(0, 1), ylim=(-1, 1))
    sc.add(Stroke(g3a.XY(xs, 0.7 * np.sin(xs * 7)), color=C["pink"], lw=2.0,
                  t0=t("Where the curve is", 0.2), dur=0.5, glow=False))
    sc.add(Write((11.3, 4.75), "slope", size=21, color=INK,
                 t0=t("How it's moving")))
    sc.add(ArrowItem((12.95, 4.65), (13.9, 5.05), color=C["pink"],
                     t0=t("How it's moving", 0.2), dur=0.4))
    sc.add(ArrowItem((14.0, 4.95), (14.75, 4.6), color=C["pink"],
                     t0=t("How it's moving", 0.45), dur=0.4))
    sc.add(Write((11.3, 3.95), "rhythm", size=21, color=INK,
                 t0=t("what rhythms it contains")))
    g3c = Graph((12.9, 3.75, 1.9, 0.75), xlim=(-0.5, 5.5), ylim=(0, 1.2))
    spectrum_bars(sc, g3c, [0.25, 0.9, 0.2, 0.55, 0.15, 0.1], C["pink"],
                  t("what rhythms it contains", 0.2), stagger=0.08, lw=2.4)
    # chain arrows
    sc.add(ArrowItem((5.35, 5.1), (5.75, 5.1), color=C["grey"],
                     t0=t("Ingredient two"), dur=0.35))
    sc.add(ArrowItem((10.25, 5.1), (10.65, 5.1), color=C["grey"],
                     t0=t("ingredient three"), dur=0.35))
    # focus ring on box 1
    th = np.linspace(0, 2 * np.pi, 60)
    sc.add(Stroke(np.column_stack([3.1 + 2.6 * np.cos(th),
                                   5.1 + 2.15 * np.sin(th)]),
                  color=C["yellow"], lw=2.6, t0=t("talk about splines"),
                  dur=0.8, amp=0.05))
    sc.add(Write((8.0, 1.6), "first stop: what is a spline?", size=25,
                 color=C["yellow"], ha="center", t0=t("talk about splines", 0.5)))


# ------------------------------------------------------------------- s06 ---
def s06(sc, T):
    t = T.t
    header(sc, "Splines: flexible curves from simple pieces",
           t0=t("Forget time series"))
    t_gone = t("The mathematical version")
    # left: gentle data + low-degree fit works
    gl = Graph((1.0, 3.9, 6.6, 3.4), xlim=(0, 1), ylim=(-0.2, 1.4))
    xs = np.linspace(0, 1, 200)
    rng = np.random.default_rng(7)
    xd = np.linspace(0.05, 0.95, 7)
    yd = 0.55 + 0.35 * np.sin(xd * 3.6) - 0.25 * xd
    t_pts = t("handful of data points")
    for i, (a, b) in enumerate(zip(xd, yd)):
        sc.add(Dot((gl.X(a), gl.Y(b)), color=INK, size=8,
                   t0=t_pts + i * 0.07).until(t_gone))
    cf = np.polyfit(xd, yd, 2)
    sc.add(Stroke(gl.XY(xs, np.polyval(cf, xs)), color=HN_C, lw=3.0,
                  t0=t("fit a polynomial"), dur=1.1, seed=60).until(t_gone))
    sc.add(Write((4.3, 3.45), "low degree: fine for gentle data", size=21,
                 color=HN_C, ha="center",
                 t0=t("works fine")).until(t_gone))
    # right: many points + high-degree oscillation
    gr = Graph((8.6, 3.9, 6.6, 3.4), xlim=(-0.02, 1.02), ylim=(-1.3, 1.9))
    t_many = t("high-degree polynomials famously oscillate")
    xd2 = np.linspace(0, 1, 14)
    yd2 = 0.5 + 0.4 * np.sin(xd2 * 9) * np.exp(-1.2 * xd2) + 0.12 * rng.normal(size=14)
    for i, (a, b) in enumerate(zip(xd2, yd2)):
        sc.add(Dot((gr.X(a), gr.Y(b)), color=INK, size=7,
                   t0=t_many - 1.6 + i * 0.05).until(t_gone))
    cf2 = np.polyfit(xd2, yd2, 13)
    yy2 = np.clip(np.polyval(cf2, xs), -1.25, 1.85)
    sc.add(Stroke(gr.XY(xs, yy2), color=C["red"], lw=3.0, t0=t_many,
                  dur=1.4, seed=61).until(t_gone))
    sc.add(Write((11.9, 3.45), "degree 13: wild edge oscillation", size=21,
                 color=C["red"], ha="center",
                 t0=t("goes wild over there")).until(t_gone))
    th = np.linspace(0, 2 * np.pi, 50)
    sc.add(Stroke(np.column_stack([gr.X(0.94) + 0.75 * np.cos(th),
                                   gr.Y(0.6) + 1.05 * np.sin(th)]),
                  color=C["red"], lw=2.4, amp=0.05,
                  t0=t("goes wild over there", 0.5), dur=0.7).until(t_gone))
    sc.add(Write((8.0, 2.65), "every coefficient bends the WHOLE curve",
                 size=22, color=C["grey"], ha="center",
                 t0=t("entire curve at once")).until(t_gone))
    # ship doodle
    t_ship = t("shipbuilders")
    pegs = [(13.0, 7.6), (13.8, 7.95), (14.6, 7.85), (15.3, 7.45)]
    bx = np.linspace(12.7, 15.5, 60)
    by = 7.35 + 0.62 * np.sin((bx - 12.7) / 2.8 * np.pi)
    sc.add(Stroke(np.column_stack([bx, by]), color=C["orange"], lw=2.6,
                  t0=t_ship, dur=0.8, seed=62))
    for i, p in enumerate(pegs):
        sc.add(Dot(p, color=INK, size=6, t0=t_ship + 0.5 + i * 0.08))
    sc.add(Write((14.1, 6.85), "a real 'spline'", size=18, color=C["grey"],
                 ha="center", t0=t_ship + 0.8))

    # the spline construction (center, appears after erase)
    g = Graph((1.6, 1.4, 12.4, 4.9), xlim=(0, 1), ylim=(-1.35, 1.55))
    knots = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    t_kn = t("chosen locations called knots")
    for i, k in enumerate(knots):
        sc.add(Dot((g.X(k), g.Y(-1.2)), color=C["orange"], size=9,
                   t0=t_kn + i * 0.12))
        sc.add(Stroke([(g.X(k), g.Y(-1.2)), (g.X(k), g.Y(1.45))],
                      color=C["dim"], lw=1.5, dashed=True, glow=False,
                      t0=t_kn + 0.1 + i * 0.12, dur=0.4))
    sc.add(Write((g.X(0.5), 1.0), "knots", size=22, color=C["orange"],
                 ha="center", t0=t_kn + 0.9))
    # smooth curve in colored pieces
    t_pc = t("usually a cubic")
    yy = 0.55 * np.sin(2 * np.pi * 1.15 * xs + 0.3) + 0.35 * np.sin(2 * np.pi * 2.6 * xs + 1.4)
    cols = [C["blue"], C["green"], C["purple"], C["yellow"], C["pink"]]
    for i in range(5):
        m = (xs >= knots[i]) & (xs <= knots[i + 1] + 1e-9)
        sc.add(Stroke(g.XY(xs[m], yy[m]), color=cols[i], lw=3.4,
                      t0=t_pc + i * 0.35, dur=0.5, seed=63 + i))
        sc.add(Write((g.X((knots[i] + knots[i + 1]) / 2), g.Y(1.15)),
                     "cubic", size=17, color=cols[i], ha="center",
                     t0=t_pc + 0.2 + i * 0.35))
    # seam highlight
    t_seam = t("seams become invisible")
    kx, ky = g.X(0.6), g.Y(float(np.interp(0.6, xs, yy)))
    sc.add(Stroke(np.column_stack([kx + 0.5 * np.cos(th), ky + 0.5 * np.sin(th)]),
                  color=C["yellow"], lw=2.4, amp=0.04, t0=t_seam, dur=0.6))
    sc.add(Write((kx + 0.7, ky - 0.85), "same value, slope, curvature",
                 size=20, color=C["yellow"], t0=t_seam + 0.4))


# ------------------------------------------------------------------- s07 ---
def s07(sc, T):
    t = T.t
    header(sc, "The Cox-de Boor recursion", t0=t("the B in B-splines"))
    xs = np.linspace(0, 1, 400)
    knots = np.linspace(0, 1, 9)
    rows = [
        ("degree 0: blocks", 0, t("Degree zero"), 6.35),
        ("degree 1: tents", 1, t("melt into little tents"), 4.75),
        ("degree 2: bumps", 2, t("Degree two"), 3.15),
        ("degree 3: cubic", 3, t("cubic bumps"), 1.55),
    ]
    cols = [C["blue"], C["green"], C["yellow"], C["pink"], C["purple"],
            C["teal"], C["orange"], C["red"]]
    for label, deg, tt, y0 in rows:
        g = Graph((2.6, y0, 9.7, 1.28), xlim=(0, 1), ylim=(0, 1.12))
        sc.add(Stroke([(g.x0 - 0.15, g.y0), (g.x0 + g.w + 0.2, g.y0)],
                      color=C["dim"], lw=1.8, glow=False, t0=tt, dur=0.4))
        if deg == 0:
            for i, k in enumerate(knots):
                sc.add(Dot((g.X(k), g.Y(0)), color=C["orange"], size=6,
                           t0=tt + 0.1 + i * 0.05))
        B = bspline_basis(xs, knots, deg)
        for i in range(B.shape[0]):
            col = cols[i % len(cols)]
            if deg == 0:
                # crisp boxes drawn as 3-sided outline
                x0k, x1k = knots[i], knots[i + 1]
                pts = [(g.X(x0k), g.Y(0)), (g.X(x0k), g.Y(0.9)),
                       (g.X(x1k), g.Y(0.9)), (g.X(x1k), g.Y(0))]
                sc.add(Stroke(pts, color=col, lw=2.4, glow=False,
                              t0=tt + 0.25 + i * 0.1, dur=0.45, amp=0.02))
            else:
                m = B[i] > 1e-4
                sc.add(Stroke(g.XY(xs[m], B[i][m] * (0.95 if deg == 1 else 1.25)),
                              color=col, lw=2.4, glow=False,
                              t0=tt + 0.25 + i * 0.08, dur=0.5,
                              seed=70 + deg * 10 + i))
        sc.add(Write((0.7, y0 + 0.55), label, size=22,
                     color=INK, t0=tt + 0.15))
        if deg < 3:
            sc.add(ArrowItem((12.75, y0 + 0.35), (12.75, y0 - 0.9),
                             color=C["grey"], t0=rows[deg + 1][2] - 0.25,
                             dur=0.45, curve=0.25))
            sc.add(Write((13.05, y0 - 0.45), "blend", size=19, color=C["grey"],
                         t0=rows[deg + 1][2] - 0.1))
    # local support highlight on degree-3 row
    t_loc = t("local support")
    g3 = Graph((2.6, 1.55, 9.7, 1.28), xlim=(0, 1), ylim=(0, 1.12))
    B3 = bspline_basis(xs, knots, 3)
    i_mid = B3.shape[0] // 2
    supp = xs[B3[i_mid] > 1e-4]
    x0s, x1s = supp.min(), supp.max()
    sc.add(Fill([(g3.X(x0s), g3.y0), (g3.X(x1s), g3.y0),
                 (g3.X(x1s), g3.y0 + g3.h + 0.15),
                 (g3.X(x0s), g3.y0 + g3.h + 0.15)], color=C["yellow"],
                alpha=0.15, t0=t_loc))
    sc.add(Stroke(g3.XY(xs[B3[i_mid] > 1e-4], B3[i_mid][B3[i_mid] > 1e-4] * 1.25),
                  color=C["yellow"], lw=3.6, t0=t_loc + 0.2, dur=0.6))
    sc.add(Write((13.15, 2.3), "one bump lives on a", size=20,
                 color=C["yellow"], t0=t("Nudge one bump")))
    sc.add(Write((13.15, 1.85), "few intervals only ->", size=20,
                 color=C["yellow"], t0=t("Nudge one bump", 0.35)))
    sc.add(Write((13.15, 1.4), "edits stay local", size=20,
                 color=C["yellow"], t0=t("changes only in that neighborhood")))


# ------------------------------------------------------------------- s08 ---
def s08(sc, T):
    t = T.t
    header(sc, "A curve = a weighted sum of bumps", t0=0.15)
    xs = np.linspace(0, 1, 300)
    target = (0.55 * np.sin(2 * np.pi * 1.05 * xs + 0.4)
              + 0.3 * np.sin(2 * np.pi * 2.7 * xs + 1.2))
    n_ctrl = 9
    yhat, B, coef = fit_bspline(xs, target, n_ctrl, 3)
    cols = [C["blue"], C["green"], C["yellow"], C["pink"], C["purple"],
            C["teal"], C["orange"], C["red"], C["blue"]]
    # bottom: raw basis + coefficient lollipops
    gb = Graph((1.3, 1.1, 8.9, 1.8), xlim=(0, 1), ylim=(-0.05, 1.15))
    t_b = t("pile of bumps")
    sc.add(Write((0.75, 2.15), "bumps", size=20, color=C["grey"], t0=t_b,
                 rotation=90))
    for i in range(B.shape[0]):
        m = B[i] > 1e-4
        sc.add(Stroke(gb.XY(xs[m], B[i][m]), color=cols[i], lw=2.0,
                      glow=False, t0=t_b + i * 0.06, dur=0.5, seed=80 + i))
    t_c = t("Give every bump a coefficient")
    for i in range(n_ctrl):
        peak = xs[np.argmax(B[i])]
        sc.add(Write((gb.X(peak), gb.Y(1.02) + 0.28), f"c{i+1}", size=16,
                     color=cols[i], ha="center", t0=t_c + i * 0.08))
    # middle: scaled bumps
    gm = Graph((1.3, 3.35, 8.9, 2.0), xlim=(0, 1), ylim=(-0.75, 0.75))
    t_s = t("Stretch each bump")
    sc.add(Write((0.75, 4.35), "scaled", size=20, color=C["grey"], t0=t_s,
                 rotation=90))
    for i in range(B.shape[0]):
        m = B[i] > 1e-4
        sc.add(Stroke(gm.XY(xs[m], coef[i] * B[i][m]), color=cols[i], lw=2.0,
                      glow=False, t0=t_s + i * 0.06, dur=0.5, seed=90 + i))
    sc.add(ArrowItem((0.85, 3.0), (0.85, 3.5), color=C["grey"], t0=t_s,
                     dur=0.4))
    # top: sum = curve through data
    gt = Graph((1.3, 5.75, 8.9, 2.0), xlim=(0, 1), ylim=(-1.1, 1.1))
    t_sum = t("add them all up")
    sc.add(ArrowItem((0.85, 5.4), (0.85, 5.9), color=C["grey"], t0=t_sum,
                     dur=0.4))
    sc.add(Write((0.68, 6.75), "sum", size=20, color=C["grey"], t0=t_sum,
                 rotation=90))
    for i, xx in enumerate(np.linspace(0.04, 0.96, 10)):
        sc.add(Dot((gt.X(xx), gt.Y(np.interp(xx, xs, target))), color=INK,
                   size=6, t0=t_sum + 0.2 + i * 0.05))
    sc.add(Stroke(gt.XY(xs, yhat), color=HN_C, lw=3.2, t0=t_sum + 0.5,
                  dur=1.4, seed=99))
    sc.add(Write((5.75, 7.95), "smooth curve through the data", size=21,
                 color=HN_C, ha="center", t0=t_sum + 1.6))
    # right column: it's linear regression
    t_lr = t("It's just linear regression")
    sc.add(BoxItem(10.9, 5.6, 4.7, 1.9, color=C["yellow"], t0=t_lr - 0.2,
                   dur=0.6))
    sc.add(Write((13.25, 6.95), "this is linear regression!", size=23,
                 color=C["yellow"], ha="center", t0=t_lr))
    sc.add(Write((13.25, 6.4), "features = the bumps", size=21, color=INK,
                 ha="center", t0=t("The bumps are the features")))
    sc.add(Write((13.25, 5.9), "targets = the series", size=21, color=INK,
                 ha="center", t0=t("The bumps are the features", 0.6)))
    sc.add(Tex((13.25, 4.7), r"$\hat{y}(t)=\sum_i c_i\,B_i(t)$", size=27,
               color=INK, ha="center", t0=t("collect the results")))
    t_ne = t("N-transpose N")
    sc.add(Tex((13.25, 3.7), r"$c=(N^{\top}N)^{-1}N^{\top}y$", size=29,
               color=HN_C, ha="center", t0=t_ne, dur=1.2))
    t_sp = t("One matrix solve")
    sc.add(Write((13.25, 2.75), "one matrix solve --", size=22, color=INK,
                 ha="center", t0=t_sp))
    sc.add(Write((13.25, 2.3), "no gradient descent", size=22, color=INK,
                 ha="center", t0=t_sp + 0.5))
    sc.add(Write((13.25, 1.55), "fast enough to run inside", size=20,
                 color=C["grey"], ha="center", t0=t("speed matters")))
    sc.add(Write((13.25, 1.1), "the loss, every batch", size=20,
                 color=C["grey"], ha="center", t0=t("speed matters", 0.5)))


BUILDERS_A = {"s01": s01, "s02": s02, "s03": s03, "s04": s04, "s05": s05,
              "s06": s06, "s07": s07, "s08": s08}
