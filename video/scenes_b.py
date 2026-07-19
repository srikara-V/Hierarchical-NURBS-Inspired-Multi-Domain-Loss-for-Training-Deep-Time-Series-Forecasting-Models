"""Scenes 9-16 for the HNMD explainer."""
import numpy as np
from engine import (Scene, Graph, Stroke, Write, Tex, Dot, Fill, ArrowItem,
                    BoxItem, C, INK, bspline_basis, open_uniform_knots,
                    fit_bspline)
from scenes_a import (header, smooth_noise, ett_series, spectrum_bars,
                      TRUE_C, PRED_C, MSE_C, TQ_C, HN_C, LEVELC)


def nurbs_curve_2d(ctrl, weights, n=260):
    """Parametric NURBS curve through 2-D control points."""
    ctrl = np.asarray(ctrl, float)
    k = len(ctrl)
    ts = np.linspace(0, 1, n)
    kn = open_uniform_knots(k, 3)
    B = bspline_basis(ts, kn, 3)
    W = B * np.asarray(weights)[:, None]
    R = W / (W.sum(axis=0, keepdims=True) + 1e-12)
    return (R.T @ ctrl)


# ------------------------------------------------------------------- s09 ---
def s09(sc, T):
    t = T.t
    header(sc, "NURBS: give every bump a weight",
           t0=t("we arrive at NURBS"))
    sc.add(Write((8, 7.62), "Non-Uniform Rational B-Spline", size=25,
                 color=C["grey"], ha="center",
                 t0=t("Non-Uniform Rational B-Spline")))
    sc.add(Tex((12.9, 6.35),
               r"$R_i(t)=\frac{w_i\,B_i(t)}{\sum_j w_j\,B_j(t)}$",
               size=33, color=C["yellow"], ha="center",
               t0=t("divide by the sum"), dur=1.4))
    sc.add(Write((12.9, 5.35), "weighted share of each bump", size=20,
                 color=C["grey"], ha="center", t0=t("adds up to one")))

    # control polygon + three curves
    ctrl = np.array([[1.6, 2.2], [2.6, 4.9], [4.1, 5.9], [5.6, 6.3],
                     [6.9, 3.2], [8.4, 2.1], [9.9, 3.4], [10.8, 5.2]])
    t_poly = t("Rational is the interesting part")
    sc.add(Stroke(ctrl, color=C["dim"], lw=1.8, dashed=True, glow=False,
                  t0=t_poly, dur=1.0))
    for i, p in enumerate(ctrl):
        sc.add(Dot(p, color=INK, size=7, t0=t_poly + 0.2 + i * 0.07))
    sc.add(Write((2.05, 5.6), "control points", size=20, color=C["grey"],
                 t0=t_poly + 0.7))
    w1 = np.ones(len(ctrl))
    sc.add(Stroke(nurbs_curve_2d(ctrl, w1), color=TRUE_C, lw=3.2,
                  t0=t("own weight"), dur=1.3, seed=90))
    # highlight P4 (index 3)
    star = ctrl[3]
    th = np.linspace(0, 2 * np.pi, 40)
    t_up = t("crank up the weight")
    sc.add(Stroke(np.column_stack([star[0] + 0.34 * np.cos(th),
                                   star[1] + 0.34 * np.sin(th)]),
                  color=C["yellow"], lw=2.4, amp=0.03, t0=t_up, dur=0.5))
    w_hi = w1.copy(); w_hi[3] = 9.0
    sc.add(Stroke(nurbs_curve_2d(ctrl, w_hi), color=PRED_C, lw=3.2,
                  t0=t("pulled toward that control point"), dur=1.3, seed=91))
    sc.add(ArrowItem((star[0] + 1.15, star[1] - 1.35), (star[0] + 0.18,
                     star[1] - 0.42), color=PRED_C,
                     t0=t("more gravity"), dur=0.5, curve=0.3))
    w_lo = w1.copy(); w_lo[3] = 0.07
    sc.add(Stroke(nurbs_curve_2d(ctrl, w_lo), color=C["teal"], lw=3.0,
                  t0=t("barely listens"), dur=1.3, seed=92))
    # legend
    t_leg = t("crank up the weight")
    for i, (txt, col, tt) in enumerate([
            ("w = 1   everywhere", TRUE_C, t("own weight", 0.8)),
            ("w = 9   on that point", PRED_C, t("pulled toward that control point", 0.5)),
            ("w = 0.1 on that point", C["teal"], t("barely listens", 0.4))]):
        y = 4.55 - i * 0.55
        sc.add(Stroke([(11.35, y + 0.1), (12.05, y + 0.1)], color=col, lw=3.2,
                      t0=tt, dur=0.3, glow=False))
        sc.add(Write((12.2, y), txt, size=20, color=col, t0=tt + 0.15))
    # CAD note
    t_cad = t("CAD and 3D modeling")
    cx = np.array([12.0, 12.3, 12.8, 13.5, 14.2, 14.8, 15.2, 15.4])
    cy = np.array([1.15, 1.6, 1.95, 2.1, 2.0, 1.7, 1.35, 1.15])
    sc.add(Stroke(np.column_stack([cx, cy]), color=C["orange"], lw=2.4,
                  t0=t_cad, dur=0.8, seed=93))
    for wx in (13.0, 14.5):
        sc.add(Stroke(np.column_stack([wx + 0.28 * np.cos(th),
                                       1.15 + 0.28 * np.sin(th)]),
                      color=C["orange"], lw=2.2, t0=t_cad + 0.6, dur=0.4))
    sc.add(Write((13.7, 0.62), "CAD, fonts, 3D surfaces", size=19,
                 color=C["orange"], ha="center", t0=t_cad + 0.8))
    # teaser
    sc.add(Write((5.0, 1.05), "classic NURBS: optimize the weights...",
                 size=22, color=C["grey"], ha="center",
                 t0=t("you optimize them")))
    sc.add(Write((5.0, 0.55), "this paper computes them from the data",
                 size=22, color=C["yellow"], ha="center",
                 t0=t("something sneakier")))


# ------------------------------------------------------------------- s10 ---
def s10(sc, T):
    t = T.t
    header(sc, "Weights = attention over the series",
           t0=t("Here's the twist"))
    # signal: calm -> stormy -> calm
    n = 500
    xs = np.linspace(0, 1, n)
    calm = 0.35 * np.sin(2 * np.pi * 2 * xs)
    burst = (np.exp(-((xs - 0.55) ** 2) / 0.012) *
             0.9 * np.sin(2 * np.pi * 26 * xs))
    y = calm + burst + 0.05 * smooth_noise(n, 3)
    g = Graph((1.3, 5.2, 10.6, 2.5), xlim=(0, 1), ylim=(-1.5, 1.5))
    sc.add(Stroke(g.XY(xs, y), color=TRUE_C, lw=2.8,
                  t0=t("fits a spline"), dur=1.6, seed=100))
    sc.add(Write((0.85, 6.45), "series", size=20, color=C["grey"],
                 rotation=90, t0=t("fits a spline", 0.4)))
    # crossed-out optimization
    t_no = t("does not optimize the weights")
    sc.add(Write((13.6, 7.0), "optimize w ?", size=24, color=C["grey"],
                 ha="center", t0=t_no))
    sc.add(Stroke([(12.35, 6.75), (14.85, 7.35)], color=C["red"], lw=3.4,
                  t0=t_no + 0.7, dur=0.4))
    sc.add(Write((13.6, 6.3), "compute w from data", size=23, color=HN_C,
                 ha="center", t0=t("It computes them")))
    # sliding windows
    t_w = t("Slide a small window")
    for (xc, col, lab, tt) in [(0.16, HN_C, "var small", t("low variance")),
                               (0.55, C["red"], "var BIG", t("high variance"))]:
        x0, x1 = g.X(xc - 0.07), g.X(xc + 0.07)
        sc.add(BoxItem(x0, g.y0 + 0.1, x1 - x0, g.h - 0.2, color=col,
                       t0=t_w if xc < 0.3 else t_w + 0.8, dur=0.6))
        sc.add(Write(((x0 + x1) / 2, g.y0 - 0.35), lab, size=19, color=col,
                     ha="center", t0=tt))
    # variance profile
    win = 25
    pad = np.pad(y, (win // 2, win // 2), mode="edge")
    var = np.array([pad[i:i + win].var() for i in range(n)])
    gv = Graph((1.3, 3.15, 10.6, 1.45), xlim=(0, 1), ylim=(0, var.max() * 1.1))
    t_v = t("How turbulent is the signal")
    sc.add(Stroke(gv.XY(xs, var), color=C["orange"], lw=2.8, t0=t_v + 0.6,
                  dur=1.4, seed=101))
    sc.add(Write((0.85, 3.6), "window", size=17, color=C["orange"],
                 rotation=90, t0=t_v + 0.8))
    sc.add(Write((12.15, 3.95), "sliding-window", size=20, color=C["orange"],
                 t0=t_v + 1.0))
    sc.add(Write((12.15, 3.5), "variance", size=20, color=C["orange"],
                 t0=t_v + 1.15))
    # softmax bars
    t_s = t("through a softmax")
    nb = 24
    centers = np.linspace(0.02, 0.98, nb)
    vs = np.interp(centers, xs, var)
    wgt = np.exp(vs / (vs.max() * 0.35))
    wgt = wgt / wgt.sum()
    gb = Graph((1.3, 0.95, 10.6, 1.7), xlim=(0, 1), ylim=(0, wgt.max() * 1.15))
    for i, (cx, wv) in enumerate(zip(centers, wgt)):
        sc.add(Stroke([(gb.X(cx), gb.Y(0)), (gb.X(cx), gb.Y(wv))],
                      color=HN_C, lw=4.0, t0=t_s + i * 0.05, dur=0.3,
                      amp=0.012, glow=False))
    sc.add(Write((0.85, 1.3), "weights", size=17, color=HN_C, rotation=90,
                 t0=t_s + 0.4))
    sc.add(ArrowItem((11.6, 3.4), (11.6, 2.2), color=C["grey"],
                     t0=t_s, dur=0.5, curve=0.25))
    sc.add(Write((12.15, 2.7), "softmax", size=21, color=HN_C, t0=t_s + 0.3))
    # attention analogy
    t_a = t("attention works in a transformer")
    sc.add(BoxItem(12.0, 0.85, 3.7, 1.5, color=C["purple"], t0=t_a - 0.2,
                   dur=0.6, fill=C["purple"], fill_alpha=0.07))
    sc.add(Write((13.85, 1.85), "just like attention:", size=21,
                 color=C["purple"], ha="center", t0=t_a))
    sc.add(Write((13.85, 1.35), "softmax picks where", size=20, color=INK,
                 ha="center", t0=t_a + 0.5))
    sc.add(Write((13.85, 0.95), "to focus", size=20, color=INK, ha="center",
                 t0=t_a + 0.8))
    # focus arrows
    t_f = t("spends its flexibility")
    sc.add(ArrowItem((gb.X(0.55), gb.Y(wgt.max()) + 0.15), (g.X(0.55), g.y0 - 0.05),
                     color=C["yellow"], t0=t_f, dur=0.7, curve=-0.6))
    sc.add(Write((g.X(0.72), 4.6), "flexibility goes here", size=20,
                 color=C["yellow"], t0=t_f + 0.5))


# ------------------------------------------------------------------- s11 ---
def s11(sc, T):
    t = T.t
    header(sc, "Peel the onion: coarse to fine", t0=t("hierarchical part"))
    xs = np.linspace(0, 1, 520)
    drift = 0.85 * np.sin(2 * np.pi * 0.55 * xs - 0.7)
    daily = 0.42 * np.sin(2 * np.pi * 6 * xs + 0.4)
    rng = np.random.default_rng(11)
    spikes = np.zeros_like(xs)
    for cx, s in [(0.22, 1), (0.47, -1), (0.71, 1), (0.9, 1)]:
        spikes += s * 0.5 * np.exp(-((xs - cx) ** 2) / 0.0006)
    noise = 0.06 * smooth_noise(len(xs), 12, k=14)
    total = drift + daily + spikes + noise
    # left: ingredient list
    comps = [("slow seasonal drift", drift, C["yellow"],
              t("slow seasonal drift")),
             ("+ daily cycle", daily, C["green"], t("daily work cycle")),
             ("+ sharp spikes", spikes, C["red"], t("Sharp load spikes")),
             ("+ noise", noise * 6, C["grey"], t("And then noise"))]
    for i, (lab, yy, col, tt) in enumerate(comps):
        gg = Graph((0.9, 6.25 - i * 1.42, 4.3, 1.05), xlim=(0, 1),
                   ylim=(-1.1, 1.1))
        sc.add(Stroke(gg.XY(xs, yy), color=col, lw=2.2, glow=False, t0=tt,
                      dur=0.9, seed=110 + i))
        sc.add(Write((5.35, 6.55 - i * 1.42), lab, size=20, color=col,
                     t0=tt + 0.2))
    sc.add(Write((3.05, 7.55), "one signal, many processes", size=22,
                 color=INK, ha="center", t0=t("pile of overlapping processes")))
    # right: iterative peeling (real fits)
    f1, _, _ = fit_bspline(xs, total, 5, 3)
    r1 = total - f1
    f2, _, _ = fit_bspline(xs, r1, 12, 3)
    r2 = r1 - f2
    f3, _, _ = fit_bspline(xs, r2, 26, 3)
    rows = [
        ("level 1: few knots", total, f1, LEVELC[0], t("Level one"),
         "captures the slow trend"),
        ("level 2: more knots", r1, f2, LEVELC[1], t("Level two"),
         "fits what level 1 missed"),
        ("level 3: finer still", r2, f3, LEVELC[2],
         t("each level more flexible"), "chases the fine wiggles"),
    ]
    for i, (lab, raw, fit, col, tt, sub) in enumerate(rows):
        m_amp = 1.18 * max(np.abs(raw).max(), np.abs(fit).max())
        gg = Graph((8.6, 5.55 - i * 2.1, 6.7, 1.55), xlim=(0, 1),
                   ylim=(-m_amp, m_amp))
        sc.add(Stroke(gg.XY(xs, raw), color=C["dim"], lw=2.0, glow=False,
                      t0=tt, dur=0.8, seed=120 + i))
        sc.add(Stroke(gg.XY(xs, fit), color=col, lw=2.8, t0=tt + 0.7,
                      dur=1.0, seed=130 + i))
        sc.add(Write((8.65, 5.35 - i * 2.1 + 1.78), lab, size=20, color=col,
                     t0=tt + 0.2))
    sc.add(Write((11.9, 7.6), "fit -> subtract -> repeat", size=22,
                 color=INK, ha="center", t0=t("peel the onion")))
    t_res = t("called the residual")
    sc.add(ArrowItem((11.9, 5.5), (11.9, 5.15), color=C["grey"], t0=t_res,
                     dur=0.4))
    sc.add(Write((13.3, 5.32), "residual", size=19, color=C["grey"],
                 t0=t_res + 0.2))
    # boosting note
    t_b = t("gradient boosting")
    sc.add(BoxItem(8.6, 0.62, 6.7, 0.78, color=TQ_C, t0=t_b - 0.2, dur=0.5))
    sc.add(Write((11.95, 0.88), "same soul as boosting: fit the leftovers",
                 size=20, color=TQ_C, ha="center", t0=t_b))
    # equations
    t_f = t("Formally")
    sc.add(Tex((2.0, 1.6), r"$R_0=y$", size=25, color=INK, ha="left",
               t0=t_f))
    sc.add(Tex((3.6, 1.6), r"$R_{\ell}=R_{\ell-1}-\hat{f}_{\ell}$", size=25,
               color=INK, ha="left", t0=t("hands the remainder down")))
    sc.add(Tex((2.0, 0.8), r"$y=\sum_{\ell}\hat{f}_{\ell}+R_{M}$", size=25,
               color=C["yellow"], ha="left", t0=t("sum of all the levels")))


# ------------------------------------------------------------------- s12 ---
def s12(sc, T):
    t = T.t
    header(sc, "The decomposition machine  S", t0=t("assemble the full"))
    steps = [
        ("input series y   ->   residual R = y", C["grey"],
         t("Set the residual equal to it"), 7.3),
        ("1. lay down knots  (denser every level)", LEVELC[0],
         t("Lay down this level's knots"), 6.35),
        ("2. weights = softmax( window variance of R )", LEVELC[1],
         t("Compute this level's weights"), 5.4),
        ("3. build the weighted NURBS basis", LEVELC[2],
         t("weighted NURBS basis"), 4.45),
        ("4. least-squares solve  ->  coefficients", LEVELC[3],
         t("least-squares solve"), 3.5),
        ("5. component = basis x coefs ;  R = R - component", LEVELC[4],
         t("Subtract the component"), 2.55),
    ]
    for i, (lab, col, tt, y0) in enumerate(steps):
        sc.add(BoxItem(0.9, y0, 7.6, 0.8, color=col, t0=tt - 0.15, dur=0.5))
        sc.add(Write((1.2, y0 + 0.26), lab, size=21, color=col, t0=tt))
        if i > 0:
            sc.add(ArrowItem((4.7, y0 + 0.95), (4.7, y0 + 0.82),
                             color=C["dim"], t0=tt - 0.1, dur=0.25, head=0.1))
    # loop arrow
    t_loop = t("carry the residual down")
    sc.add(ArrowItem((8.6, 2.95), (8.6, 6.5), color=C["yellow"], lw=3.0,
                     t0=t_loop, dur=0.9, curve=-1.0))
    sc.add(Write((9.7, 4.7), "repeat,", size=21, color=C["yellow"],
                 t0=t_loop + 0.5))
    sc.add(Write((9.7, 4.25), "5 levels max", size=21, color=C["yellow"],
                 t0=t("five levels")))
    t_stop = t("basically flat")
    sc.add(Write((9.7, 3.7), "(or stop early if", size=19, color=C["grey"],
                 t0=t_stop))
    sc.add(Write((9.7, 3.3), "residual is flat)", size=19, color=C["grey"],
                 t0=t_stop + 0.3))
    # knot density doodle
    t_k = t("denser grid than the last")
    sc.add(Write((13.3, 7.55), "knots per level", size=20, color=INK,
                 ha="center", t0=t_k))
    for li in range(3):
        nk = [5, 9, 17][li]
        y = 7.0 - li * 0.55
        for k in np.linspace(11.6, 15.2, nk):
            sc.add(Dot((k, y), color=LEVELC[li], size=5,
                       t0=t_k + 0.2 + li * 0.3 + (k - 11.6) * 0.02))
        sc.add(Stroke([(11.5, y), (15.3, y)], color=C["dim"], lw=1.4,
                      glow=False, t0=t_k + 0.1 + li * 0.3, dur=0.3))
    # output stack
    t_o = t("stack of curves")
    xs = np.linspace(0, 1, 200)
    sc.add(Write((13.3, 5.15), "output: the stack", size=21, color=INK,
                 ha="center", t0=t_o))
    for li in range(5):
        gg = Graph((11.6, 4.35 - li * 0.62, 3.4, 0.5), xlim=(0, 1),
                   ylim=(-1, 1))
        f = [0.7, 1.5, 3.2, 6.5, 13][li]
        a = [0.95, 0.8, 0.6, 0.45, 0.3][li]
        sc.add(Stroke(gg.XY(xs, a * np.sin(2 * np.pi * f * xs + li)),
                      color=LEVELC[li], lw=2.2, glow=False,
                      t0=t_o + 0.2 + li * 0.18, dur=0.6, seed=140 + li))
    t_n = t("sums of neighboring levels")
    sc.add(Write((13.3, 1.15), "+ neighbor sums:", size=20, color=C["grey"],
                 ha="center", t0=t_n))
    sc.add(Tex((13.3, 0.6), r"$D_1{+}D_2,\ D_2{+}D_3,\ \ldots$", size=21,
               color=C["grey"], ha="center", t0=t_n + 0.4))
    # exit arrow
    sc.add(ArrowItem((8.6, 2.9), (11.4, 2.4), color=C["grey"],
                     t0=t_o - 0.3, dur=0.6, curve=0.4))


# ------------------------------------------------------------------- s13 ---
def s13(sc, T):
    t = T.t
    header(sc, "One error, three lenses", t0=t("the loss itself"))
    xs = np.linspace(0, 1, 300)
    y_true = np.sin(2 * np.pi * 3 * xs) * (1 - 0.35 * xs)
    y_pred = 0.92 * np.sin(2 * np.pi * 3 * xs + 0.35) * (1 - 0.3 * xs)
    # top mini flow: pred/true -> S -> stacks
    t_s = t("exact same decomposition")
    gm = Graph((1.0, 6.95, 1.9, 0.8), xlim=(0, 1), ylim=(-1.3, 1.3))
    sc.add(Stroke(gm.XY(xs, y_pred), color=PRED_C, lw=2.0, glow=False,
                  t0=t("model's prediction"), dur=0.6, seed=150))
    gm2 = Graph((1.0, 6.0, 1.9, 0.8), xlim=(0, 1), ylim=(-1.3, 1.3))
    sc.add(Stroke(gm2.XY(xs, y_true), color=TRUE_C, lw=2.0, glow=False,
                  t0=t("the ground truth"), dur=0.6, seed=151))
    sc.add(ArrowItem((3.1, 6.85), (3.9, 6.85), color=C["grey"], t0=t_s,
                     dur=0.35))
    sc.add(BoxItem(4.0, 6.25, 1.1, 1.25, color=C["yellow"], t0=t_s + 0.2,
                   dur=0.5))
    sc.add(Write((4.55, 6.68), "S", size=34, color=C["yellow"], ha="center",
                 t0=t_s + 0.4))
    sc.add(ArrowItem((5.25, 6.85), (6.05, 6.85), color=C["grey"],
                     t0=t_s + 0.6, dur=0.35))
    for li in range(4):
        gg = Graph((6.2, 7.45 - li * 0.4, 1.7, 0.32), xlim=(0, 1),
                   ylim=(-1, 1))
        f = [0.8, 2, 4.5, 9][li]
        sc.add(Stroke(gg.XY(xs, np.sin(2 * np.pi * f * xs)),
                      color=LEVELC[li], lw=1.7, glow=False,
                      t0=t_s + 0.7 + li * 0.12, dur=0.4))
    sc.add(Write((8.35, 6.7), "compare level by level,", size=22, color=INK,
                 t0=t("level by level")))
    sc.add(Write((8.35, 6.2), "structure against structure", size=22,
                 color=INK, t0=t("Structure against structure")))
    # three panels
    panels = [
        ("time  (position)", TRUE_C, t("Lens one"), 0.9),
        ("derivative  (motion)", HN_C, t("Lens two"), 5.75),
        ("frequency  (rhythm)", C["purple"], t("lens three"), 10.6),
    ]
    for lab, col, tt, x0 in panels:
        sc.add(BoxItem(x0, 2.55, 4.5, 2.75, color=col, t0=tt - 0.1, dur=0.5))
        sc.add(Write((x0 + 2.25, 5.55), lab, size=22, color=col, ha="center",
                     t0=tt))
    # panel 1: values
    g1 = Graph((1.25, 2.8, 3.8, 2.1), xlim=(0, 1), ylim=(-1.4, 1.4))
    t1 = t("position by position")
    sc.add(Stroke(g1.XY(xs, y_true), color=TRUE_C, lw=2.2, glow=False,
                  t0=t1, dur=0.8, seed=152))
    sc.add(Stroke(g1.XY(xs, y_pred), color=PRED_C, lw=2.2, glow=False,
                  t0=t1 + 0.3, dur=0.8, seed=153))
    sc.add(Write((3.15, 2.15), "are you in the right place?", size=18,
                 color=C["grey"], ha="center", t0=t("right place")))
    # panel 2: derivative
    g2 = Graph((6.1, 2.8, 3.8, 2.1), xlim=(0, 1), ylim=(-1.4, 1.4))
    t2 = t("discrete slope")
    d_true = np.gradient(y_true, xs) / 20
    d_pred = np.gradient(y_pred, xs) / 20
    sc.add(Stroke(g2.XY(xs, d_true), color=TRUE_C, lw=2.2, glow=False,
                  t0=t2, dur=0.8, seed=154))
    sc.add(Stroke(g2.XY(xs, d_pred), color=PRED_C, lw=2.2, glow=False,
                  t0=t2 + 0.3, dur=0.8, seed=155))
    sc.add(Write((8.0, 2.15), "are you moving the right way?", size=18,
                 color=C["grey"], ha="center", t0=t("moving the right way")))
    sc.add(Tex((8.0, 5.05), r"$\Delta y_t = y_{t+1}-y_t$", size=19,
               color=HN_C, ha="center", t0=t2 + 0.5))
    # FFT primer above panel 3
    t_p = t("sum of sine waves")
    gp = Graph((10.75, 6.55, 2.1, 0.8), xlim=(0, 1), ylim=(-1.6, 1.6))
    comp = np.sin(2 * np.pi * 3 * xs) + 0.5 * np.sin(2 * np.pi * 9 * xs)
    sc.add(Stroke(gp.XY(xs, comp), color=INK, lw=1.9, glow=False, t0=t_p,
                  dur=0.6, seed=156))
    sc.add(Write((13.05, 6.85), "=", size=24, color=C["grey"], t0=t_p + 0.5))
    gp2 = Graph((13.35, 6.85, 2.0, 0.55), xlim=(0, 1), ylim=(-1.3, 1.3))
    sc.add(Stroke(gp2.XY(xs, np.sin(2 * np.pi * 3 * xs)), color=C["teal"],
                  lw=1.7, glow=False, t0=t_p + 0.6, dur=0.5))
    gp3 = Graph((13.35, 6.2, 2.0, 0.55), xlim=(0, 1), ylim=(-1.3, 1.3))
    sc.add(Stroke(gp3.XY(xs, 0.5 * np.sin(2 * np.pi * 9 * xs)),
                  color=C["orange"], lw=1.7, glow=False, t0=t_p + 0.8,
                  dur=0.5))
    sc.add(Write((14.45, 5.9), "+ ...", size=18, color=C["grey"],
                 t0=t_p + 1.0))
    sc.add(Write((12.85, 7.65), "any signal = recipe of sines", size=19,
                 color=INK, ha="center", t0=t_p - 0.3))
    # panel 3: spectrum
    g3 = Graph((10.95, 2.8, 3.8, 2.1), xlim=(-0.6, 8.5), ylim=(0, 1.25))
    t3 = t("one bright spike")
    mag_t = [0.10, 0.16, 0.2, 1.0, 0.22, 0.12, 0.3, 0.1, 0.06]
    mag_p = [0.13, 0.2, 0.25, 0.9, 0.3, 0.15, 0.22, 0.13, 0.09]
    spectrum_bars(sc, g3, mag_t, TRUE_C, t3, lw=3.4, stagger=0.06)
    spectrum_bars(sc, g3, mag_p, PRED_C, t3 + 0.5, lw=3.4, stagger=0.06,
                  dx=0.28)
    sc.add(Write((g3.X(3.2), 5.02), "daily", size=17, color=C["yellow"],
                 ha="center", t0=t3 + 0.9))
    sc.add(ArrowItem((g3.X(3.2) + 0.05, 4.92), (g3.X(3.15), g3.Y(1.05)),
                     color=C["yellow"], t0=t3 + 1.0, dur=0.4, head=0.1))
    sc.add(Write((12.85, 2.15), "do you have the right rhythms?", size=18,
                 color=C["grey"], ha="center", t0=t("right rhythms")))
    # formula + dials
    t_d = t("three dials")
    sc.add(Tex((5.4, 1.3),
               r"$\mathrm{Error}=\alpha\,\Delta_{\mathrm{time}}"
               r"+\beta\,\Delta_{\mathrm{slope}}"
               r"+\gamma\,\Delta_{\mathrm{rhythm}}$",
               size=29, color=C["yellow"], ha="center", t0=t_d, dur=1.4))
    th = np.linspace(0, 2 * np.pi, 40)
    dials = [(r"$\alpha$", "0.01", -0.75, t("turn alpha way down")),
             (r"$\beta$", "5", 0.65, t("lean hard on motion")),
             (r"$\gamma$", "1", 0.2, t("lean hard on motion", 0.5))]
    for i, (nm, val, ang, tt) in enumerate(dials):
        cx = 11.7 + i * 1.5
        sc.add(Stroke(np.column_stack([cx + 0.4 * np.cos(th),
                                       1.18 + 0.4 * np.sin(th)]),
                      color=INK, lw=2.0, t0=tt, dur=0.5))
        sc.add(Stroke([(cx, 1.18), (cx + 0.38 * np.sin(ang),
                        1.18 + 0.38 * np.cos(ang))], color=C["red"], lw=2.4,
                      t0=tt + 0.3, dur=0.3, glow=False))
        sc.add(Tex((cx - 0.62, 1.72), nm, size=21, color=INK, ha="center",
                   t0=tt + 0.1))
        sc.add(Write((cx, 0.42), val, size=18, color=C["grey"],
                     ha="center", t0=tt + 0.4))
    sc.add(Write((5.4, 0.55), "alpha, beta, gamma tuned per dataset",
                 size=18, color=C["grey"], ha="center",
                 t0=t("tuned per dataset")))


# ------------------------------------------------------------------- s14 ---
def s14(sc, T):
    t = T.t
    header(sc, "Backprop plays favorites", t0=t("one final trick"))
    xs = np.linspace(0, 1, 200)
    errs = [0.08, 0.42, 0.18, 0.22, 0.10]
    t_lv = t("every level")
    for li in range(5):
        gg = Graph((1.0, 6.6 - li * 1.18, 2.9, 0.9), xlim=(0, 1), ylim=(-1, 1))
        f = [0.7, 1.6, 3.4, 6.5, 12][li]
        sc.add(Stroke(gg.XY(xs, 0.8 * np.sin(2 * np.pi * f * xs)),
                      color=LEVELC[li], lw=2.2, glow=False,
                      t0=t_lv + li * 0.15, dur=0.6, seed=160 + li))
        sc.add(Write((0.65, 7.0 - li * 1.18), f"L{li+1}", size=20,
                     color=LEVELC[li], t0=t_lv + li * 0.15))
        # error bar
        sc.add(Stroke([(4.15, 6.65 - li * 1.18),
                       (4.15 + errs[li] * 6.2, 6.65 - li * 1.18)],
                      color=C["red"], lw=7.0, alpha=0.85,
                      t0=t("how wrong the prediction is") + li * 0.15,
                      dur=0.5, amp=0.012, glow=False))
    sc.add(Write((4.6, 7.9), "per-level error", size=20, color=C["red"],
                 t0=t("how wrong the prediction is")))
    # importance formula + bars
    t_imp = t("importance score")
    sc.add(Tex((9.5, 7.55), r"$\mathrm{imp}_{\ell}=\frac{\overline{|e_{\ell}|^{x}}}{\sum_k \overline{|e_k|^{x}}}$",
               size=27, color=C["yellow"], ha="center", t0=t("raise it to a power") , dur=1.2))
    imp = np.array(errs) ** 2
    imp = imp / imp.sum()
    for li in range(5):
        sc.add(Stroke([(8.3, 6.6 - li * 1.18), (8.3 + imp[li] * 5.2,
                        6.6 - li * 1.18)], color=C["yellow"], lw=7.0,
                      t0=t_imp + li * 0.12, dur=0.5, amp=0.012, glow=False))
        sc.add(Write((8.3 + imp[li] * 5.2 + 0.25, 6.5 - li * 1.18),
                     f"{imp[li]:.2f}", size=17, color=C["yellow"],
                     t0=t_imp + 0.4 + li * 0.12))
    sc.add(Write((8.3, 7.0 - 5 * 1.18 + 0.4), "importances sum to 1", size=18,
                 color=C["grey"], t0=t("sum to one")))
    # model + scaled gradient arrows
    t_g = t("scaled by its importance")
    sc.add(BoxItem(13.3, 3.6, 2.2, 1.5, color=C["purple"], t0=t_g - 0.3,
                   dur=0.5))
    sc.add(Write((14.4, 4.35), "MLP", size=25, color=C["purple"], ha="center",
                 t0=t_g - 0.1))
    for li in range(5):
        lw = 1.5 + imp[li] * 8
        col = C["red"] if li == 1 else C["grey"]
        y_src = 6.6 - li * 1.18
        sc.add(Dot((12.7, y_src), color=LEVELC[li], size=8,
                   t0=t_g + li * 0.15))
        sc.add(ArrowItem((12.85, y_src), (13.28, 4.9 - li * 0.28),
                         color=col, lw=lw, t0=t_g + 0.1 + li * 0.15, dur=0.6,
                         curve=0.12 * (2 - li)))
    sc.add(Write((12.7, 2.6), "gradients, scaled", size=20, color=INK,
                 t0=t_g + 0.9))
    sc.add(Write((12.7, 2.15), "by importance", size=20, color=INK,
                 t0=t_g + 1.1))
    # focus note
    t_r = t("fixing rhythm")
    sc.add(Write((5.0, 1.2), "worst level gets the biggest gradient",
                 size=22, color=C["red"], t0=t_r))
    sc.add(Write((5.0, 0.7), "re-decided on every batch", size=20,
                 color=C["grey"], t0=t("every single batch")))


# ------------------------------------------------------------------- s15 ---
def s15(sc, T):
    t = T.t
    header(sc, "Same model, three teachers", t0=t("does any of this"))
    # setup card
    t_su = t("Fix one architecture")
    sc.add(BoxItem(0.8, 4.7, 4.3, 2.9, color=INK, t0=t_su - 0.1, dur=0.6))
    lines = [("fixed 3-layer MLP", t_su + 0.2),
             ("look-back: 96", t("look-back of ninety-six")),
             ("horizons: 96 to 720", t("seven hundred twenty")),
             ("ONLY the loss changes", t("three ways"))]
    for i, (s, tt) in enumerate(lines):
        col = C["yellow"] if i == 3 else INK
        sc.add(Write((1.15, 6.95 - i * 0.62), s, size=22, color=col, t0=tt))
    losses = [("MSE", MSE_C, t("With mean squared error")),
              ("Tilde-Q", TQ_C, t("With Tilde-Q")),
              ("HNMD", HN_C, t("And with H-N-M-D"))]
    for i, (nm, col, tt) in enumerate(losses):
        sc.add(Write((1.5 + i * 1.35, 4.15), nm, size=21, color=col, t0=tt))
    # bar chart
    gb = Graph((6.0, 3.6, 6.3, 3.9), xlim=(0, 10), ylim=(0, 0.62))
    base_t = t("averaged over all four horizons")
    sc.add(Stroke([(gb.x0, gb.y0), (gb.x0 + gb.w, gb.y0)], color=C["grey"],
                  lw=2.2, t0=base_t, dur=0.5, glow=False))
    sc.add(Write((6.0, 7.75), "avg test MSE (lower = better)", size=20,
                 color=C["grey"], t0=base_t + 0.2))
    groups = [
        ("ETTh1", 1.0, [(0.487, MSE_C, t("zero point four eight seven")),
                        (0.461, TQ_C, t("zero point four six one")),
                        (0.442, HN_C, t("zero point four four two"))]),
        ("ETTh2", 6.0, [(0.557, MSE_C, t("Zero point five five seven")),
                        (0.485, TQ_C, t("zero point four eight five")),
                        (0.457, HN_C, t("zero point four five seven"))]),
    ]
    for gname, gx, bars in groups:
        for i, (v, col, tt) in enumerate(bars):
            x0 = gb.X(gx + i * 1.15)
            h = gb.Y(v) - gb.Y(0)
            sc.add(BoxItem(x0, gb.y0, 0.62, h, color=col, t0=tt, dur=0.7,
                           fill=col, fill_alpha=0.18))
            sc.add(Write((x0 + 0.31, gb.y0 + h + 0.22), f"{v:.3f}", size=18,
                         color=col, ha="center", t0=tt + 0.4))
        sc.add(Write((gb.X(gx + 1.15) + 0.3, 3.15), gname, size=22, color=INK,
                     ha="center", t0=bars[0][2]))
    t_9 = t("nine percent better")
    sc.add(Stroke([(gb.X(1.0), 2.85), (gb.X(3.3) + 0.6, 2.85)],
                  color=C["yellow"], lw=2.2, t0=t_9, dur=0.4, glow=False))
    sc.add(Write((gb.X(2.2), 2.45), "about 9% better", size=20,
                 color=C["yellow"], ha="center", t0=t_9 + 0.2))
    # wins badge
    t_w = t("every single win")
    sc.add(BoxItem(13.0, 6.6, 2.5, 1.0, color=HN_C, t0=t_w - 0.2, dur=0.5,
                   fill=HN_C, fill_alpha=0.1))
    sc.add(Write((14.25, 7.28), "wins: 4 / 4", size=22, color=HN_C,
                 ha="center", t0=t_w))
    sc.add(Write((14.25, 6.85), "(completed pairs)", size=15, color=C["grey"],
                 ha="center", t0=t_w + 0.3))
    # caveats
    t_f = t("to be fair")
    for i, s in enumerate(["biggest gains: hourly ETT",
                           "thinner on ECL / traffic / weather",
                           "PatchTST can still win some"]):
        sc.add(Write((12.85, 5.9 - i * 0.5), s, size=16, color=C["grey"],
                     t0=t_f + i * 0.9, ha="left"))
    sc.add(Write((12.85, 6.35), "honest footnotes:", size=18, color=INK,
                 t0=t_f - 0.3))
    # forecast sketch
    t_fc = t("forecasts themselves")
    gf = Graph((0.9, 0.75, 9.8, 2.0), xlim=(0, 1), ylim=(-1.5, 1.3))
    xs = np.linspace(0, 1, 300)
    y_tr = -0.15 - 1.05 * np.exp(-((xs - 0.42) ** 2) / 0.02) + \
        0.75 * np.exp(-((xs - 0.8) ** 2) / 0.01) + 0.25 * np.sin(14 * xs)
    y_mse = -0.28 + 0.1 * np.sin(6 * xs) - 0.25 * np.exp(-((xs - 0.45) ** 2) / 0.05)
    y_hn = -0.18 - 0.92 * np.exp(-((xs - 0.44) ** 2) / 0.022) + \
        0.62 * np.exp(-((xs - 0.8) ** 2) / 0.012) + 0.22 * np.sin(14 * xs + 0.3)
    sc.add(Stroke(gf.XY(xs, y_tr), color=TRUE_C, lw=2.6, t0=t_fc, dur=1.0,
                  seed=170))
    sc.add(Stroke(gf.XY(xs, y_mse), color=MSE_C, lw=2.6,
                  t0=t("cautious average"), dur=1.0, seed=171))
    sc.add(Stroke(gf.XY(xs, y_hn), color=HN_C, lw=2.6,
                  t0=t("actually commits"), dur=1.0, seed=172))
    sc.add(Write((1.15, 2.95), "truth", size=18, color=TRUE_C, t0=t_fc + 0.6))
    sc.add(Write((2.3, 2.95), "MSE-trained", size=18, color=MSE_C,
                 t0=t("cautious average", 0.6)))
    sc.add(Write((4.55, 2.95), "HNMD-trained", size=18, color=HN_C,
                 t0=t("actually commits", 0.5)))
    sc.add(Write((8.6, 2.95), "(illustrative sketch)", size=15,
                 color=C["dim"], t0=t_fc + 0.6))
    sc.add(ArrowItem((gf.X(0.55), gf.Y(-1.4)), (gf.X(0.455), gf.Y(-1.15)),
                     color=C["yellow"], t0=t("to the dip"), dur=0.5,
                     head=0.12, curve=0.2))
    sc.add(Write((gf.X(0.62), gf.Y(-1.35)), "catches the dip", size=17,
                 color=C["yellow"], t0=t("to the dip", 0.3)))


# ------------------------------------------------------------------- s16 ---
def s16(sc, T):
    t = T.t
    header(sc, "The whole machine", t0=t("back together"))
    xs = np.linspace(0, 1, 200)
    ny = 5.9
    # node 1: model
    t1 = t("model makes a forecast")
    sc.add(BoxItem(0.7, ny - 0.55, 1.7, 1.15, color=C["purple"], t0=t1,
                   dur=0.5))
    sc.add(Write((1.55, ny - 0.15), "MLP", size=22, color=C["purple"],
                 ha="center", t0=t1 + 0.2))
    # node 2: pred + true
    t2 = t("Prediction and truth")
    g2a = Graph((2.9, ny - 0.1, 1.6, 0.75), xlim=(0, 1), ylim=(-1.2, 1.2))
    sc.add(Stroke(g2a.XY(xs, np.sin(7 * xs)), color=PRED_C, lw=1.9,
                  glow=False, t0=t2, dur=0.5))
    g2b = Graph((2.9, ny - 0.95, 1.6, 0.75), xlim=(0, 1), ylim=(-1.2, 1.2))
    sc.add(Stroke(g2b.XY(xs, np.sin(7 * xs + 0.4)), color=TRUE_C, lw=1.9,
                  glow=False, t0=t2 + 0.2, dur=0.5))
    # node 3: S stack
    t3 = t("spline decomposition")
    sc.add(BoxItem(5.15, ny - 1.05, 2.1, 2.0, color=C["yellow"], t0=t3,
                   dur=0.5))
    for li in range(4):
        gg = Graph((5.35, ny + 0.55 - li * 0.42, 1.7, 0.34), xlim=(0, 1),
                   ylim=(-1, 1))
        f = [0.8, 2, 4.5, 9][li]
        sc.add(Stroke(gg.XY(xs, np.sin(2 * np.pi * f * xs)),
                      color=LEVELC[li], lw=1.7, glow=False,
                      t0=t3 + 0.2 + li * 0.1, dur=0.4))
    # node 4: lenses
    t4 = t("graded on position")
    sc.add(BoxItem(7.9, ny - 1.05, 2.3, 2.0, color=C["blue"], t0=t4, dur=0.5))
    sc.add(Write((9.05, ny + 0.55), "value", size=17, color=TRUE_C,
                 ha="center", t0=t4 + 0.15))
    sc.add(Write((9.05, ny + 0.1), "slope", size=17, color=HN_C, ha="center",
                 t0=t4 + 0.3))
    sc.add(Write((9.05, ny - 0.35), "rhythm", size=17, color=C["purple"],
                 ha="center", t0=t4 + 0.45))
    # node 5: weighted sum
    t5 = t("level errors become importance")
    sc.add(BoxItem(10.75, ny - 1.05, 2.15, 2.0, color=C["pink"], t0=t5,
                   dur=0.5))
    sc.add(Tex((11.82, ny + 0.3), r"$\alpha+\beta+\gamma$", size=20,
               color=C["pink"], ha="center", t0=t5 + 0.2))
    sc.add(Write((11.82, ny - 0.4), "importance", size=16, color=C["pink"],
                 ha="center", t0=t5 + 0.4))
    # node 6: gradients
    t6 = t("Importance shapes the gradients")
    sc.add(BoxItem(13.55, ny - 1.05, 1.9, 2.0, color=C["red"], t0=t6, dur=0.5))
    sc.add(Write((14.5, ny + 0.25), "scaled", size=17, color=C["red"],
                 ha="center", t0=t6 + 0.2))
    sc.add(Write((14.5, ny - 0.2), "gradients", size=17, color=C["red"],
                 ha="center", t0=t6 + 0.35))
    # arrows between nodes
    arrow_ts = [t2 - 0.15, t3 - 0.15, t4 - 0.15, t5 - 0.15, t6 - 0.15]
    node_x = [(2.45, 2.85), (4.6, 5.1), (7.3, 7.85), (10.25, 10.7),
              (12.95, 13.5)]
    for (xa, xb), tt in zip(node_x, arrow_ts):
        sc.add(ArrowItem((xa, ny - 0.05), (xb, ny - 0.05), color=C["grey"],
                         t0=tt, dur=0.3, head=0.12))
    # loop back
    t_loop = t("teach the model")
    sc.add(ArrowItem((14.5, ny - 1.15), (1.55, ny - 0.75), color=C["red"],
                     lw=2.8, t0=t_loop, dur=1.2, curve=1.8))
    sc.add(Write((8.0, 3.05), "batch after batch", size=20, color=C["red"],
                 ha="center", t0=t_loop + 0.6))
    # word callouts
    for word, tt, x in [("hierarchical", t("Hierarchical, because"), 6.2),
                        ("NURBS-inspired", t("NURBS-inspired, because"), 6.2),
                        ("multi-domain", t("multi-domain, because"), 9.05)]:
        pass
    sc.add(Write((6.2, 7.35), "hierarchical", size=21, color=C["yellow"],
                 ha="center", t0=t("Hierarchical, because")))
    sc.add(ArrowItem((6.2, 7.2), (6.2, 7.0), color=C["yellow"],
                     t0=t("Hierarchical, because", 0.3), dur=0.3, head=0.1))
    sc.add(Write((3.7, 7.35), "NURBS-inspired", size=21, color=HN_C,
                 ha="center", t0=t("NURBS-inspired, because")))
    sc.add(ArrowItem((4.75, 7.25), (5.5, 6.95), color=HN_C,
                     t0=t("NURBS-inspired, because", 0.4), dur=0.4,
                     head=0.11, curve=0.2))
    sc.add(Write((9.05, 7.35), "multi-domain", size=21, color=C["blue"],
                 ha="center", t0=t("multi-domain, because")))
    sc.add(ArrowItem((9.05, 7.2), (9.05, 7.0), color=C["blue"],
                     t0=t("multi-domain, because", 0.3), dur=0.3, head=0.1))
    # quote
    t_q = t("definition of truth")
    sc.add(BoxItem(2.6, 1.75, 10.8, 1.0, color=C["yellow"], t0=t_q - 0.3,
                   dur=0.6, fill=C["yellow"], fill_alpha=0.05))
    sc.add(Write((8.0, 2.08), "a loss function is a model's definition of truth",
                 size=25, color=C["yellow"], ha="center", t0=t_q - 0.1,
                 dur=1.4))
    # future work + credit
    t_n = t("what's next")
    sc.add(Write((1.0, 1.15), "next: adaptive depth,", size=19,
                 color=C["grey"], t0=t_n))
    sc.add(Write((1.0, 0.7), "learned weighting functions", size=19,
                 color=C["grey"], t0=t("Learning the weighting function")))
    sc.add(Write((15.2, 0.7), "paper: Abohwo & Vishnubhatla, Yale",
                 size=17, color=C["dim"], ha="right",
                 t0=t("core idea stands")))
    sc.add(Write((8.0, 0.62), "thanks for watching", size=22, color=INK,
                 ha="center", t0=t("Thanks for watching")))


BUILDERS_B = {"s09": s09, "s10": s10, "s11": s11, "s12": s12, "s13": s13,
              "s14": s14, "s15": s15, "s16": s16}
