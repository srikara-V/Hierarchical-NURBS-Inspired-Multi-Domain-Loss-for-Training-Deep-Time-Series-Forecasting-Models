"""
Binary Search: One Formula, Every Flavor
Khan-academy-style storyboard: narration beats + synchronized board drawing.
"""
from engine import (PAL, W, Timeline, Scene, SceneCtx, Anim, dim, ease_out,
                    sline, sarrow, draw_text, text_w, S)

LEAD = 1.0          # silence before first beat
GAP = 0.45          # pause between beats
SCENE_GAP = 1.3     # pause between scenes
TAIL = 2.2          # closing silence

VOICE_ID = "nPczCjzI2devNBz1zQrb"   # Brian: deep, resonant male
MODEL_ID = "eleven_turbo_v2_5"


class Beat:
    def __init__(self, key, text, hold=0.35):
        self.key = key
        self.text = " ".join(text.split())
        self.hold = hold
        self.t0 = 0.0
        self.dur = 2.5

    def at(self, dt):
        return self.t0 + dt

    def f(self, frac):
        return self.t0 + self.dur * frac

    @property
    def end(self):
        return self.t0 + self.dur


# ------------------------------------------------------------------ narration

SCENES = []


def scene(name, beats, build):
    SCENES.append(dict(name=name, beats=beats, build=build))


def all_beats():
    out = []
    for sc in SCENES:
        out.extend(sc["beats"])
    return out


def assign_times(durations):
    """durations: dict beat.key -> seconds. Returns total video length."""
    t = LEAD
    for sc in SCENES:
        sc["t0"] = t
        for b in sc["beats"]:
            b.t0 = t
            b.dur = durations[b.key]
            t += b.dur + b.hold + GAP
        sc["t1"] = t + SCENE_GAP - 0.5
        t += SCENE_GAP
    return t + TAIL


def build_timeline(durations):
    total = assign_times(durations)
    tl = Timeline()
    tl.duration = total
    for sdef in SCENES:
        sc = Scene(sdef["name"], sdef["t0"])
        sc.t1 = sdef["t1"]
        ctx = SceneCtx(sc)
        b = {bb.key: bb for bb in sdef["beats"]}
        sdef["build"](ctx, b)
        tl.scenes.append(sc)
    return tl


# ------------------------------------------------------------------ the grid

GRID_COLW = [430, 300, 520, 330]
GRID_HEADER = ["FLAVOR", "SPACE", "QUESTION ask(x)", "RETURN"]
GRID_ROWS = [
    ["1. sorted: find / insert", "index 0..n", "a[x] >= T   (> T right)", "lo (+ check)"],
    ["2. rotated: min / find", "index 0..n-1", "a[x] <= a[last] ?", "a[lo]; then row 1"],
    ["3. peak", "index 0..n-1", "a[x] > a[x+1] ?", "lo"],
    ["4. answer space: MIN", "values lo..hi", "feasible(x) ?", "lo"],
    ["5. answer space: MAX", "values lo..hi", "can(x)?  T,T,T,F,F", "first False - 1"],
    ["6. real numbers", "interval [lo, hi]", "same, on reals", "lo, after ~60 halvings"],
    ["7. 2D matrix", "flat 0..m*n", "A[x//n][x%n] >= T", "row x//n, col x%n"],
]
ROW_COLORS = [PAL["blue"], PAL["cyan"], PAL["pink"], PAL["gold"],
              PAL["orange"], PAL["green"], PAL["purple"]]


def grid_teaser(ctx, t, done, new, y=280, rh=62):
    g = ctx.grid(t, (170, y), GRID_COLW, GRID_HEADER, GRID_ROWS, rh=rh,
                 header_h=56, text_size=25, row_colors=ROW_COLORS)
    for i, r in enumerate(done):
        g.show_row(t + 0.15 + 0.07 * i, r, hl=False)
    for j, r in enumerate(new):
        g.show_row(t + 0.75 + 0.65 * j, r)
    return g


# ================================================================== SCENE 1

S1 = [
    Beat("s1b1", """Binary search. Most of us learn it as one narrow trick:
        find a number in a sorted array. But that picture is too small, and
        it's exactly why all the variants, rotated arrays, peaks, bananas,
        feel like separate pieces of magic. Today we rebuild binary search as
        one single formula, and then fill out a grid that covers pretty much
        every flavor you will ever meet.""", hold=0.4),
    Beat("s1b2", """Quick warm up. I'm thinking of a number from one to one
        hundred. You guess fifty, and I say: too low. With one question, you
        just deleted half the world. You guess seventy five. Too high. Half
        of what's left is gone again. Seven questions of this, and I am
        cornered. That feeling, cutting the world in half with one yes-or-no
        question, is the whole engine.""", hold=0.6),
    Beat("s1b3", """So here's the real picture. Forget arrays for a second.
        Picture a row of answers to some yes-or-no question, and suppose the
        row is sorted: all the No's come first, then all the Yes's. False,
        false, false, then true, true, true. We call a question like that
        monotone. It flips once, and it never flips back.""", hold=0.5),
    Beat("s1b4", """Binary search does exactly one job, ever. It finds that
        flip point, the first true, in about log n questions. That's it.
        That is the entire algorithm.""", hold=0.5),
    Beat("s1b5", """And every famous variant is just a different way of
        manufacturing this row. So here's our mission: one grid. For each
        flavor, we fill in the same cells. What is the search space? What is
        the yes-or-no question? And what do we return at the end? Let's fill
        it out, row by row.""", hold=0.6),
]


def build_s1(ctx, b):
    b1, b2, b3, b4, b5 = (b[k] for k in ("s1b1", "s1b2", "s1b3", "s1b4", "s1b5"))
    # b1 --- title card
    ctx.title(b1.at(0.3), "BINARY SEARCH", y=380, size=96)
    ctx.text(b1.f(0.75), (W / 2, 510), "one formula . every flavor",
             size=42, color=PAL["dim"], anchor="mm", write=False)
    ctx.clear_at(b2.t0)

    # b2 --- guessing game on 1..100
    def gx(v):
        return 360 + (v - 1) / 99 * 1200
    ctx.line(b2.at(0.2), (330, 430), (1590, 430), color=PAL["ink"], w=4, dur=0.7)
    for v in (1, 25, 50, 75, 100):
        ctx.line(b2.at(0.5), (gx(v), 416), (gx(v), 444), color=PAL["ink"], w=3, dur=0.3)
        ctx.note(b2.at(0.65), (gx(v), 478), str(v), size=24, anchor="mm")
    ctx.text(b2.at(0.1), (330, 300), "guess my number:  1 .. 100",
             size=32, color=PAL["ink"], cps=30)
    g1, g2 = b2.f(0.30), b2.f(0.55)
    # bracket of what's still possible
    bx0, bx1 = Anim(gx(1)), Anim(gx(100))
    bx0.to(g1 + 0.6, gx(51), 0.6)
    bx1.to(g2 + 0.6, gx(74), 0.6)
    sd = ctx.seed()

    def bracket(d, tt, a):
        x0, x1 = bx0(tt), bx1(tt)
        c = dim(PAL["gold"], a)
        sline(d, (x0, 512), (x1, 512), c, w=5, seed=sd)
        sline(d, (x0, 512), (x0, 496), c, w=5, seed=sd + 1)
        sline(d, (x1, 512), (x1, 496), c, w=5, seed=sd + 2)
        draw_text(d, ((x0 + x1) / 2, 545), "still possible", "hand", 24, c, "mm")
    ctx.el(b2.at(0.9), bracket, fade_in=0.4)
    ctx.circle(g1, (gx(50), 430), 26, color=PAL["blue"])
    ctx.chip(g1 + 0.15, (gx(50) - 80, 320), "50? too low", color=PAL["red"])
    ctx.circle(g2, (gx(75), 430), 26, color=PAL["blue"])
    ctx.chip(g2 + 0.15, (gx(75) - 84, 320), "75? too high", color=PAL["red"])
    ctx.chip(b2.f(0.85), (1250, 560), "7 questions beat 100 doors",
             color=PAL["blue"])
    ctx.clear_at(b3.t0)

    # b3 --- the F/T row
    vals = list("FFFFFFF") + list("TTTTT")
    bar = ctx.array(b3.at(0.15), (504, 400), vals, cw=76, ch=70,
                    idx_labels=False, val_size=30)
    for i, v in enumerate(vals):
        bar.state(b3.at(0.5 + 0.07 * i), i, v)
    ctx.note(b3.at(0.05), (504, 330), "one yes-no question, asked along a row",
             size=27)
    chip_mono = ctx.chip(b3.f(0.82), (700, 545),
                         "monotone :  flips ONCE, never back",
                         color=PAL["gold"])
    # b4 --- first true
    cx7 = 504 + 7 * 76 + 38
    ctx.arrow(b4.at(0.2), (cx7, 305), (cx7, 385), color=PAL["gold"], w=5)
    ctx.text(b4.at(0.35), (cx7, 272), "the FIRST TRUE", size=34,
             color=PAL["gold"], kind="handb", anchor="mm", cps=24)
    ctx.circle(b4.at(0.6), (cx7 + 0, 435), 47, color=PAL["gold"])
    chip_logn = ctx.chip(b4.f(0.7), (1420, 545), "~ log n questions",
                         color=PAL["blue"])
    chip_mono.t_out = b5.t0
    chip_logn.t_out = b5.t0
    # b5 --- ghost grid
    ghost = [[r[0], "", "", ""] for r in GRID_ROWS]
    g = ctx.grid(b5.at(0.35), (328, 640), [344, 240, 416, 264],
                 GRID_HEADER, ghost, rh=44, header_h=44, text_size=20,
                 row_colors=ROW_COLORS)
    for i in range(7):
        g.show_row(b5.at(0.8 + 0.12 * i), i, hl=False)
    ctx.text(b5.at(0.05), (960, 600), "the mission: fill this grid",
             size=30, color=PAL["gold"], kind="handb", anchor="mm")


scene("intro", S1, build_s1)

# ================================================================== SCENE 2

S2 = [
    Beat("s2b1", """Here is the formula, as four decisions. One: the space.
        The candidates, laid out in a row. Two: the question. A yes-or-no
        probe that goes false, false, false, true, true, true across that
        space. Three: the boundary. We always, always aim at the first true.
        And four: the return. What we hand back once we've caught it.""",
         hold=0.4),
    Beat("s2b2", """And one template executes all four. Two fingers, low and
        high, and one promise: the first true is always trapped between them.
        While they haven't met, probe the middle.""", hold=0.3),
    Beat("s2b3", """Say the probe comes back false. Then mid itself is a no,
        and everything left of mid is even deeper in no territory. That whole
        side is dead, including mid. So low jumps to mid plus one.""",
         hold=0.6),
    Beat("s2b4", """Now say a probe comes back true. Careful. Mid might be
        the very first true, so we must not skip past it. High lands right on
        mid, and mid stays alive inside the window.""", hold=0.6),
    Beat("s2b5", """False? Low hops past mid. True? High lands on mid. The
        window only shrinks, the flip point can never escape it, and when the
        fingers meet, they are standing on the first true. Return low.""",
         hold=0.9),
    Beat("s2b6", """Two tiny details make this bulletproof, and they are the
        details people get wrong. The middle uses floor division, so mid
        always lands strictly below high. And we only ever add one on the
        false side, because false cells are provably dead. Together: the
        window must shrink every round. No infinite loops, no off-by-one,
        ever.""", hold=0.5),
    Beat("s2b7", """One last guard. What if the row is all false, and no true
        exists? That's why high starts one past the end, at n, on an
        imaginary cell we simply declare to be true. If the search returns n,
        it's telling you: no real true exists. Remember that dashed cell.
        It does quiet work in every row of our grid.""", hold=0.6),
]

CODE_LINES = [
    "lo, hi = 0, n",
    "while lo < hi:",
    "    mid = (lo + hi) // 2",
    "    if ask(mid):",
    "        hi = mid          # keep mid",
    "    else:",
    "        lo = mid + 1      # mid is dead",
    "return lo                 # first True",
]


def build_s2(ctx, b):
    b1, b2, b3, b4, b5, b6, b7 = (b[f"s2b{i}"] for i in range(1, 8))
    ctx.section(b1.at(0.0), "THE FORMULA")
    # b1 four decision chips
    chips = [("1. SPACE", "the candidates", PAL["blue"], 150),
             ("2. QUESTION", "monotone yes/no", PAL["green"], 590),
             ("3. BOUNDARY", "the first TRUE", PAL["gold"], 1030),
             ("4. RETURN", "transform lo", PAL["purple"], 1470)]
    for i, (t1, t2, c, x) in enumerate(chips):
        tt = b1.f(0.12 + 0.21 * i)
        ctx.chip(tt, (x, 165), t1, color=c, size=30)
        ctx.note(tt + 0.25, (x + 8, 240), t2, size=23)
    # b2 code + bar
    code = ctx.code(b2.at(0.1), (110, 330), CODE_LINES, size=26)
    vals = [""] * 12 + [""]
    bar = ctx.array(b2.at(0.9), (960, 430), vals, cw=64, ch=64,
                    idx_labels=True, val_size=26, dashed=(12,))
    lo = bar.pointer(b2.at(1.6), "lo", 0, PAL["blue"], side="bottom")
    hi = bar.pointer(b2.at(1.8), "hi", 12, PAL["purple"], side="bottom")
    # b3 probe mid=6 -> False
    code.highlight(b3.at(0.5), 3)
    bar.probe(b3.f(0.18), 6, "F", dur=1.6, letter="F")
    code.highlight(b3.f(0.55), 6)
    lo.move(b3.f(0.72), 7)
    bar.state_range(b3.f(0.72), 0, 6, "dead", stagger=0.04)
    # b4 probe mid=9 -> True
    bar.probe(b4.f(0.15), 9, "T", dur=1.4, letter="T")
    code.highlight(b4.f(0.5), 4)
    hi.move(b4.f(0.62), 9)
    bar.state_range(b4.f(0.66), 10, 12, "dead", stagger=0.05)
    # b5 finish: probes 8T, 7T
    bar.probe(b5.f(0.10), 8, "T", dur=1.0, letter="T")
    hi.move(b5.f(0.28), 8)
    bar.probe(b5.f(0.42), 7, "T", dur=1.0, letter="T")
    hi.move(b5.f(0.60), 7)
    lo.move(b5.f(0.60), 7)
    bar.state(b5.f(0.62), 8, "dead")
    code.highlight(b5.f(0.75), 7)
    bar.state(b5.f(0.78), 7, "found")
    ctx.circle(b5.f(0.82), (960 + 7 * 64 + 32, 462), 42, color=PAL["gold"])
    ctx.text(b5.f(0.85), (960 + 7 * 64 + 32, 330), "first True!", size=30,
             color=PAL["green"], kind="handb", anchor="mm", cps=26)
    # b6 safety chips
    code.highlight(b6.f(0.15), 2)
    ctx.chip(b6.f(0.18), (140, 800), "floor mid  ->  mid < hi, always",
             color=PAL["cyan"])
    code.highlight(b6.f(0.45), 6)
    ctx.chip(b6.f(0.5), (140, 880), "+1 only on the FALSE side",
             color=PAL["green"])
    ctx.note(b6.f(0.75), (700, 815), "=> the window shrinks every round", size=25)
    # b7 sentinel
    code.highlight(b7.at(0.3), 0)
    sx = 960 + 12 * 64 + 32
    ctx.circle(b7.f(0.3), (sx, 462), 44, color=PAL["purple"])
    ctx.text(b7.f(0.38), (sx - 10, 330), "pretend TRUE", size=26,
             color=PAL["purple"], kind="handb", anchor="mm", cps=26)
    ctx.arrow(b7.f(0.5), (sx - 10, 352), (sx - 2, 420), color=PAL["purple"], w=4)
    ctx.chip(b7.f(0.62), (1130, 800), "returned n ?  =  no True exists",
             color=PAL["red"])


scene("formula", S2, build_s2)

# ================================================================== SCENE 3

S3 = [
    Beat("s3b1", """Row one of the grid: the plain sorted array. The classic:
        find target seven, LeetCode seven-oh-four. So, where is our monotone
        question hiding? Here it is: is the value at index x, at least seven?
        Down in the small values: no, no, no. From the first seven onward:
        yes, yes, yes, forever. Sorted order is exactly what makes that
        question monotone. That is the real reason binary search works on
        sorted arrays at all.""", hold=0.5),
    Beat("s3b2", """Run the exact template from before. Probe the middle: at
        least seven? True. High lands on it. Probe again: three? False. Low
        hops past it. One more probe, and the fingers meet at index three,
        the first true. One final check: is the value there actually seven?
        It is. Found, at index three. If it weren't, we'd return minus
        one.""", hold=0.9),
    Beat("s3b3", """Now watch how cheap the next problem becomes. Search
        insert position, LeetCode thirty-five: where would five go? Change
        almost nothing. Ask: at least five? The first true is index three,
        again. And that IS the insert slot. Same space, same question shape,
        same boundary. Only the return story changed.""", hold=0.6),
    Beat("s3b4", """Duplicates. LeetCode thirty-four: first and last position
        of seven. Well, the first seven is just our first true for: at least
        seven. Index three. Already done.""", hold=0.4),
    Beat("s3b5", """For the LAST seven, do not write some new backwards loop.
        Flip the question instead: strictly greater than seven? That flips
        later, at index six. So the last seven sits one step before it: index
        five. First: at least. Last: strictly greater, minus one. That minus
        one trick is worth its own little cell in the grid.""", hold=0.6),
    Beat("s3b6", """And by the way, you already know these two searches by
        name. They are lower bound and upper bound. Bisect left and bisect
        right, in Python. Two predicates, one template, zero special
        cases.""", hold=0.4),
    Beat("s3b7", """Row one of the grid: space, indices zero to n. Question:
        value at x, at least the target. Or strictly greater, for right
        edges. Return: low, with a found-check, or minus one for last
        positions.""", hold=1.0),
]


def build_s3(ctx, b):
    b1, b2, b3, b4, b5, b6, b7 = (b[f"s3b{i}"] for i in range(1, 8))
    ctx.section(b1.at(0.0), "ROW 1 . THE SORTED ARRAY", color=PAL["blue"])
    ctx.chip(b1.at(0.2), (90, 152), "LC 704 . 35 . 34", color=PAL["blue"],
             size=22, filled=False)
    A = [-4, 0, 3, 7, 7, 7, 11, 15, ""]
    arr = ctx.array(b1.at(0.5), (510, 360), A, cw=100, ch=84, val_size=33,
                    dashed=(8,))
    ctx.chip(b1.at(0.1), (1490, 168), "target T = 7", color=PAL["gold"], size=27)
    q1 = ctx.text(b1.f(0.30), (510, 196), "ask(x):   a[x] >= 7 ?", size=32,
                  color=PAL["gold"], kind="handb", cps=30)
    # letter row 1
    L1 = list("FFFTTTTT")
    ctx.note(b1.f(0.55) - 0.1, (470, 508), ">= 7 :", size=24, anchor="rm",
             color=PAL["gold"])
    for i, ch in enumerate(L1):
        ctx.text(b1.f(0.55 + 0.03 * i), (560 + 100 * i, 508), ch, size=26,
                 color=(PAL["green"] if ch == "T" else PAL["red"]),
                 kind="handb", anchor="mm", write=False)
    # b2 trace
    lo = arr.pointer(b2.at(0.2), "lo", 0, PAL["blue"], side="top")
    hi = arr.pointer(b2.at(0.4), "hi", 8, PAL["purple"], side="top")
    arr.probe(b2.f(0.12), 4, "T", dur=1.3)
    hi.move(b2.f(0.26), 4)
    arr.state_range(b2.f(0.28), 5, 8, "dead", stagger=0.04)
    arr.probe(b2.f(0.36), 2, "F", dur=1.2)
    lo.move(b2.f(0.50), 3)
    arr.state_range(b2.f(0.52), 0, 2, "dead", stagger=0.04)
    arr.probe(b2.f(0.58), 3, "T", dur=1.1)
    hi.move(b2.f(0.70), 3)
    arr.state(b2.f(0.74), 3, "found")
    res1 = ctx.text(b2.f(0.80), (1430, 320), "a[3] == 7  ->  return 3",
                    size=28, color=PAL["green"], cps=30)
    res2 = ctx.note(b2.f(0.92), (1430, 372), "(else: return -1)", size=24)
    # b3 insert position
    lo.off(b3.t0)
    hi.off(b3.t0)
    res1.t_out = b3.t0
    res2.t_out = b3.t0
    for i in range(9):
        arr.state(b3.at(0.05 + 0.02 * i), i, "base")
    t5 = ctx.chip(b3.f(0.28), (1490, 250), "target = 5", color=PAL["cyan"],
                  size=27)
    ctx.arrow(b3.f(0.45), (810, 660), (810, 462), color=PAL["cyan"], w=5)
    ctx.chip(b3.f(0.55), (690, 682), "insert slot = 3", color=PAL["cyan"])
    a5 = ctx.note(b3.f(0.72), (1430, 330), "ask: a[x] >= 5 ?", size=26,
                  color=PAL["cyan"])
    r5 = ctx.note(b3.f(0.84), (1430, 378), "first True = 3, again", size=24)
    for e in (t5, a5, r5):
        e.t_out = b4.t0
    # b4 first occurrence
    ctx.arrow(b4.f(0.5), (860, 288), (860, 348), color=PAL["green"], w=5)
    ctx.text(b4.f(0.62), (860, 262), "first(>=7) = 3", size=26,
             color=PAL["green"], kind="handb", anchor="mm", cps=30)
    # b5 last occurrence via > 7
    L2 = list("FFFFFFTT")
    ctx.note(b5.f(0.18) - 0.1, (470, 556), "> 7 :", size=24, anchor="rm",
             color=PAL["purple"])
    for i, ch in enumerate(L2):
        ctx.text(b5.f(0.18 + 0.03 * i), (560 + 100 * i, 556), ch, size=26,
                 color=(PAL["green"] if ch == "T" else PAL["red"]),
                 kind="handb", anchor="mm", write=False)
    ctx.arrow(b5.f(0.42), (1160, 288), (1160, 348), color=PAL["purple"], w=5)
    ctx.text(b5.f(0.5), (1170, 262), "first(>7) = 6", size=26,
             color=PAL["purple"], kind="handb", anchor="mm", cps=30)
    ctx.circle(b5.f(0.68), (1060, 402), 54, color=PAL["green"])
    ctx.text(b5.f(0.78), (1060, 700), "last 7  =  6 - 1  =  5", size=28,
             color=PAL["green"], kind="handb", anchor="mm", cps=30)
    # b6 bisect chips
    ctx.chip(b6.f(0.15), (250, 800), "bisect_left  =  first( >= T )",
             color=PAL["green"])
    ctx.chip(b6.f(0.4), (950, 800), "bisect_right  =  first( > T )",
             color=PAL["purple"])
    ctx.note(b6.f(0.7), (960, 890), "C++ :  lower_bound / upper_bound",
             size=24, anchor="mm")
    # b7 grid
    ctx.clear_at(b7.t0)
    grid_teaser(ctx, b7.at(0.15), [], [0])


scene("row1_sorted", S3, build_s3)

# ================================================================== SCENE 4

S4 = [
    Beat("s4b1", """Row two: the rotated sorted array. Someone took a sorted
        array and spun it. Two sorted runs now, glued at a cliff. First job,
        LeetCode one fifty-three: find the minimum. The bottom of that
        cliff.""", hold=0.4),
    Beat("s4b2", """Where's the monotone question? Look at the very last
        element. Three. Now ask every index: is your value at most that last
        element? The high run, before the cliff: no, no, no. The low run,
        after it: yes, yes, yes. The rotation point is literally a
        false-to-true boundary. We just manufactured our row.""", hold=0.5),
    Beat("s4b3", """Same template, not a character changed. Probe the middle:
        twelve, at most three? False. Low hops right. Probe: zero? True. High
        lands on it. Probe: minus four? True again. The fingers meet at index
        three. The minimum.""", hold=0.8),
    Beat("s4b4", """And now, searching a rotated array, LeetCode thirty-three,
        is just composition. Step one: find the cliff, exactly like we just
        did. Step two: the cliff splits the array into two perfectly sorted
        halves, and comparing your target to the last element tells you which
        half it lives in. Step three: run row one's classic search inside
        that half. Two binary searches, both from the same template. Still
        logarithmic.""", hold=0.5),
    Beat("s4b5", """Two footnotes for this row. Always compare against the
        LAST element; comparing mid to low breaks on arrays that were never
        rotated. And if duplicates are allowed, LeetCode one fifty-four,
        equal values can hide the cliff, and the worst case genuinely
        degrades to linear. Know that trade before an interview.""",
         hold=0.4),
    Beat("s4b6", """Row two, into the grid.""", hold=1.2),
]


def build_s4(ctx, b):
    b1, b2, b3, b4, b5, b6 = (b[f"s4b{i}"] for i in range(1, 7))
    ctx.section(b1.at(0.0), "ROW 2 . THE ROTATED ARRAY", color=PAL["cyan"])
    ctx.chip(b1.at(0.2), (90, 152), "LC 153 . 33 . 154", color=PAL["cyan"],
             size=22, filled=False)
    R = [7, 9, 12, -4, 0, 3]
    arr = ctx.array(b1.at(0.4), (630, 430), R, cw=110, ch=88, val_size=34)
    # mountain
    xs = [685 + 110 * i for i in range(6)]
    ys = [340 - (v + 4) * (140 / 16) for v in R]
    ctx.polyline(b1.f(0.3), list(zip(xs, ys)), color=PAL["cyan"], w=5, dur=1.0)
    # cliff dashes
    for k in range(5):
        ctx.line(b1.f(0.75) + 0.05 * k, (962, 190 + 34 * k), (962, 208 + 34 * k),
                 color=PAL["red"], w=4, dur=0.2)
    ctx.text(b1.f(0.8), (962, 158), "the cliff", size=27, color=PAL["red"],
             kind="handb", anchor="mm", cps=24)
    # b2 predicate
    ctx.circle(b2.at(0.35), (1235, 474), 52, color=PAL["cyan"])
    ctx.text(b2.at(0.1), (960, 690), "ask(x):   a[x] <= a[last]  (= 3) ?",
             size=31, color=PAL["gold"], kind="handb", anchor="mm", cps=30)
    L = list("FFFTTT")
    for i, ch in enumerate(L):
        ctx.text(b2.f(0.5 + 0.045 * i), (685 + 110 * i, 392), ch, size=25,
                 color=(PAL["green"] if ch == "T" else PAL["red"]),
                 kind="handb", anchor="mm", write=False)
    ctx.note(b2.f(0.85), (1360, 392), "<- the boundary!", size=24,
             color=PAL["gold"])
    # b3 trace
    lo = arr.pointer(b3.at(0.1), "lo", 0, PAL["blue"], side="bottom")
    hi = arr.pointer(b3.at(0.25), "hi", 5, PAL["purple"], side="bottom")
    arr.probe(b3.f(0.15), 2, "F", dur=1.2, label="")
    lo.move(b3.f(0.30), 3)
    arr.state_range(b3.f(0.32), 0, 2, "dead", stagger=0.05)
    arr.probe(b3.f(0.42), 4, "T", dur=1.1, label="")
    hi.move(b3.f(0.55), 4)
    arr.state(b3.f(0.57), 5, "dead")
    arr.probe(b3.f(0.65), 3, "T", dur=1.0, label="")
    hi.move(b3.f(0.78), 3)
    arr.state(b3.f(0.82), 3, "found")
    ctx.text(b3.f(0.88), (960, 745), "min  =  a[3]  =  -4", size=29,
             color=PAL["green"], kind="handb", anchor="mm", cps=30)
    # b4 composition strip
    ctx.chip(b4.f(0.18), (250, 790), "1. find the cliff", color=PAL["cyan"])
    ctx.arrow(b4.f(0.30), (560, 818), (660, 818), color=PAL["dim"], w=4)
    ctx.chip(b4.f(0.34), (680, 790), "2. pick the sorted half",
             color=PAL["blue"])
    ctx.arrow(b4.f(0.52), (1090, 818), (1190, 818), color=PAL["dim"], w=4)
    ctx.chip(b4.f(0.56), (1210, 790), "3. row-1 search", color=PAL["green"])
    ctx.note(b4.f(0.8), (1210, 862), "log + log = still log", size=23)
    # b5 footnotes
    ctx.note(b5.f(0.15), (1450, 250), "compare to LAST,", size=25,
             color=PAL["gold"])
    ctx.note(b5.f(0.22), (1450, 288), "never to lo", size=25, color=PAL["gold"])
    ctx.note(b5.f(0.6), (1450, 560), "duplicates (LC 154):", size=25,
             color=PAL["red"])
    ctx.note(b5.f(0.68), (1450, 598), "cliff can hide -> O(n)", size=25,
             color=PAL["red"])
    # b6 grid
    ctx.clear_at(b6.t0)
    grid_teaser(ctx, b6.at(0.1), [0], [1])


scene("row2_rotated", S4, build_s4)

# ================================================================== SCENE 5

S5 = [
    Beat("s5b1", """Row three: the peak, LeetCode one sixty-two. Find any
        element bigger than both neighbors. And look at this array. Nothing
        is sorted. Surely binary search is off the table. Right?""",
         hold=0.4),
    Beat("s5b2", """Ask this at every index: are we going downhill at x? Is
        the value at x greater than the value right after it? While we
        climb: no, no. Past the summit: yes, yes. And off the right edge we
        pretend the array falls to minus infinity, so the last index is
        always a yes. There's our dashed sentinel cell again, doing real
        work.""", hold=0.5),
    Beat("s5b3", """Now, one honest subtlety, because this is where the deep
        idea lives. With many hills, that downhill question can flicker. Yes,
        no, yes. Not monotone! But watch what the template actually needs.
        Probe mid. Going uphill? Then the next step is higher than mid, and
        somewhere to the right a summit must exist. Slide low past mid.
        Going downhill? Then a summit exists at mid or to its left. Pull high
        onto mid. Either way, the window still traps a peak. That is the real
        contract of binary search: never lose the answer. Monotone rows are
        just its most common costume.""", hold=0.5),
    Beat("s5b4", """Run it. Middle of the row: five against four. Downhill.
        True. High lands there. One more probe, uphill, low slides past. The
        fingers meet at index two. Value five: a peak, found in log time, in
        an unsorted array.""", hold=0.7),
    Beat("s5b5", """Row three of the grid. Space: indices. Question: downhill
        here? Return: low. Plus one idea worth its own subcell: the invariant
        is the law. Monotonicity is just the costume.""", hold=0.4),
    Beat("s5b6", """Into the grid.""", hold=1.2),
]


def build_s5(ctx, b):
    b1, b2, b3, b4, b5, b6 = (b[f"s5b{i}"] for i in range(1, 7))
    ctx.section(b1.at(0.0), "ROW 3 . THE PEAK", color=PAL["pink"])
    ctx.chip(b1.at(0.2), (90, 152), "LC 162", color=PAL["pink"], size=22,
             filled=False)
    P = [1, 3, 5, 4, 2, "-inf"]
    arr = ctx.array(b1.at(0.35), (600, 460), P, cw=120, ch=88, val_size=32,
                    dashed=(5,))
    xs = [660 + 120 * i for i in range(5)]
    ys = [400 - v * 36 for v in (1, 3, 5, 4, 2)]
    ctx.polyline(b1.f(0.45), list(zip(xs, ys)), color=PAL["pink"], w=5, dur=1.1)
    # b2 predicate letters
    ctx.text(b2.at(0.1), (900, 700), "ask(x):   a[x] > a[x+1] ?    (downhill?)",
             size=31, color=PAL["gold"], kind="handb", anchor="mm", cps=30)
    L = list("FFTTT")
    for i, ch in enumerate(L):
        ctx.text(b2.f(0.42 + 0.05 * i), (660 + 120 * i, 424), ch, size=25,
                 color=(PAL["green"] if ch == "T" else PAL["red"]),
                 kind="handb", anchor="mm", write=False)
    ctx.circle(b2.f(0.72), (600 + 5 * 120 + 60, 504), 55, color=PAL["purple"])
    ctx.note(b2.f(0.8), (1345, 496), "<- always True", size=24,
             color=PAL["purple"])
    # b3 invariant notes
    ctx.note(b3.f(0.30), (170, 300), "uphill at mid ->", size=26,
             color=PAL["green"])
    ctx.note(b3.f(0.36), (170, 338), "a peak lives RIGHT", size=26,
             color=PAL["green"])
    ctx.note(b3.f(0.52), (170, 420), "downhill at mid ->", size=26,
             color=PAL["red"])
    ctx.note(b3.f(0.58), (170, 458), "a peak: HERE or LEFT", size=26,
             color=PAL["red"])
    ctx.text(b3.f(0.82), (170, 560), "invariant:", size=28, color=PAL["gold"],
             kind="handb", cps=26)
    ctx.note(b3.f(0.88), (170, 610), "the window always", size=25)
    ctx.note(b3.f(0.92), (170, 648), "holds a peak", size=25)
    # b4 trace
    lo = arr.pointer(b4.at(0.05), "lo", 0, PAL["blue"], side="bottom")
    hi = arr.pointer(b4.at(0.2), "hi", 4, PAL["purple"], side="bottom")
    arr.probe(b4.f(0.15), 2, "T", dur=1.1, label="")
    hi.move(b4.f(0.32), 2)
    arr.state_range(b4.f(0.34), 3, 4, "dead", stagger=0.05)
    arr.probe(b4.f(0.45), 1, "F", dur=1.0, label="")
    lo.move(b4.f(0.60), 2)
    arr.state(b4.f(0.62), 0, "dead")
    arr.state(b4.f(0.63), 1, "dead")
    arr.state(b4.f(0.68), 2, "found")
    ctx.circle(b4.f(0.72), (900, 216), 26, color=PAL["gold"])
    ctx.text(b4.f(0.80), (900, 760), "peak:  a[2] = 5", size=29,
             color=PAL["green"], kind="handb", anchor="mm", cps=30)
    # b5 chips
    ctx.chip(b5.f(0.3), (330, 850), "edges fall to -inf  (sentinels)",
             color=PAL["purple"])
    ctx.chip(b5.f(0.6), (890, 850), "invariant = the LAW ; monotone = costume",
             color=PAL["gold"])
    # b6 grid
    ctx.clear_at(b6.t0)
    grid_teaser(ctx, b6.at(0.1), [0, 1], [2])


scene("row3_peak", S5, build_s5)

# ================================================================== SCENE 6

S6 = [
    Beat("s6b1", """Row four. And this is the flavor that unlocks hundreds of
        hard problems. So far, the space was indices into an array somebody
        handed us. Now the space becomes the possible ANSWERS themselves.
        Koko loves bananas, LeetCode eight seventy-five. Four piles. Eight
        hours before the guards come back. Koko picks one eating speed, k
        bananas per hour, and each hour she eats from a single pile.""",
         hold=0.4),
    Beat("s6b2", """Here's the key move: fix a speed, and just check it. Try
        speed four. The pile of three takes one hour. Six takes two. Seven
        takes two. Eleven takes three. Total: eight hours. She makes it.
        Exactly.""", hold=0.4),
    Beat("s6b3", """Now sweep every candidate speed, one through eleven, and
        ask each one: can Koko finish in time at this speed? The slow speeds:
        no, no, no. Then from four onward: yes, yes, yes. Of course it's
        monotone. Eating faster never hurts. We just built our row out of
        thin air. THAT is binary search on the answer.""", hold=0.5),
    Beat("s6b4", """So the formula instantiates like this. Space: speeds one
        up to the biggest pile. Question: does the total time at speed x fit
        in eight hours? That check is a five-line helper. Boundary: first
        yes. Return: low. The template does not change by one character.
        Only the question is new.""", hold=0.4),
    Beat("s6b5", """Run it on the speeds. Probe six: fits. Probe three: too
        slow. Probe five: fits. Probe four: fits. Low meets high at four.
        Koko eats at speed four.""", hold=0.7),
    Beat("s6b6", """And here is why this row is worth hundreds of problems.
        Watch how little changes. Ship packages within D days, LeetCode
        ten-eleven. The answer is a ship capacity. The space runs from the
        heaviest single package up to everything in one trip. The question:
        at capacity x, do the days fit? One greedy pass counts the trips.
        Same row.""", hold=0.4),
    Beat("s6b7", """Split an array into k chunks, minimizing the largest
        chunk sum. LeetCode four-ten. The answer is that largest sum. The
        question: capped at x, do we need at most k chunks? Same row. Minimum
        days for m bouquets, LeetCode fourteen eighty-two: the space is days,
        the question: enough bouquets by day x? Same row. Different story,
        different feasibility check. Identical search.""", hold=0.5),
    Beat("s6b8", """One twist left: maximize instead of minimize. Magnetic
        force, LeetCode fifteen fifty-two. Place three balls in baskets, and
        make the smallest gap as LARGE as possible. Ask: can we place them
        with gap at least x? Small gaps: yes, yes, yes. Then it breaks: no,
        no, no. The row flipped! True first, then false. And still, no new
        loop. Search for the first no, and step back one. Maximize equals
        first false, minus one.""", hold=0.6),
    Beat("s6b9", """Rows four and five, into the grid. Minimize: first yes.
        Maximize: first no, minus one. If you keep only one row of this whole
        grid, keep this one. Binary search the answer, feasibility check
        inside.""", hold=1.0),
]


def build_s6(ctx, b):
    (b1, b2, b3, b4, b5, b6, b7, b8, b9) = (b[f"s6b{i}"] for i in range(1, 10))
    ctx.section(b1.at(0.0), "ROW 4 . SEARCH THE ANSWER SPACE",
                color=PAL["gold"])
    left = []
    # b1 koko setup
    left.append(ctx.text(b1.f(0.35), (140, 170), "Koko's bananas . LC 875",
                         size=30, cps=30))
    left.append(ctx.chip(b1.f(0.55), (140, 215), "8 hours on the clock",
                         color=PAL["red"], size=25))
    piles = [3, 6, 7, 11]
    bxs = [170, 295, 420, 545]
    sd0 = ctx.seed()
    for k, (v, x) in enumerate(zip(piles, bxs)):
        hh = v * 26
        tt = b1.f(0.45 + 0.08 * k)

        def paint(d, t2, a, v=v, x=x, hh=hh, sdk=sd0 + k, tt=tt):
            p = ease_out((t2 - tt) / 0.5)
            if p <= 0:
                return
            from engine import srect
            srect(d, (x, 590 - hh * p), (80, hh * p), dim(PAL["gold"], a),
                  w=3.5, seed=sdk, fill=dim(PAL["gold"], 0.20 * a))
            if p > 0.8:
                draw_text(d, (x + 40, 590 - hh - 26), str(v), "handb", 27,
                          dim(PAL["ink"], a * (p - 0.8) / 0.2), anchor="mm")
        left.append(ctx.el(tt, paint, fade_in=0.0))
    # b2 speed-4 check
    left.append(ctx.chip(b2.f(0.12), (680, 300), "speed k = 4",
                         color=PAL["blue"], size=27))
    hrs = ["1h", "2h", "2h", "3h"]
    for k, (hlab, x) in enumerate(zip(hrs, bxs)):
        left.append(ctx.chip(b2.f(0.3 + 0.11 * k), (x + 12, 605), hlab,
                             color=PAL["cyan"], size=22, pad=9))
    left.append(ctx.text(b2.f(0.8), (170, 690),
                         "1 + 2 + 2 + 3  =  8 hours   ->  fits, exactly",
                         size=27, color=PAL["green"], cps=34))
    # b3 the k row
    hours = [27, 15, 10, 8, 8, 6, 5, 5, 5, 5, 4]
    kbar = ctx.array(b3.at(0.15), (880, 380), list(range(1, 12)), cw=74,
                     ch=64, idx_labels=False, val_size=27)
    for i in range(11):
        ctx.note(b3.f(0.40 + 0.035 * i), (880 + 74 * i + 37, 348),
                 str(hours[i]), size=20, anchor="mm")
        kbar.state(b3.f(0.42 + 0.035 * i), i, "F" if hours[i] > 8 else "T")
    ctx.note(b3.at(0.2), (1287, 480), "candidate speeds k, with hours needed",
             size=23, anchor="mm")
    # b4 instantiation card
    ctx.card(b4.at(0.1), (880, 600), 560, "THE FORMULA . LC 875",
             [("SPACE", "speeds 1 .. 11", PAL["blue"]),
              ("ASK(x)", "hours(x) <= 8 ?", PAL["gold"]),
              ("BOUNDARY", "first True", PAL["green"]),
              ("RETURN", "lo   ->   4", PAL["ink"])],
             color=PAL["gold"], stagger=0.42)
    ctx.note(b4.f(0.75), (880, 880), "hours(k) = sum of ceil(pile / k)",
             size=23)
    # b5 trace on k bar
    lo = kbar.pointer(b5.at(0.05), "lo", 0, PAL["blue"], side="bottom")
    hi = kbar.pointer(b5.at(0.15), "hi", 10, PAL["purple"], side="bottom")
    kbar.probe(b5.f(0.16), 5, "T", dur=0.9, label="")
    hi.move(b5.f(0.28), 5)
    kbar.probe(b5.f(0.38), 2, "F", dur=0.9, label="")
    lo.move(b5.f(0.50), 3)
    kbar.probe(b5.f(0.58), 4, "T", dur=0.8, label="")
    hi.move(b5.f(0.68), 4)
    kbar.probe(b5.f(0.74), 3, "T", dur=0.8, label="")
    hi.move(b5.f(0.86), 3)
    kbar.state(b5.f(0.9), 3, "found")
    ctx.circle(b5.f(0.93), (880 + 3 * 74 + 37, 412), 40, color=PAL["gold"])
    ctx.text(b5.f(0.95), (1139, 310), "k* = 4", size=30, color=PAL["gold"],
             kind="handb", anchor="mm", cps=30)
    # b6 clear left, ship card
    for e in left:
        if e.t_out is None:
            e.t_out = b6.t0 + 0.2
    ctx.card(b6.f(0.25), (150, 280), 620, "LC 1011 . SHIP IN D DAYS",
             [("SPACE", "max(w) .. sum(w)", PAL["blue"]),
              ("ASK(x)", "days at cap x <= D ?", PAL["gold"])],
             color=PAL["cyan"], stagger=0.5)
    # b7 split + bouquets cards
    ctx.card(b7.f(0.05), (150, 480), 620, "LC 410 . SPLIT ARRAY, MIN-MAX SUM",
             [("SPACE", "max(a) .. sum(a)", PAL["blue"]),
              ("ASK(x)", "chunks at cap x <= k ?", PAL["gold"])],
             color=PAL["blue"], stagger=0.45)
    ctx.card(b7.f(0.45), (150, 680), 620, "LC 1482 . M BOUQUETS",
             [("SPACE", "bloom days", PAL["blue"]),
              ("ASK(x)", "bouquets by day x >= m ?", PAL["gold"])],
             color=PAL["green"], stagger=0.45)
    ctx.chip(b7.f(0.9), (690, 495), "same row!", color=PAL["gold"], size=28)
    # b8 maximize
    ctx.clear_at(b8.t0)
    ctx.text(b8.at(0.15), (140, 240), "LC 1552 . maximize the MINIMUM gap",
             size=29, cps=32)
    ctx.chip(b8.at(0.4), (140, 290), "3 balls", color=PAL["green"], size=24)

    def px(v):
        return 240 + v * (860 / 8)
    ctx.line(b8.at(0.3), (240, 420), (1100, 420), color=PAL["ink"], w=4, dur=0.6)
    for v in (1, 2, 3, 4, 7):
        ctx.line(b8.at(0.55), (px(v), 406), (px(v), 434), color=PAL["dim"],
                 w=3, dur=0.25)
        ctx.note(b8.at(0.7), (px(v), 462), str(v), size=22, anchor="mm")
    for k, v in enumerate((1, 4, 7)):
        ctx.circle(b8.f(0.30) + 0.12 * k, (px(v), 420), 15, color=PAL["green"],
                   fill=PAL["green"], dur=0.3)
    ctx.note(b8.f(0.40), ((px(1) + px(4)) / 2, 378), "gap 3", size=23,
             color=PAL["cyan"], anchor="mm")
    ctx.note(b8.f(0.46), ((px(4) + px(7)) / 2, 378), "gap 3", size=23,
             color=PAL["cyan"], anchor="mm")
    dbar = ctx.array(b8.f(0.42), (1220, 380), [1, 2, 3, 4, 5, 6], cw=76,
                     ch=64, idx_labels=False, val_size=26)
    ctx.text(b8.f(0.36), (1448, 322), "can place, gap >= d ?", size=25,
             color=PAL["gold"], kind="handb", anchor="mm", cps=30)
    DL = list("TTTFFF")
    for i, ch in enumerate(DL):
        dbar.state(b8.f(0.55 + 0.04 * i), i, ch)
    sd = ctx.seed()

    def brk(d, tt, a):
        p = ease_out((tt - b8.f(0.78)) / 0.5)
        if p <= 0:
            return
        c = dim(PAL["gold"], a)
        sline(d, (1224, 470), (1224 + 220 * p, 470), c, w=4, seed=sd)
        if p > 0.9:
            draw_text(d, (1334, 505), "last True = 3  <-  the answer",
                      "hand", 24, c, "mm")
    ctx.el(b8.f(0.78), brk, fade_in=0.0)
    ctx.chip(b8.f(0.9), (1220, 570), "maximize  =  first False - 1",
             color=PAL["orange"])
    # b9 grid
    ctx.clear_at(b9.t0)
    grid_teaser(ctx, b9.at(0.1), [0, 1, 2], [3, 4])


scene("row45_answer", S6, build_s6)

# ================================================================== SCENE 7

S7 = [
    Beat("s7b1", """Two quick rows finish the grid. First: continuous
        answers. Square root of two. The space is the interval from one to
        two, and the question, is x squared at least two, is as monotone as
        ever. Only one thing dies: mid plus one. There is no next number on a
        continuous line. So low simply becomes mid, and instead of waiting
        for fingers to meet, we just halve a fixed number of times. Sixty
        halvings shrinks the interval below ten to the minus eighteenth.
        Sharper than any double. Run sixty, return low.""", hold=0.5),
    Beat("s7b2", """Second: LeetCode seventy-four. A matrix where each row is
        sorted, and each row starts after the last one ends. Read it row by
        row: it is one long sorted array wearing a costume. Index x maps to
        row x divided by n, column x mod n. Flatten the picture, and it is
        literally row one of our grid again.""", hold=0.5),
    Beat("s7b3", """Rows six and seven. The grid is full.""", hold=1.3),
]


def build_s7(ctx, b):
    b1, b2, b3 = b["s7b1"], b["s7b2"], b["s7b3"]
    ctx.section(b1.at(0.0), "ROWS 6 & 7 . REALS, AND 2D", color=PAL["green"])
    # ---- left: sqrt(2)
    ctx.text(b1.at(0.3), (140, 200), "continuous answers:  sqrt(2)", size=30,
             cps=30)
    ctx.text(b1.f(0.22), (140, 292), "ask(x):   x*x >= 2 ?", size=30,
             color=PAL["gold"], kind="handb", cps=30)

    def vx(v):
        return 140 + (v - 1) * 740
    loA, hiA = Anim(vx(1.0)), Anim(vx(2.0))
    hiA.to(b1.f(0.40), vx(1.5), 0.5)
    loA.to(b1.f(0.52), vx(1.25), 0.5)
    loA.to(b1.f(0.64), vx(1.375), 0.5)
    sd = ctx.seed()

    def interval(d, tt, a):
        x0, x1 = loA(tt), hiA(tt)
        d.rounded_rectangle([S(x0), S(468), S(x1), S(492)], radius=S(4),
                            fill=dim(PAL["green"], 0.22 * a))
        sline(d, (140, 480), (880, 480), dim(PAL["dim"], a), w=3, seed=sd)
        for xx, cc in ((x0, PAL["blue"]), (x1, PAL["purple"])):
            sline(d, (xx, 456), (xx, 504), dim(cc, a), w=5, seed=sd + 1)
        draw_text(d, (140, 528), "1", "hand", 24, dim(PAL["dim"], a), "mm")
        draw_text(d, (880, 528), "2", "hand", 24, dim(PAL["dim"], a), "mm")
    ctx.el(b1.at(0.5), interval, fade_in=0.4)
    for (v, lab, good, tf, yy) in ((1.5, "1.5", False, b1.f(0.40), 425),
                                   (1.25, "1.25", True, b1.f(0.52), 425),
                                   (1.375, "1.375", True, b1.f(0.64), 395)):
        c = PAL["red"] if good else PAL["green"]
        txt = lab + ("  F" if good else "  T")
        ctx.note(tf, (vx(v), yy), txt, size=22, color=c, anchor="mm")
    ctx.chip(b1.f(0.74), (140, 570), "no  mid + 1 :  just  lo = mid",
             color=PAL["purple"], size=24)
    ctx.chip(b1.f(0.84), (140, 645), "60 halvings -> width < 1e-18",
             color=PAL["green"], size=24)
    ctx.text(b1.f(0.93), (140, 725), "->  1.41421356...", size=29,
             color=PAL["ink"], cps=30)
    # ---- right: 2D
    M = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
    ctx.text(b2.at(0.1), (1060, 200), "LC 74 :  the sorted grid", size=30,
             cps=30)
    for r in range(3):
        rr = ctx.array(b2.at(0.25 + 0.12 * r), (1060, 250 + 66 * r), M[r],
                       cw=92, ch=64, idx_labels=False, val_size=25,
                       stagger=0.04)
        for c in range(4):
            ctx.note(b2.at(0.6) + 0.03 * (r * 4 + c),
                     (1060 + 92 * c + 14, 250 + 66 * r + 6),
                     str(r * 4 + c), size=14, color=PAL["purple"])
    ctx.chip(b2.f(0.55), (1480, 268), "row = x // 4", color=PAL["cyan"],
             size=24)
    ctx.chip(b2.f(0.62), (1480, 348), "col = x % 4", color=PAL["pink"],
             size=24)
    ctx.arrow(b2.f(0.5), (1240, 466), (1240, 520), color=PAL["dim"], w=4)
    flat = [v for row in M for v in row]
    ctx.array(b2.f(0.55), (1060, 540), flat, cw=58, ch=46, idx_labels=True,
              val_size=18, stagger=0.03)
    ctx.text(b2.f(0.8), (1060, 665), "one long sorted row  ->  ROW 1 again!",
             size=26, color=PAL["gold"], cps=34)
    # b3 grid
    ctx.clear_at(b3.t0)
    grid_teaser(ctx, b3.at(0.1), [0, 1, 2, 3, 4], [5, 6])


scene("rows67_reals_2d", S7, build_s7)

# ================================================================== SCENE 8

S8 = [
    Beat("s8b1", """And there it is. The whole map. Seven rows, one template.
        Read down the columns. The space is always one of three things:
        indices, candidate answers, or an interval. The question is always
        engineered to split no from yes. The boundary is always the first
        true; even last-true is just first-false minus one. And the return is
        low, wearing three different costumes.""", hold=0.6),
    Beat("s8b2", """So the next time a problem smells like binary search,
        don't start typing a while loop. Fill in a row of this grid. What is
        my space? What is my yes-or-no question, and does it flip exactly
        once? First true, or last true? And what do I return? Four cells,
        and the code writes itself. The same six lines. Every single
        time.""", hold=0.5),
    Beat("s8b3", """That's binary search. Not a trick for sorted arrays. A
        formula for finding boundaries. Happy searching.""", hold=1.4),
]


def build_s8(ctx, b):
    b1, b2, b3 = b["s8b1"], b["s8b2"], b["s8b3"]
    g = ctx.grid(b1.at(0.2), (170, 140), GRID_COLW, GRID_HEADER, GRID_ROWS,
                 rh=66, header_h=58, text_size=25, row_colors=ROW_COLORS)
    for i in range(7):
        g.show_row(b1.at(0.5 + 0.28 * i), i, hl=False)
    ctx.text(b1.at(0.05), (960, 92), "THE GRID", size=44, color=PAL["gold"],
             kind="handb", anchor="mm", cps=24)
    # b2 checklist chips
    texts = [("1. SPACE ?", PAL["blue"]),
             ("2. QUESTION - flips once ?", PAL["green"]),
             ("3. BOUNDARY - first True", PAL["gold"]),
             ("4. RETURN - transform lo", PAL["purple"])]
    widths = [text_w(s, "handb", 26) + 28 for s, _ in texts]
    total = sum(widths) + 3 * 46
    x = (W - total) / 2
    for i, ((s, c), wd) in enumerate(zip(texts, widths)):
        ctx.chip(b2.f(0.15 + 0.16 * i), (x, 742), s, color=c, size=26)
        x += wd + 46
    # b3 outro
    ctx.text(b3.f(0.25), (960, 880), "the same six lines . every time",
             size=38, color=PAL["gold"], kind="handb", anchor="mm", cps=26)
    ctx.text(b3.f(0.7), (960, 960), "happy searching.", size=32,
             color=PAL["dim"], kind="handb", anchor="mm", cps=26)


scene("finale_grid", S8, build_s8)
