"""Build + render all scenes, mux narration, concat into the final video.

Usage:
  python3 render_all.py check          # build scenes, verify anchors
  python3 render_all.py qc [sid ...]   # snapshot frames for QC
  python3 render_all.py render [sid ...]  # render scene mp4s (parallel)
  python3 render_all.py concat         # concat scene mp4s -> final
"""
import os, sys, subprocess, json
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")
OUT = os.path.join(HERE, "scenes_out")
QC = os.path.join(HERE, "qc")
os.makedirs(OUT, exist_ok=True)
os.makedirs(QC, exist_ok=True)

FPS = 24
TAIL = 1.0  # seconds of visual hold after narration ends

sys.path.insert(0, HERE)
from engine import Scene           # noqa: E402
from narration import SCENES       # noqa: E402
from sync import Timing            # noqa: E402
from scenes_a import BUILDERS_A    # noqa: E402
from scenes_b import BUILDERS_B    # noqa: E402

BUILDERS = {**BUILDERS_A, **BUILDERS_B}


def build_scene(sid):
    spec = next(s for s in SCENES if s["id"] == sid)
    T = Timing(sid, spec["text"])
    sc = Scene(sid, duration=T.audio_end, fps=FPS)
    BUILDERS[sid](sc, T)
    return sc, T


def do_check():
    ok = True
    for spec in SCENES:
        sid = spec["id"]
        try:
            sc, T = build_scene(sid)
            print(f"{sid}: OK  dur={T.audio_end:.1f}s items={len(sc.items)}")
        except Exception as e:
            ok = False
            print(f"{sid}: FAIL  {type(e).__name__}: {e}")
    return ok


def do_qc(sids):
    for sid in sids:
        sc, T = build_scene(sid)
        for frac in (0.28, 0.62, 0.97):
            t = T.audio_end * frac
            png = os.path.join(QC, f"{sid}_{int(frac*100):02d}.png")
            sc.snapshot(t, png)
        print(f"{sid}: qc frames done", flush=True)


def render_one(sid):
    sc, T = build_scene(sid)
    mp3 = os.path.join(AUDIO, f"{sid}.mp3")
    out = os.path.join(OUT, f"{sid}.mp4")
    sc.render(out, audio_path=mp3, tail=TAIL)
    sz = os.path.getsize(out) / 1e6
    return f"{sid}: rendered {T.audio_end + TAIL:.1f}s -> {sz:.1f} MB"


def do_render(sids, procs=4):
    with Pool(procs) as p:
        for msg in p.imap_unordered(render_one, sids):
            print(msg, flush=True)


def do_concat():
    lst = os.path.join(OUT, "list.txt")
    with open(lst, "w") as f:
        for spec in SCENES:
            f.write(f"file '{OUT}/{spec['id']}.mp4'\n")
    final = os.path.join(HERE, "hnmd_explainer.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", "-movflags", "+faststart", final],
                   check=True, stderr=subprocess.DEVNULL)
    sz = os.path.getsize(final) / 1e6
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "json", final],
                       capture_output=True, text=True)
    dur = float(json.loads(r.stdout)["format"]["duration"])
    print(f"FINAL: {final}  {dur/60:.1f} min  {sz:.1f} MB")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    args = sys.argv[2:]
    all_ids = [s["id"] for s in SCENES]
    if mode == "check":
        sys.exit(0 if do_check() else 1)
    elif mode == "qc":
        do_qc(args or all_ids)
    elif mode == "render":
        do_render(args or all_ids)
    elif mode == "concat":
        do_concat()
