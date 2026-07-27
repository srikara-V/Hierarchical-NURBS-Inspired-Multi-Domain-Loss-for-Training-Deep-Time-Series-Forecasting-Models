"""Build the video: TTS -> timeline -> frames -> mux.

Usage:
  python build.py stills [--fake]     render one PNG per beat for QC
  python build.py still <t> [--fake]  render a single frame at time t
  python build.py render              full video (requires real TTS)
  python build.py grid                export the master grid as grid.png
"""
import os
import subprocess
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import engine  # noqa: E402
import storyboard as sb  # noqa: E402
import tts  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
FINAL = os.path.join(OUT, "binary_search_one_formula.mp4")


def timeline(fake=False):
    res = tts.ensure_all(sb.all_beats(), sb.VOICE_ID, sb.MODEL_ID, fake=fake)
    durs = {k: d for k, (_, d) in res.items()}
    tl = sb.build_timeline(durs)
    return tl, res


def stills(fake=False):
    tl, _ = timeline(fake=fake)
    sd = os.path.join(OUT, "stills")
    os.makedirs(sd, exist_ok=True)
    shots = []
    for i, scn in enumerate(sb.SCENES):
        for bt in scn["beats"]:
            shots.append((f"{bt.key}", bt.t0 + bt.dur * 0.72))
            shots.append((f"{bt.key}_end", bt.t0 + bt.dur + bt.hold * 0.9))
    for name, t in shots:
        img = tl.draw(t)
        img.save(os.path.join(sd, f"{name}.png"))
    print(f"wrote {len(shots)} stills to {sd}  (total {tl.duration:.1f}s)")


def still_at(t, fake=False):
    tl, _ = timeline(fake=fake)
    p = os.path.join(OUT, f"frame_{t:.2f}.png")
    tl.draw(t).save(p)
    print(p)


def mix_audio(tl, res, path):
    sr = 44100
    total = int((tl.duration + 0.5) * sr)
    mixdown = np.zeros(total, dtype=np.float32)
    for scn in sb.SCENES:
        for bt in scn["beats"]:
            mp3, _ = res[bt.key]
            if mp3 is None:
                raise RuntimeError(f"no audio for {bt.key}")
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", mp3, "-f", "s16le",
                 "-ar", str(sr), "-ac", "1", "-"],
                capture_output=True).stdout
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
            i0 = int(bt.t0 * sr)
            mixdown[i0:i0 + len(x)] += x
    peak = np.abs(mixdown).max()
    if peak > 0.98:
        mixdown *= 0.98 / peak
    pcm = (mixdown * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def render():
    tl, res = timeline(fake=False)
    n_frames = int(tl.duration * engine.FPS)
    workers = max(1, (os.cpu_count() or 4))
    bounds = [round(n_frames * i / workers) for i in range(workers + 1)]
    parts = [os.path.join(OUT, f"part_{i:02d}.mp4") for i in range(workers)]

    import multiprocessing as mp
    procs = []
    for i in range(workers):
        p = mp.Process(target=engine.render_range,
                       args=(tl, bounds[i], bounds[i + 1], parts[i]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError("render worker failed")

    concat = os.path.join(OUT, "concat.txt")
    with open(concat, "w") as f:
        for pth in parts:
            f.write(f"file '{pth}'\n")
    silent = os.path.join(OUT, "video_noaudio.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe",
                    "0", "-i", concat, "-c", "copy", silent], check=True)
    wav = os.path.join(OUT, "narration.wav")
    mix_audio(tl, res, wav)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", FINAL], check=True)
    for pth in parts + [silent, concat]:
        os.remove(pth)
    sz = os.path.getsize(FINAL) / 1e6
    print(f"DONE {FINAL}  {tl.duration / 60:.1f} min  {sz:.1f} MB")


def export_grid():
    """Standalone master-grid PNG (the user's problem-type grid asset)."""
    from PIL import Image, ImageDraw
    from engine import Scene, SceneCtx, PAL, S, W, H, SS, draw_text, dim
    scn = Scene("grid", 0.0)
    ctx = SceneCtx(scn)
    g = ctx.grid(0.0, (170, 150), sb.GRID_COLW, sb.GRID_HEADER, sb.GRID_ROWS,
                 rh=78, header_h=62, text_size=26, row_colors=sb.ROW_COLORS)
    for i in range(7):
        g.show_row(0.0, i, hl=False)
    ctx.text(0.0, (960, 84), "BINARY SEARCH . ONE FORMULA, EVERY FLAVOR",
             size=40, color=PAL["gold"], kind="handb", anchor="mm",
             write=False)
    texts = [("1. SPACE ?", PAL["blue"]),
             ("2. QUESTION - flips once ?", PAL["green"]),
             ("3. BOUNDARY - first True", PAL["gold"]),
             ("4. RETURN - transform lo", PAL["purple"])]
    from engine import text_w
    widths = [text_w(s, "handb", 26) + 28 for s, _ in texts]
    total = sum(widths) + 3 * 46
    x = (W - total) / 2
    for (s, c), wd in zip(texts, widths):
        ctx.chip(0.0, (x, 900), s, color=c, size=26)
        x += wd + 46
    tmpl = "lo,hi=first,last+1   |   while lo<hi: m=(lo+hi)//2; hi=m if ask(m) else (lo:=m+1)   |   return lo"
    ctx.note(0.0, (960, 1000), tmpl, size=22, anchor="mm")
    img = Image.new("RGB", (W * SS, H * SS), PAL["bg"])
    d = ImageDraw.Draw(img)
    for el in sorted(scn.els, key=lambda e: e.z):
        el.paint(d, 10.0, 1.0)
    img = img.resize((W, H), Image.LANCZOS)
    p = os.path.join(OUT, "grid.png")
    img.save(p)
    print(p)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stills"
    fake = "--fake" in sys.argv
    if mode == "stills":
        stills(fake=fake)
    elif mode == "still":
        still_at(float(sys.argv[2]), fake=fake)
    elif mode == "render":
        render()
    elif mode == "grid":
        export_grid()
