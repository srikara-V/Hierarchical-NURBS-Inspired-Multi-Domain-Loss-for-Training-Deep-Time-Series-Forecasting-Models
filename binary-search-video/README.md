# Binary Search: One Formula, Every Flavor

A Khan-academy-style explainer video (~16 min, 1080p30) that rebuilds binary
search as a single "first-True boundary" formula, then fills out a grid of
every flavor — with canonical LeetCode problems mapped onto the same four
decisions each time (SPACE / QUESTION / BOUNDARY / RETURN).

- `out/binary_search_one_formula.mp4` — the video (male ElevenLabs voice-over)
- `out/grid.png` — the master grid as a standalone 1080p reference image
- `script.md` — full narration + the template + the grid as markdown

## Coverage

| Row | Flavor | Canonical problems |
|---|---|---|
| 1 | Sorted array: find / insert / first-last | LC 704, 35, 34 (bisect_left/right) |
| 2 | Rotated array: min / search (+ duplicates caveat) | LC 153, 33, 154 |
| 3 | Peak finding (invariant > monotonicity) | LC 162 |
| 4 | Answer-space, MINIMIZE (feasibility + greedy check) | LC 875, 1011, 410, 1482 |
| 5 | Answer-space, MAXIMIZE (first False - 1) | LC 1552, floor-sqrt |
| 6 | Real-valued answers (fixed halvings) | sqrt(2) |
| 7 | 2D matrix as flattened row | LC 74 |

## Rebuilding

```bash
export ELEVEN_LABS_API=sk_...   # ElevenLabs API key
pip install pillow numpy requests   # plus ffmpeg on PATH
python build.py stills   # QC still per narration beat -> out/stills/
python build.py render   # full video -> out/binary_search_one_formula.mp4
python build.py grid     # out/grid.png
```

Narration is cached in `out/cache/` keyed by (voice, model, text), so edits
to visuals never re-spend TTS credits; edited narration re-synthesizes only
the changed beats.

## How it's made

- `engine.py` — chalkboard renderer: wobbly hand-drawn primitives (PIL at 2x
  supersample), keyframe animation, array/pointer/code/grid widgets, and
  parallel chunked frame rendering piped straight into ffmpeg.
- `storyboard.py` — 8 scenes, 46 narration beats, each beat paired with
  synchronized board drawing (traces animate against the narration clock).
- `tts.py` / `build.py` — ElevenLabs synthesis + timeline assembly + mux.
