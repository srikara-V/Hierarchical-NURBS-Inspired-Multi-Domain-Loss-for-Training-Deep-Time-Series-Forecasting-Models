# HNMD Explainer Video (Khan-Academy style)

`hnmd_explainer.mp4` is a ~17.5-minute narrated chalkboard explainer of the
paper's methodology, built from first principles for an audience that knows
basic ML but has no time-series or spline background.

## Contents

| Scene | Topic |
|-------|-------|
| 1 | Cold open: improve the loss, not the model |
| 2 | Time-series forecasting: look-back, horizon, channels |
| 3 | Training an MLP forecaster; MSE |
| 4 | What point-wise losses can't see (flat vs shifted forecasts) |
| 5 | Roadmap: splines, hierarchy, multi-domain report card |
| 6 | Polynomials → knots → piecewise cubics (splines) |
| 7 | B-spline basis functions via Cox–de Boor |
| 8 | Fitting = linear regression on bumps, closed-form solve |
| 9 | NURBS: rational weighted basis |
| 10 | HNMD twist: softmax(window-variance) weights ("attention") |
| 11 | Complexity-based decomposition (residual peeling) |
| 12 | The full operator S, step by step |
| 13 | Multi-domain error: time, derivative, FFT + α/β/γ |
| 14 | Level-importance gradient scaling in the backward pass |
| 15 | Results: fixed MLP, MSE vs Tilde-Q vs HNMD |
| 16 | Recap and future work |

## How it was made

- **Narration**: ElevenLabs TTS (`eleven_multilingual_v2`, voice "Eric"),
  one clip per scene with character-level timestamps
  (`gen_audio.py`, needs `ELEVEN_LABS_API` env var).
- **Visuals**: a small matplotlib "chalkboard" engine (`engine.py`) that
  draws hand-drawn wobbly strokes, handwritten text, math write-ins,
  arrows and highlights, synced to narration timestamps (`sync.py`).
  All spline/NURBS curves in the diagrams are computed with a real
  Cox–de Boor implementation, and the hierarchical-decomposition figures
  are genuine least-squares residual fits, not sketches.
- **Scenes**: `scenes_a.py` / `scenes_b.py` define every diagram and its
  timing anchors into the narration text (`narration.py`).
- **Assembly**: `render_all.py` renders frames straight into ffmpeg
  (1080p, 24 fps), muxes each scene with its narration, and concatenates.

## Rebuild

```bash
pip install numpy matplotlib requests    # plus ffmpeg on PATH
cd video
mkdir -p fonts && cd fonts               # handwriting fonts (OFL licensed)
curl -sSL -o PatrickHand.ttf "https://fonts.gstatic.com/s/patrickhand/v25/LDI1apSQOAYtSuYWp8ZhfYeMWQ.ttf"
curl -sSL -o Kalam-Regular.ttf "https://fonts.gstatic.com/s/kalam/v18/YA9dr0Wd4kDdMuhW.ttf"
cd ..
export ELEVEN_LABS_API=...               # ElevenLabs API key
python3 gen_audio.py                     # narration + timestamps
python3 render_all.py check              # verify scenes/anchors
python3 render_all.py render             # render scene mp4s (parallel)
python3 render_all.py concat             # -> hnmd_explainer.mp4
```
