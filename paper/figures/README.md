# Paper figures

**Strict mode:** every numeric panel is built from your `./results/` artifacts and MSSD code paths. If anything required is missing, the script **raises an error** and writes no substitute curves.

```bash
env/bin/python paper/generate_figures.py
```

Requires: `torch`, `matplotlib`, `numpy`, `pandas`, and completed runs under `./results/` (or `./results/results/`).

## What is real vs schematic

| Figure | Numeric data? | Source |
|--------|---------------|--------|
| fig1 (a) | **No** — method diagram only (HTML/matplotlib layout, not time series) |
| fig1 (b,c) | **Yes** | `true.npy` + NURBS (`utils/mssd.py`); hyperparams from `configs/hyperparams.py` |
| fig2–3, fig5–6 | **Yes** | `pred.npy` / `true.npy` / `metrics.npy` + MSSD decomposition |
| fig4 | **Yes** | Test MSE from `metrics.npy`, averaged over horizons per dataset |

Default reference run: **ETTm1**, **MLP**, **T=96**, test sample **0**, channel **0** (`paper/figure_data.py`).

## Preflight checks

Before plotting, `validate_figure_prerequisites()` requires:

- PyTorch importable
- HNMD + MSE runs for ETTm1 / MLP / T=96 (`pred.npy`, `true.npy`, `metrics.npy`)
- For fig4: MLP runs with MSE, Tilde-Q, and HNMD for ETTm1, ETTm2, ETTh1, ETTh2, ECL, Traffic (all horizons averaged)

If ECL or Traffic experiments are not finished, generation will **fail** until those folders exist.
