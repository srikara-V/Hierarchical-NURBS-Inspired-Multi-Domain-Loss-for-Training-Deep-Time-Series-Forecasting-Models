# Preprint experiment plan (executed on Modal, July 2026)

All runs: look-back 96, horizons {96,192,336,720}, Adam lr per dataset config,
10 epochs, early stopping patience 3, seed 2024 unless noted. Artifacts live on
the Modal volume `hnmd-workspace` mirroring the repo layout.

## Phase 1 — Complete the main MLP comparison (results root `./results`)
MLP x {MSE, Tilde-Q, HNMD(tuned)} x 9 datasets x 4 horizons = 108 cells.
64 existed on the volume; the ~45 missing cells (electricity/exchange/traffic/
weather/Solar at horizons 192-720, ETTm2 mssd@720, Solar@96 remainder) are run here.

## Phase 2 — Architecture generality (results root `./results`)
DLinear and iTransformer x {MSE, Tilde-Q, HNMD} x {ETTh1, ETTh2, ETTm1, ETTm2}
x 4 horizons = 96 runs.
- DLinear HNMD hyperparameters: `HNMV_DLinear_optimized.xlsx` (tuned for DLinear).
- iTransformer HNMD hyperparameters: transferred unchanged from the MLP-tuned
  table (tests transferability of the loss configuration).

## Phase 3 — Ablation study (results root `./results_ablation/<variant>`)
MLP, seed 2024, on cells where every tuned domain weight is nonzero:
ETTh1@{96,336}, ETTm1@{96,336}, ETTm2@{96,336}. 8 variants x 6 cells = 48 runs:
1. `noalpha`   - alpha=0 (no time-domain term)
2. `nobeta`    - beta=0 (no derivative term)
3. `nogamma`   - gamma=0 (no frequency term)
4. `timeonly`  - beta=0, gamma=0 (time-domain only)
5. `levels1`   - max_levels=1 (single spline level, no hierarchy)
6. `levels3`   - max_levels=3 (shallow hierarchy)
7. `xp0`       - spline_criterion_exponent=0 (uniform level-importance gradients)
8. `nodecomp`  - max_levels=0 (multi-domain loss on the raw series, no splines)
The "Full" row of the ablation table reuses the Phase-1 tuned run.
Variants live in separate results roots because `max_levels` is not encoded in
the run's setting name.

## Phase 4 — Fixed-default HNMD (results root `./results_default`)
MLP x HNMD with one global config (alpha=0.01, beta=1, gamma=1, knot_multiplier=5,
exponent=1, max_levels=5) on all 9 datasets x 4 horizons = 36 runs. Measures how
much of the gain survives without per-dataset tuning.

## Phase 5 — Seed robustness (results root `./results_seeds/s<seed>`)
MLP x {MSE, Tilde-Q, HNMD} x {ETTh1, ETTm1} x 4 horizons x seeds {2025, 2026}
= 48 runs; combined with the seed-2024 runs from Phase 1 for mean +/- std.

## Phase 6 — Training-cost benchmark (single job)
Median wall-clock per optimizer step for MLP under each loss on ETTh1@{96,336,720}
and weather@96 (100 timed steps after warmup), one L4 GPU.

Job runner: `modal_run_jobs.py` + `make_job_list.py` (this repo).
