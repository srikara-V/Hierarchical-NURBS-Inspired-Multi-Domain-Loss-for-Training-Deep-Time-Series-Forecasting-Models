# Hierarchical NURBS-Inspired Multi-Domain Loss (HNMD / MSSD)

Authors: Jason Abohwo, Srikara Vishnubhatla

This repository trains long-horizon forecasting models with the HNMD (`mssd`) loss, plus MSE and Tilde-Q baselines.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Device selection

Training picks an accelerator automatically when `--use_gpu` is enabled (default):

1. **CUDA** (NVIDIA GPU)
2. **MPS** (Apple Silicon GPU on macOS)
3. **CPU** (fallback)

Mixed precision (`--use_amp`) is enabled only on CUDA. On Mac, training uses MPS without AMP. Pass `--use_gpu False` to force CPU.

Place datasets into `./data/` (see [scripts/multivariate_forecasting/README.md](scripts/multivariate_forecasting/README.md)). Expected layout:

```
data/
  ETT-small/          # ETTm1.csv, ETTm2.csv, ETTh1.csv, ETTh2.csv
  electricity/        # electricity.csv
  exchange_rate/      # exchange_rate.csv
  weather/            # weather.csv
  traffic/            # traffic.csv
  Solar/              # solar_AL.txt
```

## Quick start

Print the full MLP + HNMD sweep (no training):

```bash
python run_experiments.py --models MLP --datasets all --losses mssd --dry_run
```

Run tuned HNMD for MLP on all 9 datasets:

```bash
python run_experiments.py --models MLP --datasets all --losses mssd
```

Run priority cross-model comparison:

```bash
python run_experiments.py --models MLP DLinear iTransformer --datasets all --losses mssd mse tildeq --seeds 2024 --itr 3
```

## Experiment drivers

| Script | Purpose |
|--------|---------|
| `run_experiments.py` | Main CLI for all models, datasets, losses, seeds |
| `search_greek.py` | Tuned MLP + HNMD on all datasets |
| `experimentation.py` | Full MSSD grid search (knot multiplier × exponent) |
| `experimentation_mse.py` | MSE baselines (3 seeds via `--itr 3`) |
| `experimentation_tildeq.py` | Tilde-Q baselines (3 seeds via `--itr 3`) |
| `experimentation_mssdtraffic.py` | Tuned MLP + HNMD on Traffic only |
| `to_excel.py` | Aggregate `./results/*/metrics.npy` into `output.xlsx` |
| `filltables.py` | Build paper Tables 2, 3, and 4 into `tables/filled_tables.xlsx` |
| `paper/generate_tables.py` | Emit LaTeX `stats.tex` and `tables_generated.tex` from `./results/` |

Shell shortcuts live in `scripts/hnmd/`.

## Paper (LaTeX)

Source: [paper/main.tex](paper/main.tex). Build after experiments:

```bash
python filltables.py --no-run_missing
python paper/generate_tables.py
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex
```

Abstract win-counts are **auto-generated** from `./results/` (not hand-edited). PatchTST/TiDE/Time-LLM numbers in the paper are cited reported baselines, not runs from this repo unless you add them to `filltables.py` coverage.

## Supported models

`MLP`, `DLinear`, `SOFTS`, `iTransformer`, `iInformer`, `iReformer`, `iFlowformer`, `iFlashformer`, `Autoformer`, `NLinear`

## Supported losses

| Flag | Description |
|------|-------------|
| `mse` | Mean squared error |
| `mssd` | HNMD / MSSD multi-domain loss |
| `tildeq` | Tilde-Q loss |

MSSD hyperparameters for `MLP` and `DLinear` are loaded from the bundled Excel tables (`HNMV_*_optimized.xlsx`), with fallbacks for Solar.

## Ablation notes

Domain ablations can be run by setting `--alpha 0`, `--beta 0`, or `--gamma 0` on the command line. Hierarchy depth is controlled with `--max_levels`.

## Single-run example

```bash
python run.py \
  --is_training 1 \
  --root_path ./data/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_MLP \
  --model MLP \
  --data ETTm1 \
  --features M \
  --freq t \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 256 --d_ff 512 \
  --loss mssd \
  --alpha 0.01 --beta 5 --gamma 1 \
  --knot_multiplier 5 --spline_criterion_exponent 1 \
  --seed 2024 --itr 1
```

Results are written to `./results/<setting>/metrics.npy` and appended to `result_long_term_forecast.txt`.

## Stop and resume safely

Runs are designed to be interrupt-friendly (Ctrl+C or closing the terminal mid-job):

| Situation | What happens on re-run |
|-----------|-------------------------|
| Run finished (`metrics.npy` exists) | **Skipped** (`--skip_if_done`, default on) |
| Stopped mid-training | **Resumes** from last completed epoch (`training_state.pth`) |
| Training done but test never ran | **Test only** (`--recover`, default on) |

Re-run the same command to continue:

```bash
python filltables.py
# or any individual run.py / run_experiments.py command
```

To force a full retrain, pass `--no-skip_if_done --no-resume`.

## Fill paper tables

After experiments finish (or to run missing ones automatically):

```bash
# Run any missing experiments, then write tables (default)
python filltables.py --results_dir ./results --output ./tables/filled_tables.xlsx

# Preview which experiments would run, without training
python filltables.py --dry_run

# Only build tables from existing ./results (no training)
python filltables.py --no-run_missing
```

This produces Excel sheets for **Table 2**, **Table 3**, **Table 4**, plus coverage and raw result summaries. Missing cells for models not in the repo (Time-LLM, PatchTST, TiDE, etc.) stay as `-` and are listed on the **SkippedJobs** sheet. Table 4 is always populated from the bundled MSSD hyperparameter configs. Use `--aggregate best` to pick the lowest-MSE run when multiple seeds exist.

Then regenerate the PDF tables:

```bash
python paper/generate_tables.py
```
