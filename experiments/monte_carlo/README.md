# Monte Carlo evaluation

This step evaluates a finished SPOT backtest. It resamples the portfolio and AIQ
daily returns together, preserving the stored validation/test split and selected
estimation windows. Tracking RMSE remains the primary score, in daily-return
units. The optimization model, quarterly execution rule and base data pipeline
are unchanged.

## Input run and version

The reported experiment uses the archived evaluation run `261111_170130`. Put
these files under `output/eval/261111_170130/`:

| File | Fields used |
| --- | --- |
| `eval.json` | Solve-run id, estimation windows, `val_quarters`, `test_quarters` |
| `daily_returns.csv` | Alpha, strategy, window, quarter, date, portfolio/AIQ/active returns |
| `best_by_alpha.csv` | Selected window per alpha and validation/test day counts and RMSEs |
| `window_metrics.csv` | Full-period RMSE and day count for every source strategy path |

The archived run uses a split-aware evaluation format. The repository's
`eval.py` at base commit `a2346ac4162f` selects windows by full-sample RMSE and
does not write the same validation/test fields. Use the archived run for this
experiment; the current evaluator alone does not regenerate those inputs.
The script reports a clear error if the split metadata or split-specific
selection columns are absent. It does not infer a new split or silently select
windows on test data.

These inputs are generated evaluation files, not market-data downloads or
optimized stock weights. They remain in the existing gitignored `output/`
directory. Source price workbooks remain in `data/` under the original rules.

## Run

The experiment was executed with Python 3.12.3, separately from the base
project's declared Python 3.14 environment. The four analysis dependencies are
pinned in `requirements.txt`; the base `pyproject.toml` is unchanged. A separate
environment reproduces the recorded setup:

```bash
python3.12 -m venv .venv-monte-carlo
.venv-monte-carlo/bin/python -m pip install -r experiments/monte_carlo/requirements.txt
.venv-monte-carlo/bin/python experiments/monte_carlo/validate_daily.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-monte-carlo/bin/python experiments/monte_carlo/monte_carlo_daily.py \
  --input output/eval/261111_170130 \
  --output output/monte_carlo/261111_170130/results \
  --paths 10000 --seed 4004 --batch 250
.venv-monte-carlo/bin/python experiments/monte_carlo/report_daily.py \
  --package output/monte_carlo/261111_170130 \
  --pdf output/monte_carlo/261111_170130/Monte_Carlo_Report.pdf
```

The calculation and report commands accept explicit paths. The report renderer
targets the archived 5%-50% cap grid and its 15%/20% nine-month comparison; its
labels and narrative require adjustment for a different experiment.

## Method

For each historical quarter, draw consecutive circular blocks of the joint
strategy/AIQ return vector. Use the same sampled dates for every strategy and
truncate the final block to preserve that quarter's length. Join the quarters in
their original order. Report validation, test and combined results separately.

| Setting | Block length | Within-quarter volatility multiplier |
| --- | --- | --- |
| Main | 5 days | 1.0 |
| Independent-day sensitivity | 1 day | 1.0 |
| Block-length sensitivity | 10 and 20 days | 1.0 |
| Illustrative stress | 5 days | 1.5 |

Each setting uses 10,000 full-horizon joint paths, seed 4004 and batches of 250.
Keep the seed and batch size to reproduce the exact draws. The stress multiplies
each strategy's and AIQ's deviations from its own quarterly mean by 1.5. Main
and stressed settings share sampling indices.

RMSE includes mean active-return bias; TE is the sample standard deviation of
active daily returns. Neither score is annualized here. Paired comparisons use
the fraction of paths with lower RMSE and the distribution of the RMSE
difference. Percentiles describe the conditional bootstrap distribution; they
are not future confidence guarantees or formal significance tests.

## Outputs and checks

Results are written under `output/monte_carlo/261111_170130/`:

| File or directory | Contents |
| --- | --- |
| `results/observed_metrics.csv` | Recomputed validation, test and combined tracking scores |
| `results/mc_summary.csv` | Metric percentiles by strategy, setting and split |
| `results/paired_comparisons.csv` | Paired RMSE differences, path frequencies and Monte Carlo standard errors |
| `results/convergence.csv` | Mean-RMSE estimates and simulation standard errors as draws accumulate |
| `results/*_draws.npz` | Complete metric arrays for every simulated path |
| `results/manifest.json` | Input hashes, source metadata, seed, batch size, environment and data audit |
| `figures/` | Four standalone tracking and sensitivity figures |
| `Monte_Carlo_section.md` / `.tex` | Report text generated from the saved results |

The input audit recomputes all source full-period RMSEs, checks each
validation-selected window and its validation/test scores, and verifies the
active-return identity and common dates. Invalid inputs raise `ValueError`,
including when Python is run with optimization enabled. `validate_daily.py`
checks a perfect tracker, within-block continuity and deterministic simulation.

A full reproducibility check reran all five 10,000-path settings on the archived
inputs. The six result CSVs match the saved run byte for byte; all 80 saved arrays
match element for element.

## Limits

The original backtest already contains its realized position drift. This step
resamples the resulting strategy returns. It does not rebuild stock-price
paths, recompute holdings at block joins, reselect stocks or reoptimize weights.
The quarter sequence stays fixed and dependence across quarter boundaries is
not modeled.

The analysis is conditional and retrospective. It does not remove pooled-sample
cap-selection bias or include transaction costs, new market regimes, solver
uncertainty or uncertainty in the optimized holdings. The volatility stress is
applied to realized strategy returns, not to reconstructed stock positions.

Method reference: [arch circular block bootstrap documentation](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html).
