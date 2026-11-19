"""Conditional, paired block bootstrap of the team's realized strategy returns.

Uses final eval CSVs, not inferred weights or a fresh optimizer run. The observed
backtest already includes its original position drift. This bootstrap samples
strategy-return vectors; it does not recompute holdings after block boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd

DEFAULT_PATH_COUNT = 10_000
DEFAULT_SEED = 4004
DEFAULT_BATCH_SIZE = 250
SCENARIOS: tuple[tuple[str, int, float], ...] = (
    ("blocks_5", 5, 1),
    ("iid_1", 1, 1),
    ("blocks_10", 10, 1),
    ("blocks_20", 20, 1),
    ("volatility_1p5", 5, 1.5),
)


class QuarterReturns(TypedDict):
    """Synchronized benchmark and strategy returns for one historical quarter."""

    quarter: str
    values: np.ndarray
    split: str


def _circular_block_indices(
    generator: np.random.Generator,
    path_count: int,
    day_count: int,
    block_days: int,
) -> np.ndarray:
    """Sample consecutive circular blocks, truncating to the original day count."""
    block_starts = generator.integers(0, day_count, size=(path_count, (day_count + block_days - 1) // block_days))
    return ((block_starts[..., None] + np.arange(block_days)) % day_count).reshape(path_count, -1)[:, :day_count]


def _tracking_metrics(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate daily tracking and compounded-return metrics on paired paths."""
    # Portfolio shape: (paths, days, strategies); benchmark shape: (paths, days).
    active_returns = portfolio_returns - benchmark_returns[..., None]
    portfolio_centered = portfolio_returns - portfolio_returns.mean(axis=1, keepdims=True)
    benchmark_centered = benchmark_returns - benchmark_returns.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        (portfolio_centered * portfolio_centered).sum(axis=1)
        * (benchmark_centered * benchmark_centered).sum(axis=1)[:, None]
    )
    return {
        "rmse": np.sqrt((active_returns * active_returns).mean(axis=1)),
        "te": active_returns.std(axis=1, ddof=1),
        "corr": (portfolio_centered * benchmark_centered[..., None]).sum(axis=1) / denominator,
        "mean_active": active_returns.mean(axis=1),
        "terminal_gap": np.prod(1 + portfolio_returns, axis=1) - np.prod(1 + benchmark_returns, axis=1)[:, None],
    }


def _load_inputs(
    input_dir: Path, output_dir: Path
) -> tuple[dict[str, Any], list[QuarterReturns], list[str], dict[str, np.ndarray], dict[str, Any]]:
    """Audit the archived evaluation and retain its validation-selected windows."""
    evaluation_metadata = json.loads((input_dir / "eval.json").read_text())
    daily = pd.read_csv(input_dir / "daily_returns.csv", parse_dates=["date"])
    selected_windows = pd.read_csv(input_dir / "best_by_alpha.csv")
    source_metrics = pd.read_csv(input_dir / "window_metrics.csv")
    required_metadata = {"solve_run", "window_months", "val_quarters", "test_quarters"}
    missing_metadata = required_metadata - evaluation_metadata.keys()
    if missing_metadata:
        raise ValueError(
            f"eval.json is missing validation/test metadata: {sorted(missing_metadata)}. "
            "Use the archived split-aware evaluation run described in the README."
        )
    required_selection = {"alpha", "window_months", "val_n_days", "val_rmse", "test_n_days", "test_rmse"}
    missing_selection = required_selection - set(selected_windows.columns)
    if missing_selection:
        raise ValueError(f"best_by_alpha.csv is missing split-specific columns: {sorted(missing_selection)}")
    if daily.duplicated(["alpha", "strategy", "window_months", "date"]).any():
        raise ValueError("Daily returns contain duplicate strategy/date rows.")
    if not (np.isfinite(daily[["portfolio_return", "benchmark_return", "active_return"]]).all().all()):
        raise ValueError("Daily returns contain missing or nonfinite values.")
    if not (np.max(np.abs(daily.portfolio_return - daily.benchmark_return - daily.active_return)) < 1e-12):
        raise ValueError("Active returns do not equal portfolio returns minus benchmark returns.")
    if not (daily.portfolio_return.min() > -1 and daily.benchmark_return.min() > -1):
        raise ValueError("Simple returns must be greater than -1.")
    # Audit every one of the 70 source strategy paths against its published RMSE.
    max_rmse_gap = 0.0
    for row in source_metrics.itertuples(index=False):
        select = np.isclose(daily.alpha, row.alpha) & daily.strategy.eq(row.strategy)
        select &= (
            daily.window_months.isna() if pd.isna(row.window_months) else daily.window_months.eq(row.window_months)
        )
        strategy_daily = daily.loc[select].sort_values("date")
        if not (len(strategy_daily) == row.n_days):
            raise ValueError("A source strategy has an unexpected number of evaluation days.")
        rmse_difference = abs(float(np.sqrt(np.mean(strategy_daily.active_return.to_numpy() ** 2))) - row.rmse)
        max_rmse_gap = max(max_rmse_gap, rmse_difference)
        if not (rmse_difference < 1e-12):
            raise ValueError("A recomputed full-period RMSE disagrees with window_metrics.csv.")
    calendar = daily[["date", "quarter", "benchmark_return"]].drop_duplicates().sort_values("date")
    if calendar.date.duplicated().any():
        raise ValueError("Benchmark dates contain inconsistent quarter or return values.")
    if set(calendar.quarter) != set(evaluation_metadata["val_quarters"] + evaluation_metadata["test_quarters"]):
        raise ValueError("Evaluation quarters do not match the validation/test metadata.")
    names = []
    columns = []
    definitions = []
    for row in selected_windows.itertuples(index=False):
        alpha = float(row.alpha)
        window_months = int(row.window_months)
        strategy_daily = daily.loc[
            np.isclose(daily.alpha, alpha) & daily.strategy.eq("miqp") & daily.window_months.eq(window_months)
        ].sort_values("date")
        if not (np.array_equal(strategy_daily.date.to_numpy(), calendar.date.to_numpy())):
            raise ValueError("An optimized strategy does not share the benchmark calendar.")
        # Verify validation-selected windows, not an ex post re-selection.
        candidates = []
        for window in evaluation_metadata["window_months"]:
            window_daily = daily.loc[
                np.isclose(daily.alpha, alpha)
                & daily.strategy.eq("miqp")
                & daily.window_months.eq(window)
                & daily.quarter.isin(evaluation_metadata["val_quarters"])
            ]
            candidates.append((np.sqrt(np.mean(window_daily.active_return.to_numpy() ** 2)), window))
        if window_months != min(candidates)[1]:
            raise ValueError("A stored estimation window was not selected by validation RMSE.")
        for split, split_quarters in [
            ("val", evaluation_metadata["val_quarters"]),
            ("test", evaluation_metadata["test_quarters"]),
        ]:
            window_daily = strategy_daily.loc[strategy_daily.quarter.isin(split_quarters)]
            if len(window_daily) != getattr(row, f"{split}_n_days"):
                raise ValueError("A selected strategy has an unexpected split length.")
            if not (
                abs(np.sqrt(np.mean(window_daily.active_return.to_numpy() ** 2)) - getattr(row, f"{split}_rmse"))
                < 1e-12
            ):
                raise ValueError("A selected strategy RMSE disagrees with best_by_alpha.csv.")
        name = f"miqp_a{round(alpha*100):02d}_L{window_months}"
        names.append(name)
        columns.append(strategy_daily.portfolio_return.to_numpy())
        definitions.append(
            {"strategy": name, "source_strategy": "miqp", "alpha": alpha, "window_months": window_months}
        )
    # Additional baselines at the recommended and alternative cap.
    for strategy, alpha in [("ew", 0.15), ("topk_ew", 0.15), ("topk_dw", 0.15), ("topk_ew", 0.20), ("topk_dw", 0.20)]:
        strategy_daily = daily.loc[np.isclose(daily.alpha, alpha) & daily.strategy.eq(strategy)].sort_values("date")
        if not (np.array_equal(strategy_daily.date.to_numpy(), calendar.date.to_numpy())):
            raise ValueError("A comparison strategy does not share the benchmark calendar.")
        name = "ew" if strategy == "ew" else f"{strategy}_a{round(alpha*100):02d}"
        names.append(name)
        columns.append(strategy_daily.portfolio_return.to_numpy())
        definitions.append(
            {
                "strategy": name,
                "source_strategy": strategy,
                "alpha": None if strategy == "ew" else alpha,
                "window_months": None,
            }
        )
    matrix = np.column_stack([calendar.benchmark_return.to_numpy(), *columns])
    quarters = []
    for quarter in calendar.quarter.drop_duplicates():
        mask = calendar.quarter.eq(quarter).to_numpy()
        quarters.append(
            {
                "quarter": quarter,
                "values": matrix[mask],
                "split": "validation" if quarter in evaluation_metadata["val_quarters"] else "test",
            }
        )
    masks = {
        "validation": calendar.quarter.isin(evaluation_metadata["val_quarters"]).to_numpy(),
        "test": calendar.quarter.isin(evaluation_metadata["test_quarters"]).to_numpy(),
        "combined": np.ones(len(calendar), dtype=bool),
    }
    observed_rows = []
    for split, mask in masks.items():
        metric_values = _tracking_metrics(matrix[None, mask, 1:], matrix[None, mask, 0])
        for j, name in enumerate(names):
            observed_rows.append(
                {
                    "split": split,
                    "strategy": name,
                    "n_days": int(mask.sum()),
                    **{key: float(value[0, j]) for key, value in metric_values.items()},
                }
            )
    pd.DataFrame(observed_rows).to_csv(output_dir / "observed_metrics.csv", index=False)
    pd.DataFrame(definitions).to_csv(output_dir / "strategy_definitions.csv", index=False)
    selected = pd.DataFrame(matrix[:, 1:], columns=names)
    selected.insert(0, "benchmark_return", matrix[:, 0])
    selected.insert(0, "quarter", calendar.quarter.to_numpy())
    selected.insert(0, "date", calendar.date.to_numpy())
    selected.to_csv(output_dir / "selected_daily_returns.csv", index=False)
    audit = {
        "source_rows": len(daily),
        "source_paths_checked": len(source_metrics),
        "max_rmse_difference": max_rmse_gap,
        "days": len(calendar),
        "n_selected_strategies": len(names),
        "validation_selected_windows_verified": True,
        "reported_validation_and_test_rmse_verified": True,
        "periods": {
            split: {
                "start": str(calendar.loc[mask, "date"].min().date()),
                "end": str(calendar.loc[mask, "date"].max().date()),
                "n_days": int(mask.sum()),
            }
            for split, mask in masks.items()
        },
    }
    return evaluation_metadata, quarters, names, masks, audit


def _simulate_paths(
    quarter_returns: list[QuarterReturns],
    strategy_names: list[str],
    split_masks: dict[str, np.ndarray],
    path_count: int,
    seed: int,
    block_days: int,
    volatility_scale: float,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Resample joint returns within quarters and summarize each evaluation split."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, block_days]).spawn(len(quarter_returns))]
    result = {
        split: {
            key: np.empty((path_count, len(strategy_names)))
            for key in ("rmse", "te", "corr", "mean_active", "terminal_gap")
        }
        for split in split_masks
    }
    for start in range(0, path_count, batch_size):
        batch_count = min(batch_size, path_count - start)
        quarter_samples = []
        for quarter, generator in zip(quarter_returns, streams):
            values = quarter["values"]
            mean = values.mean(axis=0)
            values = mean + volatility_scale * (values - mean)
            sampled = values[_circular_block_indices(generator, batch_count, len(values), block_days)]
            if not (np.isfinite(sampled).all() and sampled.min() > -1):
                raise ValueError("Resampling produced invalid simple returns.")
            quarter_samples.append(sampled)
        sampled_returns = np.concatenate(quarter_samples, axis=1)
        for split, mask in split_masks.items():
            metric_values = _tracking_metrics(sampled_returns[:, mask, 1:], sampled_returns[:, mask, 0])
            for key, values in metric_values.items():
                if not (np.isfinite(values).all()):
                    raise ValueError("A simulated tracking metric is nonfinite.")
                result[split][key][start : start + batch_count] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=DEFAULT_PATH_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    arguments = parser.parse_args()
    if arguments.paths < 2:
        parser.error("--paths must be at least 2.")
    if arguments.batch < 1:
        parser.error("--batch must be positive.")
    if arguments.seed < 0:
        parser.error("--seed must be nonnegative.")
    output_dir = arguments.output
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_metadata, quarter_returns, strategy_names, split_masks, audit = _load_inputs(arguments.input, output_dir)
    summaries = []
    comparisons = []
    convergence = []
    primary_strategy = "miqp_a15_L9"
    alternative_strategy = "miqp_a20_L9"
    pairs = [
        (primary_strategy, "ew"),
        (primary_strategy, "topk_ew_a15"),
        (primary_strategy, "topk_dw_a15"),
        (alternative_strategy, primary_strategy),
        (alternative_strategy, "ew"),
        (alternative_strategy, "topk_ew_a20"),
        (alternative_strategy, "topk_dw_a20"),
    ]
    for scenario, length, scale in SCENARIOS:
        results_by_split = _simulate_paths(
            quarter_returns,
            strategy_names,
            split_masks,
            arguments.paths,
            arguments.seed,
            length,
            scale,
            arguments.batch,
        )
        stored_draws = {"names": np.array(strategy_names)}
        for split, values in results_by_split.items():
            for key, array in values.items():
                stored_draws[f"{split}_{key}"] = array
            for j, name in enumerate(strategy_names):
                row = {
                    "scenario": scenario,
                    "split": split,
                    "strategy": name,
                    "n_paths": arguments.paths,
                    "block_days": length,
                    "vol_scale": scale,
                }
                for key, array in values.items():
                    row.update(
                        {
                            f"{key}_{suffix}": float(v)
                            for suffix, v in zip(
                                ["p025", "p05", "median", "p95", "p975"],
                                np.quantile(array[:, j], [0.025, 0.05, 0.5, 0.95, 0.975]),
                            )
                        }
                    )
                summaries.append(row)
            for left, right in pairs:
                delta = values["rmse"][:, strategy_names.index(left)] - values["rmse"][:, strategy_names.index(right)]
                frequency = float(np.mean(delta < 0))
                comparisons.append(
                    {
                        "scenario": scenario,
                        "split": split,
                        "left": left,
                        "right": right,
                        "p_left_lower_rmse": frequency,
                        "frequency_mcse": np.sqrt(frequency * (1 - frequency) / arguments.paths),
                        **{
                            f"delta_rmse_{suffix}": float(v)
                            for suffix, v in zip(
                                ["p025", "p05", "median", "p95", "p975"],
                                np.quantile(delta, [0.025, 0.05, 0.5, 0.95, 0.975]),
                            )
                        },
                    }
                )
            if scenario == "blocks_5":
                for draw_count in [500, 1000, 2500, 5000, arguments.paths]:
                    if draw_count > arguments.paths:
                        continue
                    for name in [primary_strategy, alternative_strategy, "ew"]:
                        samples = values["rmse"][:draw_count, strategy_names.index(name)]
                        convergence.append(
                            {
                                "split": split,
                                "strategy": name,
                                "n_paths": draw_count,
                                "mean_rmse": samples.mean(),
                                "mean_rmse_mcse": samples.std(ddof=1) / np.sqrt(draw_count),
                            }
                        )
        np.savez_compressed(output_dir / f"{scenario}_draws.npz", **stored_draws)
        print(f"{scenario}: {arguments.paths:,} joint paths; validation/test/combined complete", flush=True)
    pd.DataFrame(summaries).to_csv(output_dir / "mc_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_dir / "paired_comparisons.csv", index=False)
    pd.DataFrame(convergence).drop_duplicates().to_csv(output_dir / "convergence.csv", index=False)
    manifest = {
        "source_run": evaluation_metadata["solve_run"],
        "seed": arguments.seed,
        "paths_each": arguments.paths,
        "batch": arguments.batch,
        "input_sha256": {
            input_path.name: hashlib.sha256(input_path.read_bytes()).hexdigest()
            for input_path in arguments.input.glob("*.csv")
        },
        "eval_metadata": evaluation_metadata,
        "audit": audit,
        "python": platform.python_version(),
        "packages": {package: metadata.version(package) for package in ["numpy", "pandas", "matplotlib", "reportlab"]},
        "method": "Joint circular block bootstrap of realized strategy and AIQ returns, stratified by quarter. Fixed validation-selected windows. No reoptimization or reconstruction of asset weights.",
        "interpretation": "Conditional retrospective scenario frequencies; percentile ranges are not future confidence guarantees.",
        "stress": "Each realized strategy and AIQ return deviation from its own quarterly mean is multiplied by 1.5; not an asset-level portfolio stress.",
        "source_reference": "https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
