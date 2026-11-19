"""Numerical checks for the paired realized-return block-bootstrap engine."""

from __future__ import annotations

import numpy as np

from monte_carlo_daily import _circular_block_indices, _simulate_paths, _tracking_metrics


def main() -> None:
    benchmark_returns = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.007, 0.02, -0.01])
    indices = _circular_block_indices(np.random.default_rng(123), 200, len(benchmark_returns), 3)
    assert np.all(np.diff(indices[:, :6].reshape(200, 2, 3), axis=-1) % len(benchmark_returns) == 1)
    assert np.array_equal(indices, _circular_block_indices(np.random.default_rng(123), 200, len(benchmark_returns), 3))
    sampled = benchmark_returns[indices]
    metrics = _tracking_metrics(sampled[..., None], sampled)
    assert np.max(metrics["rmse"]) == 0 and np.max(metrics["terminal_gap"]) == 0
    assert np.allclose(metrics["corr"], 1)
    quarter_returns = [
        dict(quarter="example", values=np.column_stack([benchmark_returns, benchmark_returns * 1.2]), split="test")
    ]
    simulation_arguments = (quarter_returns, ["example"], {"test": np.ones(8, dtype=bool)}, 200, 4004, 3, 1, 50)
    first_result = _simulate_paths(*simulation_arguments)
    second_result = _simulate_paths(*simulation_arguments)
    assert all(np.array_equal(first_result["test"][key], second_result["test"][key]) for key in first_result["test"])
    print("PASS: perfect tracker, circular-block continuity, seed reproducibility, full-simulation repeatability")


if __name__ == "__main__":
    main()
