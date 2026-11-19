## Monte Carlo evaluation

[`experiments/monte_carlo/`](experiments/monte_carlo/README.md) evaluates the
archived split-aware backtest `261111_170130`. It jointly resamples realized
portfolio/AIQ returns and reports paired tracking comparisons, block-length
sensitivity and an illustrative return-volatility stress.

The archived `eval.json` and `best_by_alpha.csv` contain validation/test fields
that the evaluator above does not generate. This step requires those archived
outputs; it does not infer a new split or change the optimization model. Its
README records the required schema, separate analysis environment, commands,
output tables and interpretation. Inputs and generated results remain under the
existing ignored `output/` directory.
