"""Build the final Monte Carlo figures, English insert, and verified PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = "#19374B"
TEAL = "#198B89"
GOLD = "#D69538"
GRAY = "#657783"
LABELS = {
    "miqp_a15_L9": "15% cap, 9 months",
    "miqp_a20_L9": "20% cap, 9 months",
    "ew": "Full-universe equal weight",
    "topk_ew_a15": "Top-K equal weight, 15% cap",
    "topk_dw_a15": "Top-K disclosed weight, 15% cap",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    arguments = parser.parse_args()
    package_dir = arguments.package
    result_dir = package_dir / "results"
    figure_dir = package_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    summary = pd.read_csv(result_dir / "mc_summary.csv")
    pairs = pd.read_csv(result_dir / "paired_comparisons.csv")
    observed_metrics = pd.read_csv(result_dir / "observed_metrics.csv")
    manifest = json.loads((result_dir / "manifest.json").read_text())
    path_count = manifest["paths_each"]

    def _summary_row(strategy: str, split: str = "test", scenario: str = "blocks_5") -> pd.Series:
        return summary.loc[
            summary.strategy.eq(strategy) & summary.split.eq(split) & summary.scenario.eq(scenario)
        ].iloc[0]

    def _comparison_row(left: str, right: str, split: str = "test", scenario: str = "blocks_5") -> pd.Series:
        return pairs.loc[
            pairs.left.eq(left) & pairs.right.eq(right) & pairs.split.eq(split) & pairs.scenario.eq(scenario)
        ].iloc[0]

    primary_strategy = "miqp_a15_L9"
    alternative_strategy = "miqp_a20_L9"
    strategy_names = [primary_strategy, alternative_strategy, "ew", "topk_ew_a15", "topk_dw_a15"]
    primary_summary = _summary_row(primary_strategy)
    alternative_summary = _summary_row(alternative_strategy)
    overall = _comparison_row(primary_strategy, "ew", "combined")
    primary = _comparison_row(primary_strategy, "ew")
    extra = _comparison_row(alternative_strategy, primary_strategy)
    observed_by_strategy = observed_metrics.set_index(["split", "strategy"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "savefig.dpi": 200,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for j, name in enumerate(strategy_names):
        r = _summary_row(name)
        center = r.rmse_median * 1e4
        ax.errorbar(
            center,
            j,
            xerr=[[center - r.rmse_p025 * 1e4], [r.rmse_p975 * 1e4 - center]],
            fmt="o",
            color=TEAL if j < 2 else NAVY,
            capsize=4,
            elinewidth=2,
        )
    ax.scatter(
        [observed_by_strategy.loc[("test", name), "rmse"] * 1e4 for name in strategy_names],
        np.arange(5),
        marker="D",
        s=25,
        color=GOLD,
        zorder=3,
        label="Observed test RMSE",
    )
    ax.set_yticks(np.arange(5), [LABELS[x] for x in strategy_names])
    ax.invert_yaxis()
    ax.set_xlabel("Daily tracking RMSE (basis points; lower is better)")
    ax.set_title("Test-year tracking across resampled market paths", loc="left", pad=12)
    ax.plot([], [], "o", color=TEAL, label="Bootstrap median; bars = 2.5th-97.5th percentiles")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7.4, loc="lower center", ncol=1)
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(figure_dir / "01_test_rmse.png")
    plt.close(fig)

    draws = np.load(result_dir / "blocks_5_draws.npz")
    draw_strategy_names = list(draws["names"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    settings = [
        ("test", primary_strategy, "ew", "Test: 15% vs full EW"),
        ("test", alternative_strategy, primary_strategy, "Test: 20% vs 15%"),
        ("combined", primary_strategy, "ew", "Combined: 15% vs EW"),
    ]
    for ax, (split, left, right, title) in zip(axes, settings):
        delta = (
            draws[f"{split}_rmse"][:, draw_strategy_names.index(left)]
            - draws[f"{split}_rmse"][:, draw_strategy_names.index(right)]
        ) * 1e4
        ax.hist(delta, bins=38, color=TEAL, alpha=0.85)
        ax.axvline(0, color=GOLD, lw=1.5)
        ax.set_title(title, fontsize=8.6, loc="left")
        ax.set_xlabel("RMSE difference (bp)\nNegative favors first portfolio", fontsize=8)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Simulated paths")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_paired_differences.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    for strategy, color, offset, description in [
        (primary_strategy, TEAL, -0.14, "15% cap"),
        (alternative_strategy, GOLD, 0, "20% cap"),
        ("ew", NAVY, 0.14, "Full EW"),
    ]:
        for k, scenario in enumerate(["blocks_5", "volatility_1p5"]):
            r = _summary_row(strategy, "test", scenario)
            center = r.rmse_median * 1e4
            ax.errorbar(
                k + offset,
                center,
                yerr=[[center - r.rmse_p025 * 1e4], [r.rmse_p975 * 1e4 - center]],
                fmt="o",
                color=color,
                capsize=4,
                label=description if k == 0 else None,
            )
    ax.set_xticks([0, 1], ["Historical volatility", "1.5x within-quarter return volatility"])
    ax.set_ylabel("Daily RMSE (bp)")
    ax.set_title("Illustrative strategy-return volatility stress", loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "03_volatility_stress.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    definitions = pd.read_csv(result_dir / "strategy_definitions.csv").dropna(subset=["window_months"])
    for split, color, offset in [("test", TEAL, 0.35), ("validation", NAVY, -0.35)]:
        for r in definitions.itertuples(index=False):
            m = _summary_row(r.strategy, split)
            y = m.rmse_median * 1e4
            ax.errorbar(
                r.alpha * 100 + offset,
                y,
                yerr=[[y - m.rmse_p025 * 1e4], [m.rmse_p975 * 1e4 - y]],
                fmt="o",
                color=color,
                capsize=3,
                label=split.capitalize() if r.alpha == 0.05 else None,
            )
    ax.set_xticks(np.arange(5, 51, 5))
    ax.set_xlabel("Holdings cap (%) - windows fixed at validation-selected values")
    ax.set_ylabel("Daily RMSE (bp)")
    ax.set_title("Supplement: all reported cap choices", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "04_all_caps.png")
    plt.close(fig)

    # Concise section that can be pasted into the team's report.
    section = f"""## Monte Carlo experiment

We tested the stability of the reported tracking results using a joint circular block bootstrap of the realized portfolio and AIQ daily returns. Within each historical quarter, we drew consecutive five-trading-day blocks with replacement and used identical sampled dates for all strategies. We generated {path_count:,} full-sample paths with seed 4004, retaining the validation period (October 3, 2023–October 1, 2025; 501 days) and test period (October 2, 2025–September 30, 2026; 250 days). The estimation windows were kept at the validation-selected values recorded in the evaluation files, including nine months for the 15% and 20% caps. This experiment resamples already realized strategy returns; it does not reselect stocks, reoptimize weights, or reconstruct holdings at block boundaries.

For the 15% cap, median test-period daily RMSE was **{primary_summary.rmse_median*1e4:.2f} bp**, with a 2.5th–97.5th percentile range of **{primary_summary.rmse_p025*1e4:.2f}–{primary_summary.rmse_p975*1e4:.2f} bp**. It tracked AIQ more closely than full-universe equal weight in **{primary.p_left_lower_rmse:.2%}** of paired test-period paths. However, its combined-period advantage was less stable: it beat full-universe equal weight in only **{overall.p_left_lower_rmse:.2%}** of paths. The combined-period RMSE difference (15% minus full equal weight) had a 2.5th–97.5th range of **{overall.delta_rmse_p025*1e4:+.2f} to {overall.delta_rmse_p975*1e4:+.2f} bp**, spanning zero. The observed combined-sample advantage of 1.63 bp therefore should not be described as a dependable improvement across market conditions.

Increasing the cap to 20% reduced median test-period RMSE to **{alternative_summary.rmse_median*1e4:.2f} bp** and improved on the 15% portfolio in **{extra.p_left_lower_rmse:.2%}** of paired paths. The median difference was **{extra.delta_rmse_median*1e4:.2f} bp**, with a 2.5th–97.5th range of **{extra.delta_rmse_p025*1e4:.2f} to {extra.delta_rmse_p975*1e4:+.2f} bp**; this also spans zero. Block lengths of 1, 10 and 20 days produced test-period improvement frequencies of {_comparison_row(alternative_strategy,primary_strategy,'test','iid_1').p_left_lower_rmse:.2%}, {_comparison_row(alternative_strategy,primary_strategy,'test','blocks_10').p_left_lower_rmse:.2%} and {_comparison_row(alternative_strategy,primary_strategy,'test','blocks_20').p_left_lower_rmse:.2%}, respectively. Under an illustrative 50% increase in each strategy's and AIQ's within-quarter return volatility, median test RMSE rose to **{_summary_row(primary_strategy,'test','volatility_1p5').rmse_median*1e4:.2f} bp** for the 15% cap and **{_summary_row(alternative_strategy,'test','volatility_1p5').rmse_median*1e4:.2f} bp** for the 20% cap.

These results support retaining 15% as a compact candidate while considering 20% when closer tracking justifies more holdings. They do not establish 15% as a universally optimal minimum cap. Frequencies describe conditional bootstrap paths, not future success probabilities; percentile ranges are not future confidence guarantees or formal significance tests. Cap selection used combined validation/test performance in the report, so resampling does not remove that selection bias. Trading costs and uncertainty in the optimized holdings are not included.

![Test-period RMSE distributions](figures/01_test_rmse.png)

![Paired differences; negative values favor the first portfolio](figures/02_paired_differences.png)

Method reference: [arch official circular block bootstrap documentation](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html).
"""
    section = section.replace("**", "")
    (package_dir / "Monte_Carlo_section.md").write_text(section)
    plain = section.split("\n\n")
    tex = ["\\subsection*{Monte Carlo experiment}"]
    for text in plain[1:5]:
        text = text.replace("**", "").replace("%", "\\%").replace("–", "--")
        tex.extend([text, ""])
    tex.extend(
        [
            "% Requires \\usepackage{graphicx}. Figure paths are relative to this package.",
            "\\begin{figure}[htbp]\\centering",
            "\\includegraphics[width=\\linewidth]{figures/01_test_rmse.png}",
            "\\caption{Test-period daily RMSE across 10,000 joint five-day block-bootstrap paths. Bars show the 2.5th--97.5th percentiles.}",
            "\\end{figure}",
        ]
    )
    (package_dir / "Monte_Carlo_section.tex").write_text("\n".join(tex) + "\n")

    styles = {
        "eyebrow": ParagraphStyle(
            "e", fontName="Helvetica-Bold", fontSize=8.5, leading=12, spaceAfter=8, textColor=colors.HexColor(TEAL)
        ),
        "title": ParagraphStyle(
            "t", fontName="Helvetica-Bold", fontSize=23, leading=27, spaceAfter=10, textColor=colors.HexColor(NAVY)
        ),
        "h": ParagraphStyle(
            "h",
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            spaceBefore=6,
            spaceAfter=5,
            textColor=colors.HexColor(NAVY),
        ),
        "body": ParagraphStyle(
            "b", fontName="Helvetica", fontSize=9.2, leading=12.7, spaceAfter=7, textColor=colors.HexColor(NAVY)
        ),
        "small": ParagraphStyle(
            "s", fontName="Helvetica", fontSize=7.8, leading=10.4, spaceAfter=6, textColor=colors.HexColor(GRAY)
        ),
        "cell": ParagraphStyle("c", fontName="Helvetica", fontSize=8, leading=10.5, textColor=colors.HexColor(NAVY)),
        "head": ParagraphStyle("ch", fontName="Helvetica-Bold", fontSize=8, leading=10.5, textColor=colors.white),
    }
    story = []

    def _append_paragraph(text: str, style: str = "body") -> None:
        story.append(Paragraph(text, styles[style]))

    def _append_image(name: str, height: float) -> None:
        story.extend([Image(str(figure_dir / name), width=510, height=height), Spacer(1, 5)])

    def _append_table(rows: list[list[object]], widths: list[float]) -> None:
        t = Table(
            [
                [Paragraph(escape(str(cell)), styles["head" if i == 0 else "cell"]) for cell in row]
                for i, row in enumerate(rows)
            ],
            colWidths=widths,
            hAlign="LEFT",
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EEF5F5"), colors.white]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend([t, Spacer(1, 8)])

    _append_paragraph("SPOT | MILESTONE 5 | MONTE CARLO EXPERIMENT", "eyebrow")
    _append_paragraph("How stable is the<br/>15% tracking portfolio?", "title")
    _append_paragraph(
        f'We used the team\'s final run <b>{manifest["source_run"]}</b> to generate {path_count:,} alternative paths per setting. The main comparison keeps the reported nine-month window and contrasts 15% and 20% holdings caps. Lower daily RMSE means closer tracking of AIQ.'
    )
    _append_paragraph("The compact choice has a limited overall edge", "h")
    _append_paragraph(
        f"The 15% portfolio beats full-universe equal weight in <b>{primary.p_left_lower_rmse:.2%}</b> of test-year paths, but only <b>{overall.p_left_lower_rmse:.2%}</b> of combined validation/test paths. Its small full-sample advantage is therefore sensitive to the market period. These are frequencies within resampled historical paths, not future success probabilities."
    )
    _append_image("01_test_rmse.png", 233.75)
    rows = [["Portfolio", "Observed test\nRMSE (bp)", "Bootstrap median\n[2.5th, 97.5th] (bp)"]]
    for name in strategy_names:
        r = _summary_row(name)
        rows.append(
            [
                LABELS[name],
                f'{observed_by_strategy.loc[("test",name),"rmse"]*1e4:.2f}',
                f"{r.rmse_median*1e4:.2f} [{r.rmse_p025*1e4:.2f}, {r.rmse_p975*1e4:.2f}]",
            ]
        )
    _append_table(rows, [240, 100, 170])
    _append_paragraph(
        f"<b>Decision implication.</b> A 15% cap remains a compact candidate. A 20% cap has lower test RMSE in {extra.p_left_lower_rmse:.2%} of paired paths and is worth considering when closer tracking matters more than the number of positions. Neither choice is guaranteed to dominate, and trading costs are excluded."
    )
    _append_paragraph(
        "Test period: October 2, 2025-September 30, 2026 (250 trading days). Bars and table brackets show conditional bootstrap percentiles, not future confidence bounds. All RMSE values are daily, in basis points.",
        "small",
    )

    story.append(PageBreak())
    _append_paragraph("PAIRED COMPARISONS AND ROBUSTNESS", "eyebrow")
    _append_paragraph("A better result in one period<br/>need not hold in every path.", "title")
    _append_image("02_paired_differences.png", 188)
    _append_paragraph(
        f"The 15%-minus-full-EW RMSE difference over the combined period has a median of {overall.delta_rmse_median*1e4:+.2f} bp and a 2.5th-97.5th range of <b>{overall.delta_rmse_p025*1e4:+.2f} to {overall.delta_rmse_p975*1e4:+.2f} bp</b>. That range spans zero. Even in the test period, its range is {primary.delta_rmse_p025*1e4:+.2f} to {primary.delta_rmse_p975*1e4:+.2f} bp; a high path frequency is not a formal significance result."
    )
    _append_paragraph(
        f"For 20% minus 15%, the test-period median difference is {extra.delta_rmse_median*1e4:+.2f} bp, with a range of {extra.delta_rmse_p025*1e4:+.2f} to {extra.delta_rmse_p975*1e4:+.2f} bp. The direction usually favors 20%, but some sampled paths favor 15%."
    )
    _append_paragraph("Check the dependence assumption", "h")
    rows = [["Block length", "Test: 15% beats full EW", "Test: 20% beats 15%", "Combined: 15% beats EW"]]
    for scenario, L in [("iid_1", 1), ("blocks_5", 5), ("blocks_10", 10), ("blocks_20", 20)]:
        rows.append(
            [
                f"{L} day(s)",
                f'{_comparison_row(primary_strategy,"ew","test",scenario).p_left_lower_rmse:.2%}',
                f'{_comparison_row(alternative_strategy,primary_strategy,"test",scenario).p_left_lower_rmse:.2%}',
                f'{_comparison_row(primary_strategy,"ew","combined",scenario).p_left_lower_rmse:.2%}',
            ]
        )
    _append_table(rows, [80, 142, 142, 146])
    _append_paragraph(
        f'In the illustrative 1.5x volatility setting, median test RMSE rises from {primary_summary.rmse_median*1e4:.2f} to {_summary_row(primary_strategy,"test","volatility_1p5").rmse_median*1e4:.2f} bp for 15%, and from {alternative_summary.rmse_median*1e4:.2f} to {_summary_row(alternative_strategy,"test","volatility_1p5").rmse_median*1e4:.2f} bp for 20%. This scales the realized strategy-return fluctuations around each quarterly mean; it does not simulate new stock-level positions, correlation breakdowns, or unobserved crashes.',
        "small",
    )

    story.append(PageBreak())
    _append_paragraph("METHOD, AUDIT AND INTERPRETATION", "eyebrow")
    _append_paragraph("Reproducible conditional<br/>scenario analysis", "title")
    _append_paragraph("Data and sampling", "h")
    _append_paragraph(
        "Source: daily_returns.csv, eval.json, best_by_alpha.csv and window_metrics.csv from the supplied run. The validation segment contains the first eight quarters (501 days, October 3, 2023-October 1, 2025); the test segment contains the final four quarters (250 days, October 2, 2025-September 30, 2026). Results are also reported for their 751-day combination. Estimation windows remain fixed at the values selected on validation RMSE in the evaluation metadata; no window is chosen from bootstrap outcomes."
    )
    _append_paragraph(
        "Within each historical quarter, sample fixed-length circular blocks of the synchronized daily-return vector, including AIQ and every selected portfolio. Uniform block starts and end-to-start wraparound give each date equal marginal sampling weight. Truncate the last block to preserve the number of days. Keep quarters in their original order. Use five-day blocks as the main specification, and 1, 10 and 20 days as sensitivity checks. Dependence across quarter boundaries is not modeled."
    )
    _append_paragraph(
        "The historical backtest already incorporates its own quarterly execution and position drift. This bootstrap resamples the resulting strategy returns. It does not rebuild stock-price paths, recompute drift at joins, or adapt future decisions to simulated history. It is a retrospective assessment conditional on the existing strategy-return processes, not a full portfolio-policy simulation."
    )
    _append_paragraph("Definitions and simulation precision", "h")
    _append_paragraph(
        "Active daily return is portfolio return minus AIQ return. RMSE is the square root of its mean squared value; TE is its sample standard deviation (ddof=1). One basis point is 0.01 percentage point. RMSE is not annualized. Tables also store return correlation, mean active return and the difference in compounded returns, which is a secondary path measure."
    )
    _append_paragraph(
        f"Each setting uses {path_count:,} draws, seed 4004 and batches of 250. Comparisons share identical sampled dates. Main and stressed settings use the same sampling indices. The largest binomial Monte Carlo standard error of a path frequency is {100*.5/np.sqrt(path_count):.2f} percentage point. For the test-year 20%-versus-15% frequency it is {extra.frequency_mcse*100:.2f} point. This quantifies simulation noise, not model uncertainty. No formal null-hypothesis test or multiple-comparison correction is claimed."
    )
    _append_paragraph("Verification and limits", "h")
    _append_paragraph(
        f'All {manifest["audit"]["source_paths_checked"]} source strategy paths reproduce the supplied full-period RMSEs; the largest absolute difference is {manifest["audit"]["max_rmse_difference"]:.2e} in decimal daily-return units. The validation-selected windows and their validation/test RMSEs match best_by_alpha.csv for all ten caps. Returns are finite, aligned and unique by strategy/date, and active-return identities hold. Separate numerical checks cover circular-block continuity, a perfect tracker and deterministic reruns.'
    )
    _append_paragraph(
        "The report used combined validation/test performance to justify the 15% cap. Although windows were selected on validation data, cap selection is therefore not independently validated by the test segment. Bootstrap resampling cannot remove selection bias. Only twelve historical quarters and four test quarters are available; the regime sequence is fixed, and long blocks reuse relatively few independent pieces. The simulation excludes parameter/solver uncertainty, new market regimes, and costs. Optimized weights and solver logs were not supplied, so this analysis does not certify optimization quality."
    )
    _append_paragraph("Sources and reproduction", "h")
    _append_paragraph(
        "Model context: <link href='https://github.com/Etlevn/SPOT'>github.com/Etlevn/SPOT</link>. Method definition: <link href='https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html'>arch official CircularBlockBootstrap documentation</link>. The package contains the final input CSVs, executable code, hashes, environment versions, summary tables, figures, and an English section for the team report. See the README and Chinese handoff notes for commands.",
        "small",
    )

    def _draw_footer(canvas: Canvas, doc: SimpleDocTemplate) -> None:
        canvas.setStrokeColor(colors.HexColor("#D3E0E5"))
        canvas.line(51, 37, 561, 37)
        canvas.setFont("Helvetica", 7.3)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(51, 25, "SPOT | Run 261111_170130 | Conditional block-bootstrap analysis")
        canvas.drawRightString(561, 25, str(doc.page))

    arguments.pdf.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(arguments.pdf),
        pagesize=letter,
        leftMargin=51,
        rightMargin=51,
        topMargin=39,
        bottomMargin=47,
        title="SPOT Milestone 5 - Monte Carlo Experiment",
        author="SPOT project",
    ).build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    print(arguments.pdf)


if __name__ == "__main__":
    main()
