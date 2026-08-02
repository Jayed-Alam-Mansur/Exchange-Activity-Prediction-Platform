"""Static figure export (matplotlib PNG) for the README and reports.

The dashboard renders interactive Plotly versions of these same charts. This
module produces publication-quality PNGs so the repository can be reviewed on
GitHub without running anything, and so the README has real figures rather than
placeholders.

Colour tokens, mark specs and chrome rules are shared with
:mod:`src.visualization.charts` - same palette, same ordinal-vs-categorical
discipline, same recessive grid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Settings, get_settings  # noqa: E402
from src.visualization.charts import (  # noqa: E402
    ACTIVITY_COLORS_LIGHT,
    CATEGORICAL_LIGHT,
    ORDINAL_LIGHT,
)

__all__ = ["export_all_figures", "apply_style"]

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

DPI = 150


def apply_style() -> None:
    """Apply the project's matplotlib style: recessive chrome, system sans."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "legend.frameon": False,
            "legend.labelcolor": TEXT_SECONDARY,
            "figure.autolayout": False,
        }
    )


def _finish(ax: plt.Axes, title: str, ylabel: str = "", xlabel: str = "") -> None:
    """Apply shared per-axes chrome: hide top/right spines, label axes."""
    ax.set_title(title, pad=14, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write a figure to disk and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path.name)
    return path


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------
def _figure_monthly_trend(trend: pd.DataFrame, target: str, path: Path) -> Path:
    """Monthly series with its 12-month rolling trend."""
    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(trend["date"], trend[target], color=CATEGORICAL_LIGHT[0], linewidth=2,
            marker="o", markersize=4, label="Monthly applications")
    if "rolling_12m" in trend.columns:
        ax.plot(trend["date"], trend["rolling_12m"], color=CATEGORICAL_LIGHT[1],
                linewidth=2, linestyle="--", label="12-month average")

    ax.legend(loc="upper left", ncols=2)
    _finish(ax, "Monthly exchange applications, 2022-2025", "Applications")
    return _save(fig, path)


def _figure_year_comparison(contribution: pd.DataFrame, target: str, path: Path) -> Path:
    """One line per year on a shared Jan-Dec axis."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = [pd.Timestamp(2024, m, 1).strftime("%b") for m in range(1, 13)]

    for index, year in enumerate(sorted(contribution["year"].unique())):
        subset = contribution[contribution["year"] == year].sort_values("month")
        ax.plot(subset["month"], subset[target],
                color=CATEGORICAL_LIGHT[index % len(CATEGORICAL_LIGHT)],
                linewidth=2, marker="o", markersize=5, label=str(year))

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", ncols=4)
    _finish(ax, "Year-on-year monthly comparison", "Applications")
    return _save(fig, path)


def _figure_seasonality(profile: pd.DataFrame, path: Path) -> Path:
    """Seasonal index bars against a 1.0 average line."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    values = profile["seasonal_index"].to_numpy(dtype=float)

    span = values.max() - values.min()
    indices = (
        np.zeros(len(values), dtype=int) if span == 0
        else np.clip(((values - values.min()) / span * (len(ORDINAL_LIGHT) - 1)).round().astype(int),
                     0, len(ORDINAL_LIGHT) - 1)
    )

    bars = ax.bar(profile["month_name"], values,
                  color=[ORDINAL_LIGHT[i] for i in indices],
                  edgecolor=SURFACE, linewidth=2)
    ax.bar_label(bars, fmt="%.2f", padding=3, color=TEXT_SECONDARY, fontsize=9)

    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    ax.text(11.6, 1.01, "average", color=MUTED, fontsize=9, ha="right")
    ax.set_ylim(0, max(1.5, values.max() * 1.18))
    _finish(ax, "Seasonal index by calendar month", "Index (1.0 = average)")
    return _save(fig, path)


def _figure_funnel(funnel: pd.DataFrame, path: Path) -> Path:
    """Horizontal funnel with conversion annotations."""
    fig, ax = plt.subplots(figsize=(9.5, 5))

    count = len(funnel)
    colors = [
        ORDINAL_LIGHT[min(int(i / max(count - 1, 1) * (len(ORDINAL_LIGHT) - 1)), len(ORDINAL_LIGHT) - 1)]
        for i in range(count)
    ]
    labels = [f"{row.stage}\n{row.stage_label}" for row in funnel.itertuples()]
    positions = np.arange(count)[::-1]

    ax.barh(positions, funnel["count"], color=colors, edgecolor=SURFACE,
            linewidth=2, height=0.72)

    for position, row in zip(positions, funnel.itertuples()):
        ax.text(row.count + funnel["count"].max() * 0.012, position,
                f"{row.count:,}  ({row.conversion_from_APP_pct:.1f}% of applied)",
                va="center", color=TEXT_SECONDARY, fontsize=9)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlim(0, funnel["count"].max() * 1.32)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _finish(ax, "Exchange funnel: applied to completed", "", "Candidates")
    return _save(fig, path)


def _figure_entities(entities: pd.DataFrame, target: str, path: Path, top_n: int = 12) -> Path:
    """Top LCs by application volume."""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    subset = entities.head(top_n).iloc[::-1]

    bars = ax.barh(subset["entity"], subset[target], color=CATEGORICAL_LIGHT[0],
                   edgecolor=SURFACE, linewidth=2, height=0.72)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in subset[target]], padding=4,
                 color=TEXT_SECONDARY, fontsize=9)

    ax.set_xlim(0, subset[target].max() * 1.15)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    _finish(ax, f"Top {top_n} Local Committees by applications", "", "Applications")
    return _save(fig, path)


def _figure_products(products: pd.DataFrame, target: str, path: Path) -> Path:
    """Programme volume, with conversion rate annotated."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ordered = products.sort_values("programme").reset_index(drop=True)

    bars = ax.bar(ordered["programme"], ordered[target],
                  color=[CATEGORICAL_LIGHT[i % len(CATEGORICAL_LIGHT)] for i in range(len(ordered))],
                  edgecolor=SURFACE, linewidth=2)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in ordered[target]],
                 padding=3, color=TEXT_SECONDARY, fontsize=9)

    ax.set_ylim(0, ordered[target].max() * 1.16)
    _finish(ax, "Applications by programme", "Applications")
    return _save(fig, path)


def _figure_forecast(history: pd.DataFrame, predictions: pd.DataFrame,
                     target: str, path: Path) -> Path:
    """History plus the 2026 forecast and its 95% band."""
    fig, ax = plt.subplots(figsize=(11.5, 4.8))

    forecast_dates = pd.to_datetime(predictions["date"])
    forecast_color = CATEGORICAL_LIGHT[1]

    ax.fill_between(forecast_dates, predictions["lower_95"], predictions["upper_95"],
                    color=forecast_color, alpha=0.14, linewidth=0,
                    label="95% prediction interval")
    ax.plot(history["date"], history[target], color=CATEGORICAL_LIGHT[0],
            linewidth=2, label="Historical (2022-2025)")

    bridge_x = [history["date"].iloc[-1], forecast_dates.iloc[0]]
    bridge_y = [history[target].iloc[-1], predictions["Predicted Applications"].iloc[0]]
    ax.plot(bridge_x, bridge_y, color=forecast_color, linewidth=2, linestyle="--")

    ax.plot(forecast_dates, predictions["Predicted Applications"], color=forecast_color,
            linewidth=2, linestyle="--", marker="o", markersize=5, label="Forecast (2026)")
    ax.axvline(history["date"].iloc[-1], color=MUTED, linewidth=1, linestyle=":")

    ax.legend(loc="upper left", ncols=3)
    _finish(ax, "Exchange applications: history and 2026 forecast", "Applications")
    return _save(fig, path)


def _figure_activity_levels(predictions: pd.DataFrame, path: Path) -> Path:
    """2026 monthly forecast coloured and labelled by activity level."""
    fig, ax = plt.subplots(figsize=(11, 4.5))

    colors = [ACTIVITY_COLORS_LIGHT[level] for level in predictions["Activity Level"]]
    bars = ax.bar(predictions["Month"], predictions["Predicted Applications"],
                  color=colors, edgecolor=SURFACE, linewidth=2)

    # Direct labels: identity must never rest on colour alone.
    ax.bar_label(bars, labels=predictions["Activity Level"], padding=3,
                 color=TEXT_SECONDARY, fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ACTIVITY_COLORS_LIGHT[level])
        for level in ("Low", "Medium", "High")
    ]
    ax.legend(handles, ["Low", "Medium", "High"], loc="upper left", ncols=3)

    ax.set_ylim(0, predictions["Predicted Applications"].max() * 1.22)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    _finish(ax, "Predicted 2026 activity by month", "Predicted applications")
    return _save(fig, path)


def _figure_model_comparison(evaluation: pd.DataFrame, path: Path, metric: str = "MAE") -> Path:
    """Candidate models ranked by walk-forward error."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ordered = evaluation.sort_values(metric, ascending=False)

    selected = ordered.get("selected", pd.Series(False, index=ordered.index))
    colors = [ORDINAL_LIGHT[-1] if flag else ORDINAL_LIGHT[1] for flag in selected]

    bars = ax.barh(ordered["model"], ordered[metric], color=colors,
                   edgecolor=SURFACE, linewidth=2, height=0.72)
    ax.bar_label(bars, labels=[f"{v:,.1f}" for v in ordered[metric]], padding=4,
                 color=TEXT_SECONDARY, fontsize=9)

    ax.set_xlim(0, ordered[metric].max() * 1.16)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    _finish(ax, f"Model comparison - walk-forward {metric} (lower is better)", "", metric)
    return _save(fig, path)


def _figure_backtest(backtest: pd.DataFrame, model_name: str, path: Path) -> Path:
    """Actual versus predicted across the walk-forward origins."""
    fig, ax = plt.subplots(figsize=(11, 4.3))
    subset = backtest[backtest["model"] == model_name].sort_values("date")

    ax.plot(pd.to_datetime(subset["date"]), subset["y_true"], color=CATEGORICAL_LIGHT[0],
            linewidth=2, marker="o", markersize=5, label="Actual")
    ax.plot(pd.to_datetime(subset["date"]), subset["y_pred"], color=CATEGORICAL_LIGHT[1],
            linewidth=2, linestyle="--", marker="o", markersize=5, label="Predicted")

    ax.legend(loc="upper left", ncols=2)
    _finish(ax, f"Walk-forward validation - {model_name}", "Applications")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def export_all_figures(settings: Optional[Settings] = None) -> List[Path]:
    """Regenerate every static figure from the artefacts already on disk.

    Reads the report CSVs and prediction file rather than recomputing anything,
    so this stage is fast and can be rerun independently.

    Args:
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        Paths of the figures written. Missing inputs are skipped with a warning.
    """
    settings = settings or get_settings()
    apply_style()

    reports = settings.paths.reports_dir
    figures_dir = settings.paths.figures_dir
    target = settings.target
    written: List[Path] = []

    def read(name: str) -> Optional[pd.DataFrame]:
        path = reports / f"{name}.csv"
        if not path.exists():
            logger.warning("Skipping figures needing %s (not found)", path.name)
            return None
        return pd.read_csv(path)

    tables: Dict[str, Optional[pd.DataFrame]] = {
        name: read(name)
        for name in (
            "monthly_trend", "monthly_contribution", "seasonality_profile",
            "funnel_overall", "entity_performance", "product_performance",
            "model_evaluation", "backtest_predictions",
        )
    }

    trend = tables["monthly_trend"]
    if trend is not None:
        trend["date"] = pd.to_datetime(trend["date"])
        written.append(_figure_monthly_trend(trend, target, figures_dir / "01_monthly_trend.png"))

    if tables["monthly_contribution"] is not None:
        written.append(
            _figure_year_comparison(
                tables["monthly_contribution"], target, figures_dir / "02_year_comparison.png"
            )
        )

    if tables["seasonality_profile"] is not None:
        written.append(
            _figure_seasonality(tables["seasonality_profile"], figures_dir / "03_seasonality.png")
        )

    if tables["funnel_overall"] is not None:
        written.append(_figure_funnel(tables["funnel_overall"], figures_dir / "04_funnel.png"))

    if tables["entity_performance"] is not None:
        written.append(
            _figure_entities(tables["entity_performance"], target,
                             figures_dir / "05_entity_performance.png")
        )

    if tables["product_performance"] is not None:
        written.append(
            _figure_products(tables["product_performance"], target,
                             figures_dir / "06_product_mix.png")
        )

    predictions_path = settings.paths.predictions
    if predictions_path.exists() and trend is not None:
        predictions = pd.read_csv(predictions_path)
        written.append(
            _figure_forecast(trend, predictions, target, figures_dir / "07_forecast_2026.png")
        )
        written.append(
            _figure_activity_levels(predictions, figures_dir / "08_activity_levels.png")
        )
    else:
        logger.warning("Skipping forecast figures (%s not found)", predictions_path.name)

    evaluation = tables["model_evaluation"]
    if evaluation is not None:
        written.append(
            _figure_model_comparison(evaluation, figures_dir / "09_model_comparison.png")
        )

        backtest = tables["backtest_predictions"]
        if backtest is not None:
            selected = evaluation[evaluation.get("selected", False) == True]  # noqa: E712
            model_name = (
                str(selected.iloc[0]["model"]) if not selected.empty
                else str(evaluation.sort_values("MAE").iloc[0]["model"])
            )
            written.append(
                _figure_backtest(backtest, model_name, figures_dir / "10_backtest.png")
            )

    logger.info("Exported %d figure(s) to %s", len(written), figures_dir)
    return written
