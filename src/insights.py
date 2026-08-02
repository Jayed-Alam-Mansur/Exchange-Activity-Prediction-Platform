"""Automatic narrative insight generation.

Turns the analysis and forecast tables into plain-English statements a
Member Committee can act on, e.g.:

    "December 2026 is predicted to have the highest activity, at 1,989
     applications - 10.7% of the year."

Each insight is a small, self-contained function so it can be unit-tested
against a synthetic frame, and every number quoted is computed from the data
rather than hardcoded. :func:`generate_all_insights` runs them all and skips any
that lack the inputs they need, so the dashboard degrades gracefully when only
part of the pipeline has been run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Settings, get_settings

__all__ = ["Insight", "generate_all_insights", "insights_to_frame"]

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """One generated finding.

    Attributes:
        category: Grouping label, e.g. ``"Forecast"`` or ``"Funnel"``.
        headline: One-sentence statement, safe to show on its own.
        detail: Supporting explanation or the operational implication.
        kind: ``"positive"`` / ``"negative"`` / ``"neutral"`` - drives styling.
    """

    category: str
    headline: str
    detail: str = ""
    kind: str = "neutral"


def _fmt(value: float) -> str:
    """Format a number with thousands separators and no decimals."""
    return f"{value:,.0f}"


def _pct(value: float, places: int = 1) -> str:
    """Format a percentage with an explicit sign."""
    return f"{value:+.{places}f}%"


# ---------------------------------------------------------------------------
# Historical insights
# ---------------------------------------------------------------------------
def _growth_insight(yearly: pd.DataFrame) -> Optional[Insight]:
    """Compare the most recent full year with the one before it."""
    if yearly is None or len(yearly) < 2:
        return None

    latest, previous = yearly.iloc[-1], yearly.iloc[-2]
    growth = float(latest["yoy_growth_pct"])
    direction = "increased" if growth >= 0 else "decreased"

    return Insight(
        category="Growth",
        headline=(
            f"Application volume {direction} by {abs(growth):.1f}% in "
            f"{int(latest['year'])} compared with {int(previous['year'])}."
        ),
        detail=(
            f"{int(latest['year'])} recorded {_fmt(latest['total_applications'])} "
            f"applications against {_fmt(previous['total_applications'])} the year "
            f"before, an average of {latest['avg_monthly_applications']:,.0f} per month."
        ),
        kind="positive" if growth >= 0 else "negative",
    )


def _cagr_insight(yearly: pd.DataFrame) -> Optional[Insight]:
    """Report the compound annual growth rate across the full history."""
    if yearly is None or len(yearly) < 3:
        return None

    first, last = yearly.iloc[0], yearly.iloc[-1]
    years = int(last["year"]) - int(first["year"])
    if years <= 0 or first["total_applications"] <= 0:
        return None

    cagr = ((last["total_applications"] / first["total_applications"]) ** (1 / years) - 1) * 100
    total_growth = (last["total_applications"] / first["total_applications"] - 1) * 100

    return Insight(
        category="Growth",
        headline=(
            f"Exchange activity grew at a compound annual rate of {cagr:.1f}% "
            f"between {int(first['year'])} and {int(last['year'])}."
        ),
        detail=(
            f"Total applications rose {total_growth:.0f}% over {years} years, from "
            f"{_fmt(first['total_applications'])} to {_fmt(last['total_applications'])}."
        ),
        kind="positive" if cagr >= 0 else "negative",
    )


def _peak_season_insight(profile: pd.DataFrame) -> Optional[Insight]:
    """Identify the strongest contiguous run of above-average months."""
    if profile is None or profile.empty:
        return None

    ordered = profile.sort_values("month").reset_index(drop=True)
    above = ordered["seasonal_index"] >= 1.0

    # Find the longest run of above-average months, wrapping around December.
    doubled = pd.concat([above, above], ignore_index=True)
    best_start, best_length = 0, 0
    current_start, current_length = 0, 0

    for index, value in enumerate(doubled):
        if value:
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length > best_length and current_length <= 12:
                best_start, best_length = current_start, current_length
        else:
            current_length = 0

    if best_length == 0:
        return None

    names = ordered["month_name"].tolist()
    start_name = names[best_start % 12]
    end_name = names[(best_start + best_length - 1) % 12]
    peak_row = ordered.loc[ordered["seasonal_index"].idxmax()]

    return Insight(
        category="Seasonality",
        headline=f"Peak season runs from {start_name} to {end_name}.",
        detail=(
            f"{best_length} consecutive months sit at or above the annual average. "
            f"{peak_row['month_name']} is the single strongest month at "
            f"{peak_row['seasonal_index']:.2f}x the average."
        ),
        kind="neutral",
    )


def _low_season_insight(profile: pd.DataFrame) -> Optional[Insight]:
    """Name the weakest month and quantify the gap to the strongest."""
    if profile is None or profile.empty:
        return None

    weakest = profile.loc[profile["seasonal_index"].idxmin()]
    strongest = profile.loc[profile["seasonal_index"].idxmax()]
    ratio = float(strongest["mean_applications"] / max(weakest["mean_applications"], 1))

    return Insight(
        category="Seasonality",
        headline=(
            f"{weakest['month_name']} is the quietest month, running at "
            f"{weakest['seasonal_index']:.2f}x the annual average."
        ),
        detail=(
            f"{strongest['month_name']} carries {ratio:.1f}x the volume of "
            f"{weakest['month_name']}. Recruitment and matching capacity should be "
            "planned around this swing rather than a flat monthly assumption."
        ),
        kind="neutral",
    )


# ---------------------------------------------------------------------------
# Funnel insights
# ---------------------------------------------------------------------------
def _funnel_bottleneck_insight(funnel: pd.DataFrame) -> Optional[Insight]:
    """Locate the single largest drop-off in the funnel."""
    if funnel is None or len(funnel) < 2:
        return None

    downstream = funnel.iloc[1:]
    worst = downstream.loc[downstream["dropoff_count"].idxmax()]
    previous = funnel.iloc[int(worst["stage_order"]) - 2]

    return Insight(
        category="Funnel",
        headline=(
            f"The biggest drop-off is {previous['stage']} to {worst['stage']}, "
            f"losing {worst['dropoff_pct']:.0f}% of candidates."
        ),
        detail=(
            f"{_fmt(worst['dropoff_count'])} of {_fmt(previous['count'])} "
            f"{previous['stage_label'].lower()} candidates do not reach "
            f"{worst['stage_label'].lower()}. This transition is the highest-leverage "
            "place to intervene."
        ),
        kind="negative",
    )


def _funnel_efficiency_insight(funnel: pd.DataFrame) -> Optional[Insight]:
    """Report end-to-end application-to-realization efficiency."""
    if funnel is None or funnel.empty or "RE" not in set(funnel["stage"]):
        return None

    applied = funnel.iloc[0]
    realized = funnel[funnel["stage"] == "RE"].iloc[0]
    rate = float(realized["conversion_from_APP_pct"])
    needed = 100 / rate if rate > 0 else float("nan")

    return Insight(
        category="Funnel",
        headline=f"{rate:.1f}% of applications convert all the way to a realization.",
        detail=(
            f"{_fmt(realized['count'])} realizations from {_fmt(applied['count'])} "
            f"applications. Roughly {needed:.0f} applications are needed per "
            "realization, which is the ratio to plan recruitment targets against."
        ),
        kind="neutral",
    )


# ---------------------------------------------------------------------------
# Entity and product insights
# ---------------------------------------------------------------------------
def _entity_concentration_insight(entities: pd.DataFrame) -> Optional[Insight]:
    """Quantify how concentrated volume is in the largest LCs."""
    if entities is None or len(entities) < 3:
        return None

    top_five_share = float(entities.head(5)["mc_share_pct"].sum())
    leader = entities.iloc[0]

    kind = "negative" if top_five_share > 60 else "neutral"
    return Insight(
        category="Entities",
        headline=(
            f"The top 5 Local Committees generate {top_five_share:.0f}% of all "
            "applications."
        ),
        detail=(
            f"{leader['entity']} leads with {_fmt(leader['APP'])} applications "
            f"({leader['mc_share_pct']:.1f}% of the MC) across "
            f"{len(entities)} entities."
            + (
                " That concentration is a delivery risk: a downturn in one or two "
                "LCs moves the national number."
                if kind == "negative"
                else ""
            )
        ),
        kind=kind,
    )


def _entity_growth_insight(entities: pd.DataFrame) -> Optional[Insight]:
    """Highlight the fastest-growing entity of meaningful size."""
    if entities is None or "growth_pct" not in entities.columns:
        return None

    # Restrict to LCs above median volume, so a tiny base cannot win on noise.
    median_volume = entities["APP"].median()
    candidates = entities[(entities["APP"] >= median_volume) & entities["growth_pct"].notna()]
    if candidates.empty:
        return None

    best = candidates.loc[candidates["growth_pct"].idxmax()]
    return Insight(
        category="Entities",
        headline=(
            f"{best['entity']} is the fastest-growing large LC at "
            f"{_pct(float(best['growth_pct']))} ({best.get('growth_period', 'over the period')})."
        ),
        detail=(
            f"It now contributes {best['mc_share_pct']:.1f}% of MC applications with "
            f"an application-to-realization rate of {best.get('app_to_re_pct', float('nan')):.1f}%."
        ),
        kind="positive",
    )


def _product_insight(products: pd.DataFrame) -> Optional[Insight]:
    """Describe the dominant programme and the most efficient one."""
    if products is None or products.empty:
        return None

    dominant = products.iloc[0]
    efficient = (
        products.loc[products["app_to_re_pct"].idxmax()]
        if "app_to_re_pct" in products.columns
        else None
    )

    detail = (
        f"{dominant['programme']} peaks in {dominant.get('peak_month', 'n/a')}."
    )
    if efficient is not None and efficient["programme"] != dominant["programme"]:
        detail += (
            f" {efficient['programme']} converts best, with "
            f"{efficient['app_to_re_pct']:.1f}% of applications reaching realization "
            f"versus {dominant['app_to_re_pct']:.1f}% for {dominant['programme']}."
        )

    return Insight(
        category="Products",
        headline=(
            f"{dominant['programme']} is the largest programme at "
            f"{dominant['mc_share_pct']:.0f}% of all applications."
        ),
        detail=detail,
        kind="neutral",
    )


# ---------------------------------------------------------------------------
# Forecast insights
# ---------------------------------------------------------------------------
def _forecast_peak_insight(predictions: pd.DataFrame) -> Optional[Insight]:
    """Name the predicted peak month of the forecast year."""
    if predictions is None or predictions.empty:
        return None

    peak = predictions.loc[predictions["Predicted Applications"].idxmax()]
    return Insight(
        category="Forecast",
        headline=(
            f"{peak['Month']} is predicted to have the highest activity, at "
            f"{_fmt(peak['Predicted Applications'])} applications."
        ),
        detail=(
            f"That is {peak['share_of_year_pct']:.1f}% of the forecast year's total, "
            f"with a 95% prediction interval of {_fmt(peak['lower_95'])} to "
            f"{_fmt(peak['upper_95'])}."
        ),
        kind="positive",
    )


def _forecast_high_months_insight(predictions: pd.DataFrame) -> Optional[Insight]:
    """List every month classified as high-activity."""
    if predictions is None or predictions.empty:
        return None

    high = predictions[predictions["Activity Level"] == "High"]
    if high.empty:
        return Insight(
            category="Forecast",
            headline="No month is classified as high-activity in the forecast year.",
            detail="Predicted volume stays within the normal seasonal band all year.",
            kind="neutral",
        )

    months = high["Month"].tolist()
    listed = ", ".join(months[:-1]) + (f" and {months[-1]}" if len(months) > 1 else months[0])

    return Insight(
        category="Forecast",
        headline=f"{len(high)} high-activity month(s) predicted: {listed}.",
        detail=(
            f"These months carry {high['share_of_year_pct'].sum():.0f}% of forecast "
            "annual volume. Matching capacity, partner supply and reviewer bandwidth "
            "should be staged ahead of them."
        ),
        kind="positive",
    )


def _forecast_growth_insight(
    predictions: pd.DataFrame, yearly: pd.DataFrame
) -> Optional[Insight]:
    """Compare the forecast year total with the last observed year."""
    if predictions is None or predictions.empty or yearly is None or yearly.empty:
        return None

    forecast_total = float(predictions["Predicted Applications"].sum())
    last_year = yearly.iloc[-1]
    previous_total = float(last_year["total_applications"])
    if previous_total <= 0:
        return None

    growth = (forecast_total - previous_total) / previous_total * 100
    direction = "above" if growth >= 0 else "below"

    return Insight(
        category="Forecast",
        headline=(
            f"Total 2026 applications are forecast at {_fmt(forecast_total)}, "
            f"{abs(growth):.1f}% {direction} {int(last_year['year'])}."
        ),
        detail=(
            f"{int(last_year['year'])} closed at {_fmt(previous_total)}. The forecast "
            f"averages {forecast_total / len(predictions):,.0f} applications per month."
        ),
        kind="positive" if growth >= 0 else "negative",
    )


def _forecast_quiet_period_insight(predictions: pd.DataFrame) -> Optional[Insight]:
    """Identify the quietest stretch, which is the natural capacity-building window."""
    if predictions is None or predictions.empty:
        return None

    low = predictions[predictions["Activity Level"] == "Low"]
    if low.empty:
        trough = predictions.loc[predictions["Predicted Applications"].idxmin()]
        return Insight(
            category="Forecast",
            headline=f"{trough['Month']} is the quietest forecast month.",
            detail=f"Predicted at {_fmt(trough['Predicted Applications'])} applications.",
            kind="neutral",
        )

    months = ", ".join(low["Month"].tolist())
    return Insight(
        category="Forecast",
        headline=f"Lowest predicted activity falls in {months}.",
        detail=(
            "These are the natural windows for member training, partner development "
            "and process work, since delivery pressure is at its lowest."
        ),
        kind="neutral",
    )


def _model_confidence_insight(
    evaluation: pd.DataFrame, model_name: str
) -> Optional[Insight]:
    """Report the selected model's accuracy in plain terms."""
    if evaluation is None or evaluation.empty:
        return None

    row = evaluation[evaluation["model"] == model_name]
    if row.empty:
        return None

    row = row.iloc[0]
    mape = float(row["MAPE"])
    accuracy = 100 - mape

    if mape < 10:
        quality, kind = "strong", "positive"
    elif mape < 20:
        quality, kind = "acceptable", "neutral"
    else:
        quality, kind = "weak", "negative"

    return Insight(
        category="Model",
        headline=(
            f"The selected model ({model_name}) achieves {accuracy:.1f}% accuracy "
            f"in walk-forward validation (MAPE {mape:.1f}%)."
        ),
        detail=(
            f"Mean absolute error is {row['MAE']:,.0f} applications per month across "
            f"{int(row['origins_evaluated'])} out-of-sample origins. That is "
            f"{quality} for operational planning; treat the prediction interval, not "
            "the point estimate, as the planning range."
        ),
        kind=kind,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_all_insights(
    analysis: Optional[Dict[str, pd.DataFrame]] = None,
    predictions: Optional[pd.DataFrame] = None,
    evaluation: Optional[pd.DataFrame] = None,
    model_name: str = "",
    settings: Optional[Settings] = None,
) -> List[Insight]:
    """Run every insight generator and return the ones that produced output.

    Generators that lack their inputs return ``None`` and are skipped, so this
    works with a partial pipeline (e.g. analysis complete but no model yet).

    Args:
        analysis: Mapping from ``analysis.run_full_analysis``.
        predictions: Output of ``train.generate_predictions``.
        evaluation: Model evaluation table.
        model_name: Name of the selected model.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        Insights ordered Forecast, Growth, Seasonality, Funnel, Entities,
        Products, Model.
    """
    settings = settings or get_settings()
    analysis = analysis or {}

    def table(name: str) -> Optional[pd.DataFrame]:
        value = analysis.get(name)
        return value if isinstance(value, pd.DataFrame) and not value.empty else None

    generators: List[Callable[[], Optional[Insight]]] = [
        lambda: _forecast_peak_insight(predictions),
        lambda: _forecast_high_months_insight(predictions),
        lambda: _forecast_growth_insight(predictions, table("yearly_summary")),
        lambda: _forecast_quiet_period_insight(predictions),
        lambda: _growth_insight(table("yearly_summary")),
        lambda: _cagr_insight(table("yearly_summary")),
        lambda: _peak_season_insight(table("seasonality_profile")),
        lambda: _low_season_insight(table("seasonality_profile")),
        lambda: _funnel_bottleneck_insight(table("funnel_overall")),
        lambda: _funnel_efficiency_insight(table("funnel_overall")),
        lambda: _entity_concentration_insight(table("entity_performance")),
        lambda: _entity_growth_insight(table("entity_performance")),
        lambda: _product_insight(table("product_performance")),
        lambda: _model_confidence_insight(evaluation, model_name),
    ]

    insights: List[Insight] = []
    for generate in generators:
        try:
            insight = generate()
        except Exception as exc:  # one bad generator must not sink the page
            logger.debug("Insight generator failed: %s", exc)
            continue
        if insight is not None:
            insights.append(insight)

    logger.info("Generated %d insight(s)", len(insights))
    return insights


def insights_to_frame(insights: List[Insight]) -> pd.DataFrame:
    """Convert insights to a DataFrame for CSV export.

    Args:
        insights: Output of :func:`generate_all_insights`.

    Returns:
        A DataFrame with ``category``, ``headline``, ``detail`` and ``kind``.
    """
    return pd.DataFrame(
        [
            {
                "category": item.category,
                "headline": item.headline,
                "detail": item.detail,
                "kind": item.kind,
            }
            for item in insights
        ]
    )
