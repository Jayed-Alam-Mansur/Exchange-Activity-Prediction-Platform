"""Exploratory analysis of AIESEC MC India exchange activity.

Three families of analysis, each returning a tidy DataFrame that is both
written to ``outputs/reports/`` and consumed directly by the dashboard:

1. **Time trends** - monthly volume, year-over-year growth, seasonal index,
   peak-month identification.
2. **Exchange funnel** - APP -> ACH -> ACC -> APD -> RE -> FI -> CO stage
   volumes, stage-to-stage conversion, drop-off, and end-to-end efficiency.
3. **Entity / product** - Local Committee performance, product performance and
   each month's contribution to its year.

Every function is pure: it takes a DataFrame and returns a DataFrame. Only
:func:`run_full_analysis` touches the filesystem.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Settings, get_settings

__all__ = [
    "monthly_trend",
    "yearly_summary",
    "seasonality_profile",
    "peak_months",
    "funnel_analysis",
    "funnel_by_dimension",
    "entity_analysis",
    "product_analysis",
    "monthly_contribution",
    "run_full_analysis",
]

logger = logging.getLogger(__name__)


def _stages(settings: Settings) -> List[str]:
    return settings.funnel_stages


# ---------------------------------------------------------------------------
# 1. Time trends
# ---------------------------------------------------------------------------
def monthly_trend(frame: pd.DataFrame, settings: Optional[Settings] = None) -> pd.DataFrame:
    """Aggregate to MC level per month and attach growth/rolling diagnostics.

    Args:
        frame: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        One row per month with funnel totals plus:
        ``mom_growth_pct``, ``yoy_growth_pct``, ``rolling_3m``, ``rolling_12m``,
        ``cumulative_APP`` and ``app_to_re_pct``.
    """
    settings = settings or get_settings()
    stages = _stages(settings)

    trend = frame.groupby("date", as_index=False)[stages].sum().sort_values("date")
    trend["year"] = trend["date"].dt.year.astype(int)
    trend["month"] = trend["date"].dt.month.astype(int)
    trend["month_name"] = trend["date"].dt.strftime("%b")
    trend["period"] = trend["date"].dt.strftime("%Y-%m")

    target = settings.target
    trend["mom_growth_pct"] = trend[target].pct_change() * 100
    trend["yoy_growth_pct"] = trend[target].pct_change(periods=12) * 100
    trend["rolling_3m"] = trend[target].rolling(3, min_periods=1).mean()
    trend["rolling_12m"] = trend[target].rolling(12, min_periods=1).mean()
    trend["cumulative_APP"] = trend[target].cumsum()

    # End-to-end funnel efficiency for the month.
    first, last_realized = stages[0], "RE" if "RE" in stages else stages[-1]
    trend["app_to_re_pct"] = np.where(
        trend[first] > 0, trend[last_realized] / trend[first] * 100, np.nan
    )

    return trend.reset_index(drop=True)


def yearly_summary(frame: pd.DataFrame, settings: Optional[Settings] = None) -> pd.DataFrame:
    """Summarise each calendar year: totals, averages, growth and best month.

    Args:
        frame: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        One row per year with total/average/peak applications, YoY growth and
        end-to-end conversion.
    """
    settings = settings or get_settings()
    stages = _stages(settings)
    target = settings.target

    monthly = frame.groupby(["year", "date"], as_index=False)[stages].sum()

    summary = (
        monthly.groupby("year")
        .agg(
            total_applications=(target, "sum"),
            avg_monthly_applications=(target, "mean"),
            peak_month_applications=(target, "max"),
            min_month_applications=(target, "min"),
            std_monthly_applications=(target, "std"),
            months_observed=(target, "count"),
        )
        .reset_index()
    )

    realized = monthly.groupby("year")["RE"].sum() if "RE" in stages else None
    if realized is not None:
        summary["total_realizations"] = summary["year"].map(realized).astype(int)
        summary["app_to_re_pct"] = (
            summary["total_realizations"] / summary["total_applications"] * 100
        )

    summary["yoy_growth_pct"] = summary["total_applications"].pct_change() * 100

    # Name the strongest month of each year.
    peak_idx = monthly.loc[monthly.groupby("year")[target].idxmax()]
    peak_names = dict(zip(peak_idx["year"], peak_idx["date"].dt.strftime("%b")))
    summary["peak_month"] = summary["year"].map(peak_names)

    numeric = summary.select_dtypes(include=[np.number]).columns
    summary[numeric] = summary[numeric].round(2)
    return summary


def seasonality_profile(
    frame: pd.DataFrame, settings: Optional[Settings] = None
) -> pd.DataFrame:
    """Compute an average monthly profile and a seasonal index.

    The seasonal index expresses each calendar month relative to the overall
    monthly mean: ``1.20`` means that month typically runs 20% above average.

    Args:
        frame: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        Twelve rows (Jan-Dec) with mean/median/std applications, the seasonal
        index, and a ``High``/``Medium``/``Low`` season label.
    """
    settings = settings or get_settings()
    target = settings.target

    monthly = frame.groupby(["date"], as_index=False)[target].sum()
    monthly["month"] = monthly["date"].dt.month.astype(int)
    monthly["month_name"] = monthly["date"].dt.strftime("%b")

    profile = (
        monthly.groupby(["month", "month_name"])
        .agg(
            mean_applications=(target, "mean"),
            median_applications=(target, "median"),
            std_applications=(target, "std"),
            min_applications=(target, "min"),
            max_applications=(target, "max"),
            observations=(target, "count"),
        )
        .reset_index()
        .sort_values("month")
    )

    overall_mean = monthly[target].mean()
    profile["seasonal_index"] = profile["mean_applications"] / overall_mean

    # Label seasons off the same quartile rule used for activity levels.
    low_q = profile["mean_applications"].quantile(settings.activity_levels.get("low_quantile", 0.25))
    high_q = profile["mean_applications"].quantile(
        settings.activity_levels.get("high_quantile", 0.75)
    )
    profile["season_label"] = np.select(
        [profile["mean_applications"] >= high_q, profile["mean_applications"] <= low_q],
        ["High", "Low"],
        default="Medium",
    )

    numeric = profile.select_dtypes(include=[np.number]).columns
    profile[numeric] = profile[numeric].round(3)
    return profile.reset_index(drop=True)


def peak_months(
    frame: pd.DataFrame, top_n: int = 5, settings: Optional[Settings] = None
) -> pd.DataFrame:
    """Rank individual historical months by application volume.

    Args:
        frame: Processed exchange panel.
        top_n: How many months to return.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        The ``top_n`` busiest months, highest first.
    """
    settings = settings or get_settings()
    target = settings.target

    monthly = frame.groupby("date", as_index=False)[target].sum()
    monthly["period"] = monthly["date"].dt.strftime("%b %Y")
    monthly["rank"] = monthly[target].rank(ascending=False, method="min").astype(int)

    # ``period`` already identifies the month; the raw datetime adds nothing but
    # a "00:00:00" suffix in every table that renders this frame.
    return (
        monthly.sort_values(target, ascending=False)
        .head(top_n)[["rank", "period", target]]
        .rename(columns={"period": "month", target: "applications"})
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 2. Exchange funnel
# ---------------------------------------------------------------------------
def funnel_analysis(frame: pd.DataFrame, settings: Optional[Settings] = None) -> pd.DataFrame:
    """Compute overall funnel volumes, conversion rates and drop-off.

    Definitions:
      * ``conversion_from_previous_pct`` - share of the *previous* stage that
        advanced to this one. This is the operational lever.
      * ``conversion_from_APP_pct`` - share of all applications that reached
        this stage (cumulative funnel efficiency).
      * ``dropoff_count`` / ``dropoff_pct`` - volume lost at this step.

    Args:
        frame: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        One row per funnel stage, in funnel order.
    """
    settings = settings or get_settings()
    stages = _stages(settings)
    labels = settings.funnel_labels

    totals = {stage: int(frame[stage].sum()) for stage in stages}
    top = totals[stages[0]] or 1

    rows: List[Dict[str, object]] = []
    for index, stage in enumerate(stages):
        previous = totals[stages[index - 1]] if index else None
        current = totals[stage]

        conversion = (current / previous * 100) if previous else 100.0
        dropoff_count = (previous - current) if previous is not None else 0
        dropoff_pct = (dropoff_count / previous * 100) if previous else 0.0

        rows.append(
            {
                "stage_order": index + 1,
                "stage": stage,
                "stage_label": labels.get(stage, stage),
                "count": current,
                "conversion_from_previous_pct": round(conversion, 2),
                "conversion_from_APP_pct": round(current / top * 100, 2),
                "dropoff_count": int(dropoff_count),
                "dropoff_pct": round(dropoff_pct, 2),
            }
        )

    result = pd.DataFrame(rows)

    # Flag the single worst leak - the stage transition losing the most volume.
    if len(result) > 1:
        worst = result.iloc[1:]["dropoff_count"].idxmax()
        result["is_biggest_dropoff"] = result.index == worst
    else:
        result["is_biggest_dropoff"] = False

    return result


def funnel_by_dimension(
    frame: pd.DataFrame, dimension: str, settings: Optional[Settings] = None
) -> pd.DataFrame:
    """Compute funnel efficiency broken down by any categorical column.

    Args:
        frame: Processed exchange panel.
        dimension: Column to group by, e.g. ``"entity"``, ``"product"``, ``"year"``.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        One row per dimension value with stage totals and key conversion rates,
        sorted by application volume.

    Raises:
        KeyError: if ``dimension`` is not a column of ``frame``.
    """
    settings = settings or get_settings()
    stages = _stages(settings)

    if dimension not in frame.columns:
        raise KeyError(f"'{dimension}' is not a column of the dataset")

    grouped = frame.groupby(dimension, as_index=False)[stages].sum()

    first = stages[0]
    denominator = grouped[first].replace(0, np.nan)
    for index, stage in enumerate(stages[1:], start=1):
        previous = grouped[stages[index - 1]].replace(0, np.nan)
        grouped[f"conv_{stages[index - 1]}_to_{stage}_pct"] = (
            grouped[stage] / previous * 100
        ).round(2)

    if "RE" in stages:
        grouped["app_to_re_pct"] = (grouped["RE"] / denominator * 100).round(2)
    if "CO" in stages:
        grouped["app_to_co_pct"] = (grouped["CO"] / denominator * 100).round(2)

    return grouped.sort_values(first, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Entity / product analysis
# ---------------------------------------------------------------------------
def entity_analysis(frame: pd.DataFrame, settings: Optional[Settings] = None) -> pd.DataFrame:
    """Rank Local Committees by volume, efficiency, consistency and growth.

    Args:
        frame: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        One row per LC with total/average applications, MC share, realizations,
        end-to-end conversion, month-to-month consistency and first-to-last-year
        growth, sorted by volume.
    """
    settings = settings or get_settings()
    stages = _stages(settings)
    target = settings.target

    totals = frame.groupby("entity", as_index=False)[stages].sum()
    totals["mc_share_pct"] = (totals[target] / totals[target].sum() * 100).round(2)

    if "RE" in stages:
        totals["app_to_re_pct"] = (
            totals["RE"] / totals[target].replace(0, np.nan) * 100
        ).round(2)

    # Monthly stability: coefficient of variation (lower = more consistent).
    monthly = frame.groupby(["entity", "date"], as_index=False)[target].sum()
    stability = monthly.groupby("entity")[target].agg(["mean", "std"]).reset_index()
    stability["consistency_cv"] = (stability["std"] / stability["mean"].replace(0, np.nan)).round(3)
    totals = totals.merge(
        stability[["entity", "mean", "consistency_cv"]].rename(
            columns={"mean": "avg_monthly_applications"}
        ),
        on="entity",
        how="left",
    )
    totals["avg_monthly_applications"] = totals["avg_monthly_applications"].round(1)

    # Growth from the first to the last full year in the data.
    per_year = frame.groupby(["entity", "year"], as_index=False)[target].sum()
    years = sorted(per_year["year"].unique())
    if len(years) >= 2:
        first_year = per_year[per_year["year"] == years[0]].set_index("entity")[target]
        last_year = per_year[per_year["year"] == years[-1]].set_index("entity")[target]
        growth = ((last_year - first_year) / first_year.replace(0, np.nan) * 100).round(1)
        totals["growth_pct"] = totals["entity"].map(growth)
        totals["growth_period"] = f"{years[0]}->{years[-1]}"

    totals = totals.sort_values(target, ascending=False).reset_index(drop=True)
    totals.insert(0, "rank", np.arange(1, len(totals) + 1))
    return totals


def product_analysis(frame: pd.DataFrame, settings: Optional[Settings] = None) -> pd.DataFrame:
    """Compare exchange products and directions (iGV, oGV, iGTa, ...).

    Args:
        frame: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        One row per ``programme`` with volume, share, conversion, peak month and
        growth, sorted by volume.
    """
    settings = settings or get_settings()
    stages = _stages(settings)
    target = settings.target

    totals = frame.groupby(["programme", "product", "direction"], as_index=False)[stages].sum()
    totals["mc_share_pct"] = (totals[target] / totals[target].sum() * 100).round(2)

    if "RE" in stages:
        totals["app_to_re_pct"] = (
            totals["RE"] / totals[target].replace(0, np.nan) * 100
        ).round(2)

    # Which calendar month is this programme's strongest, on average?
    monthly = frame.groupby(["programme", "month"], as_index=False)[target].sum()
    peak = monthly.loc[monthly.groupby("programme")[target].idxmax()]
    month_names = {i: pd.Timestamp(2024, i, 1).strftime("%b") for i in range(1, 13)}
    totals["peak_month"] = totals["programme"].map(
        dict(zip(peak["programme"], peak["month"].map(month_names)))
    )

    per_year = frame.groupby(["programme", "year"], as_index=False)[target].sum()
    years = sorted(per_year["year"].unique())
    if len(years) >= 2:
        first_year = per_year[per_year["year"] == years[0]].set_index("programme")[target]
        last_year = per_year[per_year["year"] == years[-1]].set_index("programme")[target]
        totals["growth_pct"] = totals["programme"].map(
            ((last_year - first_year) / first_year.replace(0, np.nan) * 100).round(1)
        )

    return totals.sort_values(target, ascending=False).reset_index(drop=True)


def monthly_contribution(
    frame: pd.DataFrame, settings: Optional[Settings] = None
) -> pd.DataFrame:
    """Express each month as a share of its own year's total.

    Normalising within the year removes the growth trend, isolating the
    seasonal shape so that years are directly comparable.

    Args:
        frame: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        One row per (year, month) with applications, the month's share of the
        year, and the deviation from an even 1/12 split.
    """
    settings = settings or get_settings()
    target = settings.target

    monthly = frame.groupby(["year", "month"], as_index=False)[target].sum()
    monthly["month_name"] = monthly["month"].map(
        {i: pd.Timestamp(2024, i, 1).strftime("%b") for i in range(1, 13)}
    )

    year_totals = monthly.groupby("year")[target].transform("sum")
    monthly["contribution_pct"] = (monthly[target] / year_totals * 100).round(2)
    monthly["vs_even_split_pp"] = (monthly["contribution_pct"] - 100 / 12).round(2)
    monthly["rank_in_year"] = (
        monthly.groupby("year")[target].rank(ascending=False, method="min").astype(int)
    )

    return monthly.sort_values(["year", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_full_analysis(
    frame: pd.DataFrame,
    settings: Optional[Settings] = None,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run every analysis and optionally persist each result to ``outputs/reports``.

    Args:
        frame: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.
        save: Write one CSV per analysis.

    Returns:
        Mapping of analysis name to its DataFrame.
    """
    settings = settings or get_settings()

    results: Dict[str, pd.DataFrame] = {
        "monthly_trend": monthly_trend(frame, settings),
        "yearly_summary": yearly_summary(frame, settings),
        "seasonality_profile": seasonality_profile(frame, settings),
        "peak_months": peak_months(frame, top_n=10, settings=settings),
        "funnel_overall": funnel_analysis(frame, settings),
        "funnel_by_product": funnel_by_dimension(frame, "programme", settings),
        "funnel_by_year": funnel_by_dimension(frame, "year", settings),
        "entity_performance": entity_analysis(frame, settings),
        "product_performance": product_analysis(frame, settings),
        "monthly_contribution": monthly_contribution(frame, settings),
    }

    if save:
        reports_dir = settings.paths.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        for name, table in results.items():
            path = reports_dir / f"{name}.csv"
            table.to_csv(path, index=False)
            logger.info("Wrote %s (%d rows)", path.name, len(table))

    return results
