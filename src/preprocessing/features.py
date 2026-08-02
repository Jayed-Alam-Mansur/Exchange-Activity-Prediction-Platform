"""Feature engineering for exchange-activity forecasting.

Design principle: **one function builds one feature row, everywhere.**

:func:`compute_feature_row` is the single place where a feature vector is
constructed. It is used to build the training matrix *and* to build each step of
a recursive multi-month forecast. Because training and serving share the exact
same code path, train/serve skew is structurally impossible rather than merely
avoided by discipline.

Feature families
----------------
**Time** - year, month, quarter, season, month index, cyclical Fourier terms,
Indian academic-calendar flags. Fully known for any future date.

**Historical** - lags (1, 2, 3, 6, 12 months), rolling means/std over 3/6/12
months, month-over-month and year-over-year growth. Every one of these is
computed from data strictly *before* the target month, so there is no leakage.

**Operational** - trailing funnel conversion quality, realization volume,
product-mix share and entity concentration (HHI). These describe the state of
the operation entering the month and are also strictly lagged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import Settings, get_settings

__all__ = [
    "SEASON_BY_MONTH",
    "compute_feature_row",
    "build_training_frame",
    "build_operational_features",
    "feature_columns",
]

logger = logging.getLogger(__name__)

#: Indian climatic/academic seasons - drives student availability for exchange.
SEASON_BY_MONTH: Dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring",
    5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Autumn",
}

#: Ordinal encoding so tree and linear models can both consume the season.
SEASON_CODES: Dict[str, int] = {
    "Winter": 0, "Spring": 1, "Summer": 2, "Monsoon": 3, "Autumn": 4
}

#: University holidays - the windows in which Indian students actually travel.
BREAK_MONTHS = {5, 6, 12, 1}

#: End-semester examination months - application activity typically dips.
EXAM_MONTHS = {3, 4, 11}

#: Semester start months - recruitment drives run here.
SEMESTER_START_MONTHS = {1, 7, 8}

#: Lag horizons in months. 12 captures "same month last year".
LAG_MONTHS: Sequence[int] = (1, 2, 3, 6, 12)

#: Rolling window sizes in months.
ROLLING_WINDOWS: Sequence[int] = (3, 6, 12)

#: Operational features carried alongside the target.
OPERATIONAL_FEATURES: Sequence[str] = (
    "op_app_to_re_pct",
    "op_app_to_ach_pct",
    "op_realizations",
    "op_outgoing_share",
    "op_gv_share",
    "op_entity_hhi",
    "op_active_entities",
)


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------
def _time_features(target_date: pd.Timestamp, origin: pd.Timestamp) -> Dict[str, float]:
    """Build calendar features for a single month.

    All of these are knowable arbitrarily far into the future, which is what
    makes a 12-month-ahead forecast possible at all.

    Args:
        target_date: The month being described (first of month).
        origin: The first month of the historical series, used as the index base.

    Returns:
        Mapping of feature name to numeric value.
    """
    month = int(target_date.month)
    season = SEASON_BY_MONTH[month]

    month_index = (target_date.year - origin.year) * 12 + (target_date.month - origin.month)

    return {
        "year": float(target_date.year),
        "month": float(month),
        "quarter": float(target_date.quarter),
        "season_code": float(SEASON_CODES[season]),
        "month_index": float(month_index),
        # Two Fourier harmonics express smooth annual seasonality with only four
        # parameters - far more sample-efficient than 11 month dummies on 48 rows.
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "month_sin2": float(np.sin(4 * np.pi * month / 12)),
        "month_cos2": float(np.cos(4 * np.pi * month / 12)),
        "is_break_season": float(month in BREAK_MONTHS),
        "is_exam_season": float(month in EXAM_MONTHS),
        "is_semester_start": float(month in SEMESTER_START_MONTHS),
        "is_year_end": float(month == 12),
        "days_in_month": float(target_date.days_in_month),
    }


# ---------------------------------------------------------------------------
# Historical features
# ---------------------------------------------------------------------------
def _history_features(history: pd.Series, target_date: pd.Timestamp) -> Dict[str, float]:
    """Build lag, rolling and growth features from prior observations only.

    Args:
        history: Target series indexed by month-start ``Timestamp``, ascending.
        target_date: The month being predicted. Everything at or after this
            date is excluded, which is what guarantees no leakage.

    Returns:
        Mapping of feature name to numeric value. ``NaN`` where insufficient
        history exists.
    """
    past = history[history.index < target_date]
    features: Dict[str, float] = {}

    # --- lags ---
    for lag in LAG_MONTHS:
        lag_date = target_date - pd.DateOffset(months=lag)
        features[f"lag_{lag}"] = float(past.get(lag_date, np.nan))

    # --- rolling statistics over the trailing window ---
    for window in ROLLING_WINDOWS:
        tail = past.tail(window)
        features[f"roll_mean_{window}"] = float(tail.mean()) if len(tail) else np.nan
        features[f"roll_std_{window}"] = float(tail.std()) if len(tail) > 1 else 0.0

    tail12 = past.tail(12)
    features["roll_max_12"] = float(tail12.max()) if len(tail12) else np.nan
    features["roll_min_12"] = float(tail12.min()) if len(tail12) else np.nan

    # --- growth ---
    lag_1, lag_2, lag_12 = features["lag_1"], features["lag_2"], features["lag_12"]
    lag_13 = float(past.get(target_date - pd.DateOffset(months=13), np.nan))

    features["mom_growth"] = _safe_growth(lag_1, lag_2)
    features["yoy_growth"] = _safe_growth(lag_1, lag_13)
    features["lag1_over_lag12"] = _safe_ratio(lag_1, lag_12)
    features["lag1_over_roll12"] = _safe_ratio(lag_1, features["roll_mean_12"])

    # Recent trend slope over the trailing six months (units per month).
    tail6 = past.tail(6)
    if len(tail6) >= 3:
        x = np.arange(len(tail6), dtype=float)
        features["trend_slope_6"] = float(np.polyfit(x, tail6.to_numpy(dtype=float), 1)[0])
    else:
        features["trend_slope_6"] = 0.0

    # Expanding mean of this calendar month across all previous years - a
    # direct, highly informative encoding of seasonality.
    same_month = past[past.index.month == target_date.month]
    features["same_month_mean"] = float(same_month.mean()) if len(same_month) else np.nan

    return features


def _safe_growth(current: float, previous: float) -> float:
    """Percentage change, returning 0.0 rather than inf/NaN on a zero base."""
    if not np.isfinite(current) or not np.isfinite(previous) or previous == 0:
        return 0.0
    return float((current - previous) / previous * 100.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio, returning 1.0 rather than inf/NaN on a zero or missing base."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return 1.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Row builder - the single shared code path
# ---------------------------------------------------------------------------
def compute_feature_row(
    history: pd.Series,
    target_date: pd.Timestamp,
    origin: pd.Timestamp,
    operational: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Build the complete feature vector for one month.

    Used identically when assembling the training matrix and when rolling a
    recursive forecast forward, so the two can never diverge.

    Args:
        history: Target series indexed by month-start, ascending. Only values
            strictly before ``target_date`` are consulted.
        target_date: The month to describe.
        origin: First month of the full series (base for ``month_index``).
        operational: Lagged operational context for this month. Missing keys
            default to ``NaN``.

    Returns:
        A flat feature mapping.
    """
    row: Dict[str, float] = {}
    row.update(_time_features(target_date, origin))
    row.update(_history_features(history, target_date))

    operational = operational or {}
    for name in OPERATIONAL_FEATURES:
        row[name] = float(operational.get(name, np.nan))

    return row


def feature_columns() -> List[str]:
    """Return the canonical, ordered list of feature column names.

    Ordering is fixed so that a persisted model always receives its columns in
    the same positions it was trained on.
    """
    probe_index = pd.date_range("2020-01-01", periods=26, freq="MS")
    probe = pd.Series(np.arange(26, dtype=float), index=probe_index)
    row = compute_feature_row(probe, probe_index[-1], probe_index[0], {})
    return list(row.keys())


# ---------------------------------------------------------------------------
# Operational features from the entity/product panel
# ---------------------------------------------------------------------------
def build_operational_features(
    panel: pd.DataFrame, settings: Optional[Settings] = None
) -> pd.DataFrame:
    """Derive monthly operational context from the entity x product panel.

    Every feature is **shifted forward by one month** before use, so the model
    only ever sees the operational state that was actually observable before the
    month it is predicting.

    Args:
        panel: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        A DataFrame indexed by ``date`` with the columns in
        :data:`OPERATIONAL_FEATURES`, already lagged by one month.
    """
    settings = settings or get_settings()
    stages = settings.funnel_stages
    target = settings.target

    monthly = panel.groupby("date", as_index=False)[stages].sum().sort_values("date")

    features = pd.DataFrame({"date": monthly["date"]})
    denominator = monthly[target].replace(0, np.nan)

    features["op_app_to_re_pct"] = (monthly.get("RE", 0) / denominator * 100).astype(float)
    features["op_app_to_ach_pct"] = (monthly.get("ACH", 0) / denominator * 100).astype(float)
    features["op_realizations"] = monthly.get("RE", pd.Series(0, index=monthly.index)).astype(float)

    # Product / direction mix.
    direction_mix = (
        panel.pivot_table(index="date", columns="direction", values=target, aggfunc="sum")
        .fillna(0.0)
    )
    total_by_date = direction_mix.sum(axis=1).replace(0, np.nan)
    outgoing_share = (direction_mix.get("outgoing", 0) / total_by_date * 100).astype(float)
    features["op_outgoing_share"] = features["date"].map(outgoing_share)

    product_mix = (
        panel.pivot_table(index="date", columns="product", values=target, aggfunc="sum")
        .fillna(0.0)
    )
    product_total = product_mix.sum(axis=1).replace(0, np.nan)
    gv_share = (product_mix.get("GV", 0) / product_total * 100).astype(float)
    features["op_gv_share"] = features["date"].map(gv_share)

    # Entity concentration: Herfindahl-Hirschman Index over LC shares.
    # High HHI = volume concentrated in a few LCs = more fragile.
    entity_monthly = panel.groupby(["date", "entity"], as_index=False)[target].sum()
    entity_totals = entity_monthly.groupby("date")[target].transform("sum").replace(0, np.nan)
    entity_monthly["share"] = entity_monthly[target] / entity_totals
    hhi = entity_monthly.groupby("date")["share"].apply(lambda s: float((s**2).sum()))
    features["op_entity_hhi"] = features["date"].map(hhi)

    active = entity_monthly[entity_monthly[target] > 0].groupby("date")["entity"].nunique()
    features["op_active_entities"] = features["date"].map(active).astype(float)

    features = features.set_index("date").sort_index()

    # Lag by one month: the state entering month t is what was observed in t-1.
    lagged = features.shift(1)
    return lagged


# ---------------------------------------------------------------------------
# Training matrix
# ---------------------------------------------------------------------------
def build_training_frame(
    panel: pd.DataFrame,
    settings: Optional[Settings] = None,
    min_history_months: int = 12,
    save: bool = False,
) -> pd.DataFrame:
    """Assemble the supervised learning matrix from the exchange panel.

    Args:
        panel: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.
        min_history_months: Rows before this many months of history are dropped,
            because their 12-month lags would be undefined.
        save: Write the matrix to ``data/processed/features.csv``.

    Returns:
        A DataFrame with a ``date`` column, every engineered feature, and the
        target column named ``y``.

    Raises:
        ValueError: if the panel yields fewer rows than ``min_history_months``.
    """
    settings = settings or get_settings()
    target = settings.target

    monthly = panel.groupby("date", as_index=False)[target].sum().sort_values("date")
    series = pd.Series(
        monthly[target].to_numpy(dtype=float),
        index=pd.DatetimeIndex(monthly["date"]),
        name=target,
    )

    if len(series) <= min_history_months:
        raise ValueError(
            f"Need more than {min_history_months} months of history to build "
            f"features; got {len(series)}"
        )

    operational = build_operational_features(panel, settings)
    origin = series.index[0]

    rows: List[Dict[str, Any]] = []
    for target_date in series.index:
        op_row = operational.loc[target_date].to_dict() if target_date in operational.index else {}
        row = compute_feature_row(series, target_date, origin, op_row)
        row["date"] = target_date
        row["y"] = float(series.loc[target_date])
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame = frame[["date", "y"] + [c for c in frame.columns if c not in ("date", "y")]]

    # Drop the warm-up period where 12-month lags are undefined.
    frame = frame.iloc[min_history_months:].reset_index(drop=True)

    if save:
        path = settings.paths.features
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        logger.info("Feature matrix written to %s (%d rows x %d cols)", path, *frame.shape)

    logger.info(
        "Built feature matrix: %d rows, %d features (target=%s)",
        len(frame),
        len(frame.columns) - 2,
        target,
    )
    return frame
