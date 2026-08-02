"""Training, model persistence and 2026 prediction generation.

Orchestrates the modelling phase end to end:

    panel -> monthly series -> backtest all candidates -> select best
          -> refit on full history -> forecast 2026 -> classify activity levels
          -> persist model + predictions_2026.csv

The artefacts produced are:

``models/trained_model.pkl``
    The selected fitted model plus the metadata needed to reproduce and audit
    it - backtest metrics, residuals, feature list and training window.

``outputs/predictions_2026.csv``
    The deliverable. First three columns match the requested format exactly
    (``Month``, ``Predicted Applications``, ``Activity Level``); the remaining
    columns add prediction intervals and alternative classifications.

``outputs/reports/model_evaluation.csv``
    Every candidate's MAE / RMSE / MAPE with the selection scoring, so the
    choice of final model is fully auditable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.models.forecasting import (
    BaseForecaster,
    SklearnRecursiveForecaster,
    assign_activity_levels,
    build_model_registry,
    forecast_with_intervals,
    rolling_origin_backtest,
    select_best_model,
)

__all__ = [
    "TrainingArtifacts",
    "build_history_series",
    "train_and_select",
    "generate_predictions",
    "save_artifacts",
    "load_artifacts",
    "run_training_pipeline",
]

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    """Everything produced by a training run."""

    best_model_name: str
    model: BaseForecaster
    evaluation: pd.DataFrame
    backtest_predictions: pd.DataFrame
    residuals: np.ndarray
    history: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_importances: Optional[pd.DataFrame] = None


def build_history_series(panel: pd.DataFrame, settings: Optional[Settings] = None) -> pd.Series:
    """Collapse the entity x product panel into the MC-level monthly target series.

    Args:
        panel: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        The target series indexed by month-start ``Timestamp``, ascending and
        gap-free.

    Raises:
        ValueError: if the resulting series has fewer than 24 observations.
    """
    settings = settings or get_settings()
    target = settings.target

    monthly = panel.groupby("date", as_index=False)[target].sum().sort_values("date")
    series = pd.Series(
        monthly[target].to_numpy(dtype=float),
        index=pd.DatetimeIndex(monthly["date"], name="date"),
        name=target,
    )

    # Reindex onto a complete monthly grid so lag arithmetic is always valid.
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    series = series.reindex(full_index)

    if series.isna().any():
        gaps = series[series.isna()].index.strftime("%Y-%m").tolist()
        logger.warning("Interpolating %d gap month(s): %s", len(gaps), gaps[:6])
        series = series.interpolate(method="linear").bfill().ffill()

    if len(series) < 24:
        raise ValueError(
            f"Need at least 24 months of history to train and backtest; got {len(series)}"
        )

    return series


def train_and_select(
    panel: pd.DataFrame, settings: Optional[Settings] = None, save_reports: bool = True
) -> TrainingArtifacts:
    """Backtest every candidate, select the best, and refit it on all history.

    Args:
        panel: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.
        save_reports: Write the evaluation tables to ``outputs/reports``.

    Returns:
        Populated :class:`TrainingArtifacts`.
    """
    settings = settings or get_settings()
    history = build_history_series(panel, settings)

    logger.info(
        "Training on %d months (%s to %s), target=%s",
        len(history),
        history.index.min().strftime("%Y-%m"),
        history.index.max().strftime("%Y-%m"),
        settings.target,
    )

    registry = build_model_registry(settings)
    backtest = rolling_origin_backtest(history, registry, settings, panel)
    best_name, evaluation = select_best_model(backtest.metrics, settings)

    # Refit the winner on the complete history before forecasting.
    best_model = registry[best_name]
    best_model.fit(history, panel)
    logger.info("Refitted '%s' on the full %d-month history", best_name, len(history))

    importances: Optional[pd.DataFrame] = None
    if isinstance(best_model, SklearnRecursiveForecaster):
        importances = best_model.feature_importances()

    metadata = {
        "trained_at": datetime.now().astimezone().isoformat(),
        "target": settings.target,
        "mc_name": settings.mc_name,
        "history_start": history.index.min().strftime("%Y-%m"),
        "history_end": history.index.max().strftime("%Y-%m"),
        "history_months": int(len(history)),
        "candidates_evaluated": int(len(backtest.metrics)),
        "best_model": best_name,
        "backtest_MAE": float(evaluation.loc[evaluation["model"] == best_name, "MAE"].iloc[0]),
        "backtest_RMSE": float(evaluation.loc[evaluation["model"] == best_name, "RMSE"].iloc[0]),
        "backtest_MAPE": float(evaluation.loc[evaluation["model"] == best_name, "MAPE"].iloc[0]),
    }

    if save_reports:
        reports = settings.paths.reports_dir
        reports.mkdir(parents=True, exist_ok=True)

        evaluation.round(4).to_csv(reports / "model_evaluation.csv", index=False)
        backtest.predictions.to_csv(reports / "backtest_predictions.csv", index=False)
        logger.info("Wrote model_evaluation.csv and backtest_predictions.csv")

        if importances is not None:
            importances.round(6).to_csv(reports / "feature_importance.csv", index=False)
            logger.info("Wrote feature_importance.csv")

    return TrainingArtifacts(
        best_model_name=best_name,
        model=best_model,
        evaluation=evaluation,
        backtest_predictions=backtest.predictions,
        residuals=backtest.residuals.get(best_name, np.array([])),
        history=history,
        metadata=metadata,
        feature_importances=importances,
    )


def generate_predictions(
    artifacts: TrainingArtifacts,
    settings: Optional[Settings] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Forecast the target year and classify each month's activity level.

    Args:
        artifacts: Output of :func:`train_and_select`.
        settings: Loaded settings. Loaded from disk when omitted.
        save: Write ``outputs/predictions_2026.csv``.

    Returns:
        The prediction table. The first three columns are exactly
        ``Month``, ``Predicted Applications``, ``Activity Level``.
    """
    settings = settings or get_settings()
    horizon = settings.horizon
    year = settings.forecast_year
    history = artifacts.history

    # Forecast far enough ahead to cover the whole target year, in case the
    # history ends before December of the preceding year.
    last_observed = history.index.max()
    target_start = pd.Timestamp(year=year, month=1, day=1)
    lead_months = max(
        0, (target_start.year - last_observed.year) * 12 + (target_start.month - last_observed.month) - 1
    )
    total_steps = lead_months + horizon

    forecast = forecast_with_intervals(
        artifacts.model,
        horizon=total_steps,
        residuals=artifacts.residuals if artifacts.residuals.size else None,
        confidence=0.95,
    )

    dates = pd.date_range(last_observed + pd.DateOffset(months=1), periods=total_steps, freq="MS")
    forecast["date"] = dates
    forecast = forecast[forecast["date"].dt.year == year].reset_index(drop=True)

    if forecast.empty:
        raise ValueError(f"Forecast produced no months in {year}")

    values = forecast["forecast"].to_numpy(dtype=float)

    # Primary label, plus both alternatives for transparency.
    primary_labels, thresholds = assign_activity_levels(values, history, settings)
    vs_history, _ = assign_activity_levels(
        values, history, settings, method="historical_quartiles"
    )
    within_year, _ = assign_activity_levels(
        values, history, settings, method="forecast_quartiles"
    )

    table = pd.DataFrame(
        {
            "Month": forecast["date"].dt.strftime("%b %Y"),
            "Predicted Applications": np.rint(values).astype(int),
            "Activity Level": primary_labels,
            # --- supporting detail beyond the required three columns ---
            "date": forecast["date"].dt.strftime("%Y-%m-%d"),
            "month_number": forecast["date"].dt.month,
            "lower_95": np.rint(forecast["lower"].to_numpy()).astype(int),
            "upper_95": np.rint(forecast["upper"].to_numpy()).astype(int),
            "activity_level_vs_history": vs_history,
            "activity_level_within_2026": within_year,
            "model": artifacts.best_model_name,
        }
    )

    # Rank and share, useful for the dashboard's peak-month callouts.
    table["rank_in_year"] = table["Predicted Applications"].rank(ascending=False, method="min").astype(int)
    total = int(table["Predicted Applications"].sum())
    table["share_of_year_pct"] = (table["Predicted Applications"] / total * 100).round(2)

    if save:
        path = settings.paths.predictions
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        logger.info("Predictions written to %s", path)

        summary = {
            "forecast_year": year,
            "model": artifacts.best_model_name,
            "total_predicted_applications": total,
            "peak_month": table.loc[table["rank_in_year"] == 1, "Month"].iloc[0],
            "peak_value": int(table["Predicted Applications"].max()),
            "trough_month": table.loc[
                table["Predicted Applications"].idxmin(), "Month"
            ],
            "high_activity_months": ", ".join(
                table.loc[table["Activity Level"] == "High", "Month"].tolist()
            ),
            **{k: v for k, v in thresholds.items()},
        }
        pd.DataFrame([summary]).to_csv(
            settings.paths.reports_dir / "prediction_summary.csv", index=False
        )
        logger.info("Wrote prediction_summary.csv")

    logger.info(
        "%d forecast: total=%d, peak=%s (%d)",
        year,
        total,
        table.loc[table["rank_in_year"] == 1, "Month"].iloc[0],
        int(table["Predicted Applications"].max()),
    )
    return table


def save_artifacts(
    artifacts: TrainingArtifacts, settings: Optional[Settings] = None
) -> Path:
    """Persist the fitted model and its provenance to ``models/trained_model.pkl``.

    Args:
        artifacts: Output of :func:`train_and_select`.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        The path written.
    """
    settings = settings or get_settings()
    path = settings.paths.trained_model
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": artifacts.model,
        "model_name": artifacts.best_model_name,
        "metadata": artifacts.metadata,
        "evaluation": artifacts.evaluation,
        "residuals": artifacts.residuals,
        "history": artifacts.history,
        "feature_importances": artifacts.feature_importances,
    }

    joblib.dump(payload, path, compress=3)
    logger.info("Model persisted to %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def load_artifacts(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Load a persisted model bundle.

    Args:
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        The payload dict written by :func:`save_artifacts`.

    Raises:
        FileNotFoundError: if no model has been trained yet.
    """
    settings = settings or get_settings()
    path = settings.paths.trained_model

    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python run_pipeline.py --step train"
        )

    return joblib.load(path)


def run_training_pipeline(
    panel: pd.DataFrame, settings: Optional[Settings] = None
) -> tuple[TrainingArtifacts, pd.DataFrame]:
    """Run backtest, selection, persistence and prediction in one call.

    Args:
        panel: Processed exchange panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        ``(artifacts, predictions)``.
    """
    settings = settings or get_settings()
    artifacts = train_and_select(panel, settings)
    save_artifacts(artifacts, settings)
    predictions = generate_predictions(artifacts, settings)
    return artifacts, predictions
