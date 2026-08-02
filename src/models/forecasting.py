"""Forecasting model suite, backtesting harness and model selection.

Philosophy
----------
Start simple, escalate only if the data earns it. With 48 monthly observations,
a deep network is not a modelling choice - it is a way to overfit confidently.
So the suite spans four tiers and lets a walk-forward backtest decide:

1. **Baselines** - last value, 3/6-month moving average, seasonal naive
   (same month last year), seasonal naive with drift.
2. **Linear** - ordinary least squares and ridge over engineered features.
3. **Classical time series** - Holt-Winters exponential smoothing, SARIMA and
   (optionally) Prophet.
4. **Tree ensembles** - random forest, gradient boosting, XGBoost.

Every model implements the same :class:`BaseForecaster` interface, so the
backtest, selection and forecasting code is written once.

Evaluation
----------
Rolling-origin (walk-forward) cross-validation with an expanding window. At each
origin the model is refit on everything observed so far and predicts one month
ahead. This mirrors how the platform would actually be used and, unlike a random
K-fold split, never lets the model see the future.

Reported per model: **MAE**, **RMSE**, **MAPE**, sMAPE, error standard deviation
(stability) and worst-case absolute error.

Selection
---------
Accuracy first: models within a configurable tolerance of the best MAE are
treated as tied. Ties are broken by a transparent weighted score over accuracy,
stability and explainability, and the full scoring table is written to
``outputs/reports/model_evaluation.csv`` so the choice can be audited.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.preprocessing.features import (
    OPERATIONAL_FEATURES,
    build_operational_features,
    compute_feature_row,
)

__all__ = [
    "BaseForecaster",
    "NaiveLastForecaster",
    "MovingAverageForecaster",
    "SeasonalNaiveForecaster",
    "SeasonalNaiveDriftForecaster",
    "SklearnRecursiveForecaster",
    "HoltWintersForecaster",
    "SarimaForecaster",
    "ProphetForecaster",
    "build_model_registry",
    "evaluate_metrics",
    "rolling_origin_backtest",
    "select_best_model",
    "assign_activity_levels",
    "forecast_with_intervals",
]

logger = logging.getLogger(__name__)

# Statsmodels emits convergence chatter that drowns the pipeline log. The
# backtest already reports which models failed, so the warnings add no signal.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:  # pragma: no cover - depends on the installed statsmodels version
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=ValueWarning)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute the standard forecast error metrics.

    Args:
        y_true: Observed values.
        y_pred: Predicted values, same length as ``y_true``.

    Returns:
        Mapping with ``MAE``, ``RMSE``, ``MAPE``, ``sMAPE``, ``error_std``,
        ``max_abs_error``, ``bias`` and ``n``.

    Raises:
        ValueError: if the inputs differ in length or are empty.
    """
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {truth.shape} vs y_pred {pred.shape}")
    if truth.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")

    errors = truth - pred
    abs_errors = np.abs(errors)

    # MAPE is undefined at zero; mask those points rather than returning inf.
    nonzero = truth != 0
    mape = float(np.mean(abs_errors[nonzero] / np.abs(truth[nonzero])) * 100) if nonzero.any() else float("nan")

    denominator = (np.abs(truth) + np.abs(pred)) / 2.0
    valid = denominator != 0
    smape = float(np.mean(abs_errors[valid] / denominator[valid]) * 100) if valid.any() else float("nan")

    return {
        "MAE": float(np.mean(abs_errors)),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE": mape,
        "sMAPE": smape,
        "error_std": float(np.std(abs_errors)),
        "max_abs_error": float(np.max(abs_errors)),
        "bias": float(np.mean(errors)),
        "n": int(truth.size),
    }


# ---------------------------------------------------------------------------
# Forecaster interface
# ---------------------------------------------------------------------------
class BaseForecaster(ABC):
    """Common interface for every forecasting model in the suite.

    Attributes:
        name: Registry key, also used in reports.
        family: One of ``baseline`` / ``linear`` / ``timeseries`` / ``ensemble``.
        explainability: 1 (fully transparent) to 4 (black box).
    """

    name: str = "base"
    family: str = "baseline"
    explainability: int = 1

    def __init__(self) -> None:
        self._history: Optional[pd.Series] = None
        self._fitted: bool = False

    @abstractmethod
    def fit(self, history: pd.Series, panel: Optional[pd.DataFrame] = None) -> "BaseForecaster":
        """Fit the model.

        Args:
            history: Target series indexed by month-start ``Timestamp``, ascending.
            panel: Optional entity x product panel, for operational features.

        Returns:
            ``self``, to allow chaining.
        """

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Predict the next ``horizon`` months after the fitted history.

        Args:
            horizon: Number of months to forecast.

        Returns:
            Array of length ``horizon``.
        """

    def _check_fitted(self) -> None:
        if not self._fitted or self._history is None:
            raise RuntimeError(f"{self.name} must be fitted before calling predict()")

    @staticmethod
    def _clip(values: np.ndarray) -> np.ndarray:
        """Clip to non-negative - application counts cannot be negative."""
        return np.clip(np.asarray(values, dtype=float), 0.0, None)


# ---------------------------------------------------------------------------
# Tier 1: baselines
# ---------------------------------------------------------------------------
class NaiveLastForecaster(BaseForecaster):
    """Carry the most recent observation forward. The floor every model must beat."""

    name = "naive_last"
    family = "baseline"
    explainability = 1

    def fit(self, history: pd.Series, panel: Optional[pd.DataFrame] = None) -> "NaiveLastForecaster":
        self._history = history.astype(float)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        assert self._history is not None
        return self._clip(np.repeat(self._history.iloc[-1], horizon))


class MovingAverageForecaster(BaseForecaster):
    """Average of the trailing ``window`` months, held flat across the horizon.

    Args:
        window: Number of trailing months to average.
    """

    family = "baseline"
    explainability = 1

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        self.window = int(window)
        self.name = f"moving_average_{self.window}"

    def fit(self, history: pd.Series, panel: Optional[pd.DataFrame] = None) -> "MovingAverageForecaster":
        self._history = history.astype(float)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        assert self._history is not None
        value = float(self._history.tail(self.window).mean())
        return self._clip(np.repeat(value, horizon))


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeat the value from the same month one year earlier.

    A strong, hard-to-beat baseline for any series with a dominant annual cycle -
    which exchange activity, tied to the academic calendar, certainly has.
    """

    name = "seasonal_naive"
    family = "baseline"
    explainability = 1

    def __init__(self, seasonal_period: int = 12) -> None:
        super().__init__()
        self.seasonal_period = int(seasonal_period)

    def fit(self, history: pd.Series, panel: Optional[pd.DataFrame] = None) -> "SeasonalNaiveForecaster":
        self._history = history.astype(float)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        assert self._history is not None
        values = self._history.to_numpy(dtype=float)
        period = self.seasonal_period

        if len(values) < period:  # not enough history for a full cycle
            return self._clip(np.repeat(values[-1], horizon))

        season = values[-period:]
        return self._clip(np.array([season[i % period] for i in range(horizon)]))


class SeasonalNaiveDriftForecaster(SeasonalNaiveForecaster):
    """Seasonal naive scaled by the most recent year-over-year growth rate.

    Captures both the annual shape *and* the underlying growth trend while
    remaining completely transparent: ``forecast = last_year_same_month x growth``.
    """

    name = "seasonal_naive_drift"
    family = "baseline"
    explainability = 1

    def __init__(self, seasonal_period: int = 12, damping: float = 0.85) -> None:
        super().__init__(seasonal_period)
        self.damping = float(damping)
        self._growth: float = 1.0

    def fit(self, history: pd.Series, panel: Optional[pd.DataFrame] = None) -> "SeasonalNaiveDriftForecaster":
        super().fit(history, panel)
        values = history.astype(float).to_numpy()
        period = self.seasonal_period

        if len(values) >= 2 * period:
            recent = float(values[-period:].sum())
            previous = float(values[-2 * period : -period].sum())
            raw_growth = recent / previous if previous > 0 else 1.0
            # Damped: growth rarely persists undiminished, and damping protects
            # the 12-month horizon from compounding an optimistic single year.
            self._growth = 1.0 + (raw_growth - 1.0) * self.damping
        else:
            self._growth = 1.0

        return self

    def predict(self, horizon: int) -> np.ndarray:
        base = super().predict(horizon)
        return self._clip(base * self._growth)


# ---------------------------------------------------------------------------
# Tier 2 & 4: feature-based regressors (linear and tree ensembles)
# ---------------------------------------------------------------------------
class SklearnRecursiveForecaster(BaseForecaster):
    """Wrap any scikit-learn-style regressor as a recursive monthly forecaster.

    The regressor is trained on the engineered feature matrix. To forecast month
    ``t + k`` it predicts one month at a time, appending each prediction to the
    working history so that the next step's lag and rolling features are
    available. Features are built by the *same* :func:`compute_feature_row` used
    at training time, which rules out train/serve skew.

    Operational features cannot be observed for future months, so they are held
    at their trailing 12-month mean - a documented, deliberately conservative
    assumption.

    Args:
        name: Registry key.
        estimator: An unfitted regressor exposing ``fit`` / ``predict``.
        family: Model family label for reporting.
        explainability: 1 (transparent) to 4 (black box).
        scale_features: Standardise features before fitting - needed for
            penalised linear models, harmful to nothing.
    """

    def __init__(
        self,
        name: str,
        estimator: Any,
        family: str = "ensemble",
        explainability: int = 4,
        scale_features: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.estimator = estimator
        self.family = family
        self.explainability = explainability
        self.scale_features = scale_features

        self._feature_names: List[str] = []
        self._origin: Optional[pd.Timestamp] = None
        self._operational_defaults: Dict[str, float] = {}
        self._scaler: Optional[Any] = None
        self._min_history: int = 12

    def _training_matrix(
        self, history: pd.Series, operational: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build the (X, y) pair for the supplied history."""
        assert self._origin is not None
        rows: List[Dict[str, float]] = []
        targets: List[float] = []

        for target_date in history.index:
            op_row: Dict[str, float] = {}
            if operational is not None and target_date in operational.index:
                op_row = operational.loc[target_date].to_dict()
            rows.append(compute_feature_row(history, target_date, self._origin, op_row))
            targets.append(float(history.loc[target_date]))

        frame = pd.DataFrame(rows).iloc[self._min_history :]
        target_array = np.asarray(targets[self._min_history :], dtype=float)
        return frame, target_array

    def fit(
        self, history: pd.Series, panel: Optional[pd.DataFrame] = None
    ) -> "SklearnRecursiveForecaster":
        self._history = history.astype(float)
        self._origin = history.index[0]

        operational: Optional[pd.DataFrame] = None
        if panel is not None:
            try:
                operational = build_operational_features(panel)
            except Exception as exc:  # operational features are a nice-to-have
                logger.debug("Operational features unavailable for %s: %s", self.name, exc)

        # Use as much warm-up as the history affords, but never all of it.
        self._min_history = min(12, max(3, len(history) // 3))

        features, targets = self._training_matrix(self._history, operational)
        if len(features) < 4:
            raise ValueError(
                f"{self.name}: only {len(features)} usable training rows after warm-up"
            )

        features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
        self._feature_names = list(features.columns)

        # Freeze the operational context used for future months.
        if operational is not None and not operational.empty:
            tail = operational.tail(12).mean(numeric_only=True)
            self._operational_defaults = {
                name: float(tail.get(name, 0.0)) for name in OPERATIONAL_FEATURES
            }
        else:
            self._operational_defaults = {name: 0.0 for name in OPERATIONAL_FEATURES}

        matrix = features.to_numpy(dtype=float)
        if self.scale_features:
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            matrix = self._scaler.fit_transform(matrix)

        self.estimator.fit(matrix, targets)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        assert self._history is not None and self._origin is not None

        working = self._history.copy()
        predictions: List[float] = []

        for _ in range(horizon):
            next_date = working.index[-1] + pd.DateOffset(months=1)
            row = compute_feature_row(
                working, next_date, self._origin, self._operational_defaults
            )
            frame = pd.DataFrame([row])[self._feature_names].fillna(0.0)

            matrix = frame.to_numpy(dtype=float)
            if self._scaler is not None:
                matrix = self._scaler.transform(matrix)

            value = float(self.estimator.predict(matrix)[0])
            value = max(value, 0.0)
            predictions.append(value)

            # Append the prediction so the next step has its lag features.
            working = pd.concat([working, pd.Series([value], index=[next_date])])

        return self._clip(np.array(predictions))

    def feature_importances(self) -> Optional[pd.DataFrame]:
        """Return per-feature importance or coefficient magnitude, if available.

        Returns:
            A DataFrame sorted by importance, or ``None`` for estimators that
            expose neither ``feature_importances_`` nor ``coef_``.
        """
        if not self._fitted:
            return None

        if hasattr(self.estimator, "feature_importances_"):
            values = np.asarray(self.estimator.feature_importances_, dtype=float)
        elif hasattr(self.estimator, "coef_"):
            values = np.abs(np.asarray(self.estimator.coef_, dtype=float)).ravel()
        else:
            return None

        if len(values) != len(self._feature_names):
            return None

        return (
            pd.DataFrame({"feature": self._feature_names, "importance": values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Tier 3: classical time series
# ---------------------------------------------------------------------------
class HoltWintersForecaster(BaseForecaster):
    """Holt-Winters triple exponential smoothing (level + trend + seasonality).

    Args:
        trend: ``"add"``, ``"mul"`` or ``None``.
        seasonal: ``"add"``, ``"mul"`` or ``None``.
        seasonal_periods: Length of the seasonal cycle in months.
    """

    name = "holt_winters"
    family = "timeseries"
    explainability = 3

    def __init__(
        self, trend: Optional[str] = "add", seasonal: Optional[str] = "add",
        seasonal_periods: int = 12
    ) -> None:
        super().__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = int(seasonal_periods)
        self._model: Any = None

    def fit(self, history: pd.Series, panel: Optional[pd.DataFrame] = None) -> "HoltWintersForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._history = history.astype(float)
        series = self._history.copy()
        series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()

        seasonal = self.seasonal if len(series) >= 2 * self.seasonal_periods else None

        self._model = ExponentialSmoothing(
            series,
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=self.seasonal_periods if seasonal else None,
            initialization_method="estimated",
        ).fit(optimized=True)

        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        return self._clip(np.asarray(self._model.forecast(horizon), dtype=float))


class SarimaForecaster(BaseForecaster):
    """Seasonal ARIMA via ``statsmodels`` SARIMAX.

    Args:
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``.
    """

    name = "sarima"
    family = "timeseries"
    explainability = 3

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
    ) -> None:
        super().__init__()
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self._result: Any = None

    def fit(self, history: pd.Series, panel: Optional[pd.DataFrame] = None) -> "SarimaForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._history = history.astype(float)
        series = self._history.copy()
        series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()
        series = series.asfreq("MS")

        seasonal_order = self.seasonal_order
        # A seasonal difference needs at least two complete cycles.
        if len(series) < 2 * seasonal_order[3]:
            seasonal_order = (0, 0, 0, 0)

        self._result = SARIMAX(
            series,
            order=self.order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        return self._clip(np.asarray(self._result.forecast(steps=horizon), dtype=float))

    def prediction_interval(self, horizon: int, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Return model-native lower/upper prediction bounds.

        Args:
            horizon: Number of months ahead.
            alpha: Significance level; ``0.05`` gives a 95% interval.

        Returns:
            ``(lower, upper)`` arrays, each of length ``horizon``.
        """
        self._check_fitted()
        forecast = self._result.get_forecast(steps=horizon)
        interval = forecast.conf_int(alpha=alpha)
        return (
            self._clip(interval.iloc[:, 0].to_numpy()),
            self._clip(interval.iloc[:, 1].to_numpy()),
        )


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet, used only when the optional dependency is installed.

    Prophet is a natural fit for this problem (additive trend plus yearly
    seasonality on monthly data), but it drags in a cmdstan toolchain that
    frequently fails to build. The registry therefore probes for it and skips
    the candidate cleanly when it is absent, rather than making the whole suite
    un-runnable.
    """

    name = "prophet"
    family = "timeseries"
    explainability = 3

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    @staticmethod
    def is_available() -> bool:
        """Return ``True`` when ``prophet`` can be imported."""
        try:
            import prophet  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, history: pd.Series, panel: Optional[pd.DataFrame] = None) -> "ProphetForecaster":
        from prophet import Prophet

        self._history = history.astype(float)
        frame = pd.DataFrame({"ds": history.index, "y": history.to_numpy(dtype=float)})

        self._model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.95,
        )
        self._model.fit(frame)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._check_fitted()
        future = self._model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = self._model.predict(future)
        return self._clip(forecast["yhat"].tail(horizon).to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def build_model_registry(settings: Optional[Settings] = None) -> Dict[str, BaseForecaster]:
    """Instantiate every enabled candidate model from configuration.

    Unavailable optional dependencies (Prophet, XGBoost) are skipped with a
    warning rather than raising, so the suite always runs.

    Args:
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        Mapping of model name to an unfitted forecaster instance.
    """
    settings = settings or get_settings()
    candidates = settings.modeling.get("candidates", {}) or {}
    seasonal_period = int(settings.modeling.get("seasonal_period", 12))
    random_state = settings.random_state

    def enabled(key: str) -> bool:
        return bool((candidates.get(key) or {}).get("enabled", False))

    def params(key: str) -> Dict[str, Any]:
        cfg = dict(candidates.get(key) or {})
        cfg.pop("enabled", None)
        return cfg

    registry: Dict[str, BaseForecaster] = {}

    # --- Tier 1: baselines ---
    if enabled("naive_last"):
        registry["naive_last"] = NaiveLastForecaster()

    for key in ("moving_average_3", "moving_average_6"):
        if enabled(key):
            window = int(params(key).get("window", int(key.rsplit("_", 1)[1])))
            registry[key] = MovingAverageForecaster(window=window)

    if enabled("seasonal_naive"):
        registry["seasonal_naive"] = SeasonalNaiveForecaster(seasonal_period)

    if enabled("seasonal_naive_drift"):
        registry["seasonal_naive_drift"] = SeasonalNaiveDriftForecaster(seasonal_period)

    # --- Tier 2: linear ---
    if enabled("linear_regression"):
        from sklearn.linear_model import LinearRegression

        registry["linear_regression"] = SklearnRecursiveForecaster(
            "linear_regression", LinearRegression(), "linear", 2, scale_features=True
        )

    if enabled("ridge"):
        from sklearn.linear_model import Ridge

        registry["ridge"] = SklearnRecursiveForecaster(
            "ridge",
            Ridge(alpha=float(params("ridge").get("alpha", 1.0)), random_state=random_state),
            "linear",
            2,
            scale_features=True,
        )

    # --- Tier 4: tree ensembles ---
    if enabled("random_forest"):
        from sklearn.ensemble import RandomForestRegressor

        cfg = params("random_forest")
        registry["random_forest"] = SklearnRecursiveForecaster(
            "random_forest",
            RandomForestRegressor(
                n_estimators=int(cfg.get("n_estimators", 500)),
                max_depth=cfg.get("max_depth"),
                random_state=random_state,
                n_jobs=-1,
            ),
            "ensemble",
            4,
        )

    if enabled("gradient_boosting"):
        from sklearn.ensemble import GradientBoostingRegressor

        cfg = params("gradient_boosting")
        registry["gradient_boosting"] = SklearnRecursiveForecaster(
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=int(cfg.get("n_estimators", 400)),
                max_depth=int(cfg.get("max_depth", 3)),
                learning_rate=float(cfg.get("learning_rate", 0.05)),
                random_state=random_state,
            ),
            "ensemble",
            4,
        )

    if enabled("xgboost"):
        try:
            from xgboost import XGBRegressor

            cfg = params("xgboost")
            registry["xgboost"] = SklearnRecursiveForecaster(
                "xgboost",
                XGBRegressor(
                    n_estimators=int(cfg.get("n_estimators", 600)),
                    max_depth=int(cfg.get("max_depth", 3)),
                    learning_rate=float(cfg.get("learning_rate", 0.05)),
                    subsample=float(cfg.get("subsample", 0.9)),
                    random_state=random_state,
                    objective="reg:squarederror",
                    verbosity=0,
                    n_jobs=-1,
                ),
                "ensemble",
                4,
            )
        except ImportError:
            logger.warning("xgboost is not installed - skipping the XGBoost candidate")

    # --- Tier 3: classical time series ---
    if enabled("holt_winters"):
        cfg = params("holt_winters")
        registry["holt_winters"] = HoltWintersForecaster(
            trend=cfg.get("trend", "add"),
            seasonal=cfg.get("seasonal", "add"),
            seasonal_periods=seasonal_period,
        )

    if enabled("sarima"):
        cfg = params("sarima")
        registry["sarima"] = SarimaForecaster(
            order=tuple(cfg.get("order", (1, 1, 1))),
            seasonal_order=tuple(cfg.get("seasonal_order", (1, 1, 1, 12))),
        )

    if enabled("prophet"):
        if ProphetForecaster.is_available():
            registry["prophet"] = ProphetForecaster()
        else:
            logger.warning(
                "prophet is not installed - skipping the Prophet candidate "
                "(pip install prophet to enable it)"
            )

    logger.info("Model registry: %d candidate(s) -> %s", len(registry), sorted(registry))
    return registry


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Walk-forward evaluation output for the whole registry."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    residuals: Dict[str, np.ndarray] = field(default_factory=dict)


def rolling_origin_backtest(
    history: pd.Series,
    registry: Dict[str, BaseForecaster],
    settings: Optional[Settings] = None,
    panel: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Evaluate every model with expanding-window walk-forward validation.

    At each origin the model is refit on all data observed up to that point and
    asked for a ``horizon``-step forecast. This is strictly out-of-sample and is
    the only honest way to evaluate a forecaster.

    Args:
        history: Target series indexed by month-start, ascending.
        registry: Candidate models, from :func:`build_model_registry`.
        settings: Loaded settings. Loaded from disk when omitted.
        panel: Optional entity x product panel for operational features.

    Returns:
        A :class:`BacktestResult` with a per-model metrics table, the full
        prediction log and per-model residual arrays.

    Raises:
        ValueError: if the history is too short for the configured backtest.
    """
    settings = settings or get_settings()
    backtest_cfg = settings.modeling.get("backtest", {}) or {}

    initial = int(backtest_cfg.get("initial_train_months", 24))
    step = int(backtest_cfg.get("step", 1))
    horizon = int(backtest_cfg.get("horizon", 1))
    min_points = int(backtest_cfg.get("min_test_points", 12))

    origins = list(range(initial, len(history) - horizon + 1, step))
    if len(origins) < min_points:
        raise ValueError(
            f"Backtest needs at least {min_points} origins but only {len(origins)} are "
            f"available from {len(history)} months of history. Lower "
            "modeling.backtest.initial_train_months in config/config.yaml."
        )

    logger.info(
        "Backtesting %d model(s) over %d origin(s), %d-step-ahead",
        len(registry),
        len(origins),
        horizon,
    )

    prediction_rows: List[Dict[str, Any]] = []

    for name, model in registry.items():
        failures = 0

        for origin in origins:
            train = history.iloc[:origin]
            actuals = history.iloc[origin : origin + horizon]

            try:
                # A fresh instance per origin prevents state leaking between folds.
                fitted = _clone_forecaster(model).fit(train, panel)
                predicted = fitted.predict(horizon)
            except Exception as exc:
                failures += 1
                logger.debug("%s failed at origin %d: %s", name, origin, exc)
                continue

            for step_index, (date, actual) in enumerate(actuals.items()):
                prediction_rows.append(
                    {
                        "model": name,
                        "origin": int(origin),
                        "origin_date": history.index[origin - 1],
                        "date": date,
                        "step": step_index + 1,
                        "y_true": float(actual),
                        "y_pred": float(predicted[step_index]),
                    }
                )

        if failures:
            logger.warning("%s failed at %d/%d origin(s)", name, failures, len(origins))

    if not prediction_rows:
        raise ValueError("Backtest produced no predictions - every model failed")

    predictions = pd.DataFrame(prediction_rows)

    metric_rows: List[Dict[str, Any]] = []
    residuals: Dict[str, np.ndarray] = {}

    for name, group in predictions.groupby("model"):
        metrics = evaluate_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        model = registry[name]
        metric_rows.append(
            {
                "model": name,
                "family": model.family,
                "explainability": model.explainability,
                "origins_evaluated": int(group["origin"].nunique()),
                **metrics,
            }
        )
        residuals[name] = (group["y_true"] - group["y_pred"]).to_numpy(dtype=float)

    metrics_frame = (
        pd.DataFrame(metric_rows).sort_values("MAE").reset_index(drop=True)
    )

    return BacktestResult(metrics=metrics_frame, predictions=predictions, residuals=residuals)


def _clone_forecaster(model: BaseForecaster) -> BaseForecaster:
    """Return an unfitted copy of a forecaster.

    Args:
        model: A (possibly fitted) forecaster.

    Returns:
        A fresh instance with the same hyper-parameters and no fitted state.
    """
    import copy

    if isinstance(model, SklearnRecursiveForecaster):
        from sklearn.base import clone as sklearn_clone

        try:
            estimator = sklearn_clone(model.estimator)
        except Exception:
            estimator = copy.deepcopy(model.estimator)

        return SklearnRecursiveForecaster(
            name=model.name,
            estimator=estimator,
            family=model.family,
            explainability=model.explainability,
            scale_features=model.scale_features,
        )

    return copy.deepcopy(_reset_state(model))


def _reset_state(model: BaseForecaster) -> BaseForecaster:
    """Strip fitted state from a forecaster in place, returning it."""
    model._history = None
    model._fitted = False
    for attribute in ("_model", "_result"):
        if hasattr(model, attribute):
            setattr(model, attribute, None)
    return model


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def select_best_model(
    metrics: pd.DataFrame, settings: Optional[Settings] = None
) -> Tuple[str, pd.DataFrame]:
    """Pick the final model from accuracy, stability and explainability.

    Procedure:
      1. Rank by the primary metric (MAE by default).
      2. Every model within ``tolerance_pct`` of the best is considered tied on
         accuracy - a 2% MAE difference over 24 origins is noise, not signal.
      3. Among the tied set, score ``0.5 x accuracy + 0.3 x stability +
         0.2 x explainability`` on min-max normalised values (lower is better)
         and take the winner.

    Args:
        metrics: Per-model metrics table from the backtest.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        ``(best_model_name, scored_table)`` where the table carries the tie
        flags and composite scores for auditability.

    Raises:
        ValueError: if ``metrics`` is empty.
    """
    settings = settings or get_settings()
    selection = settings.modeling.get("selection", {}) or {}

    primary = str(selection.get("primary_metric", "MAE"))
    tolerance = float(selection.get("tolerance_pct", 5.0))
    stability_col = "error_std"

    if metrics.empty:
        raise ValueError("Cannot select a model from an empty metrics table")

    table = metrics.copy().sort_values(primary).reset_index(drop=True)

    best_score = float(table[primary].iloc[0])
    threshold = best_score * (1.0 + tolerance / 100.0)
    table["within_tolerance"] = table[primary] <= threshold

    explainability_cfg = selection.get("explainability_rank", {}) or {}
    table["explainability_rank"] = table.apply(
        lambda row: float(explainability_cfg.get(row["model"], row.get("explainability", 3))),
        axis=1,
    )

    def normalise(series: pd.Series) -> pd.Series:
        """Min-max normalise to [0, 1]; a constant column maps to all zeros."""
        span = series.max() - series.min()
        if span == 0 or not np.isfinite(span):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.min()) / span

    tied = table[table["within_tolerance"]].copy()

    # Accuracy is scored against the TOLERANCE BAND, not min-max normalised
    # within the tied set. Min-max would stretch whatever accuracy spread
    # happens to exist inside the band back out to the full [0, 1] range -
    # re-inflating a 2% MAE gap that was just declared negligible, and undoing
    # the purpose of having a tolerance at all. Scoring against the band means
    # a model at the band edge costs a full unit while one near the best costs
    # almost nothing.
    band_width = threshold - best_score
    if band_width > 0:
        tied["accuracy_norm"] = ((tied[primary] - best_score) / band_width).clip(0.0, 1.0)
    else:
        tied["accuracy_norm"] = 0.0

    # Stability and explainability are the actual discriminators inside the
    # band, so they keep full min-max spread.
    tied["stability_norm"] = normalise(tied[stability_col])
    tied["explainability_norm"] = normalise(tied["explainability_rank"])
    tied["composite_score"] = (
        0.5 * tied["accuracy_norm"]
        + 0.3 * tied["stability_norm"]
        + 0.2 * tied["explainability_norm"]
    )

    best = str(tied.sort_values("composite_score").iloc[0]["model"])

    table = table.merge(
        tied[["model", "accuracy_norm", "stability_norm", "explainability_norm", "composite_score"]],
        on="model",
        how="left",
    )
    table["selected"] = table["model"] == best

    logger.info(
        "Selected '%s' (%s=%.2f, %d model(s) within %.0f%% tolerance)",
        best,
        primary,
        float(table.loc[table["model"] == best, primary].iloc[0]),
        int(table["within_tolerance"].sum()),
        tolerance,
    )
    return best, table


# ---------------------------------------------------------------------------
# Forecasting with uncertainty
# ---------------------------------------------------------------------------
def forecast_with_intervals(
    model: BaseForecaster,
    horizon: int,
    residuals: Optional[np.ndarray] = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Produce point forecasts plus empirical prediction intervals.

    Intervals come from the model's own backtest residual distribution, which
    reflects the errors it actually made on this series - more honest than a
    parametric interval built on assumptions the residuals may not satisfy.
    Uncertainty is widened by ``sqrt(step)`` to reflect compounding error over
    the horizon (the standard random-walk scaling).

    Models exposing a native interval (SARIMA) are used directly when no
    residuals are supplied.

    Args:
        model: A fitted forecaster.
        horizon: Months to forecast.
        residuals: Backtest residuals (``y_true - y_pred``) for this model.
        confidence: Interval coverage, e.g. ``0.95``.

    Returns:
        A DataFrame with ``step``, ``forecast``, ``lower``, ``upper``.
    """
    point = np.asarray(model.predict(horizon), dtype=float)
    steps = np.arange(1, horizon + 1)

    if residuals is not None and len(residuals) >= 5:
        alpha = 1.0 - confidence
        lower_q = float(np.quantile(residuals, alpha / 2.0))
        upper_q = float(np.quantile(residuals, 1.0 - alpha / 2.0))
        widening = np.sqrt(steps)
        lower = point + lower_q * widening
        upper = point + upper_q * widening
    elif isinstance(model, SarimaForecaster):
        lower, upper = model.prediction_interval(horizon, alpha=1.0 - confidence)
    else:
        # Last resort: a flat +/-20% band, clearly wide enough to signal that
        # this is a fallback rather than a calibrated interval.
        lower, upper = point * 0.8, point * 1.2

    return pd.DataFrame(
        {
            "step": steps,
            "forecast": np.clip(point, 0, None),
            "lower": np.clip(lower, 0, None),
            "upper": np.clip(upper, 0, None),
        }
    )


# ---------------------------------------------------------------------------
# Activity levels
# ---------------------------------------------------------------------------
def assign_activity_levels(
    forecast_values: Sequence[float],
    history: pd.Series,
    settings: Optional[Settings] = None,
    method: Optional[str] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Classify forecast months as Low / Medium / High activity.

    Three methods are supported, because the naive one has a real failure mode:

    ``historical_quartiles``
        Literal reading of the brief - thresholds are the 25th/75th percentile
        of *all* historical monthly values. On a strongly growing series every
        future month clears the historical 75th percentile, so all twelve months
        come out "High" and the label carries no information.

    ``trend_adjusted`` (default)
        Classical **ratio-to-moving-average** decomposition. Each historical
        month is divided by a centred 12-month moving average, yielding a pure
        seasonal factor with the growth trend removed; each forecast month is
        divided by the forecast year's own mean, which is the same quantity for
        a full year. Quartiles of the historical seasonal factors then classify
        the forecast. This compares months on *seasonal* strength, which is what
        "high-activity month" operationally means. Recommended.

    ``forecast_quartiles``
        Quartiles of the twelve forecast values themselves - answers "which
        months of 2026 are the peaks" with no reference to history.

    Args:
        forecast_values: Predicted values, in chronological order.
        history: Historical target series indexed by month-start.
        settings: Loaded settings. Loaded from disk when omitted.
        method: Override for ``activity_levels.method``.

    Returns:
        ``(labels, thresholds)`` where ``labels`` aligns with ``forecast_values``
        and ``thresholds`` records the cut points actually used.
    """
    settings = settings or get_settings()
    cfg = settings.activity_levels
    method = method or str(cfg.get("method", "trend_adjusted"))

    low_q = float(cfg.get("low_quantile", 0.25))
    high_q = float(cfg.get("high_quantile", 0.75))
    labels_cfg = cfg.get("labels", {}) or {}
    low_label = str(labels_cfg.get("low", "Low"))
    medium_label = str(labels_cfg.get("medium", "Medium"))
    high_label = str(labels_cfg.get("high", "High"))

    values = np.asarray(forecast_values, dtype=float)
    historical = history.to_numpy(dtype=float)

    if method == "forecast_quartiles":
        comparison_basis = values
        scored = values
    elif method == "historical_quartiles":
        comparison_basis = historical
        scored = values
    else:  # trend_adjusted
        # Classical ratio-to-moving-average: divide each observation by a
        # CENTRED 12-month mean to strip the trend, leaving a seasonal factor.
        # min_periods=12 keeps partially-filled warm-up windows - whose ratios
        # collapse toward 1.0 and would compress the quartiles - out of the basis.
        level = history.rolling(12, center=True, min_periods=12).mean()
        ratios = (history / level).replace([np.inf, -np.inf], np.nan).dropna()

        if len(ratios) >= 8:
            comparison_basis = ratios.to_numpy(dtype=float)
            # For a full forecast year the centred 12-month mean of any month is
            # simply that year's mean, so the forecast ratio is value / year mean.
            forecast_level = float(np.mean(values))
        else:
            # Too little history for a centred window - fall back to the
            # trailing level, which is noisier but always defined.
            trailing = history.rolling(12, min_periods=1).mean().to_numpy(dtype=float)
            comparison_basis = np.divide(
                historical, trailing, out=np.ones_like(historical), where=trailing > 0
            )
            forecast_level = float(history.tail(12).mean())

        scored = values / forecast_level if forecast_level > 0 else values

    low_cut = float(np.quantile(comparison_basis, low_q))
    high_cut = float(np.quantile(comparison_basis, high_q))

    labels = [
        high_label if value >= high_cut else low_label if value <= low_cut else medium_label
        for value in scored
    ]

    thresholds = {
        "method": method,
        "low_threshold": low_cut,
        "high_threshold": high_cut,
        "low_quantile": low_q,
        "high_quantile": high_q,
    }
    return labels, thresholds
