"""Test suite for the exchange activity prediction pipeline.

Run with::

    pytest -q

The tests exercise the pieces most likely to break silently:

* **Parsing** - the tolerant aggregation walker, including nesting-order changes.
* **Validation** - that funnel and coverage violations are actually caught.
* **Feature engineering** - the no-leakage guarantee, which is the single most
  important correctness property in a forecasting pipeline.
* **Models** - that every forecaster honours the interface and produces sane
  output, and that the metrics are right on hand-checkable numbers.
* **Activity levels** - including the documented failure mode of the naive
  historical-quartile method on a growing series.
* **Charts** - that every figure serialises in both light and dark themes.

No test hits the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.api.aiesec_api import month_windows, redact_token
from src.config import get_settings
from src.insights import generate_all_insights
from src.models.forecasting import (
    MovingAverageForecaster,
    NaiveLastForecaster,
    SeasonalNaiveDriftForecaster,
    SeasonalNaiveForecaster,
    assign_activity_levels,
    build_model_registry,
    evaluate_metrics,
    rolling_origin_backtest,
    select_best_model,
)
from src.preprocessing.analysis import (
    funnel_analysis,
    monthly_contribution,
    seasonality_profile,
    yearly_summary,
)
from src.preprocessing.cleaning import parse_envelope, parse_payload, validate_dataset
from src.preprocessing.features import build_training_frame, compute_feature_row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def settings():
    """Project settings loaded once per module."""
    return get_settings()


@pytest.fixture(scope="module")
def panel(settings) -> pd.DataFrame:
    """A deterministic exchange panel built from the reference generator.

    Uses the real generator + real parser, so these tests double as an
    integration check of the offline path.
    """
    from src.api.reference_data import generate_reference_responses

    envelope = generate_reference_responses(settings, save=False)
    return parse_envelope(envelope, settings)


@pytest.fixture(scope="module")
def history(panel, settings) -> pd.Series:
    """MC-level monthly target series."""
    from src.models.train import build_history_series

    return build_history_series(panel, settings)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
class TestApiHelpers:
    def test_month_windows_covers_every_month_inclusively(self):
        windows = month_windows("2022-01-01", "2025-12-31")
        assert len(windows) == 48
        assert windows[0] == (pd.Timestamp("2022-01-01").date(), pd.Timestamp("2022-01-31").date())
        assert windows[-1] == (pd.Timestamp("2025-12-01").date(), pd.Timestamp("2025-12-31").date())

    def test_month_windows_clips_to_partial_range(self):
        windows = month_windows("2024-01-15", "2024-03-10")
        assert windows[0][0].day == 15   # start is not widened backwards
        assert windows[-1][1].day == 10  # end is not widened forwards
        assert len(windows) == 3

    def test_month_windows_rejects_reversed_range(self):
        with pytest.raises(ValueError):
            month_windows("2025-01-01", "2024-01-01")

    def test_redact_token_scrubs_secret_from_urls(self):
        url = "https://api.example.com/x?access_token=SUPERSECRET&page=1"
        assert "SUPERSECRET" not in redact_token(url, "SUPERSECRET")
        assert "***REDACTED***" in redact_token(url, "SUPERSECRET")

    def test_redact_token_is_noop_without_a_token(self):
        assert redact_token("plain text", None) == "plain text"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
class TestParsing:
    STATUS_MAP = {
        "applied": "APP", "achieved": "ACH", "accepted": "ACC",
        "approved": "APD", "realized": "RE", "finished": "FI", "completed": "CO",
    }

    def test_parses_nested_office_direction_status(self):
        payload = {
            "analytics": {
                "offices": {
                    "buckets": [
                        {
                            "key": 1, "key_as_string": "AIESEC in Testville",
                            "directions": {
                                "buckets": [
                                    {
                                        "key": "outgoing",
                                        "statuses": {
                                            "buckets": [
                                                {"key": "applied", "doc_count": 100},
                                                {"key": "realized", "doc_count": 15},
                                            ]
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        }
        result = parse_payload(payload, self.STATUS_MAP)
        assert result[("AIESEC in Testville", None, "outgoing")] == {"APP": 100, "RE": 15}

    def test_parser_is_tolerant_of_reversed_nesting(self):
        """Direction above office must parse to the same cell as office above direction."""
        payload = {
            "directions": {
                "buckets": [
                    {
                        "key": "incoming",
                        "offices": {
                            "buckets": [
                                {
                                    "key": 7, "key_as_string": "AIESEC in Elsewhere",
                                    "statuses": {"buckets": [{"key": "applied", "doc_count": 42}]},
                                }
                            ]
                        },
                    }
                ]
            }
        }
        result = parse_payload(payload, self.STATUS_MAP)
        assert result[("AIESEC in Elsewhere", None, "incoming")] == {"APP": 42}

    def test_direction_aliases_normalise(self):
        for alias in ("og", "OUT", "Sending"):
            payload = {
                "directions": {
                    "buckets": [
                        {
                            "key": alias,
                            "statuses": {"buckets": [{"key": "applied", "doc_count": 5}]},
                        }
                    ]
                }
            }
            result = parse_payload(payload, self.STATUS_MAP)
            assert list(result)[0][2] == "outgoing", f"alias {alias!r} did not normalise"

    def test_unknown_statuses_are_ignored_not_crashed_on(self):
        payload = {
            "statuses": {
                "buckets": [
                    {"key": "applied", "doc_count": 10},
                    {"key": "some_future_status", "doc_count": 99},
                ]
            }
        }
        result = parse_payload(payload, self.STATUS_MAP)
        assert result[(None, None, None)] == {"APP": 10}

    def test_empty_payload_yields_nothing(self):
        assert parse_payload({}, self.STATUS_MAP) == {}

    def test_parse_envelope_rejects_empty_records(self, settings):
        with pytest.raises(ValueError, match="no records"):
            parse_envelope({"records": []}, settings)


# ---------------------------------------------------------------------------
# Dataset shape and validation
# ---------------------------------------------------------------------------
class TestDataset:
    def test_panel_has_expected_shape(self, panel, settings):
        assert len(panel) == 48 * len(settings.entities) * 6  # months x LCs x programmes
        assert panel["date"].dt.to_period("M").nunique() == 48
        for stage in settings.funnel_stages:
            assert stage in panel.columns

    def test_funnel_is_monotonically_non_increasing(self, panel, settings):
        stages = settings.funnel_stages
        for earlier, later in zip(stages, stages[1:]):
            assert (panel[later] <= panel[earlier]).all(), f"{later} exceeds {earlier}"

    def test_no_negative_counts(self, panel, settings):
        for stage in settings.funnel_stages:
            assert (panel[stage] >= 0).all()

    def test_validation_passes_on_good_data(self, panel, settings):
        assert validate_dataset(panel, settings).passed

    def test_validation_catches_funnel_violation(self, panel, settings):
        broken = panel.copy()
        broken.loc[0, "CO"] = broken.loc[0, "APP"] + 1000
        report = validate_dataset(broken, settings)
        assert not report.passed
        assert any("monotonicity" in error for error in report.errors)

    def test_validation_catches_missing_months(self, panel, settings):
        gapped = panel[panel["date"] != panel["date"].max()]
        report = validate_dataset(gapped, settings)
        assert not report.passed
        assert any("missing" in error for error in report.errors)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
class TestFeatures:
    def test_no_target_leakage(self):
        """Features for month t must not change when the value AT t changes.

        This is the property that makes the backtest meaningful. If it fails,
        every reported metric is optimistic and the model is useless in production.
        """
        index = pd.date_range("2022-01-01", periods=30, freq="MS")
        series = pd.Series(np.arange(30, dtype=float) * 10 + 100, index=index)
        target_date = index[24]

        baseline = compute_feature_row(series, target_date, index[0])

        tampered = series.copy()
        tampered.loc[target_date] = 999_999.0
        after = compute_feature_row(tampered, target_date, index[0])

        assert baseline == after, "feature row changed when only the target changed"

    def test_future_values_do_not_leak_either(self):
        index = pd.date_range("2022-01-01", periods=30, freq="MS")
        series = pd.Series(np.arange(30, dtype=float) * 10 + 100, index=index)
        target_date = index[20]

        baseline = compute_feature_row(series, target_date, index[0])

        tampered = series.copy()
        tampered.iloc[21:] = 999_999.0
        after = compute_feature_row(tampered, target_date, index[0])

        assert baseline == after, "future observations leaked into the feature row"

    def test_lag_features_take_the_correct_values(self):
        index = pd.date_range("2022-01-01", periods=24, freq="MS")
        series = pd.Series(np.arange(24, dtype=float), index=index)
        target_date = index[13]  # value 13

        row = compute_feature_row(series, target_date, index[0])
        assert row["lag_1"] == 12.0
        assert row["lag_12"] == 1.0
        assert row["month_index"] == 13.0

    def test_seasonal_flags_match_the_calendar(self):
        index = pd.date_range("2022-01-01", periods=30, freq="MS")
        series = pd.Series(np.ones(30), index=index)

        december = compute_feature_row(series, pd.Timestamp("2023-12-01"), index[0])
        assert december["is_break_season"] == 1.0
        assert december["is_year_end"] == 1.0

        march = compute_feature_row(series, pd.Timestamp("2023-03-01"), index[0])
        assert march["is_exam_season"] == 1.0
        assert march["is_break_season"] == 0.0

    def test_training_frame_has_no_nans(self, panel, settings):
        frame = build_training_frame(panel, settings, save=False)
        assert not frame.isna().any().any()
        assert "y" in frame.columns
        assert len(frame) == 36  # 48 months minus the 12-month warm-up


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_perfect_forecast_scores_zero(self):
        metrics = evaluate_metrics([100.0, 200.0], [100.0, 200.0])
        assert metrics["MAE"] == 0.0
        assert metrics["RMSE"] == 0.0
        assert metrics["MAPE"] == 0.0

    def test_metrics_match_hand_computed_values(self):
        metrics = evaluate_metrics([100.0, 200.0], [110.0, 180.0])
        assert metrics["MAE"] == pytest.approx(15.0)             # (10 + 20) / 2
        assert metrics["RMSE"] == pytest.approx(np.sqrt(250.0))  # sqrt((100+400)/2)
        assert metrics["MAPE"] == pytest.approx(10.0)            # (10% + 10%) / 2
        assert metrics["bias"] == pytest.approx(5.0)             # (-10 + 20) / 2

    def test_mape_skips_zero_actuals_instead_of_returning_inf(self):
        metrics = evaluate_metrics([0.0, 100.0], [10.0, 110.0])
        assert np.isfinite(metrics["MAPE"])
        assert metrics["MAPE"] == pytest.approx(10.0)

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            evaluate_metrics([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
class TestForecasters:
    @staticmethod
    def _series() -> pd.Series:
        index = pd.date_range("2022-01-01", periods=36, freq="MS")
        seasonal = 100 * np.sin(2 * np.pi * np.arange(36) / 12)
        trend = np.arange(36, dtype=float) * 5
        return pd.Series(1000 + trend + seasonal, index=index)

    def test_naive_last_repeats_the_final_value(self):
        series = self._series()
        predictions = NaiveLastForecaster().fit(series).predict(6)
        assert np.allclose(predictions, series.iloc[-1])

    def test_moving_average_uses_the_configured_window(self):
        series = self._series()
        predictions = MovingAverageForecaster(window=3).fit(series).predict(4)
        assert np.allclose(predictions, series.tail(3).mean())

    def test_seasonal_naive_repeats_last_year(self):
        series = self._series()
        predictions = SeasonalNaiveForecaster(12).fit(series).predict(12)
        assert np.allclose(predictions, series.tail(12).to_numpy())

    def test_seasonal_naive_drift_scales_by_growth(self):
        series = self._series()
        plain = SeasonalNaiveForecaster(12).fit(series).predict(12)
        drifted = SeasonalNaiveDriftForecaster(12).fit(series).predict(12)
        # The series trends upward, so the drifted variant must sit above it.
        assert drifted.sum() > plain.sum()

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            NaiveLastForecaster().predict(3)

    def test_all_registry_models_produce_valid_output(self, history, panel, settings):
        registry = build_model_registry(settings)
        assert len(registry) >= 8, "expected a broad candidate suite"

        for name, model in registry.items():
            model.fit(history, panel)
            predictions = model.predict(12)

            assert len(predictions) == 12, f"{name} returned the wrong horizon"
            assert np.all(np.isfinite(predictions)), f"{name} produced non-finite values"
            assert np.all(predictions >= 0), f"{name} produced negative applications"


# ---------------------------------------------------------------------------
# Backtesting and selection
# ---------------------------------------------------------------------------
class TestBacktest:
    def test_backtest_is_out_of_sample_and_complete(self, history, panel, settings):
        registry = {"seasonal_naive": SeasonalNaiveForecaster(12)}
        result = rolling_origin_backtest(history, registry, settings, panel)

        assert len(result.metrics) == 1
        assert result.metrics.iloc[0]["origins_evaluated"] == 24

        # Every backtested date must be strictly after the origin it was predicted from.
        assert (result.predictions["date"] > result.predictions["origin_date"]).all()

    def test_selection_prefers_accuracy_then_explainability(self, settings):
        metrics = pd.DataFrame(
            [
                # Black box, marginally best MAE.
                {"model": "xgboost", "family": "ensemble", "explainability": 4,
                 "MAE": 100.0, "RMSE": 120.0, "MAPE": 8.0, "error_std": 50.0,
                 "max_abs_error": 200.0, "bias": 0.0, "n": 24, "origins_evaluated": 24},
                # Transparent, within the 5% tolerance band and more stable.
                {"model": "seasonal_naive", "family": "baseline", "explainability": 1,
                 "MAE": 102.0, "RMSE": 121.0, "MAPE": 8.1, "error_std": 40.0,
                 "max_abs_error": 190.0, "bias": 0.0, "n": 24, "origins_evaluated": 24},
            ]
        )
        best, table = select_best_model(metrics, settings)

        assert best == "seasonal_naive", "tie-break should favour the explainable, stable model"
        assert bool(table.loc[table["model"] == best, "selected"].iloc[0])
        assert int(table["within_tolerance"].sum()) == 2

    def test_clear_accuracy_winner_is_not_overridden(self, settings):
        metrics = pd.DataFrame(
            [
                {"model": "xgboost", "family": "ensemble", "explainability": 4,
                 "MAE": 50.0, "RMSE": 60.0, "MAPE": 4.0, "error_std": 30.0,
                 "max_abs_error": 100.0, "bias": 0.0, "n": 24, "origins_evaluated": 24},
                {"model": "naive_last", "family": "baseline", "explainability": 1,
                 "MAE": 200.0, "RMSE": 240.0, "MAPE": 16.0, "error_std": 20.0,
                 "max_abs_error": 300.0, "bias": 0.0, "n": 24, "origins_evaluated": 24},
            ]
        )
        best, _ = select_best_model(metrics, settings)
        assert best == "xgboost", "a 4x MAE gap must not be overridden by explainability"

    def test_backtest_rejects_too_short_history(self, settings):
        short = pd.Series(
            np.arange(20, dtype=float), index=pd.date_range("2024-01-01", periods=20, freq="MS")
        )
        with pytest.raises(ValueError, match="at least"):
            rolling_origin_backtest(short, {"naive": NaiveLastForecaster()}, settings)


# ---------------------------------------------------------------------------
# Activity levels
# ---------------------------------------------------------------------------
class TestActivityLevels:
    def test_trend_adjusted_labels_are_all_three_and_seasonal(self, history, settings):
        # A clearly seasonal forecast year: peaks in Jun and Dec, trough in Feb.
        values = [1600, 1150, 1300, 1400, 1750, 1850, 1550, 1400, 1420, 1520, 1600, 2000]
        labels, thresholds = assign_activity_levels(values, history, settings, "trend_adjusted")

        assert len(labels) == 12
        assert set(labels) <= {"Low", "Medium", "High"}
        assert labels[11] == "High", "December is the strongest month and must be High"
        assert labels[1] == "Low", "February is the weakest month and must be Low"
        assert thresholds["method"] == "trend_adjusted"

    def test_historical_quartiles_degenerate_on_a_growing_series(self, history, settings):
        """The documented failure mode of the naive method, pinned by a test.

        On a strongly growing series every forecast month clears the historical
        75th percentile, so the label collapses to a constant and carries no
        information. This is exactly why trend_adjusted is the default.
        """
        values = [float(history.max()) * 1.3] * 12
        labels, _ = assign_activity_levels(values, history, settings, "historical_quartiles")
        assert set(labels) == {"High"}

    def test_forecast_quartiles_always_span_the_range(self, history, settings):
        values = list(np.linspace(1000, 2000, 12))
        labels, _ = assign_activity_levels(values, history, settings, "forecast_quartiles")
        assert labels[0] == "Low"
        assert labels[-1] == "High"

    def test_levels_align_with_the_input_length(self, history, settings):
        for size in (3, 6, 12):
            labels, _ = assign_activity_levels([1500.0] * size, history, settings)
            assert len(labels) == size


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
class TestAnalysis:
    def test_funnel_conversions_are_internally_consistent(self, panel, settings):
        funnel = funnel_analysis(panel, settings)

        assert list(funnel["stage"]) == settings.funnel_stages
        assert funnel.iloc[0]["conversion_from_previous_pct"] == 100.0
        # Counts must decrease monotonically down the funnel.
        assert funnel["count"].is_monotonic_decreasing

        # conversion_from_APP must equal count / APP.
        applied = funnel.iloc[0]["count"]
        for row in funnel.itertuples():
            assert row.conversion_from_APP_pct == pytest.approx(row.count / applied * 100, abs=0.01)

    def test_exactly_one_stage_is_flagged_as_the_biggest_dropoff(self, panel, settings):
        funnel = funnel_analysis(panel, settings)
        assert int(funnel["is_biggest_dropoff"].sum()) == 1

    def test_seasonality_index_averages_to_one(self, panel, settings):
        profile = seasonality_profile(panel, settings)
        assert len(profile) == 12
        assert profile["seasonal_index"].mean() == pytest.approx(1.0, abs=0.01)

    def test_monthly_contributions_sum_to_100_per_year(self, panel, settings):
        contribution = monthly_contribution(panel, settings)
        totals = contribution.groupby("year")["contribution_pct"].sum()
        for total in totals:
            assert total == pytest.approx(100.0, abs=0.1)

    def test_yearly_summary_covers_every_year(self, panel, settings):
        summary = yearly_summary(panel, settings)
        assert list(summary["year"]) == [2022, 2023, 2024, 2025]
        assert (summary["months_observed"] == 12).all()
        assert pd.isna(summary.iloc[0]["yoy_growth_pct"])  # no prior year to compare


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------
class TestInsights:
    def test_insights_generate_from_a_full_analysis(self, panel, settings):
        from src.preprocessing.analysis import run_full_analysis

        analysis = run_full_analysis(panel, settings, save=False)
        predictions = pd.DataFrame(
            {
                "Month": [f"{m} 2026" for m in
                          ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]],
                "Predicted Applications": [1600, 1150, 1300, 1400, 1750, 1850,
                                           1550, 1400, 1420, 1520, 1600, 2000],
                "Activity Level": ["Medium", "Low", "Low", "Medium", "High", "High",
                                   "Medium", "Medium", "Medium", "Medium", "Medium", "High"],
                "share_of_year_pct": [8.5] * 12,
                "lower_95": [1000] * 12,
                "upper_95": [2200] * 12,
            }
        )

        insights = generate_all_insights(analysis, predictions, None, "ridge", settings)

        assert len(insights) >= 8
        categories = {insight.category for insight in insights}
        assert {"Forecast", "Growth", "Seasonality", "Funnel"} <= categories
        # Every insight must carry a real sentence.
        for insight in insights:
            assert insight.headline and insight.headline.endswith(".")

    def test_insights_degrade_gracefully_without_inputs(self, settings):
        assert generate_all_insights({}, None, None, "", settings) == []


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
class TestCharts:
    def test_every_chart_renders_in_both_themes(self, panel, settings):
        """Serialisation catches the invalid-property and datetime-axis bugs."""
        from src.preprocessing.analysis import run_full_analysis
        from src.visualization import charts

        analysis = run_full_analysis(panel, settings, save=False)
        predictions = pd.DataFrame(
            {
                "Month": ["Jan 2026", "Feb 2026", "Mar 2026"],
                "Predicted Applications": [1600, 1150, 1300],
                "Activity Level": ["Medium", "Low", "High"],
                "date": ["2026-01-01", "2026-02-01", "2026-03-01"],
                "lower_95": [1400, 950, 1100],
                "upper_95": [1800, 1350, 1500],
            }
        )

        for dark in (False, True):
            figures = [
                charts.chart_monthly_trend(analysis["monthly_trend"], dark),
                charts.chart_year_comparison(analysis["monthly_contribution"], dark),
                charts.chart_seasonality(analysis["seasonality_profile"], dark),
                charts.chart_monthly_contribution_heatmap(analysis["monthly_contribution"], dark),
                charts.chart_funnel(analysis["funnel_overall"], dark),
                charts.chart_entity_performance(analysis["entity_performance"], dark),
                charts.chart_product_mix(analysis["product_performance"], dark),
                charts.chart_forecast(analysis["monthly_trend"], predictions, dark),
                charts.chart_activity_levels(predictions, dark),
            ]
            for figure in figures:
                assert figure.to_json(), "figure failed to serialise"

    def test_ink_contrast_flips_with_background_lightness(self):
        from src.visualization.charts import _ink_on

        assert _ink_on("#b7d3f6") == "#0b0b0b"  # light fill -> dark ink
        assert _ink_on("#0d366b") == "#ffffff"  # dark fill -> light ink
