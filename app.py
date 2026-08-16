"""Streamlit dashboard for the AIESEC MC India exchange activity platform.

Run with::

    streamlit run app.py

Four pages, matching the analytical narrative:

1. **Overview** - headline KPIs, growth trend, predicted peak month.
2. **Historical Analysis** - monthly trend, year comparison, seasonality,
   funnel and entity/product performance.
3. **2026 Prediction** - forecast chart with uncertainty, high-activity months,
   full prediction table and model diagnostics.
4. **Insights** - automatically generated findings and recommendations.

Every artefact is loaded from disk and cached. If the pipeline has not been run,
the app says so and shows the exact command instead of failing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Ensure the project root is importable when Streamlit launches from elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import configure_logging, get_settings  # noqa: E402
from src.insights import Insight, generate_all_insights  # noqa: E402
from src.preprocessing.analysis import run_full_analysis  # noqa: E402
from src.visualization import charts  # noqa: E402

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AIESEC MC India - Exchange Activity Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ANALYSIS_TABLES = (
    "monthly_trend", "yearly_summary", "seasonality_profile", "peak_months",
    "funnel_overall", "funnel_by_product", "funnel_by_year",
    "entity_performance", "product_performance", "monthly_contribution",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_everything() -> Dict[str, Any]:
    """Load every artefact the dashboard needs.

    Returns:
        A dict with ``panel``, ``analysis``, ``predictions``, ``evaluation``,
        ``backtest``, ``importances``, ``metadata`` and ``is_reference_data``.
        Missing optional artefacts are ``None`` rather than an error.
    """
    settings = get_settings()
    configure_logging(settings)

    bundle: Dict[str, Any] = {
        "settings": settings,
        "panel": None,
        "analysis": {},
        "predictions": None,
        "evaluation": None,
        "backtest": None,
        "importances": None,
        "metadata": {},
        "is_reference_data": False,
        "errors": [],
    }

    # --- processed panel ---
    panel_path = settings.paths.processed_dataset
    if not panel_path.exists():
        bundle["errors"].append(f"Processed dataset not found at {panel_path}")
        return bundle

    panel = pd.read_csv(panel_path, parse_dates=["date"])
    bundle["panel"] = panel

    # --- provenance ---
    # Prefer the raw envelope; fall back to the committed provenance mirror so a
    # deployed dashboard still reports the source (data/raw/ is git-ignored).
    import json

    raw_path = settings.paths.raw_responses
    provenance_path = settings.paths.data_processed / "provenance.json"

    for path, extract in (
        (raw_path, lambda blob: blob.get("metadata", {})),
        (provenance_path, lambda blob: blob),
    ):
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                metadata = extract(json.load(handle))
            bundle["metadata"] = metadata
            bundle["is_reference_data"] = bool(metadata.get("is_reference_data"))
            break
        except Exception as exc:  # provenance is informational, never fatal
            logger.debug("Could not read provenance from %s: %s", path, exc)

    # --- analysis tables: read from disk, recompute if absent ---
    reports = settings.paths.reports_dir
    analysis: Dict[str, pd.DataFrame] = {}
    for name in ANALYSIS_TABLES:
        path = reports / f"{name}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"])
            analysis[name] = frame

    if len(analysis) < len(ANALYSIS_TABLES):
        analysis = run_full_analysis(panel, settings, save=False)
    bundle["analysis"] = analysis

    # --- predictions and model diagnostics ---
    if settings.paths.predictions.exists():
        bundle["predictions"] = pd.read_csv(settings.paths.predictions)

    for key, filename in (
        ("evaluation", "model_evaluation.csv"),
        ("backtest", "backtest_predictions.csv"),
        ("importances", "feature_importance.csv"),
    ):
        path = reports / filename
        if path.exists():
            bundle[key] = pd.read_csv(path)

    return bundle


@st.cache_data(show_spinner=False)
def compute_insights(
    _analysis: Dict[str, pd.DataFrame],
    _predictions: Optional[pd.DataFrame],
    _evaluation: Optional[pd.DataFrame],
    model_name: str,
) -> List[Insight]:
    """Generate insights, cached on the model name.

    Args:
        _analysis: Analysis tables (leading underscore keeps Streamlit from
            trying to hash the DataFrames).
        _predictions: Prediction table.
        _evaluation: Model evaluation table.
        model_name: Selected model, which is the cache key.

    Returns:
        The generated insights.
    """
    return generate_all_insights(
        analysis=_analysis,
        predictions=_predictions,
        evaluation=_evaluation,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------
def render_metric(
    label: str,
    value: str,
    delta: Optional[str] = None,
    help_text: str = "",
    delta_color: str = "normal",
) -> None:
    """Render one KPI tile.

    Args:
        label: Tile caption.
        value: The headline figure.
        delta: Optional secondary line beneath the value.
        help_text: Tooltip content.
        delta_color: ``"normal"`` when the delta really is a change (green up /
            red down), or ``"off"`` when it is just a supporting figure - a
            green up-arrow beside "quietest month" would assert growth that the
            number does not mean.
    """
    st.metric(
        label=label, value=value, delta=delta, help=help_text or None,
        delta_color=delta_color,
    )


def dark_mode() -> bool:
    """Detect whether Streamlit is rendering on a dark surface.

    ``st.context.theme.type`` reports the *active* theme, including the case
    where dark mode comes from the browser/OS preference rather than from
    ``config.toml``. ``st.get_option("theme.base")`` only sees the configured
    value and reports light in that case, so it is used purely as a fallback.

    Returns:
        ``True`` when the active theme is dark.
    """
    try:
        theme_type = getattr(st.context.theme, "type", None)
        if theme_type:
            return str(theme_type).lower() == "dark"
    except Exception:  # older Streamlit, or no script run context
        pass

    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return False


def data_source_label(bundle: Dict[str, Any]) -> str:
    """Return a short label naming where the loaded data came from.

    Data provenance is documented in full in the README. In the dashboard it is
    surfaced as a single sidebar line rather than a page banner, so it stays
    visible without dominating every screen.

    Args:
        bundle: The loaded artefact bundle.

    Returns:
        A one-line provenance description.
    """
    if bundle.get("is_reference_data"):
        return "Reference dataset (offline)"
    return "AIESEC Analytics API"


def show_table(frame: pd.DataFrame, caption: str = "", height: Optional[int] = None) -> None:
    """Render a DataFrame as the accessible table view behind a chart.

    Args:
        frame: Data to display.
        caption: Optional source note rendered beneath the table.
        height: Fixed pixel height. Omitted entirely when ``None``, since
            Streamlit rejects an explicit ``height=None``.
    """
    kwargs: Dict[str, Any] = {"width": "stretch", "hide_index": True}
    if height is not None:
        kwargs["height"] = height

    st.dataframe(frame, **kwargs)
    if caption:
        st.caption(caption)


# ---------------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------------
def page_overview(bundle: Dict[str, Any]) -> None:
    """Headline KPIs, the growth trend, and the predicted peak month."""
    settings = bundle["settings"]
    analysis = bundle["analysis"]
    predictions = bundle["predictions"]
    target = settings.target

    st.title("Exchange Activity Overview")
    st.caption(f"{settings.mc_name} · exchange applications, 2022-2025 · 2026 forecast")

    trend = analysis["monthly_trend"]
    yearly = analysis["yearly_summary"]

    total = int(trend[target].sum())
    average = float(trend[target].mean())
    latest_year = yearly.iloc[-1]
    latest_growth = float(latest_year["yoy_growth_pct"])

    columns = st.columns(4)
    with columns[0]:
        render_metric(
            "Total applications (2022-2025)", f"{total:,}",
            help_text="Sum of monthly applications across the full history",
        )
    with columns[1]:
        render_metric(
            "Average monthly activity", f"{average:,.0f}",
            help_text="Mean applications per month across 48 months",
        )
    with columns[2]:
        render_metric(
            f"{int(latest_year['year'])} growth", f"{latest_growth:+.1f}%",
            delta=f"{latest_growth:+.1f}% vs prior year",
            help_text="Year-over-year change in total applications",
        )
    with columns[3]:
        if predictions is not None:
            peak = predictions.loc[predictions["Predicted Applications"].idxmax()]
            render_metric(
                "Predicted 2026 peak month", str(peak["Month"]),
                delta=f"{int(peak['Predicted Applications']):,} applications",
                help_text="Month with the highest forecast application volume",
                delta_color="off",
            )
        else:
            render_metric("Predicted 2026 peak month", "—", help_text="Run the training step")

    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            charts.chart_monthly_trend(trend, dark_mode(), target=target),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            charts.chart_seasonality(analysis["seasonality_profile"], dark_mode()),
            width="stretch",
        )

    st.subheader("Year-by-year summary")
    display = yearly.copy()
    display.columns = [c.replace("_", " ").title() for c in display.columns]
    show_table(display, "Source: outputs/reports/yearly_summary.csv")

    if predictions is not None:
        st.subheader("2026 at a glance")
        forecast_total = int(predictions["Predicted Applications"].sum())
        last_total = float(latest_year["total_applications"])
        change = (forecast_total - last_total) / last_total * 100
        high_months = predictions[predictions["Activity Level"] == "High"]["Month"].tolist()

        summary = st.columns(3)
        with summary[0]:
            render_metric("Forecast total 2026", f"{forecast_total:,}", delta=f"{change:+.1f}% vs {int(latest_year['year'])}")
        with summary[1]:
            render_metric("High-activity months", str(len(high_months)),
                          help_text=", ".join(high_months) if high_months else "None")
        with summary[2]:
            render_metric("Model", str(predictions["model"].iloc[0]),
                          help_text="Chosen by walk-forward validation")


# ---------------------------------------------------------------------------
# Page 2: Historical Analysis
# ---------------------------------------------------------------------------
def page_historical(bundle: Dict[str, Any]) -> None:
    """Monthly trend, year comparison, seasonality, funnel and entity views."""
    settings = bundle["settings"]
    analysis = bundle["analysis"]
    target = settings.target

    st.title("Historical Analysis")
    st.caption("Exchange activity trends, seasonality, funnel performance and entity contribution")

    tabs = st.tabs(["Trends", "Seasonality", "Exchange funnel", "Entities & products"])

    # --- Trends ---
    with tabs[0]:
        trend = analysis["monthly_trend"]
        st.plotly_chart(
            charts.chart_monthly_trend(trend, dark_mode(), target=target), width="stretch"
        )

        st.subheader("Year-on-year comparison")
        st.plotly_chart(
            charts.chart_year_comparison(analysis["monthly_contribution"], dark_mode(), target),
            width="stretch",
        )

        st.subheader("Busiest months on record")
        show_table(analysis["peak_months"], "Source: outputs/reports/peak_months.csv")

        with st.expander("Monthly detail table"):
            columns = ["period", target, "mom_growth_pct", "yoy_growth_pct",
                       "rolling_3m", "rolling_12m", "app_to_re_pct"]
            available = [c for c in columns if c in trend.columns]
            show_table(trend[available].round(2))

    # --- Seasonality ---
    with tabs[1]:
        st.plotly_chart(
            charts.chart_seasonality(analysis["seasonality_profile"], dark_mode()),
            width="stretch",
        )
        st.plotly_chart(
            charts.chart_monthly_contribution_heatmap(analysis["monthly_contribution"], dark_mode()),
            width="stretch",
        )
        st.caption(
            "The heatmap normalises each year to 100%, so the growth trend is removed "
            "and the seasonal shape can be compared directly across years."
        )
        show_table(analysis["seasonality_profile"], "Source: outputs/reports/seasonality_profile.csv")

    # --- Funnel ---
    with tabs[2]:
        funnel = analysis["funnel_overall"]
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(charts.chart_funnel(funnel, dark_mode()), width="stretch")
        with right:
            st.subheader("Conversion and drop-off")
            display = funnel[
                ["stage", "stage_label", "count", "conversion_from_previous_pct",
                 "conversion_from_APP_pct", "dropoff_count", "dropoff_pct"]
            ].copy()
            display.columns = ["Stage", "Label", "Count", "From previous %",
                               "From APP %", "Dropped", "Drop-off %"]
            show_table(display)

            worst = funnel.iloc[1:].loc[funnel.iloc[1:]["dropoff_count"].idxmax()]
            upstream = funnel.iloc[int(worst["stage_order"]) - 2]
            st.error(
                f"**Largest leak: {upstream['stage']} → {worst['stage']}.** "
                f"{int(worst['dropoff_count']):,} candidates "
                f"({worst['dropoff_pct']:.0f}% of {upstream['stage']}) are lost at this "
                "transition — the highest-leverage place to intervene.",
                icon="🚨",
            )

        st.subheader("Funnel efficiency by programme")
        show_table(analysis["funnel_by_product"].round(2))
        st.subheader("Funnel efficiency by year")
        show_table(analysis["funnel_by_year"].round(2))

    # --- Entities & products ---
    with tabs[3]:
        entities = analysis["entity_performance"]
        top_n = st.slider("Local Committees to display", 5, min(25, len(entities)), 12)

        st.plotly_chart(
            charts.chart_entity_performance(entities, dark_mode(), top_n, target),
            width="stretch",
        )
        show_table(entities.round(2), "Source: outputs/reports/entity_performance.csv")

        st.subheader("Programme performance")
        st.plotly_chart(
            charts.chart_product_mix(analysis["product_performance"], dark_mode(), target),
            width="stretch",
        )
        show_table(analysis["product_performance"].round(2))


# ---------------------------------------------------------------------------
# Page 3: 2026 Prediction
# ---------------------------------------------------------------------------
def page_prediction(bundle: Dict[str, Any]) -> None:
    """Forecast chart, high-activity months, prediction table and diagnostics."""
    settings = bundle["settings"]
    analysis = bundle["analysis"]
    predictions = bundle["predictions"]
    target = settings.target

    st.title(f"{settings.forecast_year} Forecast")

    if predictions is None:
        st.error(
            "No predictions found. Generate them with:\n\n"
            "```bash\npython run_pipeline.py --step train\n```"
        )
        return

    st.caption(
        f"Predicted monthly exchange applications for {settings.forecast_year}, "
        f"produced by the **{predictions['model'].iloc[0]}** model"
    )

    total = int(predictions["Predicted Applications"].sum())
    peak = predictions.loc[predictions["Predicted Applications"].idxmax()]
    trough = predictions.loc[predictions["Predicted Applications"].idxmin()]
    high_months = predictions[predictions["Activity Level"] == "High"]

    columns = st.columns(4)
    with columns[0]:
        render_metric("Total forecast applications", f"{total:,}")
    with columns[1]:
        render_metric("Peak month", str(peak["Month"]),
                      delta=f"{int(peak['Predicted Applications']):,} applications",
                      delta_color="off")
    with columns[2]:
        render_metric("Quietest month", str(trough["Month"]),
                      delta=f"{int(trough['Predicted Applications']):,} applications",
                      delta_color="off")
    with columns[3]:
        render_metric("High-activity months", str(len(high_months)))

    st.divider()

    trend = analysis["monthly_trend"]
    st.plotly_chart(
        charts.chart_forecast(trend, predictions, dark_mode(), target), width="stretch"
    )
    st.caption(
        "The shaded band is a 95% prediction interval derived from the model's own "
        "walk-forward residuals, widened by sqrt(h) as the horizon extends. Plan "
        "against the band, not the central line."
    )

    st.subheader("High-activity months")
    if high_months.empty:
        st.info("No month is classified as high-activity in the forecast year.")
    else:
        st.success(
            "**" + ", ".join(high_months["Month"].tolist()) + "** are predicted to be "
            f"high-activity, carrying {high_months['share_of_year_pct'].sum():.0f}% of "
            "the year's volume between them.",
            icon="📈",
        )

    st.plotly_chart(charts.chart_activity_levels(predictions, dark_mode()), width="stretch")

    st.subheader("Prediction table")
    core = predictions[["Month", "Predicted Applications", "Activity Level"]]
    detailed = predictions[
        ["Month", "Predicted Applications", "Activity Level", "lower_95", "upper_95",
         "rank_in_year", "share_of_year_pct", "activity_level_vs_history",
         "activity_level_within_2026"]
    ]

    view = st.radio("View", ["Deliverable format", "With detail"], horizontal=True)
    show_table(core if view == "Deliverable format" else detailed)

    st.download_button(
        "Download predictions_2026.csv",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="predictions_2026.csv",
        mime="text/csv",
    )

    st.caption(
        "Activity levels use a trend-adjusted classification: history is detrended by "
        "a centred 12-month moving average so months are compared on *seasonal* "
        "strength. High = top 25%, Medium = middle 50%, Low = bottom 25%. The two "
        "alternative classifications are shown in the detailed view."
    )

    # --- diagnostics ---
    st.divider()
    st.subheader("Model diagnostics")

    evaluation = bundle["evaluation"]
    backtest = bundle["backtest"]

    if evaluation is not None:
        metric = st.selectbox("Error metric", ["MAE", "RMSE", "MAPE", "sMAPE"], index=0)
        st.plotly_chart(
            charts.chart_model_comparison(evaluation, dark_mode(), metric), width="stretch"
        )
        display = evaluation[
            ["model", "family", "MAE", "RMSE", "MAPE", "error_std",
             "max_abs_error", "origins_evaluated", "selected"]
        ].round(3)
        show_table(display, "Source: outputs/reports/model_evaluation.csv")

    if backtest is not None:
        model_name = str(predictions["model"].iloc[0])
        st.plotly_chart(
            charts.chart_backtest(backtest, model_name, dark_mode()), width="stretch"
        )

    importances = bundle["importances"]
    if importances is not None and not importances.empty:
        st.plotly_chart(
            charts.chart_feature_importance(importances, dark_mode()), width="stretch"
        )


# ---------------------------------------------------------------------------
# Page 4: Insights
# ---------------------------------------------------------------------------
def page_insights(bundle: Dict[str, Any]) -> None:
    """Automatically generated findings, grouped by category."""
    st.title("Automated Insights")
    st.caption("Generated directly from the analysis and forecast - no hand-written numbers")

    predictions = bundle["predictions"]
    model_name = str(predictions["model"].iloc[0]) if predictions is not None else ""

    insights = compute_insights(
        bundle["analysis"], predictions, bundle["evaluation"], model_name
    )

    if not insights:
        st.info("No insights available yet. Run `python run_pipeline.py --step all`.")
        return

    icons = {"positive": "📈", "negative": "⚠️", "neutral": "📊"}

    categories: Dict[str, List[Insight]] = {}
    for insight in insights:
        categories.setdefault(insight.category, []).append(insight)

    for category, items in categories.items():
        st.subheader(category)
        for insight in items:
            icon = icons.get(insight.kind, "📊")
            with st.container(border=True):
                st.markdown(f"{icon}&nbsp;&nbsp;**{insight.headline}**")
                if insight.detail:
                    st.caption(insight.detail)

    st.divider()
    st.subheader("Recommended actions")
    st.markdown(
        """
These follow directly from the findings above:

1. **Stage capacity ahead of the predicted peaks.** Matching and reviewer bandwidth
   should be in place one month *before* each high-activity month, not during it.
2. **Attack the largest funnel leak first.** The biggest single drop-off is worth
   more than a uniform improvement everywhere else - a few points recovered at the
   worst transition compounds through every downstream stage.
3. **Use the quiet months for capability, not idling.** The lowest-activity months
   are the natural window for member training, partner development and process work.
4. **Watch entity concentration.** When a handful of LCs carry most of the volume,
   national performance inherits their risk; growing the mid-tier is a hedge.
5. **Plan against the interval, not the point estimate.** The prediction band is the
   honest planning range; the central line is only its midpoint.
        """
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the dashboard."""
    bundle = load_everything()

    with st.sidebar:
        st.title("AIESEC MC India")
        st.caption("Exchange Activity Prediction Platform")
        st.divider()

        page = st.radio(
            "Page",
            ["Overview", "Historical Analysis", "2026 Prediction", "Insights"],
            label_visibility="collapsed",
        )

        st.divider()
        metadata = bundle.get("metadata", {})
        st.markdown("**Data source**")
        st.caption(data_source_label(bundle))
        st.caption(f"Window: {metadata.get('collection_start', '?')} to {metadata.get('collection_end', '?')}")

        if bundle.get("predictions") is not None:
            st.markdown("**Model**")
            st.caption(str(bundle["predictions"]["model"].iloc[0]))

        st.divider()
        if st.button("Reload data", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    if bundle["errors"]:
        st.title("Pipeline artefacts not found")
        for error in bundle["errors"]:
            st.error(error)
        st.markdown(
            "Run the pipeline first:\n\n"
            "```bash\npython run_pipeline.py --step all --use-reference-data\n```"
        )
        return

    if page == "Overview":
        page_overview(bundle)
    elif page == "Historical Analysis":
        page_historical(bundle)
    elif page == "2026 Prediction":
        page_prediction(bundle)
    else:
        page_insights(bundle)


if __name__ == "__main__":
    main()
