"""Interactive Plotly charts for the exchange-activity dashboard.

Design rules applied throughout (see the project README's Visualization notes):

* **Colour by job.** Categorical hues in a *fixed* order for identity (products,
  entities); a single-hue *ordinal* ramp for anything ordered (funnel stages,
  Low/Medium/High activity). Rank never picks a colour - an entity keeps its hue
  when a filter changes the series count.
* **One axis, always.** No chart in this module has a secondary y-axis.
  Different scales get separate charts or a shared index.
* **Identity never rests on colour alone.** Every multi-series chart carries a
  legend, and the dashboard pairs each chart with the table behind it.
* **Recessive chrome.** Hairline horizontal gridlines only, muted axis ink,
  2px lines, 8px markers.
* **Hover by default.** Time series use a unified crosshair; categorical marks
  use per-mark tooltips.

Every function returns a ``plotly.graph_objects.Figure`` and takes a ``dark``
flag so the same code renders correctly on both surfaces - the dark values are
separately chosen steps, not an automatic inversion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

__all__ = [
    "theme",
    "CATEGORICAL_LIGHT",
    "CATEGORICAL_DARK",
    "chart_monthly_trend",
    "chart_year_comparison",
    "chart_seasonality",
    "chart_monthly_contribution_heatmap",
    "chart_funnel",
    "chart_entity_performance",
    "chart_product_mix",
    "chart_forecast",
    "chart_activity_levels",
    "chart_model_comparison",
    "chart_backtest",
    "chart_feature_importance",
]

logger = logging.getLogger(__name__)

# --- Categorical slots, in fixed assignment order -------------------------
CATEGORICAL_LIGHT: List[str] = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
CATEGORICAL_DARK: List[str] = [
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
]

# --- Single-hue ordinal ramp (blue). Steps chosen to clear 2:1 on each surface.
ORDINAL_LIGHT: List[str] = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
ORDINAL_DARK: List[str] = ["#184f95", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"]

#: Low / Medium / High is an ordered magnitude, so it takes the ordinal ramp -
#: not three unrelated categorical hues.
ACTIVITY_COLORS_LIGHT: Dict[str, str] = {"Low": "#86b6ef", "Medium": "#3987e5", "High": "#184f95"}
ACTIVITY_COLORS_DARK: Dict[str, str] = {"Low": "#184f95", "Medium": "#3987e5", "High": "#b7d3f6"}


def theme(dark: bool = False) -> Dict[str, Any]:
    """Return the colour and chrome tokens for the requested surface.

    Args:
        dark: Render for the dark surface.

    Returns:
        Mapping of role name to colour, plus the categorical and ordinal ramps.
    """
    if dark:
        return {
            "surface": "#1a1a19",
            "text_primary": "#ffffff",
            "text_secondary": "#c3c2b7",
            "muted": "#898781",
            "grid": "#2c2c2a",
            "axis": "#383835",
            "categorical": CATEGORICAL_DARK,
            "ordinal": ORDINAL_DARK,
            "activity": ACTIVITY_COLORS_DARK,
            "good": "#0ca30c",
            "critical": "#d03b3b",
        }
    return {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "categorical": CATEGORICAL_LIGHT,
        "ordinal": ORDINAL_LIGHT,
        "activity": ACTIVITY_COLORS_LIGHT,
        "good": "#006300",
        "critical": "#d03b3b",
    }


def _apply_layout(
    figure: go.Figure,
    tokens: Dict[str, Any],
    title: str = "",
    y_title: str = "",
    x_title: str = "",
    height: int = 420,
    show_legend: bool = True,
    unified_hover: bool = False,
) -> go.Figure:
    """Apply the shared chrome: recessive grid, muted ink, legend on top.

    Args:
        figure: The figure to restyle, modified in place.
        tokens: Output of :func:`theme`.
        title: Chart title. Empty string omits it.
        y_title: Y axis label.
        x_title: X axis label.
        height: Figure height in pixels.
        show_legend: Whether to render the legend.
        unified_hover: Use an ``x unified`` crosshair (for time series).

    Returns:
        The same figure, for chaining.
    """
    figure.update_layout(
        title={"text": title, "font": {"size": 16, "color": tokens["text_primary"]}} if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
              "size": 12, "color": tokens["text_secondary"]},
        height=height,
        margin={"l": 60, "r": 24, "t": 76 if title else 24, "b": 48},
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            # Sits clear of the title, which occupies the top of the margin.
            "y": 1.04,
            "xanchor": "left",
            "x": 0,
            "font": {"color": tokens["text_secondary"]},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified" if unified_hover else "closest",
        hoverlabel={"font": {"family": 'system-ui, -apple-system, sans-serif', "size": 12}},
    )
    # Horizontal hairlines only - vertical gridlines add noise without signal.
    figure.update_xaxes(
        title_text=x_title,
        showgrid=False,
        zeroline=False,
        linecolor=tokens["axis"],
        tickfont={"color": tokens["muted"]},
        title_font={"color": tokens["text_secondary"]},
    )
    figure.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=tokens["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont={"color": tokens["muted"]},
        title_font={"color": tokens["text_secondary"]},
    )
    return figure


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba()`` string at the given alpha."""
    value = color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _ink_on(background: str) -> str:
    """Pick readable ink for text sitting on a coloured fill.

    Both ends of the ordinal ramp are used for fills, so a single hardcoded
    text colour is unreadable at one end. Relative luminance (WCAG) decides.

    Args:
        background: The fill colour as ``#rrggbb``.

    Returns:
        ``"#0b0b0b"`` on light fills, ``"#ffffff"`` on dark ones.
    """
    value = background.lstrip("#")
    channels = []
    for offset in (0, 2, 4):
        channel = int(value[offset : offset + 2], 16) / 255.0
        channels.append(
            channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4
        )
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return "#0b0b0b" if luminance > 0.45 else "#ffffff"


# ---------------------------------------------------------------------------
# Historical analysis
# ---------------------------------------------------------------------------
def chart_monthly_trend(
    trend: pd.DataFrame,
    dark: bool = False,
    show_rolling: bool = True,
    target: str = "APP",
) -> go.Figure:
    """Plot the monthly application series with a 12-month rolling mean.

    Args:
        trend: Output of ``analysis.monthly_trend``.
        dark: Render for the dark surface.
        show_rolling: Overlay the 12-month rolling average.
        target: Column holding the plotted measure.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=trend["date"],
            y=trend[target],
            mode="lines+markers",
            name="Monthly applications",
            line={"color": tokens["categorical"][0], "width": 2},
            marker={"size": 6, "color": tokens["categorical"][0]},
            hovertemplate="%{y:,.0f} applications<extra></extra>",
        )
    )

    if show_rolling and "rolling_12m" in trend.columns:
        figure.add_trace(
            go.Scatter(
                x=trend["date"],
                y=trend["rolling_12m"],
                mode="lines",
                name="12-month average (trend)",
                line={"color": tokens["categorical"][1], "width": 2, "dash": "dot"},
                hovertemplate="%{y:,.0f} trend<extra></extra>",
            )
        )

    return _apply_layout(
        figure, tokens, "Monthly exchange applications", "Applications", "",
        unified_hover=True,
    )


def chart_year_comparison(
    contribution: pd.DataFrame, dark: bool = False, target: str = "APP"
) -> go.Figure:
    """Overlay each year's monthly profile on a shared Jan-Dec axis.

    One line per year makes both the growth (vertical separation) and the
    stability of the seasonal shape (parallel movement) visible at once.

    Args:
        contribution: Output of ``analysis.monthly_contribution``.
        dark: Render for the dark surface.
        target: Column holding the plotted measure.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    figure = go.Figure()
    month_labels = [pd.Timestamp(2024, m, 1).strftime("%b") for m in range(1, 13)]

    # Colour follows the year, assigned in chronological order and never cycled.
    for index, year in enumerate(sorted(contribution["year"].unique())):
        subset = contribution[contribution["year"] == year].sort_values("month")
        figure.add_trace(
            go.Scatter(
                x=[month_labels[m - 1] for m in subset["month"]],
                y=subset[target],
                mode="lines+markers",
                name=str(year),
                line={"color": tokens["categorical"][index % len(tokens["categorical"])], "width": 2},
                marker={"size": 8},
                hovertemplate=f"{year}: %{{y:,.0f}}<extra></extra>",
            )
        )

    return _apply_layout(
        figure, tokens, "Year-on-year monthly comparison", "Applications", "",
        unified_hover=True,
    )


def chart_seasonality(profile: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Show the average monthly seasonal index as bars against a 1.0 baseline.

    Bars are shaded by the ordinal ramp because the seasonal index is an ordered
    magnitude, and the reference line at 1.0 makes above/below average instant.

    Args:
        profile: Output of ``analysis.seasonality_profile``.
        dark: Render for the dark surface.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    ramp = tokens["ordinal"]

    values = profile["seasonal_index"].to_numpy(dtype=float)
    span = values.max() - values.min()
    # Map each bar onto the ordinal ramp by its magnitude.
    indices = (
        np.zeros(len(values), dtype=int)
        if span == 0
        else np.clip(((values - values.min()) / span * (len(ramp) - 1)).round().astype(int), 0, len(ramp) - 1)
    )

    figure = go.Figure(
        go.Bar(
            x=profile["month_name"],
            y=values,
            marker={
                "color": [ramp[i] for i in indices],
                "line": {"width": 2, "color": tokens["surface"]},  # 2px surface gap
                "cornerradius": 4,
            },
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont={"color": tokens["text_secondary"], "size": 11},
            hovertemplate="%{x}: index %{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    # Unlabelled: the y-axis title already reads "1.0 = average", so an
    # annotation here only collides with the January bar label.
    figure.add_hline(y=1.0, line={"color": tokens["muted"], "width": 1, "dash": "dash"})

    figure = _apply_layout(
        figure, tokens, "Seasonal index by calendar month", "Index (1.0 = average)", "",
        show_legend=False,
    )
    figure.update_yaxes(range=[0, max(1.5, float(values.max()) * 1.18)])
    return figure


def chart_monthly_contribution_heatmap(
    contribution: pd.DataFrame, dark: bool = False
) -> go.Figure:
    """Render year x month contribution as a single-hue sequential heatmap.

    Args:
        contribution: Output of ``analysis.monthly_contribution``.
        dark: Render for the dark surface.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    month_labels = [pd.Timestamp(2024, m, 1).strftime("%b") for m in range(1, 13)]

    matrix = contribution.pivot(index="year", columns="month", values="contribution_pct").sort_index()
    scale = [[i / (len(tokens["ordinal"]) - 1), c] for i, c in enumerate(tokens["ordinal"])]

    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=[month_labels[m - 1] for m in matrix.columns],
            y=[str(y) for y in matrix.index],
            colorscale=scale,
            xgap=2,  # 2px surface gap between cells
            ygap=2,
            colorbar={
                "title": {"text": "% of year", "font": {"color": tokens["text_secondary"], "size": 11}},
                "tickfont": {"color": tokens["muted"], "size": 11},
                "outlinewidth": 0,
                "thickness": 12,
            },
            hovertemplate="%{y} %{x}: %{z:.1f}% of the year<extra></extra>",
        )
    )

    figure = _apply_layout(
        figure, tokens, "Share of annual applications by month", "", "",
        show_legend=False, height=300,
    )
    figure.update_yaxes(showgrid=False, autorange="reversed")
    return figure


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------
def chart_funnel(funnel: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Draw the APP -> CO exchange funnel with stage-to-stage conversion labels.

    Stages are an ordered sequence, so they take the single-hue ordinal ramp
    (darkening down the funnel) rather than unrelated categorical hues.

    Args:
        funnel: Output of ``analysis.funnel_analysis``.
        dark: Render for the dark surface.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    ramp = tokens["ordinal"]
    count = len(funnel)
    colors = [ramp[min(int(i / max(count - 1, 1) * (len(ramp) - 1)), len(ramp) - 1)] for i in range(count)]

    labels = [f"{row.stage} - {row.stage_label}" for row in funnel.itertuples()]

    figure = go.Figure(
        go.Funnel(
            y=labels,
            x=funnel["count"],
            textposition="inside",
            textinfo="value+percent initial",
            # Per-bar ink: the ramp spans light to dark, so one fixed colour
            # would be unreadable at one end of the funnel.
            textfont={"color": [_ink_on(c) for c in colors], "size": 12},
            marker={"color": colors, "line": {"width": 2, "color": tokens["surface"]}},
            connector={"line": {"color": tokens["axis"], "width": 1}},
            hovertemplate="%{y}<br>%{x:,.0f} applications<extra></extra>",
        )
    )

    return _apply_layout(
        figure, tokens, "Exchange funnel: applied to completed", "", "",
        show_legend=False, height=460,
    )


# ---------------------------------------------------------------------------
# Entity and product
# ---------------------------------------------------------------------------
def chart_entity_performance(
    entities: pd.DataFrame, dark: bool = False, top_n: int = 12, target: str = "APP"
) -> go.Figure:
    """Rank Local Committees by application volume as a horizontal bar chart.

    Args:
        entities: Output of ``analysis.entity_analysis``.
        dark: Render for the dark surface.
        top_n: How many LCs to show.
        target: Column holding the plotted measure.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    subset = entities.head(top_n).iloc[::-1]  # largest at the top after flip

    figure = go.Figure(
        go.Bar(
            x=subset[target],
            y=subset["entity"],
            orientation="h",
            marker={
                "color": tokens["categorical"][0],
                "line": {"width": 2, "color": tokens["surface"]},
                "cornerradius": 4,
            },
            text=[f"{v:,.0f}" for v in subset[target]],
            textposition="outside",
            textfont={"color": tokens["text_secondary"], "size": 11},
            customdata=subset[["mc_share_pct"]].to_numpy(),
            hovertemplate="%{y}<br>%{x:,.0f} applications (%{customdata[0]:.1f}% of MC)<extra></extra>",
            showlegend=False,
        )
    )

    figure = _apply_layout(
        figure, tokens, f"Top {top_n} Local Committees by applications", "", "Applications",
        show_legend=False, height=max(360, 30 * len(subset)),
    )
    figure.update_xaxes(showgrid=True, gridcolor=tokens["grid"])
    figure.update_yaxes(showgrid=False)
    figure.update_layout(margin={"l": 190, "r": 70, "t": 56, "b": 48})
    return figure


def chart_product_mix(
    products: pd.DataFrame, dark: bool = False, target: str = "APP"
) -> go.Figure:
    """Compare programmes (iGV, oGV, iGTa, ...) by volume.

    Programme identity is categorical, so hues are assigned in fixed slot order
    and never re-assigned when the set is filtered.

    Args:
        products: Output of ``analysis.product_analysis``.
        dark: Render for the dark surface.
        target: Column holding the plotted measure.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    ordered = products.sort_values("programme").reset_index(drop=True)
    colors = [tokens["categorical"][i % len(tokens["categorical"])] for i in range(len(ordered))]

    figure = go.Figure(
        go.Bar(
            x=ordered["programme"],
            y=ordered[target],
            marker={
                "color": colors,
                "line": {"width": 2, "color": tokens["surface"]},
                "cornerradius": 4,
            },
            text=[f"{v:,.0f}" for v in ordered[target]],
            textposition="outside",
            textfont={"color": tokens["text_secondary"], "size": 11},
            customdata=ordered[["mc_share_pct", "app_to_re_pct"]].to_numpy(),
            hovertemplate=(
                "%{x}<br>%{y:,.0f} applications"
                "<br>%{customdata[0]:.1f}% of MC"
                "<br>%{customdata[1]:.1f}% reach realization<extra></extra>"
            ),
            showlegend=False,
        )
    )

    return _apply_layout(
        figure, tokens, "Applications by programme", "Applications", "", show_legend=False
    )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
def chart_forecast(
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    dark: bool = False,
    target: str = "APP",
) -> go.Figure:
    """Plot history and the 2026 forecast with its 95% prediction band.

    History and forecast are the same measure on one axis, distinguished by hue
    and a dashed forecast line, with the uncertainty band drawn beneath.

    Args:
        history: Monthly historical frame with ``date`` and ``target``.
        predictions: Output of ``train.generate_predictions``.
        dark: Render for the dark surface.
        target: Column holding the plotted measure.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    figure = go.Figure()

    forecast_dates = pd.to_datetime(predictions["date"])
    forecast_color = tokens["categorical"][1]

    # Uncertainty band first, so the lines draw on top of it.
    figure.add_trace(
        go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(predictions["upper_95"]) + list(predictions["lower_95"][::-1]),
            fill="toself",
            fillcolor=_hex_to_rgba(forecast_color, 0.14),
            line={"color": "rgba(0,0,0,0)"},
            name="95% prediction interval",
            hoverinfo="skip",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history[target],
            mode="lines",
            name="Historical (2022-2025)",
            line={"color": tokens["categorical"][0], "width": 2},
            hovertemplate="%{y:,.0f} actual<extra></extra>",
        )
    )

    # Join the two lines so the series reads as continuous.
    bridge_x = [history["date"].iloc[-1], forecast_dates.iloc[0]]
    bridge_y = [history[target].iloc[-1], predictions["Predicted Applications"].iloc[0]]

    figure.add_trace(
        go.Scatter(
            x=bridge_x, y=bridge_y, mode="lines",
            line={"color": forecast_color, "width": 2, "dash": "dash"},
            showlegend=False, hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=predictions["Predicted Applications"],
            mode="lines+markers",
            name="Forecast (2026)",
            line={"color": forecast_color, "width": 2, "dash": "dash"},
            marker={"size": 8, "color": forecast_color},
            hovertemplate="%{y:,.0f} forecast<extra></extra>",
        )
    )

    # Plotly 6 computes the annotation offset arithmetically from ``x``, which
    # fails for Timestamp and ISO-string values on a datetime axis. Milliseconds
    # since epoch is the one form that works with an attached annotation.
    boundary = pd.Timestamp(history["date"].iloc[-1]).value // 10**6
    figure.add_vline(
        x=boundary,
        line={"color": tokens["muted"], "width": 1, "dash": "dot"},
        annotation_text="forecast starts",
        annotation_position="top left",
        annotation_font={"color": tokens["muted"], "size": 11},
    )

    return _apply_layout(
        figure, tokens, "Exchange applications: history and 2026 forecast",
        "Applications", "", height=460, unified_hover=True,
    )


def chart_activity_levels(predictions: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Show the 2026 monthly forecast coloured by predicted activity level.

    Low/Medium/High is an ordered scale, so it takes the single-hue ordinal ramp.
    Because colour alone must not carry meaning, each bar is also directly
    labelled with its level.

    Args:
        predictions: Output of ``train.generate_predictions``.
        dark: Render for the dark surface.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    colors = tokens["activity"]

    figure = go.Figure()

    # One trace per level so the legend maps level -> colour explicitly.
    for level in ("Low", "Medium", "High"):
        subset = predictions[predictions["Activity Level"] == level]
        if subset.empty:
            continue
        figure.add_trace(
            go.Bar(
                x=subset["Month"],
                y=subset["Predicted Applications"],
                name=level,
                marker={
                    "color": colors[level],
                    "line": {"width": 2, "color": tokens["surface"]},
                    "cornerradius": 4,
                },
                text=[level] * len(subset),
                textposition="outside",
                textfont={"color": tokens["text_secondary"], "size": 11},
                hovertemplate="%{x}<br>%{y:,.0f} applications<br>" + level + " activity<extra></extra>",
            )
        )

    figure = _apply_layout(
        figure,
        tokens,
        "Predicted 2026 activity by month",
        "Predicted applications",
        "",
        height=440,
    )
    figure.update_layout(
        barmode="group",
        xaxis={"categoryorder": "array", "categoryarray": predictions["Month"].tolist()},
    )
    figure.update_yaxes(range=[0, float(predictions["Predicted Applications"].max()) * 1.2])
    return figure


# ---------------------------------------------------------------------------
# Model diagnostics
# ---------------------------------------------------------------------------
def chart_model_comparison(
    evaluation: pd.DataFrame, dark: bool = False, metric: str = "MAE"
) -> go.Figure:
    """Compare candidate models on one error metric, best at the top.

    The selected model is emphasised with the strongest ordinal step; the rest
    recede. All bars share one hue because this is a single measure, not eight
    identities.

    Args:
        evaluation: Output of ``forecasting.select_best_model``.
        dark: Render for the dark surface.
        metric: Which error column to plot.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    ordered = evaluation.sort_values(metric, ascending=False)

    selected = ordered.get("selected", pd.Series(False, index=ordered.index))
    colors = [
        tokens["ordinal"][-1] if is_best else tokens["ordinal"][1]
        for is_best in selected
    ]

    figure = go.Figure(
        go.Bar(
            x=ordered[metric],
            y=ordered["model"],
            orientation="h",
            marker={
                "color": colors,
                "line": {"width": 2, "color": tokens["surface"]},
                "cornerradius": 4,
            },
            text=[f"{v:,.1f}" for v in ordered[metric]],
            textposition="outside",
            textfont={"color": tokens["text_secondary"], "size": 11},
            customdata=ordered[["family"]].to_numpy(),
            hovertemplate="%{y} (%{customdata[0]})<br>" + metric + ": %{x:,.2f}<extra></extra>",
            showlegend=False,
        )
    )

    figure = _apply_layout(
        figure,
        tokens,
        f"Model comparison - walk-forward {metric} (lower is better)",
        "",
        metric,
        show_legend=False,
        height=max(360, 32 * len(ordered)),
    )
    figure.update_xaxes(showgrid=True, gridcolor=tokens["grid"])
    figure.update_yaxes(showgrid=False)
    figure.update_layout(margin={"l": 160, "r": 70, "t": 56, "b": 48})
    return figure


def chart_backtest(
    backtest: pd.DataFrame, model_name: str, dark: bool = False
) -> go.Figure:
    """Overlay actual versus predicted values across the walk-forward origins.

    Args:
        backtest: Output of ``forecasting.rolling_origin_backtest``.
        model_name: Which model's fold predictions to draw.
        dark: Render for the dark surface.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    subset = backtest[backtest["model"] == model_name].sort_values("date")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=subset["date"], y=subset["y_true"], mode="lines+markers", name="Actual",
            line={"color": tokens["categorical"][0], "width": 2}, marker={"size": 7},
            hovertemplate="Actual: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=subset["date"], y=subset["y_pred"], mode="lines+markers", name="Predicted",
            line={"color": tokens["categorical"][1], "width": 2, "dash": "dash"},
            marker={"size": 7},
            hovertemplate="Predicted: %{y:,.0f}<extra></extra>",
        )
    )

    return _apply_layout(
        figure, tokens, f"Walk-forward validation - {model_name}", "Applications", "",
        unified_hover=True,
    )


def chart_feature_importance(
    importances: pd.DataFrame, dark: bool = False, top_n: int = 15
) -> go.Figure:
    """Plot the most influential engineered features for the selected model.

    Args:
        importances: Two-column frame of ``feature`` and ``importance``.
        dark: Render for the dark surface.
        top_n: How many features to show.

    Returns:
        A Plotly figure.
    """
    tokens = theme(dark)
    subset = importances.head(top_n).iloc[::-1]

    figure = go.Figure(
        go.Bar(
            x=subset["importance"],
            y=subset["feature"],
            orientation="h",
            marker={
                "color": tokens["categorical"][0],
                "line": {"width": 2, "color": tokens["surface"]},
                "cornerradius": 4,
            },
            hovertemplate="%{y}: %{x:,.3f}<extra></extra>",
            showlegend=False,
        )
    )

    figure = _apply_layout(
        figure, tokens, f"Top {top_n} features by influence", "", "Importance",
        show_legend=False, height=max(360, 26 * len(subset)),
    )
    figure.update_xaxes(showgrid=True, gridcolor=tokens["grid"])
    figure.update_yaxes(showgrid=False)
    figure.update_layout(margin={"l": 170, "r": 40, "t": 56, "b": 48})
    return figure
