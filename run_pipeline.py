#!/usr/bin/env python3
"""Command-line entry point for the exchange activity prediction pipeline.

Usage
-----
Run everything (collects live data if ``AIESEC_ACCESS_TOKEN`` is set, otherwise
falls back to the offline reference dataset)::

    python run_pipeline.py --step all

Force the offline reference dataset even when credentials exist::

    python run_pipeline.py --step all --use-reference-data

Run a single stage (each stage reads the previous stage's artefacts from disk,
so they can be run independently)::

    python run_pipeline.py --step collect
    python run_pipeline.py --step process
    python run_pipeline.py --step analyze
    python run_pipeline.py --step train

Exit codes: ``0`` success, ``1`` pipeline failure, ``2`` bad arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import pandas as pd

from src.api.aiesec_api import AiesecAPIError, AuthenticationError, collect_exchange_data
from src.api.reference_data import generate_reference_responses
from src.config import Settings, configure_logging, get_settings
from src.insights import generate_all_insights, insights_to_frame
from src.models.train import generate_predictions, save_artifacts, train_and_select
from src.preprocessing.analysis import run_full_analysis
from src.preprocessing.cleaning import build_exchange_dataset, load_exchange_dataset
from src.preprocessing.features import build_training_frame
from src.visualization.figures import export_all_figures

logger = logging.getLogger("pipeline")

STEPS = ("collect", "process", "analyze", "features", "train", "figures", "all")


def _banner(title: str) -> None:
    """Log a visually distinct stage header."""
    logger.info("=" * 72)
    logger.info(title)
    logger.info("=" * 72)


def step_collect(settings: Settings, use_reference: bool, args: argparse.Namespace) -> None:
    """Collect raw data from the API, or generate the offline reference dataset.

    Args:
        settings: Loaded settings.
        use_reference: Skip the API and generate reference data.
        args: Parsed CLI arguments (for date overrides).
    """
    _banner("STEP 1/5  DATA COLLECTION")

    if use_reference:
        logger.warning("Reference mode: generating SIMULATED data (no API call)")
        generate_reference_responses(settings)
        return

    if not settings.has_credentials:
        logger.warning(
            "AIESEC_ACCESS_TOKEN is not set - falling back to the offline reference "
            "dataset. Set it in .env for live data."
        )
        generate_reference_responses(settings)
        return

    try:
        result = collect_exchange_data(
            settings, start_date=args.start_date, end_date=args.end_date
        )
    except AuthenticationError as exc:
        logger.error("Authentication failed: %s", exc)
        logger.warning("Falling back to the offline reference dataset")
        generate_reference_responses(settings)
        return
    except AiesecAPIError as exc:
        logger.error("Collection failed: %s", exc)
        raise

    if result.windows_succeeded == 0:
        logger.error("No windows succeeded; falling back to the reference dataset")
        generate_reference_responses(settings)


def step_process(settings: Settings) -> pd.DataFrame:
    """Parse raw payloads into the validated processed dataset.

    Args:
        settings: Loaded settings.

    Returns:
        The processed exchange panel.
    """
    _banner("STEP 2/5  PARSING, CLEANING AND VALIDATION")
    return build_exchange_dataset(settings, save=True, strict=False)


def step_analyze(settings: Settings, panel: pd.DataFrame) -> dict:
    """Run exploratory analysis and write the report tables.

    Args:
        settings: Loaded settings.
        panel: Processed exchange panel.

    Returns:
        Mapping of analysis name to DataFrame.
    """
    _banner("STEP 3/5  EXPLORATORY DATA ANALYSIS")
    results = run_full_analysis(panel, settings, save=True)

    funnel = results["funnel_overall"]
    realized = funnel[funnel["stage"] == "RE"]
    if not realized.empty:
        logger.info(
            "Application-to-realization conversion: %.1f%%",
            float(realized.iloc[0]["conversion_from_APP_pct"]),
        )
    return results


def step_features(settings: Settings, panel: pd.DataFrame) -> pd.DataFrame:
    """Build and persist the supervised feature matrix.

    Args:
        settings: Loaded settings.
        panel: Processed exchange panel.

    Returns:
        The feature matrix.
    """
    _banner("STEP 4/5  FEATURE ENGINEERING")
    return build_training_frame(panel, settings, save=True)


def step_train(settings: Settings, panel: pd.DataFrame, analysis: Optional[dict]) -> tuple:
    """Backtest, select, persist and forecast.

    Args:
        settings: Loaded settings.
        panel: Processed exchange panel.
        analysis: Analysis tables, used for insight generation.

    Returns:
        ``(artifacts, predictions)``.
    """
    _banner("STEP 5/5  MODEL TRAINING, SELECTION AND FORECASTING")

    artifacts = train_and_select(panel, settings)
    save_artifacts(artifacts, settings)
    predictions = generate_predictions(artifacts, settings)

    insights = generate_all_insights(
        analysis=analysis,
        predictions=predictions,
        evaluation=artifacts.evaluation,
        model_name=artifacts.best_model_name,
        settings=settings,
    )
    if insights:
        path = settings.paths.reports_dir / "insights.csv"
        insights_to_frame(insights).to_csv(path, index=False)
        logger.info("Wrote %d insight(s) to %s", len(insights), path.name)

        logger.info("-" * 72)
        for insight in insights[:6]:
            logger.info("[%s] %s", insight.category, insight.headline)
        logger.info("-" * 72)

    return artifacts, predictions


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="AIESEC MC India exchange activity prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step", choices=STEPS, default="all", help="Which stage to run (default: all)"
    )
    parser.add_argument(
        "--use-reference-data",
        action="store_true",
        help="Force the offline simulated dataset instead of calling the API",
    )
    parser.add_argument("--start-date", help="Override collection.start_date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Override collection.end_date (YYYY-MM-DD)")
    parser.add_argument("--config", help="Path to an alternative config YAML")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: from config / LOG_LEVEL)",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip static figure export"
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the pipeline.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    try:
        settings = get_settings(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(settings, args.log_level)

    logger.info("AIESEC Exchange Activity Prediction Platform")
    logger.info("MC: %s | target: %s | forecast year: %s",
                settings.mc_name, settings.target, settings.forecast_year)
    logger.info("Credentials present: %s", settings.has_credentials)

    step = args.step
    panel: Optional[pd.DataFrame] = None
    analysis: Optional[dict] = None

    try:
        if step in ("collect", "all"):
            step_collect(settings, args.use_reference_data, args)

        if step in ("process", "all"):
            panel = step_process(settings)

        if step in ("analyze", "features", "train", "figures") and panel is None:
            panel = load_exchange_dataset(settings)

        if step in ("analyze", "all"):
            analysis = step_analyze(settings, panel)

        if step in ("features", "all"):
            step_features(settings, panel)

        if step in ("train", "all"):
            if analysis is None:
                analysis = run_full_analysis(panel, settings, save=False)
            step_train(settings, panel, analysis)

        if step in ("figures", "all") and not args.no_figures:
            _banner("STATIC FIGURE EXPORT")
            export_all_figures(settings)

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level guard, full trace logged
        logger.exception("Pipeline failed: %s", exc)
        return 1

    _banner("PIPELINE COMPLETE")
    logger.info("Processed dataset : %s", settings.paths.processed_dataset)
    logger.info("Predictions       : %s", settings.paths.predictions)
    logger.info("Reports           : %s", settings.paths.reports_dir)
    logger.info("Figures           : %s", settings.paths.figures_dir)
    logger.info("Launch dashboard  : streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
