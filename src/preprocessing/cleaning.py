"""Parse, clean and validate raw AIESEC Analytics payloads.

Turns the nested Elasticsearch-style aggregation JSON saved by
:mod:`src.api.aiesec_api` into the tidy monthly panel written to
``data/processed/exchange_data.csv``.

Parsing strategy
----------------
The exact nesting order of the Analytics API's aggregation tree is not
guaranteed to be stable, and differs between analytics families. Rather than
hardcoding a path like ``analytics.offices.buckets[].directions.buckets[]``,
:func:`parse_payload` performs a **tolerant recursive walk**: it descends any
``{"buckets": [...]}`` node it finds, classifies each bucket as an entity, a
direction, a product or a funnel status by inspecting its key, and accumulates
counts against whatever context it has gathered so far.

That means a change in nesting order - or an extra grouping level - does not
break ingestion. Only a change in the *status vocabulary* would, and that
vocabulary is itself configuration (``funnel.api_status_map``).

Output schema (``data/processed/exchange_data.csv``)
----------------------------------------------------
============  ==========================================================
column        meaning
============  ==========================================================
date          First day of the month (``YYYY-MM-01``)
year          Calendar year
month         Calendar month number, 1-12
month_name    Short month name, e.g. ``Jan``
quarter       Calendar quarter, 1-4
entity        Local Committee name
product       ``GV`` / ``GTa`` / ``GTe``
direction     ``incoming`` / ``outgoing``
programme     Direction-prefixed product, e.g. ``oGV``
APP..CO       Funnel stage counts, monotonically non-increasing
============  ==========================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.api.aiesec_api import load_raw_responses
from src.config import Settings, get_settings

__all__ = [
    "parse_payload",
    "parse_envelope",
    "build_exchange_dataset",
    "load_exchange_dataset",
    "build_monthly_series",
    "ValidationReport",
    "validate_dataset",
]

logger = logging.getLogger(__name__)

#: Aggregation keys that introduce an entity (Local Committee) grouping.
ENTITY_KEYS = {"offices", "office", "entities", "entity", "committees", "committee", "lc", "lcs"}

#: Aggregation keys that introduce an incoming/outgoing grouping.
DIRECTION_KEYS = {"directions", "direction", "types", "type", "flow", "opportunity_type"}

#: Aggregation keys that introduce a product/programme grouping.
PRODUCT_KEYS = {"programmes", "programme", "products", "product", "programs", "program"}

#: Aggregation keys that introduce a funnel status grouping.
STATUS_KEYS = {"statuses", "status", "stages", "stage", "application_status"}

#: Normalised direction spellings.
_DIRECTION_ALIASES = {
    "outgoing": "outgoing", "og": "outgoing", "o": "outgoing", "out": "outgoing",
    "sending": "outgoing", "export": "outgoing",
    "incoming": "incoming", "ic": "incoming", "i": "incoming", "in": "incoming",
    "hosting": "incoming", "import": "incoming",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@dataclass
class _BucketContext:
    """Dimension values accumulated while descending the aggregation tree."""

    entity: Optional[str] = None
    entity_id: Optional[int] = None
    product: Optional[str] = None
    direction: Optional[str] = None

    def merged(self, **updates: Any) -> "_BucketContext":
        """Return a copy with ``updates`` applied - keeps the walk side-effect free."""
        return _BucketContext(
            entity=updates.get("entity", self.entity),
            entity_id=updates.get("entity_id", self.entity_id),
            product=updates.get("product", self.product),
            direction=updates.get("direction", self.direction),
        )

    def key(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return (self.entity, self.product, self.direction)


def _coerce_int(value: Any) -> int:
    """Convert a ``doc_count``-like value to ``int``, defaulting to 0."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalise_direction(raw: Any) -> Optional[str]:
    """Map an API direction key onto ``incoming`` / ``outgoing``."""
    if raw is None:
        return None
    return _DIRECTION_ALIASES.get(str(raw).strip().lower())


def _bucket_label(bucket: Dict[str, Any]) -> str:
    """Prefer the human-readable bucket label, falling back to the raw key."""
    return str(bucket.get("key_as_string") or bucket.get("key") or "")


def parse_payload(
    payload: Any,
    status_map: Dict[str, str],
    programme_mapping: Optional[Dict[int, str]] = None,
    context: Optional[_BucketContext] = None,
    sink: Optional[Dict[Tuple[Optional[str], Optional[str], Optional[str]], Dict[str, int]]] = None,
) -> Dict[Tuple[Optional[str], Optional[str], Optional[str]], Dict[str, int]]:
    """Recursively extract funnel counts from one aggregation payload.

    Args:
        payload: Any node of the decoded JSON tree.
        status_map: API status key (lowercased) -> canonical stage code.
        programme_mapping: GIS programme id -> product short name.
        context: Dimension values gathered from ancestor buckets.
        sink: Accumulator, keyed by ``(entity, product, direction)``.

    Returns:
        The accumulator, mapping each dimension combination to stage counts.
    """
    context = context or _BucketContext()
    sink = sink if sink is not None else {}
    programme_mapping = programme_mapping or {}

    if isinstance(payload, list):
        for item in payload:
            parse_payload(item, status_map, programme_mapping, context, sink)
        return sink

    if not isinstance(payload, dict):
        return sink

    for key, value in payload.items():
        key_lower = str(key).lower()

        # A grouping node: {"<dimension>": {"buckets": [...]}}
        if isinstance(value, dict) and isinstance(value.get("buckets"), list):
            for bucket in value["buckets"]:
                if not isinstance(bucket, dict):
                    continue
                _consume_bucket(
                    dimension=key_lower,
                    bucket=bucket,
                    status_map=status_map,
                    programme_mapping=programme_mapping,
                    context=context,
                    sink=sink,
                )
            continue

        # A bare buckets list, or any other nested structure worth descending.
        if isinstance(value, (dict, list)):
            parse_payload(value, status_map, programme_mapping, context, sink)

    return sink


def _consume_bucket(
    dimension: str,
    bucket: Dict[str, Any],
    status_map: Dict[str, str],
    programme_mapping: Dict[int, str],
    context: _BucketContext,
    sink: Dict[Tuple[Optional[str], Optional[str], Optional[str]], Dict[str, int]],
) -> None:
    """Classify a single bucket, update context, record counts, then recurse.

    Args:
        dimension: Lowercased name of the aggregation the bucket belongs to.
        bucket: The bucket object itself.
        status_map: API status key -> canonical stage code.
        programme_mapping: GIS programme id -> product short name.
        context: Context inherited from ancestor buckets.
        sink: Accumulator to write counts into.
    """
    raw_key = bucket.get("key")
    label = _bucket_label(bucket)
    key_lower = str(raw_key).strip().lower()

    # 1. Funnel status bucket - this is a leaf measurement.
    stage = status_map.get(key_lower)
    if stage is not None or dimension in STATUS_KEYS:
        stage = stage or status_map.get(str(label).strip().lower())
        if stage is not None:
            cell = sink.setdefault(context.key(), {})
            cell[stage] = cell.get(stage, 0) + _coerce_int(bucket.get("doc_count"))
            # A status bucket can still nest further breakdowns.
            parse_payload(bucket, status_map, programme_mapping, context, sink)
            return

    # 2. Dimension buckets - widen the context and descend.
    updates: Dict[str, Any] = {}
    if dimension in ENTITY_KEYS:
        updates["entity"] = label or None
        try:
            updates["entity_id"] = int(raw_key)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            updates["entity_id"] = None
    elif dimension in DIRECTION_KEYS:
        updates["direction"] = _normalise_direction(raw_key) or _normalise_direction(label)
    elif dimension in PRODUCT_KEYS:
        try:
            updates["product"] = programme_mapping.get(int(raw_key), label or None)
        except (TypeError, ValueError):
            updates["product"] = label or None

    parse_payload(bucket, status_map, programme_mapping, context.merged(**updates), sink)


def parse_envelope(envelope: Dict[str, Any], settings: Settings) -> pd.DataFrame:
    """Convert a raw-response envelope into the tidy monthly panel.

    Args:
        envelope: Envelope produced by collection or reference generation.
        settings: Loaded settings (funnel vocabulary, product mapping).

    Returns:
        A long-format DataFrame, one row per (month, entity, product, direction).

    Raises:
        ValueError: if the envelope contains no records.
    """
    records = envelope.get("records") or []
    if not records:
        raise ValueError("Raw envelope contains no records - nothing to parse")

    status_map = {k.lower(): v for k, v in settings.api_status_map.items()}
    programme_mapping = settings.programme_mapping
    stages = settings.funnel_stages

    rows: List[Dict[str, Any]] = []

    for record in records:
        month = str(record.get("month", ""))
        record_product = record.get("product")

        for page in record.get("pages", []):
            counts = parse_payload(page, status_map, programme_mapping)

            for (entity, product, direction), stage_counts in counts.items():
                if not stage_counts:
                    continue
                row: Dict[str, Any] = {
                    "month_key": month,
                    "entity": entity or settings.mc_name,
                    "product": product or record_product or "UNKNOWN",
                    "direction": direction or "unknown",
                }
                for stage in stages:
                    row[stage] = int(stage_counts.get(stage, 0))
                rows.append(row)

    if not rows:
        raise ValueError(
            "Parsed zero rows from the raw envelope. The aggregation shape may "
            "differ from expectations - check funnel.api_status_map in "
            "config/config.yaml against a sample response in data/raw/."
        )

    frame = pd.DataFrame(rows)

    # Collapse duplicates that arise when the same cell appears across pages.
    group_cols = ["month_key", "entity", "product", "direction"]
    frame = frame.groupby(group_cols, as_index=False)[stages].sum()

    return _add_calendar_columns(frame)


def _add_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive date/calendar columns and the ``programme`` label; drop the raw key."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["month_key"] + "-01", format="%Y-%m-%d")
    frame["year"] = frame["date"].dt.year.astype(int)
    frame["month"] = frame["date"].dt.month.astype(int)
    frame["month_name"] = frame["date"].dt.strftime("%b")
    frame["quarter"] = frame["date"].dt.quarter.astype(int)

    prefix = frame["direction"].map({"outgoing": "o", "incoming": "i"}).fillna("")
    frame["programme"] = prefix + frame["product"].astype(str)

    ordered = [
        "date", "year", "month", "month_name", "quarter",
        "entity", "product", "direction", "programme",
    ]
    stages = [c for c in frame.columns if c not in ordered + ["month_key"]]
    return frame[ordered + stages].sort_values(["date", "entity", "programme"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Result of validating the processed dataset."""

    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def log(self) -> None:
        """Emit the report through the logging system at appropriate levels."""
        for warning in self.warnings:
            logger.warning("Validation warning: %s", warning)
        for error in self.errors:
            logger.error("Validation error: %s", error)
        if self.passed:
            logger.info("Dataset validation passed (%s)", self.stats)


def validate_dataset(frame: pd.DataFrame, settings: Settings) -> ValidationReport:
    """Check the processed panel for structural and semantic integrity.

    Checks performed:
      1. Required columns are present.
      2. No negative counts.
      3. The funnel is monotonically non-increasing within every row.
      4. Every month in the configured collection window is represented.
      5. No duplicate (month, entity, programme) cells.

    Args:
        frame: The processed panel.
        settings: Loaded settings (funnel stages, collection window).

    Returns:
        A :class:`ValidationReport`. Monotonicity and coverage gaps are errors;
        anything recoverable is a warning.
    """
    errors: List[str] = []
    warnings: List[str] = []
    stages = settings.funnel_stages

    required = ["date", "entity", "product", "direction", *stages]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return ValidationReport(passed=False, errors=errors)

    if frame.empty:
        errors.append("Processed dataset is empty")
        return ValidationReport(passed=False, errors=errors)

    # 2. Negative counts.
    negatives = {stage: int((frame[stage] < 0).sum()) for stage in stages}
    if any(negatives.values()):
        errors.append(f"Negative counts found: { {k: v for k, v in negatives.items() if v} }")

    # 3. Funnel monotonicity.
    violations = 0
    for earlier, later in zip(stages, stages[1:]):
        violations += int((frame[later] > frame[earlier]).sum())
    if violations:
        errors.append(
            f"{violations} row(s) violate funnel monotonicity "
            "(a downstream stage exceeds its upstream stage)"
        )

    # 4. Month coverage.
    expected = pd.period_range(
        start=str(settings.collection.get("start_date", "2022-01-01")),
        end=str(settings.collection.get("end_date", "2025-12-31")),
        freq="M",
    )
    present = pd.PeriodIndex(pd.to_datetime(frame["date"]).dt.to_period("M").unique())
    gaps = sorted(set(expected) - set(present))
    if gaps:
        errors.append(f"{len(gaps)} month(s) missing from the panel, e.g. {gaps[:6]}")

    # 5. Duplicate cells.
    dupes = int(frame.duplicated(subset=["date", "entity", "programme"]).sum())
    if dupes:
        warnings.append(f"{dupes} duplicate (date, entity, programme) row(s) found")

    zero_months = int(frame.groupby("date")[stages[0]].sum().eq(0).sum())
    if zero_months:
        warnings.append(f"{zero_months} month(s) have zero applications")

    stats = {
        "rows": int(len(frame)),
        "months": int(present.size),
        "entities": int(frame["entity"].nunique()),
        "programmes": int(frame["programme"].nunique()),
        "total_APP": int(frame[stages[0]].sum()),
    }

    report = ValidationReport(passed=not errors, errors=errors, warnings=warnings, stats=stats)
    report.log()
    return report


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def build_exchange_dataset(
    settings: Optional[Settings] = None,
    save: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """Read raw responses, parse, validate and write the processed dataset.

    Args:
        settings: Loaded settings. Loaded from disk when omitted.
        save: Write ``data/processed/exchange_data.csv`` and the monthly series.
        strict: Raise on validation errors instead of logging them.

    Returns:
        The processed monthly panel.

    Raises:
        ValueError: if ``strict`` is set and validation fails.
    """
    settings = settings or get_settings()
    envelope = load_raw_responses(settings)

    if envelope.get("metadata", {}).get("is_reference_data"):
        logger.warning(
            "Building dataset from SIMULATED reference data - results are illustrative only"
        )

    frame = parse_envelope(envelope, settings)
    report = validate_dataset(frame, settings)

    if not report.passed and strict:
        raise ValueError(f"Dataset validation failed: {report.errors}")

    if save:
        path = settings.paths.processed_dataset
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        logger.info("Processed dataset written to %s (%d rows)", path, len(frame))

        monthly = build_monthly_series(frame, settings)
        monthly.to_csv(settings.paths.monthly_series, index=False)
        logger.info("Monthly MC-level series written to %s", settings.paths.monthly_series)

    return frame


def load_exchange_dataset(settings: Optional[Settings] = None) -> pd.DataFrame:
    """Load the processed panel from disk with proper dtypes.

    Args:
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        The processed monthly panel.

    Raises:
        FileNotFoundError: if the dataset has not been built yet.
    """
    settings = settings or get_settings()
    path = settings.paths.processed_dataset

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run: python run_pipeline.py --step process"
        )

    frame = pd.read_csv(path, parse_dates=["date"])
    return frame


def build_monthly_series(
    frame: pd.DataFrame, settings: Optional[Settings] = None
) -> pd.DataFrame:
    """Aggregate the panel to one MC-level row per month.

    This is the series the forecasting models consume.

    Args:
        frame: The processed panel.
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        A DataFrame indexed by month with total funnel counts plus calendar
        columns, sorted ascending and gap-free.
    """
    settings = settings or get_settings()
    stages = settings.funnel_stages

    monthly = frame.groupby("date", as_index=False)[stages].sum().sort_values("date")
    monthly["year"] = monthly["date"].dt.year.astype(int)
    monthly["month"] = monthly["date"].dt.month.astype(int)
    monthly["month_name"] = monthly["date"].dt.strftime("%b")
    monthly["quarter"] = monthly["date"].dt.quarter.astype(int)

    return monthly.reset_index(drop=True)
