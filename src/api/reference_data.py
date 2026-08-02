"""Offline reference dataset generator.

WHY THIS EXISTS
---------------
The AIESEC Analytics API requires a short-lived GIS ``access_token`` bound to a
real AIESEC account. Anyone reviewing this repository without those credentials
would otherwise be unable to run a single line of the pipeline.

This module synthesises a **clearly-labelled reference dataset** that:

  * emits the *same nested aggregation JSON shape* the live API returns, so the
    parser in :mod:`src.preprocessing.cleaning` is exercised identically on both
    the live and offline paths;
  * is deterministic (seeded), so results are reproducible;
  * is stamped ``is_reference_data: true`` in the raw envelope, which propagates
    into every downstream artefact and the dashboard banner.

IMPORTANT
---------
This is **simulated data**, not real AIESEC operational data. Every number
produced by the pipeline when running in reference mode is illustrative. It
exists to demonstrate that the pipeline is correct and complete - not to make
claims about AIESEC in India's actual performance. Supply a real token to
replace it.

GENERATIVE MODEL
----------------
Monthly applications per (entity, product, direction) are drawn from a Poisson
whose rate is the product of four documented factors::

    lambda = base_scale
           x entity_weight        (Zipf-like LC size distribution)
           x product_weight       (GV >> GTa > GTe)
           x direction_weight     (India is a net-sending MC)
           x trend(t)             (post-COVID recovery 2022 -> 2025)
           x seasonality(month)   (Indian academic calendar)
           x lognormal noise

Downstream funnel stages are then drawn as Binomials using stage-to-stage
conversion rates, which is how a real funnel behaves (each stage is a subset of
the previous one, so counts are monotonically non-increasing by construction).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.api.aiesec_api import month_windows
from src.config import Settings, get_settings

__all__ = ["generate_reference_responses", "REFERENCE_DATA_WARNING"]

logger = logging.getLogger(__name__)

REFERENCE_DATA_WARNING = (
    "SIMULATED REFERENCE DATA - not real AIESEC operational data. "
    "Set AIESEC_ACCESS_TOKEN in .env to collect live data from the Analytics API."
)

# --- documented generative parameters --------------------------------------

#: Relative product volume. AIESEC India's exchange mix is Global Volunteer
#: dominant, with Global Talent second and Global Teacher a small programme.
PRODUCT_WEIGHTS: Dict[str, float] = {"GV": 1.00, "GTa": 0.34, "GTe": 0.11, "GE": 0.05}

#: India is a net *sending* MC: outgoing exchange far exceeds incoming.
DIRECTION_WEIGHTS: Dict[str, float] = {"outgoing": 0.74, "incoming": 0.26}

#: Year-level activity multipliers: 2022 is still COVID-suppressed, 2023 is the
#: sharp recovery year, 2024-2025 consolidate into steadier growth.
YEAR_TREND: Dict[int, float] = {2022: 1.00, 2023: 1.44, 2024: 1.61, 2025: 1.79}

#: Monthly seasonality, driven by the Indian academic calendar.
#: Outgoing peaks in the winter (Dec-Jan) and summer (May-Jun) breaks.
SEASONALITY_OUTGOING: Dict[int, float] = {
    1: 1.18, 2: 0.84, 3: 0.93, 4: 1.04, 5: 1.36, 6: 1.28,
    7: 1.02, 8: 0.88, 9: 0.97, 10: 1.09, 11: 1.21, 12: 1.44,
}

#: Incoming exchange peaks slightly later - partners arrive over Jun-Aug and Dec-Jan.
SEASONALITY_INCOMING: Dict[int, float] = {
    1: 1.22, 2: 0.86, 3: 0.90, 4: 0.95, 5: 1.14, 6: 1.34,
    7: 1.31, 8: 1.12, 9: 0.94, 10: 0.92, 11: 1.05, 12: 1.31,
}

#: Stage-to-stage conversion rates for the APP -> ACH -> ACC -> APD -> RE -> FI -> CO
#: funnel. Compounded APP -> RE is ~15%, consistent with published AIESEC funnels.
BASE_CONVERSIONS: List[Tuple[str, str, float]] = [
    ("APP", "ACH", 0.42),
    ("ACH", "ACC", 0.66),
    ("ACC", "APD", 0.78),
    ("APD", "RE", 0.72),
    ("RE", "FI", 0.93),
    ("FI", "CO", 0.88),
]

#: Overall MC scale: expected monthly applications per (entity, product, direction)
#: cell before weighting. Tuned so MC-level monthly totals land in the low
#: thousands, the right order of magnitude for a large sending MC.
BASE_SCALE: float = 26.0

#: Canonical stage code -> API status key (inverse of ``funnel.api_status_map``).
STAGE_TO_API_STATUS: Dict[str, str] = {
    "APP": "applied",
    "ACH": "achieved",
    "ACC": "accepted",
    "APD": "approved",
    "RE": "realized",
    "FI": "finished",
    "CO": "completed",
}


def _entity_weights(entities: List[str], rng: np.random.Generator) -> Dict[str, float]:
    """Assign each LC a stable size weight following a Zipf-like distribution.

    Real MC entity portfolios are highly unequal - a handful of large LCs drive
    most of the volume. A ``1 / rank^0.65`` decay reproduces that shape.

    Args:
        entities: Ordered LC names (order defines rank).
        rng: Seeded random generator, for the small per-LC jitter.

    Returns:
        Mapping of LC name to a positive weight.
    """
    ranks = np.arange(1, len(entities) + 1, dtype=float)
    weights = 1.0 / np.power(ranks, 0.65)
    jitter = rng.uniform(0.85, 1.15, size=len(entities))
    weights = weights * jitter
    weights = weights / weights.mean()  # keep the mean at 1.0
    return dict(zip(entities, weights.tolist()))


def _trend_multiplier(target: date) -> float:
    """Interpolate the annual trend smoothly across months.

    A step change every January would create an artificial discontinuity that
    the seasonal models would happily learn. Linear interpolation between year
    anchors keeps the trend smooth and realistic.

    Args:
        target: The month being generated.

    Returns:
        A positive growth multiplier.
    """
    years = sorted(YEAR_TREND)
    if target.year <= years[0]:
        return YEAR_TREND[years[0]]
    if target.year >= years[-1]:
        # Extrapolate the final year-over-year slope through the last year.
        final_growth = YEAR_TREND[years[-1]] / YEAR_TREND[years[-2]]
        progress = (target.month - 1) / 12.0
        return YEAR_TREND[years[-1]] * (1.0 + (final_growth - 1.0) * progress * 0.35)

    current = YEAR_TREND[target.year]
    nxt = YEAR_TREND.get(target.year + 1, current)
    progress = (target.month - 1) / 12.0
    return current + (nxt - current) * progress


def _conversion_rates(
    product: str, direction: str, target: date, rng: np.random.Generator
) -> List[Tuple[str, str, float]]:
    """Perturb the base funnel conversions per product, direction and month.

    Two structural effects are modelled on top of noise:
      * Global Talent converts better than Global Volunteer (longer, more
        deliberate applications).
      * Funnel efficiency improves slowly year over year as processes mature.

    Args:
        product: Product short name, e.g. ``"GV"``.
        direction: ``"incoming"`` or ``"outgoing"``.
        target: The month being generated.
        rng: Seeded random generator.

    Returns:
        ``(from_stage, to_stage, rate)`` triples with rates clipped to ``[0.05, 0.97]``.
    """
    product_quality = {"GV": 1.00, "GTa": 1.12, "GTe": 1.05, "GE": 0.95}.get(product, 1.0)
    direction_quality = 1.0 if direction == "outgoing" else 0.94
    maturity = 1.0 + 0.02 * (target.year - 2022)

    rates: List[Tuple[str, str, float]] = []
    for source, dest, base in BASE_CONVERSIONS:
        rate = base * product_quality * direction_quality * maturity * rng.normal(1.0, 0.05)
        rates.append((source, dest, float(np.clip(rate, 0.05, 0.97))))
    return rates


def _simulate_funnel(
    applications: int,
    product: str,
    direction: str,
    target: date,
    rng: np.random.Generator,
) -> Dict[str, int]:
    """Draw the full funnel downstream of a known application count.

    Each stage is drawn as ``Binomial(previous_stage, conversion_rate)``, which
    guarantees the funnel is monotonically non-increasing - a real funnel
    invariant that the validation layer later asserts.

    Args:
        applications: Stage-0 count (APP).
        product: Product short name.
        direction: ``"incoming"`` or ``"outgoing"``.
        target: The month being generated.
        rng: Seeded random generator.

    Returns:
        Mapping of stage code to count, always including every stage.
    """
    counts: Dict[str, int] = {"APP": int(applications)}
    for source, dest, rate in _conversion_rates(product, direction, target, rng):
        counts[dest] = int(rng.binomial(max(counts[source], 0), rate))
    return counts


def _status_buckets(counts: Dict[str, int]) -> List[Dict[str, Any]]:
    """Render stage counts as API-style status buckets."""
    return [
        {"key": STAGE_TO_API_STATUS[stage], "doc_count": int(count)}
        for stage, count in counts.items()
        if stage in STAGE_TO_API_STATUS
    ]


def generate_reference_responses(
    settings: Optional[Settings] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Generate a complete offline reference dataset in raw-API envelope form.

    Args:
        settings: Loaded settings. Loaded from disk when omitted.
        save: Write the envelope to ``data/raw/api_responses.json``.

    Returns:
        The envelope dict, matching the structure produced by live collection
        except that ``metadata.is_reference_data`` is ``True``.
    """
    settings = settings or get_settings()
    seed = int(settings.reference_data.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    entities = settings.entities
    if not entities:
        raise ValueError("reference_data.entities is empty in config/config.yaml")

    products = settings.products or ["GV"]
    directions = settings.directions or ["outgoing", "incoming"]

    start = str(settings.collection.get("start_date", "2022-01-01"))
    end = str(settings.collection.get("end_date", "2025-12-31"))
    windows = month_windows(start, end)

    entity_weight = _entity_weights(entities, rng)
    # Stable pseudo office ids so entity identity survives a re-run.
    entity_ids = {name: 90000 + idx for idx, name in enumerate(entities)}

    logger.warning(REFERENCE_DATA_WARNING)
    logger.info(
        "Generating reference data: %d months x %d entities x %d products x %d directions",
        len(windows),
        len(entities),
        len(products),
        len(directions),
    )

    records: List[Dict[str, Any]] = []

    for win_start, win_end in windows:
        trend = _trend_multiplier(win_start)

        for product in products:
            office_buckets: List[Dict[str, Any]] = []

            for entity in entities:
                direction_buckets: List[Dict[str, Any]] = []

                for direction in directions:
                    seasonal = (
                        SEASONALITY_OUTGOING if direction == "outgoing" else SEASONALITY_INCOMING
                    )[win_start.month]

                    rate = (
                        BASE_SCALE
                        * entity_weight[entity]
                        * PRODUCT_WEIGHTS.get(product, 0.1)
                        * DIRECTION_WEIGHTS.get(direction, 0.5)
                        * trend
                        * seasonal
                        * rng.lognormal(mean=0.0, sigma=0.12)
                    )
                    applications = int(rng.poisson(max(rate, 0.01)))
                    counts = _simulate_funnel(applications, product, direction, win_start, rng)

                    direction_buckets.append(
                        {
                            "key": direction,
                            "doc_count": counts["APP"],
                            "statuses": {"buckets": _status_buckets(counts)},
                        }
                    )

                office_buckets.append(
                    {
                        "key": entity_ids[entity],
                        "key_as_string": entity,
                        "doc_count": sum(b["doc_count"] for b in direction_buckets),
                        "directions": {"buckets": direction_buckets},
                    }
                )

            page = {
                "analytics": {"offices": {"buckets": office_buckets}},
                "meta": {"page": 1, "total_pages": 1, "generated": "reference"},
            }
            records.append(
                {
                    "start_date": win_start.isoformat(),
                    "end_date": win_end.isoformat(),
                    "month": win_start.strftime("%Y-%m"),
                    "programme_id": None,
                    "product": product,
                    "office_id": settings.office_id,
                    "pages": [page],
                }
            )

    envelope = {
        "metadata": {
            "source": "SIMULATED reference dataset (no API credentials available)",
            "warning": REFERENCE_DATA_WARNING,
            "endpoint": None,
            "mc_name": settings.mc_name,
            "office_id": settings.office_id,
            "collection_start": start,
            "collection_end": end,
            "collected_at": datetime.now().astimezone().isoformat(),
            "record_count": len(records),
            "failed_windows": [],
            "is_reference_data": True,
            "random_seed": seed,
        },
        "records": records,
    }

    if save:
        path: Path = settings.paths.raw_responses
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2, ensure_ascii=False)
        logger.info("Reference raw responses written to %s", path)

    return envelope
