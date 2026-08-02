"""Client for the AIESEC Analytics API.

Target endpoint
---------------
``GET https://analytics.api.aiesec.org/v2/applications/analyze.json``

Authentication
--------------
AIESEC GIS authenticates with a short-lived ``access_token`` supplied as a
**query parameter** (not an ``Authorization`` header). The token is read from
the ``AIESEC_ACCESS_TOKEN`` environment variable and is never logged, never
written to disk, and never echoed in an exception message - see
:func:`redact_token`.

Filters
-------
The endpoint uses nested bracket syntax, e.g.::

    performance[office_id]=1589
    start_date=2022-01-01
    end_date=2022-01-31

The filter namespace (``performance`` / ``recruitment`` / ``advancement``)
differs per analytics family and is therefore configuration-driven
(``api.filter_namespace``) rather than hardcoded.

Response
--------
Responses are Elasticsearch-style aggregations: nested ``buckets`` arrays
carrying ``doc_count`` per funnel status and per child office. Parsing lives in
:mod:`src.preprocessing.cleaning` so that this module stays a pure transport
layer.

Robustness
----------
* Exponential backoff with jitter on ``429`` and ``5xx``.
* ``Retry-After`` is honoured when the server sends it.
* Cursor/page pagination is drained until the last page.
* Every raw payload is persisted so that downstream parsing never needs the
  network again.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests

from src.config import Settings, get_settings

__all__ = [
    "AiesecAPIError",
    "AuthenticationError",
    "RateLimitError",
    "AiesecAnalyticsClient",
    "CollectionResult",
    "collect_exchange_data",
    "month_windows",
    "redact_token",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class AiesecAPIError(RuntimeError):
    """Base class for every AIESEC API failure."""


class AuthenticationError(AiesecAPIError):
    """Raised on HTTP 401/403 - the token is missing, expired or unauthorised."""


class RateLimitError(AiesecAPIError):
    """Raised when the API keeps returning 429 after the retry budget is spent."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def redact_token(text: str, token: Optional[str]) -> str:
    """Remove an access token from arbitrary text before it is logged.

    Args:
        text: Text that may embed the token (typically a URL).
        token: The secret to scrub. ``None`` is a no-op.

    Returns:
        The text with every occurrence of the token replaced by ``***REDACTED***``.
    """
    if not token:
        return text
    return text.replace(token, "***REDACTED***")


def month_windows(start: str | date, end: str | date) -> List[Tuple[date, date]]:
    """Split an inclusive date range into calendar-month windows.

    Monthly windowing keeps each API response small, makes partial failures
    recoverable, and gives the forecasting layer its natural monthly grain.

    Args:
        start: First day of the range (``YYYY-MM-DD`` or ``date``).
        end: Last day of the range (``YYYY-MM-DD`` or ``date``).

    Returns:
        Ordered ``(window_start, window_end)`` pairs, both inclusive.

    Raises:
        ValueError: if ``start`` is after ``end``.
    """
    start_date = date.fromisoformat(start) if isinstance(start, str) else start
    end_date = date.fromisoformat(end) if isinstance(end, str) else end

    if start_date > end_date:
        raise ValueError(f"start ({start_date}) must not be after end ({end_date})")

    windows: List[Tuple[date, date]] = []
    year, month = start_date.year, start_date.month

    while date(year, month, 1) <= end_date:
        first = date(year, month, 1)
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        last = date(next_year, next_month, 1) - timedelta(days=1)
        windows.append((max(first, start_date), min(last, end_date)))
        year, month = next_year, next_month

    return windows


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
@dataclass
class AiesecAnalyticsClient:
    """Thin, resilient HTTP client for the AIESEC Analytics API.

    Args:
        settings: Loaded project settings (provides token, base URL, retry policy).
        session: Optional pre-built ``requests.Session`` - useful for testing.

    Example:
        >>> client = AiesecAnalyticsClient(get_settings())          # doctest: +SKIP
        >>> payload = client.analyze_applications(                  # doctest: +SKIP
        ...     start_date="2024-01-01", end_date="2024-01-31"
        ... )
    """

    settings: Settings
    session: requests.Session = field(default_factory=requests.Session)

    # populated in __post_init__
    _base_url: str = field(init=False, default="")
    _endpoint: str = field(init=False, default="")
    _timeout: int = field(init=False, default=60)
    _max_retries: int = field(init=False, default=5)
    _backoff: float = field(init=False, default=1.5)
    _retry_statuses: Tuple[int, ...] = field(init=False, default=(429, 500, 502, 503, 504))
    _throttle: float = field(init=False, default=0.35)
    _per_page: int = field(init=False, default=100)
    _last_request_at: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        api_cfg = self.settings.api
        self._base_url = self.settings.api_base_url
        self._endpoint = str(api_cfg.get("analyze_endpoint", "/v2/applications/analyze.json"))
        self._timeout = int(api_cfg.get("timeout_seconds", 60))
        self._max_retries = int(api_cfg.get("max_retries", 5))
        self._backoff = float(api_cfg.get("backoff_factor", 1.5))
        self._retry_statuses = tuple(api_cfg.get("retry_on_status", [429, 500, 502, 503, 504]))
        self._throttle = float(api_cfg.get("rate_limit_sleep_seconds", 0.35))
        self._per_page = int(api_cfg.get("per_page", 100))
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "aiesec-exchange-prediction/1.0"}
        )

    # -- internals ---------------------------------------------------------
    @property
    def _url(self) -> str:
        return f"{self._base_url}{self._endpoint}"

    def _respect_rate_limit(self) -> None:
        """Sleep just enough to keep a polite, constant request spacing."""
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self._throttle:
            time.sleep(self._throttle - elapsed)
        self._last_request_at = time.monotonic()

    def _sleep_for_retry(self, attempt: int, response: Optional[requests.Response]) -> None:
        """Wait before a retry, honouring ``Retry-After`` when present.

        Args:
            attempt: Zero-based attempt index, used for exponential growth.
            response: The failed response, if one was received.
        """
        delay: Optional[float] = None
        if response is not None:
            header = response.headers.get("Retry-After")
            if header:
                try:
                    delay = float(header)
                except ValueError:
                    delay = None

        if delay is None:
            # Exponential backoff with jitter to avoid synchronised retries.
            delay = (self._backoff**attempt) + random.uniform(0, 0.5)

        logger.warning("Retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, self._max_retries)
        time.sleep(delay)

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Issue one GET with retry/backoff and return the decoded JSON body.

        Args:
            params: Fully-formed query parameters, including ``access_token``.

        Returns:
            The parsed JSON response body.

        Raises:
            AuthenticationError: on 401/403.
            RateLimitError: when 429 persists past the retry budget.
            AiesecAPIError: on any other unrecoverable transport/decode failure.
        """
        token = str(params.get("access_token", ""))
        safe_url = redact_token(self._url, token)
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            self._respect_rate_limit()
            try:
                response = self.session.get(self._url, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Network error calling %s: %s", safe_url, exc)
                self._sleep_for_retry(attempt, None)
                continue

            # 402 is what this API actually returns for a *missing* token
            # (body: {"error":"Unauthorized"}); 401 is an invalid/expired one
            # (sub_code "invalid_token"); 403 is a valid token without rights to
            # the requested office. All three are credential problems, and
            # retrying any of them is pointless.
            if response.status_code in (401, 402, 403):
                raise AuthenticationError(
                    f"AIESEC API returned {response.status_code} for {safe_url}: "
                    f"{redact_token(response.text[:200], token)}\n"
                    "The access token is missing, expired, or lacks permission for "
                    "this office. Set a fresh AIESEC_ACCESS_TOKEN in .env and retry."
                )

            if response.status_code in self._retry_statuses:
                logger.warning(
                    "AIESEC API returned %s for %s", response.status_code, safe_url
                )
                last_error = AiesecAPIError(f"HTTP {response.status_code}")
                self._sleep_for_retry(attempt, response)
                continue

            if not response.ok:
                raise AiesecAPIError(
                    f"AIESEC API returned HTTP {response.status_code} for {safe_url}: "
                    f"{redact_token(response.text[:500], token)}"
                )

            try:
                return response.json()
            except ValueError as exc:
                raise AiesecAPIError(
                    f"Could not decode JSON from {safe_url}: "
                    f"{redact_token(response.text[:500], token)}"
                ) from exc

        if isinstance(last_error, AiesecAPIError) and "429" in str(last_error):
            raise RateLimitError(
                f"Rate limited by the AIESEC API after {self._max_retries} attempts. "
                "Increase api.rate_limit_sleep_seconds in config/config.yaml."
            )
        raise AiesecAPIError(
            f"Request to {safe_url} failed after {self._max_retries} attempts: {last_error}"
        )

    def _build_params(
        self,
        start_date: str,
        end_date: str,
        programme_id: Optional[int] = None,
        page: int = 1,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble the query string for one analyze call.

        Args:
            start_date: Inclusive window start, ``YYYY-MM-DD``.
            end_date: Inclusive window end, ``YYYY-MM-DD``.
            programme_id: Optional GIS programme id to scope the pull.
            page: 1-based page number.
            extra: Additional caller-supplied parameters.

        Returns:
            A parameter dict ready to hand to ``requests``.
        """
        namespace = str(self.settings.api.get("filter_namespace", "performance"))
        office_id = self.settings.require_office_id()

        params: Dict[str, Any] = {
            "access_token": self.settings.require_token(),
            "start_date": start_date,
            "end_date": end_date,
            f"{namespace}[office_id]": office_id,
            "page": page,
            "per_page": self._per_page,
        }

        if self.settings.api.get("include_child_offices", True):
            params[f"{namespace}[include_child_offices]"] = "true"

        if programme_id is not None:
            params[f"{namespace}[programmes][]"] = programme_id

        if extra:
            params.update(extra)
        return params

    # -- public API --------------------------------------------------------
    def analyze_applications(
        self,
        start_date: str,
        end_date: str,
        programme_id: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch every page of application analytics for one window.

        Args:
            start_date: Inclusive window start, ``YYYY-MM-DD``.
            end_date: Inclusive window end, ``YYYY-MM-DD``.
            programme_id: Optional GIS programme id.
            extra_params: Additional query parameters merged into the request.

        Returns:
            One decoded JSON body per page, in page order.
        """
        pages: List[Dict[str, Any]] = []
        page = 1

        while True:
            params = self._build_params(start_date, end_date, programme_id, page, extra_params)
            payload = self._request(params)
            pages.append(payload)

            if not self._has_next_page(payload, page):
                break
            page += 1

        logger.debug(
            "Fetched %d page(s) for %s..%s programme=%s",
            len(pages),
            start_date,
            end_date,
            programme_id,
        )
        return pages

    @staticmethod
    def _has_next_page(payload: Dict[str, Any], current_page: int) -> bool:
        """Decide whether another page should be requested.

        The Analytics API is not fully consistent about where it reports
        paging, so several known shapes are probed before giving up. Aggregation
        responses carry no paging block at all, which correctly yields ``False``.

        Args:
            payload: A decoded response body.
            current_page: The page number that produced ``payload``.

        Returns:
            ``True`` if at least one more page exists.
        """
        paging = payload.get("paging") or payload.get("meta") or {}
        if isinstance(paging, dict):
            total_pages = paging.get("total_pages") or paging.get("pages")
            if total_pages is not None:
                try:
                    return current_page < int(total_pages)
                except (TypeError, ValueError):
                    return False
            if "next_page" in paging:
                return bool(paging["next_page"])
        return False

    def check_connection(self) -> bool:
        """Probe the API with a one-day window to validate credentials early.

        Returns:
            ``True`` when the endpoint answers successfully.

        Raises:
            AuthenticationError: if the token is rejected.
            AiesecAPIError: on other transport failures.
        """
        self.analyze_applications("2025-01-01", "2025-01-01")
        logger.info("AIESEC Analytics API connection OK")
        return True


# ---------------------------------------------------------------------------
# Collection orchestration
# ---------------------------------------------------------------------------
@dataclass
class CollectionResult:
    """Outcome of a full collection run."""

    records: List[Dict[str, Any]]
    windows_attempted: int
    windows_succeeded: int
    windows_failed: List[Dict[str, str]]
    raw_path: Optional[Path] = None

    @property
    def success_rate(self) -> float:
        """Fraction of windows that returned data, in ``[0, 1]``."""
        if not self.windows_attempted:
            return 0.0
        return self.windows_succeeded / self.windows_attempted


def collect_exchange_data(
    settings: Optional[Settings] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save: bool = True,
    fail_fast: bool = False,
) -> CollectionResult:
    """Pull MC India application analytics month by month and persist the raw JSON.

    Each ``(month, programme)`` combination is requested separately. A failure in
    one window is logged and recorded but does not abort the run unless
    ``fail_fast`` is set - a four-year backfill should not be lost to one
    transient 502.

    Args:
        settings: Loaded settings. Loaded from disk when omitted.
        start_date: Override for ``collection.start_date``.
        end_date: Override for ``collection.end_date``.
        save: Write the envelope to ``data/raw/api_responses.json``.
        fail_fast: Re-raise the first window failure instead of continuing.

    Returns:
        A :class:`CollectionResult` describing what was fetched.

    Raises:
        AuthenticationError: if credentials are rejected (always fatal).
    """
    settings = settings or get_settings()
    client = AiesecAnalyticsClient(settings)

    window_start = start_date or str(settings.collection.get("start_date", "2022-01-01"))
    window_end = end_date or str(settings.collection.get("end_date", "2025-12-31"))
    windows = month_windows(window_start, window_end)

    programme_mapping = settings.programme_mapping
    active_products = set(settings.products)
    programmes: List[Tuple[Optional[int], str]] = [
        (pid, name) for pid, name in programme_mapping.items() if name in active_products
    ] or [(None, "ALL")]

    logger.info(
        "Collecting %s: %d month(s) x %d programme(s) = %d request group(s)",
        settings.mc_name,
        len(windows),
        len(programmes),
        len(windows) * len(programmes),
    )

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    attempted = 0
    succeeded = 0

    for win_start, win_end in windows:
        for programme_id, product in programmes:
            attempted += 1
            try:
                pages = client.analyze_applications(
                    start_date=win_start.isoformat(),
                    end_date=win_end.isoformat(),
                    programme_id=programme_id,
                )
            except AuthenticationError:
                raise  # credentials are broken - continuing is pointless
            except AiesecAPIError as exc:
                logger.error(
                    "Window %s..%s programme=%s failed: %s",
                    win_start,
                    win_end,
                    product,
                    exc,
                )
                failures.append(
                    {
                        "start_date": win_start.isoformat(),
                        "end_date": win_end.isoformat(),
                        "product": product,
                        "error": str(exc),
                    }
                )
                if fail_fast:
                    raise
                continue

            succeeded += 1
            records.append(
                {
                    "start_date": win_start.isoformat(),
                    "end_date": win_end.isoformat(),
                    "month": win_start.strftime("%Y-%m"),
                    "programme_id": programme_id,
                    "product": product,
                    "office_id": settings.office_id,
                    "pages": pages,
                }
            )
            logger.info(
                "  fetched %s %-4s (%d page(s))", win_start.strftime("%Y-%m"), product, len(pages)
            )

    result = CollectionResult(
        records=records,
        windows_attempted=attempted,
        windows_succeeded=succeeded,
        windows_failed=failures,
    )

    if save:
        result.raw_path = save_raw_responses(records, settings, failures)

    logger.info(
        "Collection finished: %d/%d windows OK (%.0f%%)",
        succeeded,
        attempted,
        result.success_rate * 100,
    )
    return result


def save_raw_responses(
    records: List[Dict[str, Any]],
    settings: Settings,
    failures: Optional[List[Dict[str, str]]] = None,
) -> Path:
    """Persist raw API payloads with a provenance envelope.

    The envelope records *when*, *what* and *how* the data was pulled so that a
    reviewer can audit the dataset without rerunning the collection.

    Args:
        records: Per-window raw payload records.
        settings: Loaded settings (used for paths and metadata).
        failures: Windows that failed, recorded for transparency.

    Returns:
        The path the envelope was written to.
    """
    output_path = settings.paths.raw_responses
    output_path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "metadata": {
            "source": "AIESEC Analytics API",
            "endpoint": f"{settings.api_base_url}"
            f"{settings.api.get('analyze_endpoint', '/v2/applications/analyze.json')}",
            "mc_name": settings.mc_name,
            "office_id": settings.office_id,
            "collection_start": settings.collection.get("start_date"),
            "collection_end": settings.collection.get("end_date"),
            "collected_at": datetime.now().astimezone().isoformat(),
            "record_count": len(records),
            "failed_windows": failures or [],
            "is_reference_data": False,
        },
        "records": records,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=2, ensure_ascii=False)

    logger.info("Raw API responses written to %s", output_path)
    return output_path


def load_raw_responses(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Read the saved raw-response envelope back from disk.

    Args:
        settings: Loaded settings. Loaded from disk when omitted.

    Returns:
        The full envelope, including its ``metadata`` block.

    Raises:
        FileNotFoundError: if no raw responses have been collected yet.
    """
    settings = settings or get_settings()
    path = settings.paths.raw_responses

    if not path.exists():
        raise FileNotFoundError(
            f"No raw API responses at {path}. Run the collection step first "
            "(python run_pipeline.py --step collect)."
        )

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
