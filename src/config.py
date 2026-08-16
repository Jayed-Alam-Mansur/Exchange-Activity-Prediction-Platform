"""Configuration and secret loading.

Single source of truth for every path, tunable and credential in the project.

Design rules enforced here:
  * No path is hardcoded anywhere else in the codebase - everything resolves
    through :class:`Settings`.
  * No secret ever appears in a config file. Secrets come from the process
    environment, optionally seeded from a git-ignored ``.env``.
  * Environment variables always win over ``config/config.yaml`` so that the
    same code runs unchanged in local, CI and container environments.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = ["Settings", "get_settings", "configure_logging", "ProjectPaths"]

# Project root = parent of the ``src`` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------
def load_dotenv_file(path: Optional[Path] = None) -> None:
    """Seed ``os.environ`` from a ``.env`` file without overwriting real env vars.

    Uses ``python-dotenv`` when available and falls back to a small built-in
    parser otherwise, so the project never hard-fails on a missing optional
    dependency.

    Args:
        path: Explicit ``.env`` location. Defaults to ``<project root>/.env``.
    """
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except ImportError:  # pragma: no cover - exercised only without python-dotenv
        logger.debug("python-dotenv not installed; using built-in .env parser")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Path container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectPaths:
    """Absolute paths for every artefact the pipeline reads or writes."""

    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    raw_responses: Path
    committees: Path
    processed_dataset: Path
    monthly_series: Path
    features: Path
    models_dir: Path
    trained_model: Path
    outputs_dir: Path
    predictions: Path
    figures_dir: Path
    reports_dir: Path

    def ensure_directories(self) -> None:
        """Create every directory the pipeline writes into. Idempotent."""
        for directory in (
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.models_dir,
            self.outputs_dir,
            self.figures_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    """Typed view over ``config/config.yaml`` plus environment secrets."""

    raw: Dict[str, Any]
    paths: ProjectPaths
    config_path: Path

    # --- secrets / env-driven values (never persisted to disk) --------------
    access_token: Optional[str] = field(default=None, repr=False)
    office_id: Optional[int] = None
    api_base_url: str = ""

    # ---------------- convenience accessors ----------------
    @property
    def project(self) -> Dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def api(self) -> Dict[str, Any]:
        return self.raw.get("api", {})

    @property
    def collection(self) -> Dict[str, Any]:
        return self.raw.get("collection", {})

    @property
    def modeling(self) -> Dict[str, Any]:
        return self.raw.get("modeling", {})

    @property
    def activity_levels(self) -> Dict[str, Any]:
        return self.raw.get("activity_levels", {})

    @property
    def reference_data(self) -> Dict[str, Any]:
        return self.raw.get("reference_data", {})

    @property
    def mc_name(self) -> str:
        return str(self.project.get("mc_name", "AIESEC in India"))

    @property
    def funnel_stages(self) -> List[str]:
        return list(self.raw.get("funnel", {}).get("stages", []))

    @property
    def funnel_labels(self) -> Dict[str, str]:
        return dict(self.raw.get("funnel", {}).get("labels", {}))

    @property
    def api_status_map(self) -> Dict[str, str]:
        return dict(self.raw.get("funnel", {}).get("api_status_map", {}))

    @property
    def non_stage_statuses(self) -> Dict[str, str]:
        """API statuses that are drop-outs/side-channels, not funnel stages."""
        return dict(self.raw.get("funnel", {}).get("non_stage_statuses", {}))

    @property
    def committees_endpoint(self) -> str:
        return str(
            self.api.get(
                "committees_endpoint", "https://gis-api.aiesec.org/v2/committees/{id}.json"
            )
        )

    @property
    def products(self) -> List[str]:
        return list(self.raw.get("products", {}).get("active", []))

    @property
    def programme_mapping(self) -> Dict[int, str]:
        mapping = self.raw.get("products", {}).get("mapping", {}) or {}
        return {int(k): str(v) for k, v in mapping.items()}

    @property
    def directions(self) -> List[str]:
        return list(self.raw.get("directions", []))

    @property
    def entities(self) -> List[str]:
        return list(self.reference_data.get("entities", []))

    @property
    def target(self) -> str:
        return str(self.modeling.get("target", "APP"))

    @property
    def forecast_year(self) -> int:
        return int(self.modeling.get("forecast_year", 2026))

    @property
    def horizon(self) -> int:
        return int(self.modeling.get("horizon", 12))

    @property
    def random_state(self) -> int:
        return int(self.modeling.get("random_state", 42))

    @property
    def has_credentials(self) -> bool:
        """True when a non-empty access token is present in the environment."""
        return bool(self.access_token and self.access_token.strip())

    def require_token(self) -> str:
        """Return the access token or raise a clear, actionable error.

        Raises:
            RuntimeError: if ``AIESEC_ACCESS_TOKEN`` is unset or empty.
        """
        if not self.has_credentials:
            raise RuntimeError(
                "AIESEC_ACCESS_TOKEN is not set. Copy .env.example to .env and "
                "provide a valid AIESEC GIS access token, or run the pipeline "
                "with --use-reference-data to use the offline reference dataset."
            )
        return str(self.access_token)

    def require_office_id(self) -> int:
        """Return the MC office id or raise a clear, actionable error.

        Raises:
            RuntimeError: if the office id is configured nowhere.
        """
        if self.office_id is None:
            raise RuntimeError(
                "No office_id configured. Set AIESEC_OFFICE_ID in .env or "
                "api.office_id in config/config.yaml to the EXPA office id of "
                f"{self.mc_name}."
            )
        return int(self.office_id)


def _build_paths(root: Path, path_cfg: Dict[str, str]) -> ProjectPaths:
    """Resolve every configured relative path against the project root."""

    def resolve(key: str, default: str) -> Path:
        return root / path_cfg.get(key, default)

    return ProjectPaths(
        root=root,
        data_raw=resolve("data_raw", "data/raw"),
        data_interim=resolve("data_interim", "data/interim"),
        data_processed=resolve("data_processed", "data/processed"),
        raw_responses=resolve("raw_responses", "data/raw/api_responses.json"),
        committees=resolve("committees", "data/raw/committees.json"),
        processed_dataset=resolve("processed_dataset", "data/processed/exchange_data.csv"),
        monthly_series=resolve("monthly_series", "data/processed/monthly_applications.csv"),
        features=resolve("features", "data/processed/features.csv"),
        models_dir=resolve("models_dir", "models"),
        trained_model=resolve("trained_model", "models/trained_model.pkl"),
        outputs_dir=resolve("outputs_dir", "outputs"),
        predictions=resolve("predictions", "outputs/predictions_2026.csv"),
        figures_dir=resolve("figures_dir", "outputs/figures"),
        reports_dir=resolve("reports_dir", "outputs/reports"),
    )


def _coerce_office_id(value: Any) -> Optional[int]:
    """Best-effort conversion of an office id to ``int``; ``None`` when blank."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        logger.warning("Ignoring non-numeric office id %r", value)
        return None


def get_settings(config_path: Optional[Path | str] = None) -> Settings:
    """Load configuration from YAML, then overlay environment variables.

    Args:
        config_path: Optional override for ``config/config.yaml``.

    Returns:
        A fully resolved :class:`Settings` instance with directories created.

    Raises:
        FileNotFoundError: if the configuration file does not exist.
        ValueError: if the configuration file is not a YAML mapping.
    """
    load_dotenv_file()

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping")

    paths = _build_paths(PROJECT_ROOT, raw.get("paths", {}) or {})
    paths.ensure_directories()

    # Environment overrides config; config provides the default.
    office_id = _coerce_office_id(
        os.environ.get("AIESEC_OFFICE_ID") or raw.get("api", {}).get("office_id")
    )
    base_url = (
        os.environ.get("AIESEC_API_BASE_URL")
        or raw.get("api", {}).get("base_url")
        or "https://analytics.api.aiesec.org"
    ).rstrip("/")

    return Settings(
        raw=raw,
        paths=paths,
        config_path=path,
        access_token=os.environ.get("AIESEC_ACCESS_TOKEN"),
        office_id=office_id,
        api_base_url=base_url,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging(settings: Optional[Settings] = None, level: Optional[str] = None) -> None:
    """Configure root logging once, from config with an env override.

    Precedence: explicit ``level`` argument > ``LOG_LEVEL`` env > config file.

    Args:
        settings: Loaded settings (optional - falls back to sane defaults).
        level: Explicit level name, e.g. ``"DEBUG"``.
    """
    log_cfg = (settings.raw.get("logging", {}) if settings else {}) or {}
    resolved = (level or os.environ.get("LOG_LEVEL") or log_cfg.get("level") or "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format=log_cfg.get(
            "format", "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
        ),
        datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
        force=True,
    )
    # Third-party libraries are noisy at DEBUG; keep them at WARNING.
    for noisy in ("urllib3", "matplotlib", "requests", "cmdstanpy", "prophet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
