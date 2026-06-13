"""Configuration for production SLM provider routing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from openharness.orchestration.subroutine_models import HF_SUBROUTINE_MODELS
from openharness.orchestration.types import TaskType

try:  # pragma: no cover - exercised only on Python <3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


SlmBackend = Literal["auto", "deterministic", "local_transformers", "local_onnx", "remote_http", "fake"]
FallbackPolicy = Literal[
    "deterministic_first",
    "model_first",
    "model_only",
    "deterministic_only",
]

SUBROUTINE_TASK_TYPES: tuple[TaskType, ...] = (
    "json_repair",
    "trace_localize",
    "search_query",
    "search_rank",
    "failure_classify",
    "patch_risk",
)


class SlmDefaults(BaseModel):
    """Default execution settings shared by subroutine model routes."""

    backend: SlmBackend = "auto"
    timeout_ms: int = Field(default=30_000, ge=1)
    concurrency: int = Field(default=1, ge=1)
    cost_per_call_usd: float = Field(default=0.0, ge=0.0)
    reliability: float = Field(default=0.82, ge=0.0, le=1.0)
    fallback_policy: FallbackPolicy = "deterministic_first"
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    model_path: str | None = None
    remote_url: str | None = None


class SlmRouteConfig(BaseModel):
    """Execution settings for one typed SLM subroutine."""

    task_type: TaskType
    model_id: str
    backend: SlmBackend = "auto"
    timeout_ms: int = Field(default=30_000, ge=1)
    concurrency: int = Field(default=1, ge=1)
    cost_per_call_usd: float = Field(default=0.0, ge=0.0)
    reliability: float = Field(default=0.82, ge=0.0, le=1.0)
    fallback_policy: FallbackPolicy = "deterministic_first"
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    model_path: str | None = None
    remote_url: str | None = None
    enabled: bool = True


class SlmHarnessConfig(BaseModel):
    """Resolved SLM harness config after file and environment overrides."""

    config_path: str | None = None
    defaults: SlmDefaults = Field(default_factory=SlmDefaults)
    subroutines: dict[str, SlmRouteConfig]

    def route_for(self, task_type: TaskType) -> SlmRouteConfig:
        """Return the configured route for a task type."""
        key = str(task_type)
        route = self.subroutines.get(key)
        if route is not None:
            return route
        return build_route_config(task_type, self.defaults)


def load_slm_harness_config(*, cwd: str | Path | None = None) -> SlmHarnessConfig:
    """Load `.slm-harness.toml`, then apply environment overrides."""
    config_path = _resolve_config_path(cwd=cwd)
    raw = _load_toml(config_path) if config_path is not None else {}
    defaults = _defaults_from_raw(raw.get("defaults", {}))
    routes: dict[str, SlmRouteConfig] = {
        task_type: build_route_config(task_type, defaults)
        for task_type in SUBROUTINE_TASK_TYPES
    }

    raw_subroutines = raw.get("subroutines", {})
    if isinstance(raw_subroutines, dict):
        for key, value in raw_subroutines.items():
            if key in routes and isinstance(value, dict):
                routes[key] = _merge_route(routes[key], value)

    config = SlmHarnessConfig(
        config_path=str(config_path) if config_path is not None else None,
        defaults=defaults,
        subroutines=routes,
    )
    return _apply_env_overrides(config)


def build_route_config(task_type: TaskType, defaults: SlmDefaults | None = None) -> SlmRouteConfig:
    """Build a route config from defaults and the known model catalog."""
    resolved = defaults or SlmDefaults()
    model_id = HF_SUBROUTINE_MODELS.get(str(task_type), f"local/{task_type}-rules-v1")
    return SlmRouteConfig(
        task_type=task_type,
        model_id=model_id,
        backend=resolved.backend,
        timeout_ms=resolved.timeout_ms,
        concurrency=resolved.concurrency,
        cost_per_call_usd=resolved.cost_per_call_usd,
        reliability=resolved.reliability,
        fallback_policy=resolved.fallback_policy,
        min_confidence=resolved.min_confidence,
        model_path=resolved.model_path,
        remote_url=resolved.remote_url,
    )


def slm_config_status(config: SlmHarnessConfig | None = None) -> dict[str, Any]:
    """Return JSON-ready model routing status for health and MCP probes."""
    resolved = config or load_slm_harness_config()
    return {
        "config_path": resolved.config_path,
        "defaults": resolved.defaults.model_dump(mode="json"),
        "subroutines": {
            name: _route_status(route) for name, route in sorted(resolved.subroutines.items())
        },
    }


def _route_status(route: SlmRouteConfig) -> dict[str, Any]:
    model_path = Path(route.model_path).expanduser() if route.model_path else None
    return {
        **route.model_dump(mode="json"),
        "model_path_exists": bool(model_path and model_path.exists()),
        "remote_configured": bool(route.remote_url),
    }


def _resolve_config_path(*, cwd: str | Path | None) -> Path | None:
    env_path = os.getenv("SLM_HARNESS_CONFIG")
    if env_path:
        path = Path(env_path).expanduser()
        return path if path.exists() else None

    start = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd()
    candidates = [start, *start.parents]
    for directory in candidates:
        path = directory / ".slm-harness.toml"
        if path.exists():
            return path
    return None


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        parsed = tomllib.load(handle)
    return parsed if isinstance(parsed, dict) else {}


def _defaults_from_raw(raw: Any) -> SlmDefaults:
    if not isinstance(raw, dict):
        return SlmDefaults()
    return SlmDefaults(**_known_fields(raw, SlmDefaults))


def _merge_route(route: SlmRouteConfig, raw: dict[str, Any]) -> SlmRouteConfig:
    payload = route.model_dump()
    payload.update(_known_fields(raw, SlmRouteConfig))
    payload["task_type"] = route.task_type
    return SlmRouteConfig(**payload)


def _known_fields(raw: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if key in model.model_fields}


def _apply_env_overrides(config: SlmHarnessConfig) -> SlmHarnessConfig:
    global_updates = _env_updates("SLM_HARNESS")
    routes: dict[str, SlmRouteConfig] = {}
    for key, route in config.subroutines.items():
        updates = dict(global_updates)
        updates.update(_env_updates(f"SLM_HARNESS_{key.upper()}"))
        if updates:
            routes[key] = route.model_copy(update=updates)
        else:
            routes[key] = route
    defaults = config.defaults.model_copy(update=global_updates) if global_updates else config.defaults
    return config.model_copy(update={"defaults": defaults, "subroutines": routes})


def _env_updates(prefix: str) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    _maybe_set(updates, "backend", _env_backend(os.getenv(f"{prefix}_BACKEND")))
    _maybe_set(updates, "model_path", os.getenv(f"{prefix}_MODEL_PATH") or os.getenv(f"{prefix}_MODEL_DIR"))
    _maybe_set(updates, "remote_url", os.getenv(f"{prefix}_REMOTE_URL"))
    _maybe_set(updates, "model_id", os.getenv(f"{prefix}_MODEL_ID"))
    _maybe_set(updates, "timeout_ms", _env_int(os.getenv(f"{prefix}_TIMEOUT_MS")))
    _maybe_set(updates, "concurrency", _env_int(os.getenv(f"{prefix}_CONCURRENCY")))
    _maybe_set(updates, "cost_per_call_usd", _env_float(os.getenv(f"{prefix}_COST_PER_CALL_USD")))
    _maybe_set(updates, "reliability", _env_float(os.getenv(f"{prefix}_RELIABILITY")))
    _maybe_set(updates, "min_confidence", _env_float(os.getenv(f"{prefix}_MIN_CONFIDENCE")))
    fallback = os.getenv(f"{prefix}_FALLBACK_POLICY")
    if fallback in {"deterministic_first", "model_first", "model_only", "deterministic_only"}:
        updates["fallback_policy"] = fallback
    return updates


def _maybe_set(updates: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        updates[key] = value


def _env_backend(value: str | None) -> SlmBackend | None:
    if value in {"auto", "deterministic", "local_transformers", "local_onnx", "remote_http", "fake"}:
        return cast(SlmBackend, value)
    return None


def _env_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
