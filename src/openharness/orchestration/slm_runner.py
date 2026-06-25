"""SLM provider runner abstraction for local and cloud subroutine execution."""

from __future__ import annotations

import json
import importlib
import re
import time
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from openharness.orchestration.slm_config import (
    SlmBackend,
    SlmHarnessConfig,
    SlmRouteConfig,
    load_slm_harness_config,
)
from openharness.orchestration.subroutine_models import (
    HF_SUBROUTINE_MODELS,
    SubroutineRun,
    classify_failure_subroutine,
    classify_patch_risk_subroutine,
    generate_search_queries_subroutine,
    localize_traceback_subroutine,
    rank_search_hits_subroutine,
    repair_json_subroutine,
)
from openharness.orchestration.types import TaskType


class SlmRunRequest(BaseModel):
    """Request sent from an executor to a local or cloud SLM backend."""

    model_config = ConfigDict(populate_by_name=True)

    task_type: TaskType
    input: Any
    output_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    model_id: str | None = None
    backend: SlmBackend = "auto"
    timeout_ms: int = Field(default=30_000, ge=1)
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlmRunResponse(BaseModel):
    """Normalized response returned by every SLM backend."""

    output: Any = ""
    schema_valid: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    backend: SlmBackend
    model_id: str
    raw_text: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlmRunner(Protocol):
    """Common interface for deterministic, local, remote, and test runners."""

    async def run(self, request: SlmRunRequest) -> SlmRunResponse:
        """Run one typed SLM subroutine."""


class DeterministicSlmRunner:
    """Runs the deterministic baseline for a subroutine."""

    backend: SlmBackend = "deterministic"

    async def run(self, request: SlmRunRequest) -> SlmRunResponse:
        started = time.perf_counter()
        run = _run_builtin_subroutine(request, allow_slm=False)
        return _response_from_subroutine_run(
            run,
            backend=self.backend,
            started=started,
            cost_usd=0.0,
            schema=request.output_schema,
        )


class FakeSlmRunner:
    """Deterministic fake model runner for CI and no-model local development."""

    backend: SlmBackend = "fake"

    async def run(self, request: SlmRunRequest) -> SlmRunResponse:
        started = time.perf_counter()
        run = _run_builtin_subroutine(request, allow_slm=True)
        response = _response_from_subroutine_run(
            run,
            backend=self.backend,
            started=started,
            cost_usd=0.0,
            schema=request.output_schema,
        )
        response.metadata["synthetic_runner"] = True
        return response


class RemoteHttpSlmRunner:
    """HTTP runner for hosted vLLM/SLM endpoints."""

    backend: SlmBackend = "remote_http"

    def __init__(self, route: SlmRouteConfig) -> None:
        self.route = route

    async def run(self, request: SlmRunRequest) -> SlmRunResponse:
        started = time.perf_counter()
        if not self.route.remote_url:
            return _error_response(
                request,
                backend=self.backend,
                route=self.route,
                started=started,
                error="remote_http backend requires remote_url",
            )
        payload = {
            "task_type": request.task_type,
            "input": request.input,
            "schema": request.output_schema,
            "model": request.model_id or self.route.model_id,
            "trace_id": request.trace_id,
            "metadata": request.metadata,
        }
        timeout = httpx.Timeout(request.timeout_ms / 1000.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.route.remote_url, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return _error_response(
                request,
                backend=self.backend,
                route=self.route,
                started=started,
                error=str(exc),
            )
        return _response_from_remote_payload(
            data,
            request=request,
            route=self.route,
            started=started,
        )


class LocalTransformersSlmRunner:
    """Optional local Hugging Face causal-LM runner."""

    backend: SlmBackend = "local_transformers"

    def __init__(self, route: SlmRouteConfig) -> None:
        self.route = route
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    async def run(self, request: SlmRunRequest) -> SlmRunResponse:
        started = time.perf_counter()
        try:
            self._ensure_loaded()
            tokenizer = self._tokenizer
            model = self._model
            torch = self._torch
            if tokenizer is None or model is None or torch is None:
                raise RuntimeError("local_transformers backend failed to load")
            prompt = _render_prompt(request)
            encoded = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(**encoded, max_new_tokens=256, do_sample=False)
            raw_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            raw_text = raw_text[len(prompt) :] if raw_text.startswith(prompt) else raw_text
            output = _parse_model_output(raw_text)
            schema_valid = _schema_valid(output, request.output_schema)
            return SlmRunResponse(
                output=output,
                schema_valid=schema_valid,
                confidence=0.78 if schema_valid else 0.32,
                latency_ms=_elapsed_ms(started),
                cost_usd=self.route.cost_per_call_usd,
                backend=self.backend,
                model_id=request.model_id or self.route.model_id,
                raw_text=raw_text,
                error=None if schema_valid else "model output did not satisfy schema",
                metadata={"model_path": self.route.model_path},
            )
        except Exception as exc:
            return _error_response(
                request,
                backend=self.backend,
                route=self.route,
                started=started,
                error=str(exc),
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        model_ref = self.route.model_path or self.route.model_id
        if not model_ref:
            raise RuntimeError("local_transformers backend requires model_path or model_id")
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for local_transformers") from exc
        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_ref)
        self._model = AutoModelForCausalLM.from_pretrained(model_ref)
        self._model.to(device)
        self._model.eval()


class LocalOnnxSlmRunner:
    """Optional local ONNX runner slot for tiny classifiers/generators."""

    backend: SlmBackend = "local_onnx"

    def __init__(self, route: SlmRouteConfig) -> None:
        self.route = route

    async def run(self, request: SlmRunRequest) -> SlmRunResponse:
        started = time.perf_counter()
        if not self.route.model_path:
            return _error_response(
                request,
                backend=self.backend,
                route=self.route,
                started=started,
                error="local_onnx backend requires model_path",
            )
        try:
            importlib.import_module("onnxruntime")
        except ImportError as exc:
            return _error_response(
                request,
                backend=self.backend,
                route=self.route,
                started=started,
                error=f"onnxruntime is required for local_onnx: {exc}",
            )
        return _error_response(
            request,
            backend=self.backend,
            route=self.route,
            started=started,
            error="local_onnx model IO binding is not configured for this subroutine",
        )


class ConfiguredSlmRunner:
    """Config-aware runner that applies deterministic/model fallback policy."""

    def __init__(self, config: SlmHarnessConfig | None = None) -> None:
        self.config = config or load_slm_harness_config()

    async def run(self, request: SlmRunRequest) -> SlmRunResponse:
        route = self.config.route_for(request.task_type)
        request = request.model_copy(
            update={
                "model_id": request.model_id or route.model_id,
                "timeout_ms": request.timeout_ms or route.timeout_ms,
            }
        )
        attempts: list[dict[str, Any]] = []

        if route.fallback_policy in {"deterministic_first", "deterministic_only"}:
            deterministic = await DeterministicSlmRunner().run(request)
            attempts.append(_attempt_from_response(deterministic))
            if _acceptable(deterministic, route) or route.fallback_policy == "deterministic_only":
                deterministic.metadata["runner_attempts"] = attempts
                return deterministic

        target_backend = _select_backend(request.backend, route)
        model_response = await _runner_for_backend(target_backend, route).run(
            request.model_copy(update={"backend": target_backend})
        )
        attempts.append(_attempt_from_response(model_response))
        if _acceptable(model_response, route) or route.fallback_policy == "model_only":
            model_response.metadata["runner_attempts"] = attempts
            return model_response

        if route.fallback_policy == "model_first":
            deterministic = await DeterministicSlmRunner().run(request)
            attempts.append(_attempt_from_response(deterministic))
            deterministic.metadata["runner_attempts"] = attempts
            return deterministic

        model_response.metadata["runner_attempts"] = attempts
        return model_response


async def run_slm_subroutine(
    request: SlmRunRequest,
    *,
    config: SlmHarnessConfig | None = None,
) -> SlmRunResponse:
    """Run a typed subroutine through the configured provider stack."""
    return await ConfiguredSlmRunner(config=config).run(request)


def slm_response_to_subroutine_output(
    task_type: TaskType,
    response: SlmRunResponse,
) -> dict[str, Any]:
    """Render an SLM runner response in the legacy subroutine output shape."""
    success = response.error is None and response.schema_valid
    return {
        "subroutine": str(task_type),
        "output": response.output,
        "success": success,
        "schema_valid": response.schema_valid,
        "verifier_pass": success,
        "fallback_path": response.metadata.get("fallback_path", response.backend),
        "model_id": response.model_id,
        "local": response.backend != "remote_http",
        "estimated_cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
        "error": response.error,
        "attempts": response.metadata.get("runner_attempts", response.metadata.get("attempts", [])),
        "backend": response.backend,
        "raw_text": response.raw_text,
    }


def _run_builtin_subroutine(request: SlmRunRequest, *, allow_slm: bool) -> SubroutineRun:
    payload = request.input
    task_type = request.task_type
    if task_type == "json_repair":
        raw = str(payload.get("raw", payload) if isinstance(payload, dict) else payload)
        return repair_json_subroutine(raw, allow_slm=allow_slm)
    if task_type == "trace_localize":
        if isinstance(payload, dict):
            traceback = str(payload.get("traceback", ""))
            project_prefix = str(payload.get("project_prefix", ""))
        else:
            traceback = str(payload)
            project_prefix = ""
        return localize_traceback_subroutine(traceback, project_prefix=project_prefix)
    if task_type == "search_query":
        task = str(payload.get("task", payload) if isinstance(payload, dict) else payload)
        return generate_search_queries_subroutine(task)
    if task_type == "search_rank":
        if isinstance(payload, dict):
            query = str(payload.get("query", ""))
            hits_raw = payload.get("hits", [])
            hits = hits_raw if isinstance(hits_raw, list) else []
        else:
            query = str(payload)
            hits = []
        return rank_search_hits_subroutine(query, hits)
    if task_type == "failure_classify":
        text = str(payload.get("text", payload) if isinstance(payload, dict) else payload)
        return classify_failure_subroutine(text)
    if task_type == "patch_risk":
        if isinstance(payload, dict):
            diff = str(payload.get("diff", ""))
            tests = str(payload.get("tests", ""))
        else:
            diff = str(payload)
            tests = ""
        return classify_patch_risk_subroutine(diff, tests=tests)
    return SubroutineRun(
        subroutine=str(task_type),
        output={},
        success=False,
        schema_valid=False,
        verifier_pass=False,
        fallback_path="unsupported_task_type",
        model_id=HF_SUBROUTINE_MODELS.get(str(task_type), "unknown"),
        error=f"unsupported SLM task type: {task_type}",
    )


def _response_from_subroutine_run(
    run: SubroutineRun,
    *,
    backend: SlmBackend,
    started: float,
    cost_usd: float,
    schema: dict[str, Any] | None,
) -> SlmRunResponse:
    schema_valid = run.schema_valid and _schema_valid(run.output, schema)
    return SlmRunResponse(
        output=run.output,
        schema_valid=schema_valid,
        confidence=_confidence_for_run(run),
        latency_ms=max(run.latency_ms, _elapsed_ms(started)),
        cost_usd=cost_usd,
        backend=backend,
        model_id=run.model_id,
        raw_text=json.dumps(run.output, ensure_ascii=False, sort_keys=True),
        error=run.error if not schema_valid else None,
        metadata={
            "fallback_path": run.fallback_path,
            "attempts": run.attempts,
            "success": run.success,
            "verifier_pass": run.verifier_pass,
        },
    )


def _response_from_remote_payload(
    data: Any,
    *,
    request: SlmRunRequest,
    route: SlmRouteConfig,
    started: float,
) -> SlmRunResponse:
    if isinstance(data, dict) and {"output", "schema_valid", "confidence"}.issubset(data):
        output = data.get("output")
        schema_valid = bool(data.get("schema_valid")) and _schema_valid(output, request.output_schema)
        return SlmRunResponse(
            output=output,
            schema_valid=schema_valid,
            confidence=_bounded_float(data.get("confidence"), 0.0),
            latency_ms=_elapsed_ms(started),
            cost_usd=_bounded_cost(data.get("cost_usd"), route.cost_per_call_usd),
            backend="remote_http",
            model_id=str(data.get("model_id") or request.model_id or route.model_id),
            raw_text=str(data.get("raw_text") or ""),
            error=str(data["error"]) if data.get("error") else None,
            metadata={"remote_payload_shape": "slm_run_response"},
        )

    raw_text = _remote_text(data)
    output = _parse_model_output(raw_text)
    schema_valid = _schema_valid(output, request.output_schema)
    return SlmRunResponse(
        output=output,
        schema_valid=schema_valid,
        confidence=0.78 if schema_valid else 0.28,
        latency_ms=_elapsed_ms(started),
        cost_usd=route.cost_per_call_usd,
        backend="remote_http",
        model_id=request.model_id or route.model_id,
        raw_text=raw_text,
        error=None if schema_valid else "remote model output did not satisfy schema",
        metadata={"remote_payload_shape": "text_generation"},
    )


def _runner_for_backend(backend: SlmBackend, route: SlmRouteConfig) -> SlmRunner:
    if backend == "deterministic":
        return DeterministicSlmRunner()
    if backend == "remote_http":
        return RemoteHttpSlmRunner(route)
    if backend == "local_transformers":
        return LocalTransformersSlmRunner(route)
    if backend == "local_onnx":
        return LocalOnnxSlmRunner(route)
    return FakeSlmRunner()


def _select_backend(requested: SlmBackend, route: SlmRouteConfig) -> SlmBackend:
    if requested != "auto":
        return requested
    if route.backend != "auto":
        return route.backend
    if route.remote_url:
        return "remote_http"
    if route.model_path:
        suffix = Path(route.model_path).suffix.lower()
        return "local_onnx" if suffix == ".onnx" else "local_transformers"
    return "fake"


def _acceptable(response: SlmRunResponse, route: SlmRouteConfig) -> bool:
    return response.error is None and response.schema_valid and response.confidence >= route.min_confidence


def _attempt_from_response(response: SlmRunResponse) -> dict[str, Any]:
    return {
        "backend": response.backend,
        "ok": response.error is None and response.schema_valid,
        "confidence": response.confidence,
        "latency_ms": response.latency_ms,
        "error": response.error,
        "fallback_path": response.metadata.get("fallback_path"),
    }


def _error_response(
    request: SlmRunRequest,
    *,
    backend: SlmBackend,
    route: SlmRouteConfig,
    started: float,
    error: str,
) -> SlmRunResponse:
    return SlmRunResponse(
        output={},
        schema_valid=False,
        confidence=0.0,
        latency_ms=_elapsed_ms(started),
        cost_usd=route.cost_per_call_usd,
        backend=backend,
        model_id=request.model_id or route.model_id,
        raw_text="",
        error=error,
    )


def _render_prompt(request: SlmRunRequest) -> str:
    return (
        "Return only JSON matching the requested schema.\n"
        f"TASK_TYPE: {request.task_type}\n"
        f"SCHEMA: {json.dumps(request.output_schema or {}, sort_keys=True)}\n"
        f"INPUT: {json.dumps(request.input, ensure_ascii=False, sort_keys=True)}\n"
        "JSON:"
    )


def _parse_model_output(raw_text: str) -> Any:
    text = raw_text.strip()
    parsed = _try_json(text)
    if parsed is not None:
        return parsed
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        parsed = _try_json(match.group(0))
        if parsed is not None:
            return parsed
    return {"text": text} if text else {}


def _remote_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return str(message["content"])
            if isinstance(first.get("text"), str):
                return str(first["text"])
    if isinstance(data.get("generated_text"), str):
        return str(data["generated_text"])
    if isinstance(data.get("text"), str):
        return str(data["text"])
    return json.dumps(data, ensure_ascii=False)


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _schema_valid(output: Any, schema: dict[str, Any] | None) -> bool:
    if not isinstance(output, dict):
        return False
    if not schema:
        return True
    if schema.get("type") == "object" and not isinstance(output, dict):
        return False
    required = schema.get("required", [])
    if isinstance(required, list):
        if not all(isinstance(key, str) and key in output for key in required):
            return False
    props = schema.get("properties") or {}
    for field, field_schema in props.items():
        if field not in output:
            continue
        items_schema = field_schema.get("items") if field_schema.get("type") == "array" else None
        if items_schema and isinstance(output[field], list):
            expected_type = items_schema.get("type")
            if expected_type == "string" and not all(isinstance(v, str) for v in output[field]):
                return False
    return True


def _confidence_for_run(run: SubroutineRun) -> float:
    raw = run.output.get("confidence") if isinstance(run.output, dict) else None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return max(0.0, min(1.0, float(raw)))
    if run.success and run.schema_valid:
        return 0.86
    return 0.0


def _bounded_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def _bounded_cost(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return default


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
