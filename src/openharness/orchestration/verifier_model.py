"""Verifier/escalation classifier models for orchestration results."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, Field

from openharness.orchestration.types import ExecutorKind, ExecutorResult, Subtask, TaskType


DEFAULT_VERIFIER_LABELS: tuple[str, ...] = (
    "accept",
    "escalate_empty_output",
    "escalate_error",
    "escalate_low_confidence",
    "escalate_risky_diff",
    "escalate_visual_failure",
    "escalate_incomplete",
)

ACCEPT_LABEL = "accept"
DEFAULT_ACCEPT_THRESHOLD = 0.65

_EXECUTOR_KINDS = {"deterministic", "classifier", "task_slm", "frontier_llm", "mcp_tool"}
_TASK_TYPES = {
    "route",
    "classify",
    "extract",
    "verify",
    "code",
    "tool",
    "reason",
    "json_repair",
    "trace_localize",
    "search_query",
    "search_rank",
    "failure_classify",
    "patch_risk",
    "unknown",
}

_FAILURE_RE = re.compile(
    r"\b("
    r"traceback|exception|modulenotfounderror|importerror|syntaxerror|typeerror|"
    r"assertionerror|referenceerror|fatal|npm err|pytest failed|segmentation fault"
    r")\b|(?<!0 )\bfailed\b|error:",
    re.IGNORECASE,
)
_PASS_RE = re.compile(
    r"\b("
    r"passed|all tests pass|0 failed|build succeeded|typecheck passed|lint passed|"
    r"compiled successfully"
    r")\b",
    re.IGNORECASE,
)
_INCOMPLETE_RE = re.compile(
    r"\b(todo|fixme|placeholder|not implemented|omitted|cannot complete|unable to complete)\b",
    re.IGNORECASE,
)
_VISUAL_FAILURE_RE = re.compile(
    r"\b(blank|white screen|overlap|clipped|not visible|spinner|loading forever|"
    r"console error|layout broken)\b",
    re.IGNORECASE,
)
_RISKY_DIFF_RE = re.compile(
    r"\b(auth|authentication|authorization|payment|billing|password|secret|token|"
    r"delete|drop table|subprocess|shell=True)\b|rm -rf|eval\(|exec\(",
    re.IGNORECASE,
)


class VerifierInput(BaseModel):
    """Normalized verifier input from an executor result and surrounding evidence."""

    task: str
    task_type: TaskType | None = None
    executor_name: str = "external.candidate"
    executor_kind: ExecutorKind = "mcp_tool"
    executor_output: Any = ""
    executor_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    logs: str = ""
    diff: str = ""
    screenshot_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        subtask: Subtask,
        result: ExecutorResult,
        *,
        logs: str = "",
        diff: str = "",
        screenshot_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> VerifierInput:
        """Build verifier input from an executor result."""
        return cls(
            task=subtask.input or subtask.goal,
            task_type=subtask.task_type,
            executor_name=result.executor_name,
            executor_kind=result.executor_kind,
            executor_output=result.output,
            executor_confidence=result.confidence,
            logs=logs,
            diff=diff,
            screenshot_summary=screenshot_summary,
            metadata=metadata or {},
        )

    def to_model_text(self, *, max_chars: int = 6000) -> str:
        """Render the verifier input as compact text for a sequence classifier."""
        output = self._stringify(self.executor_output)
        sections = [
            f"TASK_TYPE: {self.task_type or 'unknown'}",
            f"TASK: {self.task}",
            f"EXECUTOR: {self.executor_name} ({self.executor_kind})",
            f"EXECUTOR_CONFIDENCE: {self.executor_confidence:.3f}",
            f"OUTPUT: {output}",
            f"LOGS: {self.logs}",
            f"DIFF: {self.diff}",
            f"SCREENSHOT_SUMMARY: {self.screenshot_summary}",
        ]
        rendered = "\n".join(section for section in sections if section.split(": ", 1)[-1] != "")
        return rendered[:max_chars]

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)


class VerifierPrediction(BaseModel):
    """Accepted/escalate decision emitted by a verifier classifier."""

    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    escalate: bool
    label: str
    reason: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierClassifier(Protocol):
    """Classifier interface shared by heuristic and tiny transformer backends."""

    def predict(self, item: VerifierInput) -> VerifierPrediction:
        """Predict whether an executor result is good enough."""


class HeuristicVerifierModel:
    """Deterministic verifier used until a trained tiny classifier is configured."""

    source = "heuristic.verifier_escalation"

    def __init__(self, *, accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD) -> None:
        self.accept_threshold = accept_threshold

    def predict(self, item: VerifierInput) -> VerifierPrediction:
        """Classify the result using high-precision local signals."""
        output_text = VerifierInput._stringify(item.executor_output)
        combined = "\n".join(
            part
            for part in (
                item.task,
                output_text,
                item.logs,
                item.diff,
                item.screenshot_summary,
            )
            if part
        )
        output_empty = item.executor_output in ("", None, {}, []) or not output_text.strip()

        if output_empty:
            return self._escalate("escalate_empty_output", 0.93, "executor output was empty")
        if _FAILURE_RE.search(combined):
            return self._escalate("escalate_error", 0.91, "failure marker found in output or logs")
        if item.screenshot_summary and _VISUAL_FAILURE_RE.search(item.screenshot_summary):
            return self._escalate(
                "escalate_visual_failure",
                0.88,
                "screenshot summary indicates a visual failure",
            )
        if _INCOMPLETE_RE.search(combined):
            return self._escalate("escalate_incomplete", 0.84, "result appears incomplete")
        if item.diff and _RISKY_DIFF_RE.search(item.diff) and not _PASS_RE.search(combined):
            return self._escalate(
                "escalate_risky_diff",
                0.82,
                "risky diff needs stronger verification evidence",
            )
        if item.executor_confidence < self.accept_threshold:
            return self._escalate(
                "escalate_low_confidence",
                max(0.7, 1.0 - item.executor_confidence),
                "executor confidence was below verifier threshold",
            )

        confidence = min(0.96, max(0.74, 0.72 + (item.executor_confidence * 0.22)))
        if _PASS_RE.search(combined):
            confidence = max(confidence, 0.92)
        return VerifierPrediction(
            accepted=True,
            confidence=confidence,
            escalate=False,
            label=ACCEPT_LABEL,
            reason="result has output, sufficient confidence, and no failure markers",
            source=self.source,
            metadata={"accept_threshold": self.accept_threshold},
        )

    def _escalate(self, label: str, confidence: float, reason: str) -> VerifierPrediction:
        return VerifierPrediction(
            accepted=False,
            confidence=confidence,
            escalate=True,
            label=label,
            reason=reason,
            source=self.source,
            metadata={"accept_threshold": self.accept_threshold},
        )


class TransformersVerifierModel:
    """Tiny HF sequence-classifier backend for verifier/escalation decisions."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_dir = str(model_dir)
        self.accept_threshold = accept_threshold
        self.max_length = max_length

        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for TransformersVerifierModel"
            ) from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id_to_label = self._load_labels()

    def predict(self, item: VerifierInput) -> VerifierPrediction:
        """Run the configured tiny classifier and normalize its label."""
        encoded = self.tokenizer(
            item.to_model_text(),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.no_grad():
            logits = self.model(**encoded).logits
            probs = self._torch.softmax(logits, dim=-1)[0]
            score, index = self._torch.max(probs, dim=-1)
        label = self.id_to_label.get(int(index.item()), str(int(index.item())))
        confidence = float(score.item())
        accepted = label == ACCEPT_LABEL and confidence >= self.accept_threshold
        return VerifierPrediction(
            accepted=accepted,
            confidence=confidence,
            escalate=not accepted,
            label=label,
            reason=(
                "tiny verifier classifier accepted result"
                if accepted
                else f"tiny verifier classifier selected {label}"
            ),
            source=f"transformers:{self.model_dir}",
            metadata={"accept_threshold": self.accept_threshold, "device": self.device},
        )

    def _load_labels(self) -> dict[int, str]:
        config = getattr(self.model, "config", None)
        raw = getattr(config, "id2label", None)
        labels: dict[int, str] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    labels[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
        if labels:
            return labels
        return {index: label for index, label in enumerate(DEFAULT_VERIFIER_LABELS)}


def build_verifier_model_from_env() -> VerifierClassifier:
    """Build the configured verifier backend, falling back to heuristics for local use."""
    backend = os.getenv("SLM_HARNESS_VERIFIER_BACKEND", "auto").strip().lower()
    model_dir = os.getenv("SLM_HARNESS_VERIFIER_MODEL_DIR")
    threshold = _env_float("SLM_HARNESS_VERIFIER_THRESHOLD", DEFAULT_ACCEPT_THRESHOLD)

    if model_dir and backend in {"auto", "transformers", "tiny_transformer"}:
        try:
            return TransformersVerifierModel(model_dir, accept_threshold=threshold)
        except Exception:
            if backend in {"transformers", "tiny_transformer"}:
                raise
    return HeuristicVerifierModel(accept_threshold=threshold)


def verifier_input_from_context(
    subtask: Subtask,
    *,
    shared: dict[str, Any],
) -> VerifierInput:
    """Normalize MCP/engine context into classifier input."""
    raw_result = shared.get("executor_result")
    if isinstance(raw_result, ExecutorResult):
        return VerifierInput.from_result(
            subtask,
            raw_result,
            logs=_string(shared.get("logs")),
            diff=_string(shared.get("diff")),
            screenshot_summary=_string(shared.get("screenshot_summary")),
            metadata={"context_source": "executor_result"},
        )
    if isinstance(raw_result, dict):
        return VerifierInput(
            task=_string(shared.get("task"), subtask.input or subtask.goal),
            task_type=_coerce_task_type(shared.get("candidate_task_type")),
            executor_name=_string(raw_result.get("executor_name"), "external.candidate"),
            executor_kind=_coerce_executor_kind(raw_result.get("executor_kind")),
            executor_output=raw_result.get("output", ""),
            executor_confidence=_bounded_float(raw_result.get("confidence"), 0.0),
            logs=_string(shared.get("logs")),
            diff=_string(shared.get("diff")),
            screenshot_summary=_string(shared.get("screenshot_summary")),
            metadata={"context_source": "executor_result_dict"},
        )

    return VerifierInput(
        task=_string(shared.get("task"), subtask.input or subtask.goal),
        task_type=_coerce_task_type(shared.get("candidate_task_type")),
        executor_name=_string(shared.get("executor_name"), "external.candidate"),
        executor_kind=_coerce_executor_kind(shared.get("executor_kind")),
        executor_output=shared.get("candidate_output", subtask.input),
        executor_confidence=_bounded_float(shared.get("executor_confidence"), 0.0),
        logs=_string(shared.get("logs")),
        diff=_string(shared.get("diff")),
        screenshot_summary=_string(shared.get("screenshot_summary")),
        metadata={"context_source": "shared_fields"},
    )


def quantize_transformer_linear_layers(model_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Export a CPU dynamic-int8 state dict for a trained tiny classifier."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for quantization") from exc

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to("cpu")
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized.state_dict(), destination)
    return {
        "model_dir": str(model_dir),
        "output_path": str(destination),
        "size_bytes": destination.stat().st_size,
    }


def _coerce_executor_kind(value: Any) -> ExecutorKind:
    text = _string(value, "mcp_tool")
    if text not in _EXECUTOR_KINDS:
        text = "mcp_tool"
    return cast(ExecutorKind, text)


def _coerce_task_type(value: Any) -> TaskType | None:
    if value is None:
        return None
    text = _string(value)
    if text not in _TASK_TYPES:
        return None
    return cast(TaskType, text)


def _bounded_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)
