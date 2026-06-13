"""Executor adapters used by the orchestration router."""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any

from openharness.orchestration.decomposer import WorkflowDecomposer
from openharness.orchestration.subroutine_models import (
    generate_search_queries_subroutine,
    localize_traceback_subroutine,
    rank_search_hits_subroutine,
    repair_json_subroutine,
)
from openharness.orchestration.types import (
    ExecutorKind,
    ExecutorProfile,
    ExecutorResult,
    Subtask,
    TaskContext,
)
from openharness.orchestration.verifier_model import (
    VerifierClassifier,
    build_verifier_model_from_env,
    verifier_input_from_context,
)


class BaseExecutor(ABC):
    """Base class for deterministic, classifier, SLM, MCP, and frontier executors."""

    profile: ExecutorProfile

    @abstractmethod
    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        """Run the executor for a subtask."""

    def _result(
        self,
        subtask: Subtask,
        *,
        output: Any,
        confidence: float,
        started_at: float,
        escalated: bool = False,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutorResult:
        elapsed_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        return ExecutorResult(
            subtask_id=subtask.id,
            executor_name=self.profile.name,
            executor_kind=self.profile.kind,
            output=output,
            confidence=confidence,
            cost_usd=self.profile.cost_per_call_usd,
            latency_ms=max(elapsed_ms, self.profile.p50_latency_ms),
            escalated=escalated,
            error=error,
            metadata=metadata or {},
        )


class KeywordClassifierExecutor(BaseExecutor):
    """Local classifier that maps text to the harness task taxonomy."""

    def __init__(self) -> None:
        self._decomposer = WorkflowDecomposer()
        self.profile = ExecutorProfile(
            name="local.keyword_classifier",
            kind="classifier",
            description="CPU-local keyword classifier for intent and subtask typing.",
            supported_task_types=["route", "classify", "unknown"],
            local=True,
            reliability=0.78,
            cost_per_call_usd=0.0,
            p50_latency_ms=2,
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        started = time.perf_counter()
        inferred = self._decomposer.infer_task_type(subtask.input or subtask.goal, context=context)
        confidence = 0.84 if inferred != "unknown" else 0.52
        return self._result(
            subtask,
            output={
                "task_type": inferred,
                "labels": [inferred],
                "rationale": "Matched local routing keywords.",
            },
            confidence=confidence,
            started_at=started,
        )


class RegexExtractorExecutor(BaseExecutor):
    """Local deterministic extractor for common structured fields."""

    _EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
    _URL_RE = re.compile(r"https?://[^\s)>\"]+")
    _PATH_RE = re.compile(r"(?:^|\s)((?:\.{1,2}/|/)?[\w.-]+(?:/[\w.@+-]+)+)")

    def __init__(self) -> None:
        self.profile = ExecutorProfile(
            name="local.regex_extractor",
            kind="deterministic",
            description="CPU-local regex extractor for emails, URLs, and filesystem-like paths.",
            supported_task_types=["extract"],
            local=True,
            reliability=0.82,
            cost_per_call_usd=0.0,
            p50_latency_ms=2,
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        del context
        started = time.perf_counter()
        text = subtask.input or subtask.goal
        output = {
            "emails": self._EMAIL_RE.findall(text),
            "urls": self._URL_RE.findall(text),
            "paths": [match.strip() for match in self._PATH_RE.findall(text)],
        }
        matches = sum(len(value) for value in output.values())
        return self._result(
            subtask,
            output=output,
            confidence=0.9 if matches else 0.42,
            started_at=started,
            metadata={"match_count": matches},
        )


class HeuristicVerifierExecutor(BaseExecutor):
    """Local verifier that accepts non-empty, non-error-like outputs."""

    def __init__(self) -> None:
        self.profile = ExecutorProfile(
            name="local.heuristic_verifier",
            kind="deterministic",
            description="CPU-local verifier for non-empty outputs and simple failure markers.",
            supported_task_types=["verify"],
            local=True,
            reliability=0.73,
            cost_per_call_usd=0.0,
            p50_latency_ms=2,
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        started = time.perf_counter()
        candidate = str(context.shared.get("candidate_output") or subtask.input or "")
        lowered = candidate.lower()
        accepted = bool(candidate.strip()) and "error" not in lowered and "failed" not in lowered
        return self._result(
            subtask,
            output={"accepted": accepted, "issues": [] if accepted else ["empty_or_error_like_output"]},
            confidence=0.78 if accepted else 0.48,
            started_at=started,
        )


class VerifierEscalationClassifierExecutor(BaseExecutor):
    """Verifier/escalation classifier for deciding whether frontier review is needed."""

    def __init__(self, model: VerifierClassifier | None = None) -> None:
        self._model = model or build_verifier_model_from_env()
        self.profile = ExecutorProfile(
            name="local.verifier_escalation_classifier",
            kind="classifier",
            description=(
                "CPU-local verifier/escalation classifier over task, result, logs, diff, "
                "and screenshot summary."
            ),
            supported_task_types=["verify"],
            local=True,
            reliability=0.86,
            cost_per_call_usd=0.0,
            p50_latency_ms=1,
            metadata={
                "output_schema": {
                    "accepted": "bool",
                    "confidence": "float",
                    "escalate": "bool",
                },
                "model_slot": "BERT-Tiny compatible <10M parameter verifier",
                "env_model_dir": "SLM_HARNESS_VERIFIER_MODEL_DIR",
            },
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        started = time.perf_counter()
        verifier_input = verifier_input_from_context(subtask, shared=context.shared)
        prediction = self._model.predict(verifier_input)
        return self._result(
            subtask,
            output=prediction.model_dump(mode="json"),
            confidence=prediction.confidence,
            started_at=started,
            escalated=prediction.escalate,
            metadata={
                "verifier_source": prediction.source,
                "verifier_label": prediction.label,
                "input": verifier_input.model_dump(mode="json"),
            },
        )


class JsonRepairSlmExecutor(BaseExecutor):
    """JSON repair executor with deterministic parse before SLM-style fallback."""

    def __init__(self) -> None:
        self.profile = ExecutorProfile(
            name="local.json_repair_slm",
            kind="task_slm",
            description="Repair malformed JSON tool payloads with deterministic-first SLM fallback semantics.",
            supported_task_types=["json_repair"],
            local=True,
            reliability=0.9,
            cost_per_call_usd=0.0,
            p50_latency_ms=3,
            metadata={"model_id": "ishaanranjan/slm-agent-json-repair-smollm2-360m"},
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        started = time.perf_counter()
        raw = str(context.shared.get("raw") or context.shared.get("candidate_output") or subtask.input or subtask.goal)
        run = repair_json_subroutine(raw)
        return self._subroutine_result(subtask, run.as_output(), started, run.success)

    def _subroutine_result(self, subtask: Subtask, output: dict[str, Any], started: float, success: bool) -> ExecutorResult:
        return self._result(
            subtask,
            output=output,
            confidence=0.9 if success else 0.2,
            started_at=started,
            escalated=not success,
            error=output.get("error") if not success else None,
            metadata=_subroutine_metadata(output, subtask),
        )


class TraceLocalizerSlmExecutor(BaseExecutor):
    """Traceback localizer executor for culprit project frame detection."""

    def __init__(self) -> None:
        self.profile = ExecutorProfile(
            name="local.trace_localizer_slm",
            kind="task_slm",
            description="Localize Python traceback culprit frame and failure category.",
            supported_task_types=["trace_localize"],
            local=True,
            reliability=0.88,
            cost_per_call_usd=0.0,
            p50_latency_ms=3,
            metadata={"model_id": "ishaanranjan/slm-agent-trace-localizer-qwen2-5-0-5b"},
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        started = time.perf_counter()
        traceback = str(context.shared.get("traceback") or subtask.input or subtask.goal)
        project_prefix = str(context.shared.get("project_prefix") or "")
        run = localize_traceback_subroutine(traceback, project_prefix=project_prefix)
        return self._result(
            subtask,
            output=run.as_output(),
            confidence=float(run.output.get("confidence", 0.0)),
            started_at=started,
            escalated=not run.success,
            error=run.error,
            metadata=_subroutine_metadata(run.as_output(), subtask),
        )


class SearchQueryGenSlmExecutor(BaseExecutor):
    """Search-query generation executor for coding tasks."""

    def __init__(self) -> None:
        self.profile = ExecutorProfile(
            name="local.search_query_gen_slm",
            kind="task_slm",
            description="Generate code-search query candidates for a coding task.",
            supported_task_types=["search_query"],
            local=True,
            reliability=0.82,
            cost_per_call_usd=0.0,
            p50_latency_ms=3,
            metadata={"model_id": "ishaanranjan/slm-agent-search-query-gen-qwen2-5-0-5b"},
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        started = time.perf_counter()
        task = str(context.shared.get("task") or subtask.input or subtask.goal)
        run = generate_search_queries_subroutine(task)
        return self._result(
            subtask,
            output=run.as_output(),
            confidence=float(run.output.get("confidence", 0.0)),
            started_at=started,
            escalated=not run.success,
            error=run.error,
            metadata=_subroutine_metadata(run.as_output(), subtask),
        )


class SearchHitRankerSlmExecutor(BaseExecutor):
    """Search-hit ranking executor for selecting definition-like results."""

    def __init__(self) -> None:
        self.profile = ExecutorProfile(
            name="local.search_hit_ranker_slm",
            kind="task_slm",
            description="Rank code-search hits and prefer definition sites.",
            supported_task_types=["search_rank"],
            local=True,
            reliability=0.9,
            cost_per_call_usd=0.0,
            p50_latency_ms=3,
            metadata={"model_id": "ishaanranjan/slm-agent-search-hit-ranker-qwen2-5-0-5b"},
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        started = time.perf_counter()
        query = str(context.shared.get("query") or subtask.input or subtask.goal)
        raw_hits = context.shared.get("hits") or []
        hits = raw_hits if isinstance(raw_hits, list) else []
        run = rank_search_hits_subroutine(query, hits)
        return self._result(
            subtask,
            output=run.as_output(),
            confidence=float(run.output.get("confidence", 0.0)),
            started_at=started,
            escalated=not run.success,
            error=run.error,
            metadata=_subroutine_metadata(run.as_output(), subtask),
        )


def _subroutine_metadata(output: dict[str, Any], subtask: Subtask) -> dict[str, Any]:
    return {
        "executor_name": output.get("subroutine"),
        "model_id": output.get("model_id"),
        "local": output.get("local", True),
        "success": output.get("success"),
        "verifier_result": output.get("verifier_pass"),
        "fallback_path": output.get("fallback_path"),
        "trace_id": subtask.metadata.get("trace_id"),
    }


class TinySlmStubExecutor(BaseExecutor):
    """A local task-SLM adapter placeholder with the same contract as a real tiny model."""

    def __init__(self) -> None:
        self.profile = ExecutorProfile(
            name="local.tiny_slm_stub",
            kind="task_slm",
            description="Pluggable local tiny-SLM adapter placeholder for narrow reasoning tasks.",
            supported_task_types=["reason", "classify", "verify", "unknown"],
            local=True,
            reliability=0.7,
            cost_per_call_usd=0.00001,
            p50_latency_ms=15,
            metadata={"model_params": "<10M-compatible adapter slot"},
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        del context
        started = time.perf_counter()
        text = subtask.input or subtask.goal
        output = {
            "summary": text[:240],
            "decision": "handled_by_local_tiny_slm_adapter",
            "needs_frontier": len(text) > 1200 or subtask.task_type == "unknown",
        }
        confidence = 0.72 if not output["needs_frontier"] else 0.55
        return self._result(subtask, output=output, confidence=confidence, started_at=started)


class CodeHeuristicExecutor(BaseExecutor):
    """Local code-task triage executor that deliberately escalates implementation work."""

    def __init__(self) -> None:
        self.profile = ExecutorProfile(
            name="local.code_triage",
            kind="deterministic",
            description="CPU-local coding triage that extracts intent but does not edit code.",
            supported_task_types=["code"],
            local=True,
            reliability=0.56,
            cost_per_call_usd=0.0,
            p50_latency_ms=3,
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        del context
        started = time.perf_counter()
        return self._result(
            subtask,
            output={
                "classification": "coding_task",
                "recommendation": "escalate_to_coding_agent_or_frontier_llm",
            },
            confidence=0.5,
            started_at=started,
        )


class FrontierHandoffExecutor(BaseExecutor):
    """Frontier LLM executor slot used when local executors are not reliable enough."""

    def __init__(self) -> None:
        self.profile = ExecutorProfile(
            name="frontier.openharness_llm",
            kind="frontier_llm",
            description="Escalation slot for the existing OpenHarness LLM/tool loop.",
            supported_task_types=["route", "classify", "extract", "verify", "code", "tool", "reason", "unknown"],
            local=False,
            reliability=0.96,
            cost_per_call_usd=0.02,
            p50_latency_ms=1200,
            metadata={"adapter": "openharness.query_engine"},
        )

    async def execute(self, subtask: Subtask, context: TaskContext) -> ExecutorResult:
        started = time.perf_counter()
        return self._result(
            subtask,
            output={
                "handoff_required": True,
                "target": "openharness_llm",
                "prompt": subtask.input or subtask.goal,
                "trace_id": context.trace_id,
            },
            confidence=0.96,
            started_at=started,
            escalated=True,
            metadata={"reason": "selected_frontier_executor"},
        )


def executor_kind_name(kind: ExecutorKind) -> str:
    """Return a display-safe executor kind name."""
    return str(kind)
