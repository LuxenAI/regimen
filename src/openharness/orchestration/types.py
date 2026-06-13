"""Typed orchestration contracts for local SLM routing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


TaskType = Literal[
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
]
ExecutorKind = Literal["deterministic", "classifier", "task_slm", "frontier_llm", "mcp_tool"]
TraceEventType = Literal["decompose", "route", "execute", "verify", "escalate", "complete"]


def new_id(prefix: str) -> str:
    """Return a compact trace/subtask id."""
    return f"{prefix}_{uuid4().hex[:12]}"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class TaskContext(BaseModel):
    """Shared blackboard context carried through an orchestration run."""

    root_goal: str
    shared: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    budget_usd: float | None = None
    trace_id: str = Field(default_factory=lambda: new_id("trace"))


class Subtask(BaseModel):
    """A typed unit of work that can be routed to an executor."""

    id: str = Field(default_factory=lambda: new_id("subtask"))
    task_type: TaskType = "unknown"
    goal: str
    input: str = ""
    priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutorProfile(BaseModel):
    """Public routing metadata for one executor."""

    name: str
    kind: ExecutorKind
    description: str = ""
    supported_task_types: list[TaskType]
    local: bool = True
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    cost_per_call_usd: float = Field(default=0.0, ge=0.0)
    p50_latency_ms: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def supports(self, task_type: TaskType) -> bool:
        """Return whether this executor can handle the supplied subtask type."""
        return task_type in self.supported_task_types


class RouteCandidate(BaseModel):
    """A scored executor option considered by the router."""

    executor_name: str
    kind: ExecutorKind
    reliability: float
    expected_cost_usd: float
    expected_latency_ms: int
    local: bool
    reliable: bool
    reason: str


class RouteDecision(BaseModel):
    """The selected executor and the alternatives considered."""

    subtask_id: str
    selected_executor: str
    selected_kind: ExecutorKind
    should_escalate: bool
    reason: str
    candidates: list[RouteCandidate] = Field(default_factory=list)


class ExecutorResult(BaseModel):
    """Normalized output from any executor."""

    subtask_id: str
    executor_name: str
    executor_kind: ExecutorKind
    output: Any = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)
    escalated: bool = False
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Verifier judgment for an executor result."""

    subtask_id: str
    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float
    verifier: str
    reason: str


class TraceEvent(BaseModel):
    """One timestamped orchestration event."""

    at: str = Field(default_factory=utc_now_iso)
    event_type: TraceEventType
    payload: dict[str, Any] = Field(default_factory=dict)


class TraceMetrics(BaseModel):
    """Aggregated cost, latency, and escalation metrics."""

    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    subtasks: int = 0
    accepted: int = 0
    rejected: int = 0
    escalations: int = 0
    acceptance_rate: float = 0.0
    executor_counts: dict[str, int] = Field(default_factory=dict)


class OrchestrationTrace(BaseModel):
    """Complete record for one orchestration run."""

    trace_id: str
    root_goal: str
    created_at: str = Field(default_factory=utc_now_iso)
    subtasks: list[Subtask] = Field(default_factory=list)
    decisions: list[RouteDecision] = Field(default_factory=list)
    results: list[ExecutorResult] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    metrics: TraceMetrics = Field(default_factory=TraceMetrics)

    def add_event(self, event_type: TraceEventType, payload: dict[str, Any]) -> None:
        """Append a trace event."""
        self.events.append(TraceEvent(event_type=event_type, payload=payload))

    def recompute_metrics(self) -> None:
        """Refresh aggregate metrics from recorded results and verifications."""
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.executor_name] = counts.get(result.executor_name, 0) + 1
        accepted = sum(1 for item in self.verifications if item.accepted)
        rejected = sum(1 for item in self.verifications if not item.accepted)
        subtasks = len(self.subtasks)
        self.metrics = TraceMetrics(
            total_cost_usd=sum(result.cost_usd for result in self.results),
            total_latency_ms=sum(result.latency_ms for result in self.results),
            subtasks=subtasks,
            accepted=accepted,
            rejected=rejected,
            escalations=sum(1 for result in self.results if result.escalated),
            acceptance_rate=accepted / subtasks if subtasks else 0.0,
            executor_counts=counts,
        )
