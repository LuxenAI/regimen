"""Scenario loading and summary metrics for production evals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from openharness.evals.agent_drivers import AgentSessionSpec, AgentSessionTrace


class EvalScenario(BaseModel):
    """Portable JSONL scenario used for subroutine and agent-session evals."""

    id: str
    kind: str
    prompt: str
    input: Any = ""
    expected: Any = None
    fixture_path: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_agent_spec(self, *, use_mcp: bool) -> AgentSessionSpec:
        """Convert the scenario to an agent-session request."""
        return AgentSessionSpec(
            scenario_id=self.id,
            prompt=self.prompt,
            fixture_path=self.fixture_path,
            use_mcp=use_mcp,
            expected_success=bool(self.metadata.get("expected_success", True)),
            metadata=self.metadata,
        )


def load_jsonl_scenarios(path: str | Path) -> list[EvalScenario]:
    """Load eval scenarios from a JSONL file."""
    scenarios: list[EvalScenario] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                scenarios.append(EvalScenario(**json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"Invalid eval scenario at {path}:{line_no}: {exc}") from exc
    return scenarios


def write_traces_jsonl(traces: Iterable[AgentSessionTrace], path: str | Path) -> None:
    """Write normalized traces as JSONL."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.model_dump(mode="json"), sort_keys=True) + "\n")


def summarize_traces(traces: list[AgentSessionTrace]) -> dict[str, Any]:
    """Summarize agent-session deltas for a set of traces."""
    if not traces:
        return {
            "sessions": 0,
            "success_rate": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "cost_per_success_usd": None,
            "mcp_call_rate": 0.0,
            "estimated_token_savings": 0,
        }
    successes = sum(1 for trace in traces if trace.success)
    total_cost = sum(trace.cost_usd for trace in traces)
    latencies = [float(trace.wall_time_ms) for trace in traces]
    return {
        "sessions": len(traces),
        "success_rate": successes / len(traces),
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "cost_per_success_usd": total_cost / successes if successes else None,
        "mcp_call_rate": sum(1 for trace in traces if trace.mcp_call_count > 0) / len(traces),
        "estimated_token_savings": sum(trace.estimated_token_savings for trace in traces),
        "tool_call_count": sum(trace.tool_call_count for trace in traces),
        "mcp_call_count": sum(trace.mcp_call_count for trace in traces),
    }


def _percentile(values: list[float], frac: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * frac))]
