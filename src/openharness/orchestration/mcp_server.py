"""MCP server exposing the local-first orchestration layer."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from openharness.orchestration.codex import build_codex_mcp_config_snippet
from openharness.orchestration.engine import build_default_orchestration_engine
from openharness.orchestration.types import ExecutorResult, Subtask, TaskContext, TaskType


INSTRUCTIONS = (
    "slm-harness routes typed agent subtasks to the cheapest reliable executor first. "
    "Use slm_route_task before expensive model work, slm_run_task for local execution with "
    "verification/escalation telemetry, and slm_get_trace to inspect cost, latency, and "
    "escalation behavior. Frontier outputs are handoff markers unless the host wires a live "
    "OpenHarness LLM executor."
)

_ENGINE = build_default_orchestration_engine()


def create_server() -> Any:
    """Create the FastMCP server lazily so normal imports do not require mcp."""
    from mcp.server.fastmcp import FastMCP

    try:
        server = FastMCP("slm-harness", instructions=INSTRUCTIONS)
    except TypeError:
        server = FastMCP("slm-harness")

    @server.tool()
    def slm_list_executors() -> str:
        """List registered local, SLM, MCP, and frontier executor slots."""
        return _json({"executors": _ENGINE.list_executors()})

    @server.tool()
    def slm_decompose_workflow(
        goal: str,
        task_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Decompose an agent workflow into typed subtasks."""
        typed = _parse_task_type(task_type)
        task_context = TaskContext(root_goal=goal, shared=dict(context or {}))
        subtasks = _ENGINE.decompose(goal, task_type=typed, context=task_context)
        return _json({"subtasks": [subtask.model_dump(mode="json") for subtask in subtasks]})

    @server.tool()
    def slm_route_task(
        goal: str,
        task_type: str | None = None,
        context: dict[str, Any] | None = None,
        min_reliability: float = 0.7,
        max_cost_usd: float | None = None,
    ) -> str:
        """Route a typed subtask to the cheapest reliable executor without executing it."""
        subtask, decision = _ENGINE.route_goal(
            goal,
            task_type=_parse_task_type(task_type),
            min_reliability=min_reliability,
            max_cost_usd=max_cost_usd,
            context_data=context,
        )
        return _json(
            {
                "subtask": subtask.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }
        )

    @server.tool()
    async def slm_run_task(
        goal: str,
        task_type: str | None = None,
        context: dict[str, Any] | None = None,
        min_reliability: float = 0.7,
        verify: bool = True,
        max_escalations: int = 1,
    ) -> str:
        """Execute typed subtasks locally where reliable, verify, and trace escalation."""
        trace = await _ENGINE.run_goal(
            goal,
            task_type=_parse_task_type(task_type),
            min_reliability=min_reliability,
            verify=verify,
            max_escalations=max(0, max_escalations),
            context_data=context,
        )
        return _json({"trace": trace.model_dump(mode="json")})

    @server.tool()
    def slm_verify_result(
        goal: str,
        output: str,
        task_type: str | None = None,
        confidence: float = 0.8,
    ) -> str:
        """Verify an externally produced result against the same local verifier contract."""
        subtask = Subtask(goal=goal, input=goal, task_type=_parse_task_type(task_type) or "unknown")
        result = ExecutorResult(
            subtask_id=subtask.id,
            executor_name="external.candidate",
            executor_kind="mcp_tool",
            output=output,
            confidence=max(0.0, min(1.0, confidence)),
        )
        verification = _ENGINE.verifier.verify(subtask, result)
        return _json({"verification": verification.model_dump(mode="json")})

    @server.tool()
    def slm_get_trace(trace_id: str | None = None, limit: int = 5) -> str:
        """Fetch a trace by id or list recent traces for eval/debugging."""
        if trace_id:
            trace = _ENGINE.trace_store.get(trace_id)
            return _json({"trace": trace.model_dump(mode="json") if trace else None})
        traces = _ENGINE.trace_store.recent(limit=max(1, min(limit, 25)))
        return _json({"traces": [trace.model_dump(mode="json") for trace in traces]})

    @server.tool()
    def slm_codex_probe() -> str:
        """Return Codex integration metadata and a local config.toml snippet."""
        return _json(
            {
                "server": "slm-harness",
                "transport": "stdio",
                "codex_config_toml": build_codex_mcp_config_snippet(),
                "tools": [
                    "slm_list_executors",
                    "slm_decompose_workflow",
                    "slm_route_task",
                    "slm_run_task",
                    "slm_verify_result",
                    "slm_get_trace",
                    "slm_codex_probe",
                ],
            }
        )

    return server


def run_stdio() -> None:
    """Run the MCP server over stdio."""
    create_server().run("stdio")


def main() -> None:
    """Console entrypoint for the MCP server."""
    run_stdio()


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _parse_task_type(value: str | None) -> TaskType | None:
    if value is None or value == "":
        return None
    allowed = {"route", "classify", "extract", "verify", "code", "tool", "reason", "unknown"}
    if value not in allowed:
        raise ValueError(f"Unsupported task_type: {value}")
    return cast(TaskType, value)


if __name__ == "__main__":
    try:
        run_stdio()
    except KeyboardInterrupt:
        asyncio.run(asyncio.sleep(0))
