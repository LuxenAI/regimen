"""MCP server exposing the local-first orchestration layer."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, cast

from openharness.orchestration.codex import (
    build_claude_mcp_add_command,
    build_claude_mcp_json,
    build_codex_mcp_config_snippet,
)
from openharness.orchestration.engine import build_default_orchestration_engine
from openharness.orchestration.slm_config import slm_config_status
from openharness.orchestration.types import Subtask, TaskContext, TaskType


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
    async def slm_verify_result(
        goal: str,
        output: str,
        task_type: str | None = None,
        confidence: float = 0.8,
    ) -> str:
        """Verify an externally produced result against the same local verifier contract."""
        payload = await _run_verifier_escalation(
            goal=goal,
            output=output,
            task_type=task_type,
            confidence=confidence,
            executor_name="external.candidate",
            executor_kind="mcp_tool",
            logs="",
            diff="",
            screenshot_summary="",
        )
        return _json(payload)

    @server.tool()
    async def slm_verify_escalation(
        task: str,
        output: str,
        task_type: str | None = None,
        confidence: float = 0.8,
        executor_name: str = "external.candidate",
        executor_kind: str = "mcp_tool",
        logs: str = "",
        diff: str = "",
        screenshot_summary: str = "",
    ) -> str:
        """Classify whether a task result is accepted or needs frontier escalation."""
        payload = await _run_verifier_escalation(
            goal=task,
            output=output,
            task_type=task_type,
            confidence=confidence,
            executor_name=executor_name,
            executor_kind=executor_kind,
            logs=logs,
            diff=diff,
            screenshot_summary=screenshot_summary,
        )
        return _json(payload)



    @server.tool()
    async def slm_repair_json(raw: str) -> str:
        """Repair malformed JSON through the JSON repair executor."""
        result = await _run_subroutine_tool("Repair JSON", "json_repair", {"raw": raw})
        return _json(result)

    @server.tool()
    async def slm_localize_traceback(traceback: str, project_prefix: str = "") -> str:
        """Find the likely culprit project frame in a Python traceback."""
        result = await _run_subroutine_tool(
            "Localize traceback",
            "trace_localize",
            {"traceback": traceback, "project_prefix": project_prefix},
        )
        return _json(result)

    @server.tool()
    async def slm_generate_search_queries(task: str) -> str:
        """Generate code-search query candidates for a coding task."""
        result = await _run_subroutine_tool("Generate search queries", "search_query", {"task": task})
        return _json(result)

    @server.tool()
    async def slm_rank_search_hits(query: str, hits: list[dict[str, Any]]) -> str:
        """Rank code-search hits, preferring likely definition sites."""
        result = await _run_subroutine_tool(
            "Rank search hits",
            "search_rank",
            {"query": query, "hits": hits},
        )
        return _json(result)

    @server.tool()
    async def slm_classify_failure(text: str) -> str:
        """Classify logs, stack traces, compiler output, or CI failures."""
        result = await _run_subroutine_tool(
            "Classify failure",
            "failure_classify",
            {"text": text},
        )
        return _json(result)

    @server.tool()
    async def slm_classify_patch_risk(diff: str, tests: str = "") -> str:
        """Classify patch risk, affected subsystem, and review/test requirements."""
        result = await _run_subroutine_tool(
            "Classify patch risk",
            "patch_risk",
            {"diff": diff, "tests": tests},
        )
        return _json(result)

    @server.tool()
    def slm_get_trace(trace_id: str | None = None, limit: int = 5) -> str:
        """Fetch a trace by id or list recent traces for eval/debugging."""
        if trace_id:
            trace = _ENGINE.trace_store.get(trace_id)
            return _json({"trace": trace.model_dump(mode="json") if trace else None})
        traces = _ENGINE.trace_store.recent(limit=max(1, min(limit, 25)))
        return _json({"traces": [trace.model_dump(mode="json") for trace in traces]})

    @server.tool()
    def slm_export_trace(trace_id: str | None = None, limit: int = 25) -> str:
        """Export one trace or recent traces for eval harness ingestion."""
        if trace_id:
            trace = _ENGINE.trace_store.get(trace_id)
            return _json({"format": "slm-harness-trace-v1", "trace": trace.model_dump(mode="json") if trace else None})
        traces = _ENGINE.trace_store.recent(limit=max(1, min(limit, 100)))
        return _json(
            {
                "format": "slm-harness-trace-v1",
                "traces": [trace.model_dump(mode="json") for trace in traces],
            }
        )

    @server.tool()
    def slm_health() -> str:
        """Return MCP runtime, executor, and model-route health."""
        return _json(
            {
                "ok": True,
                "server": "slm-harness",
                "transports": ["stdio", "streamable-http"],
                "executor_count": len(_ENGINE.list_executors()),
                "executors": _ENGINE.list_executors(),
                "model_status": slm_config_status(),
            }
        )

    @server.tool()
    def slm_model_status() -> str:
        """Return resolved local/cloud SLM route configuration."""
        return _json({"model_status": slm_config_status()})

    @server.tool()
    def slm_eval_manifest() -> str:
        """Return the production eval plan, metrics, and client install snippets."""
        return _json(
            {
                "manifest_version": "agent-session-evals-v1",
                "drivers": ["replay", "codex_cli", "claude_code_cli"],
                "baselines": [
                    "no_mcp",
                    "mcp_deterministic_only",
                    "mcp_local_slm",
                    "mcp_cloud_slm",
                    "mcp_cloud_slm_frontier_fallback",
                ],
                "scenario_types": [
                    "json_repair",
                    "failure_classify",
                    "trace_localize",
                    "search_query",
                    "search_rank",
                    "patch_risk",
                    "agent_session",
                ],
                "metrics": [
                    "task_success",
                    "schema_valid_rate",
                    "verifier_pass_rate",
                    "escalation_rate",
                    "fallback_rate",
                    "tool_call_count",
                    "estimated_token_savings",
                    "latency_p50_ms",
                    "latency_p95_ms",
                    "cost_per_success_usd",
                ],
                "codex_config_toml": build_codex_mcp_config_snippet(),
                "claude_mcp_json": build_claude_mcp_json(),
                "claude_mcp_add_command": build_claude_mcp_add_command(),
            }
        )

    @server.tool()
    def slm_codex_probe() -> str:
        """Return Codex integration metadata and a local config.toml snippet."""
        return _json(
            {
                "server": "slm-harness",
                "transport": "stdio",
                "codex_config_toml": build_codex_mcp_config_snippet(),
                "claude_mcp_json": build_claude_mcp_json(),
                "claude_mcp_add_command": build_claude_mcp_add_command(),
                "tools": [
                    "slm_list_executors",
                    "slm_decompose_workflow",
                    "slm_route_task",
                    "slm_run_task",
                    "slm_verify_result",
                    "slm_verify_escalation",
                    "slm_repair_json",
                    "slm_localize_traceback",
                    "slm_generate_search_queries",
                    "slm_rank_search_hits",
                    "slm_classify_failure",
                    "slm_classify_patch_risk",
                    "slm_get_trace",
                    "slm_export_trace",
                    "slm_health",
                    "slm_model_status",
                    "slm_eval_manifest",
                    "slm_codex_probe",
                ],
            }
        )

    # Opt-in lexical code-context retriever. Disabled by default so existing
    # behavior is unchanged unless SLM_HARNESS_ENABLE_CODE_RETRIEVER is set.
    if os.getenv("SLM_HARNESS_ENABLE_CODE_RETRIEVER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:

        @server.tool()
        def slm_code_context_search(query: str, root: str, top_k: int = 5) -> str:
            """Rank repository symbols for a query and return the files to read.

            Opt-in (set SLM_HARNESS_ENABLE_CODE_RETRIEVER). Deterministic BM25 over
            symbol name + docstring tokens; reduces context to the top_k symbols'
            files before any model call. No network, no model.
            """
            from openharness.orchestration.code_retriever import build_retriever

            retriever = build_retriever([root])
            results = retriever.query(query, top_k=max(1, top_k))
            files: list[str] = []
            for result in results:
                if result.entry.file_path not in files:
                    files.append(result.entry.file_path)
            return _json(
                {
                    "query": query,
                    "root": root,
                    "results": [
                        {
                            "name": r.entry.name,
                            "file": r.entry.file_path,
                            "line": r.entry.line,
                            "kind": r.entry.kind,
                            "score": round(r.score, 4),
                        }
                        for r in results
                    ],
                    "files_to_read": files,
                    "retriever": {
                        "stemmer": retriever.stemmer_name,
                        "splitter": retriever.splitter_name,
                        "symbols": len(retriever.entries),
                    },
                }
            )

    return server


def run_transport(transport: str = "stdio") -> None:
    """Run the MCP server over stdio or streamable-http."""
    if transport not in {"stdio", "streamable-http", "sse"}:
        raise ValueError(f"Unsupported MCP transport: {transport}")
    create_server().run(cast(Any, transport))


def run_stdio() -> None:
    """Run the MCP server over stdio."""
    run_transport("stdio")


def main() -> None:
    """Console entrypoint for the MCP server."""
    transport = os.getenv("SLM_HARNESS_MCP_TRANSPORT", "stdio")
    if len(sys.argv) > 1:
        transport = sys.argv[1]
    run_transport(transport)


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


async def _run_verifier_escalation(
    *,
    goal: str,
    output: str,
    task_type: str | None,
    confidence: float,
    executor_name: str,
    executor_kind: str,
    logs: str,
    diff: str,
    screenshot_summary: str,
) -> dict[str, Any]:
    candidate_type = _parse_task_type(task_type)
    subtask = Subtask(
        goal=f"Verify result for: {goal}",
        input=goal,
        task_type="verify",
        metadata={"candidate_task_type": candidate_type or "unknown"},
    )
    context = TaskContext(
        root_goal=goal,
        shared={
            "task": goal,
            "candidate_task_type": candidate_type or "unknown",
            "candidate_output": output,
            "executor_confidence": max(0.0, min(1.0, confidence)),
            "executor_name": executor_name,
            "executor_kind": executor_kind,
            "logs": logs,
            "diff": diff,
            "screenshot_summary": screenshot_summary,
        },
    )
    executor = _ENGINE.registry.require("local.verifier_escalation_classifier")
    result = await executor.execute(subtask, context)
    verification = _ENGINE.verifier.verify(subtask, result)
    return {
        "result": result.model_dump(mode="json"),
        "verification": verification.model_dump(mode="json"),
        "decision": result.output,
    }


async def _run_subroutine_tool(goal: str, task_type: TaskType, shared: dict[str, Any]) -> dict[str, Any]:
    subtask = Subtask(goal=goal, input=str(shared.get("raw") or shared.get("task") or shared.get("query") or goal), task_type=task_type)
    context = TaskContext(root_goal=goal, shared=shared)
    decision = _ENGINE.route_subtask(subtask)
    result = await _ENGINE._execute_decision(subtask, context, decision)
    verification = _ENGINE.verifier.verify(subtask, result)
    return {
        "subtask": subtask.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
        "verification": verification.model_dump(mode="json"),
    }


def _parse_task_type(value: str | None) -> TaskType | None:
    if value is None or value == "":
        return None
    allowed = {
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
    if value not in allowed:
        raise ValueError(f"Unsupported task_type: {value}")
    return cast(TaskType, value)


if __name__ == "__main__":
    try:
        run_stdio()
    except KeyboardInterrupt:
        asyncio.run(asyncio.sleep(0))
