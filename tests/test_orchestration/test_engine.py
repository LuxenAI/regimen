"""Tests for the local-first orchestration layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openharness.orchestration.codex import build_codex_mcp_config_snippet
from openharness.orchestration.engine import build_default_orchestration_engine


def test_route_prefers_cheapest_reliable_local_executor_for_extraction() -> None:
    engine = build_default_orchestration_engine()

    subtask, decision = engine.route_goal(
        "Extract email addresses from contact ops@example.com",
        task_type="extract",
        min_reliability=0.7,
    )

    assert subtask.task_type == "extract"
    assert decision.selected_executor == "local.regex_extractor"
    assert decision.should_escalate is False
    assert decision.candidates[0].executor_name == "local.regex_extractor"


def test_route_prefers_verifier_escalation_classifier_for_verify_tasks() -> None:
    engine = build_default_orchestration_engine()

    subtask, decision = engine.route_goal(
        "Verify whether a tool result should be accepted",
        task_type="verify",
        min_reliability=0.7,
    )

    assert subtask.task_type == "verify"
    assert decision.selected_executor == "local.verifier_escalation_classifier"
    assert decision.should_escalate is False


@pytest.mark.asyncio
async def test_run_goal_records_local_result_metrics_and_trace() -> None:
    engine = build_default_orchestration_engine()

    trace = await engine.run_goal(
        "Extract the email admin@example.com and url https://example.com/docs",
        task_type="extract",
    )

    assert trace.metrics.subtasks == 1
    assert trace.metrics.accepted == 1
    assert trace.metrics.escalations == 0
    assert trace.results[0].executor_name == "local.regex_extractor"
    assert trace.results[0].output["emails"] == ["admin@example.com"]
    assert trace.trace_id in {item.trace_id for item in engine.trace_store.recent()}


@pytest.mark.asyncio
async def test_run_goal_escalates_to_frontier_slot_for_high_reliability_code_task() -> None:
    engine = build_default_orchestration_engine()

    trace = await engine.run_goal(
        "Implement the repo changes and run tests for a coding agent workflow",
        task_type="code",
        min_reliability=0.9,
    )

    assert trace.decisions[0].selected_executor == "frontier.openharness_llm"
    assert trace.results[0].escalated is True
    assert trace.verifications[0].accepted is False
    assert trace.metrics.escalations == 1


def test_codex_config_snippet_is_stdio_mcp_config() -> None:
    snippet = build_codex_mcp_config_snippet(
        server_name="slm_harness_test",
        python="/usr/bin/python3",
        root="/tmp/slm-harness",
    )

    assert "[mcp_servers.slm_harness_test]" in snippet
    assert 'command = "/usr/bin/python3"' in snippet
    assert 'args = ["-m", "openharness.orchestration.mcp_server"]' in snippet
    resolved_root = Path("/tmp/slm-harness").resolve()
    assert f'cwd = "{resolved_root}"' in snippet
    assert "[mcp_servers.slm_harness_test.env]" in snippet
    assert f'PYTHONPATH = "{resolved_root / "src"}"' in snippet


def test_executor_profiles_are_json_ready() -> None:
    engine = build_default_orchestration_engine()

    encoded = json.dumps({"executors": engine.list_executors()})

    assert "local.verifier_escalation_classifier" in encoded
    assert "local.tiny_slm_stub" in encoded
    assert "frontier.openharness_llm" in encoded
