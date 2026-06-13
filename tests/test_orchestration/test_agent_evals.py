"""Tests for production agent-session eval utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals.agent_drivers import ReplayDriver
from openharness.evals.harness import load_jsonl_scenarios, summarize_traces


def test_load_agent_session_fixture() -> None:
    path = Path(__file__).resolve().parents[2] / "evals" / "fixtures" / "agent_sessions.jsonl"
    scenarios = load_jsonl_scenarios(path)

    assert scenarios
    assert scenarios[0].kind == "agent_session"


@pytest.mark.asyncio
async def test_replay_driver_emits_agent_session_delta_metrics() -> None:
    path = Path(__file__).resolve().parents[2] / "evals" / "fixtures" / "agent_sessions.jsonl"
    scenarios = load_jsonl_scenarios(path)
    driver = ReplayDriver()
    traces = [await driver.run(scenario.to_agent_spec(use_mcp=True)) for scenario in scenarios]
    summary = summarize_traces(traces)

    assert summary["success_rate"] == 1.0
    assert summary["mcp_call_count"] >= len(scenarios)
    assert summary["estimated_token_savings"] > 0
