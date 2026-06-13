"""MCP integration tests for the orchestration server."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from openharness.mcp.client import McpClientManager
from openharness.mcp.types import McpStdioServerConfig


@pytest.mark.asyncio
async def test_orchestration_mcp_server_exposes_codex_facing_tools() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manager = McpClientManager(
        {
            "slm_harness": McpStdioServerConfig(
                command=sys.executable,
                args=["-m", "openharness.orchestration.mcp_server"],
                cwd=str(repo_root),
                env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
            )
        }
    )
    await manager.connect_all()
    try:
        statuses = manager.list_statuses()
        assert statuses[0].state == "connected"
        tool_names = {tool.name for tool in statuses[0].tools}
        assert "slm_route_task" in tool_names
        assert "slm_run_task" in tool_names
        assert "slm_verify_escalation" in tool_names
        assert "slm_codex_probe" in tool_names

        route_output = await manager.call_tool(
            "slm_harness",
            "slm_route_task",
            {
                "goal": "Extract email user@example.com",
                "task_type": "extract",
                "min_reliability": 0.7,
            },
        )
        route_payload = json.loads(route_output)
        assert route_payload["decision"]["selected_executor"] == "local.regex_extractor"

        probe_output = await manager.call_tool("slm_harness", "slm_codex_probe", {})
        probe_payload = json.loads(probe_output)
        assert probe_payload["transport"] == "stdio"
        assert "[mcp_servers.slm_harness]" in probe_payload["codex_config_toml"]
        assert "slm_verify_escalation" in probe_payload["tools"]

        accepted_output = await manager.call_tool(
            "slm_harness",
            "slm_verify_escalation",
            {
                "task": "Extract email admin@example.com",
                "task_type": "extract",
                "output": '{"emails": ["admin@example.com"]}',
                "confidence": 0.92,
                "logs": "1 passed",
            },
        )
        accepted_payload = json.loads(accepted_output)
        assert accepted_payload["decision"]["accepted"] is True
        assert accepted_payload["decision"]["escalate"] is False

        escalate_output = await manager.call_tool(
            "slm_harness",
            "slm_verify_escalation",
            {
                "task": "Run tests",
                "task_type": "tool",
                "output": "pytest output",
                "confidence": 0.9,
                "logs": "Traceback: SyntaxError: invalid syntax",
            },
        )
        escalate_payload = json.loads(escalate_output)
        assert escalate_payload["decision"]["accepted"] is False
        assert escalate_payload["decision"]["escalate"] is True
    finally:
        await manager.close()
