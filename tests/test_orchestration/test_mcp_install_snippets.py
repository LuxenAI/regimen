"""Tests for Codex and Claude Code MCP install snippets."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from openharness.orchestration.codex import (
    build_claude_mcp_add_command,
    build_claude_mcp_json,
    build_codex_mcp_config_snippet,
)


def test_codex_config_snippet_is_valid_toml(tmp_path: Path) -> None:
    snippet = build_codex_mcp_config_snippet(python=sys.executable, root=tmp_path)
    parsed = tomllib.loads(snippet)

    assert parsed["mcp_servers"]["slm_harness"]["command"] == sys.executable
    assert parsed["mcp_servers"]["slm_harness"]["args"] == [
        "-m",
        "openharness.orchestration.mcp_server",
    ]


def test_claude_mcp_json_and_command_are_valid(tmp_path: Path) -> None:
    snippet = build_claude_mcp_json(python=sys.executable, root=tmp_path)
    parsed = json.loads(snippet)
    server = parsed["mcpServers"]["slm-harness"]

    assert server["type"] == "stdio"
    assert server["command"] == sys.executable
    assert "PYTHONPATH" in server["env"]
    assert build_claude_mcp_add_command(python=sys.executable, root=tmp_path).startswith(
        "claude mcp add"
    )
