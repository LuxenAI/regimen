"""Codex-facing install snippets for the orchestration MCP server."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root for a source checkout."""
    return Path(__file__).resolve().parents[3]


def build_codex_mcp_config_snippet(
    *,
    server_name: str = "slm_harness",
    python: str | None = None,
    root: str | Path | None = None,
) -> str:
    """Return a config.toml snippet for Codex stdio MCP setup."""
    resolved_root = Path(root).expanduser().resolve() if root is not None else repo_root()
    src_dir = resolved_root / "src"
    command = python or sys.executable
    return "\n".join(
        [
            f"[mcp_servers.{server_name}]",
            f'command = "{_toml_escape(command)}"',
            'args = ["-m", "openharness.orchestration.mcp_server"]',
            f'cwd = "{_toml_escape(str(resolved_root))}"',
            "startup_timeout_sec = 10",
            "tool_timeout_sec = 60",
            "",
            f"[mcp_servers.{server_name}.env]",
            f'PYTHONPATH = "{_toml_escape(str(src_dir))}"',
        ]
    )


def build_claude_mcp_json(
    *,
    server_name: str = "slm-harness",
    python: str | None = None,
    root: str | Path | None = None,
) -> str:
    """Return a project-scoped `.mcp.json` snippet for Claude Code stdio setup."""
    resolved_root = Path(root).expanduser().resolve() if root is not None else repo_root()
    command = python or sys.executable
    payload = {
        "mcpServers": {
            server_name: {
                "type": "stdio",
                "command": str(command),
                "args": ["-m", "openharness.orchestration.mcp_server"],
                "cwd": str(resolved_root),
                "env": {"PYTHONPATH": str(resolved_root / "src")},
            }
        }
    }
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def build_claude_mcp_add_command(
    *,
    server_name: str = "slm-harness",
    python: str | None = None,
    root: str | Path | None = None,
) -> str:
    """Return a Claude Code command for local stdio MCP installation."""
    resolved_root = Path(root).expanduser().resolve() if root is not None else repo_root()
    command = python or sys.executable
    return (
        "claude mcp add --transport stdio "
        f"--env PYTHONPATH={_shell_quote(str(resolved_root / 'src'))} "
        f"{_shell_quote(server_name)} -- "
        f"{_shell_quote(str(command))} -m openharness.orchestration.mcp_server"
    )


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _shell_quote(value: str) -> str:
    if not value or any(char.isspace() or char in "\"'\\$`" for char in value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value
