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


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
