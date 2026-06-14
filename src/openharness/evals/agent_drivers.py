"""Agent-session drivers for Codex, Claude Code, and offline replay evals."""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


class AgentSessionSpec(BaseModel):
    """One agent-session eval request."""

    scenario_id: str
    prompt: str
    fixture_path: str | None = None
    use_mcp: bool = True
    timeout_ms: int = Field(default=120_000, ge=1)
    expected_success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSessionTrace(BaseModel):
    """Normalized trace emitted by any agent-session driver."""

    driver: str
    scenario_id: str
    prompt: str
    use_mcp: bool
    success: bool
    wall_time_ms: int
    cost_usd: float = 0.0
    tool_call_count: int = 0
    mcp_call_count: int = 0
    estimated_token_savings: int = 0
    logs: str = ""
    patch: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AgentDriver(Protocol):
    """Runs one eval scenario through an agent surface."""

    name: str

    async def run(self, spec: AgentSessionSpec) -> AgentSessionTrace:
        """Run one eval session."""


class ReplayDriver:
    """Offline deterministic driver for CI without paid agent calls."""

    name = "replay"

    async def run(self, spec: AgentSessionSpec) -> AgentSessionTrace:
        started = time.perf_counter()
        mcp_calls = int(spec.metadata.get("mcp_call_count", 3 if spec.use_mcp else 0))
        token_savings = int(spec.metadata.get("estimated_token_savings", 700 if spec.use_mcp else 0))
        tool_calls = int(spec.metadata.get("tool_call_count", 4 if spec.use_mcp else 1))
        await asyncio.sleep(0)
        return AgentSessionTrace(
            driver=self.name,
            scenario_id=spec.scenario_id,
            prompt=spec.prompt,
            use_mcp=spec.use_mcp,
            success=spec.expected_success,
            wall_time_ms=_elapsed_ms(started),
            tool_call_count=tool_calls,
            mcp_call_count=mcp_calls,
            estimated_token_savings=token_savings,
            logs="offline replay completed",
            metrics={
                "fixture_path": spec.fixture_path,
                "baseline": "mcp" if spec.use_mcp else "no_mcp",
            },
        )


class CodexCliDriver:
    """Driver for real Codex CLI sessions when available."""

    name = "codex_cli"

    def __init__(self, command: str | None = None, args: tuple[str, ...] | None = None) -> None:
        self.command: str = command if command is not None else os.getenv("SLM_HARNESS_CODEX_COMMAND", "codex")
        self.args = args or tuple(shlex.split(os.getenv("SLM_HARNESS_CODEX_ARGS", "exec")))

    async def run(self, spec: AgentSessionSpec) -> AgentSessionTrace:
        return await _run_cli_driver(
            name=self.name,
            command=self.command,
            args=(*self.args, spec.prompt),
            spec=spec,
        )


class ClaudeCodeCliDriver:
    """Driver for real Claude Code CLI sessions when available."""

    name = "claude_code_cli"

    def __init__(self, command: str | None = None, args: tuple[str, ...] | None = None) -> None:
        self.command: str = command if command is not None else os.getenv("SLM_HARNESS_CLAUDE_COMMAND", "claude")
        self.args = args or tuple(shlex.split(os.getenv("SLM_HARNESS_CLAUDE_ARGS", "-p")))

    async def run(self, spec: AgentSessionSpec) -> AgentSessionTrace:
        return await _run_cli_driver(
            name=self.name,
            command=self.command,
            args=(*self.args, spec.prompt),
            spec=spec,
        )


async def _run_cli_driver(
    *,
    name: str,
    command: str,
    args: tuple[str, ...],
    spec: AgentSessionSpec,
) -> AgentSessionTrace:
    started = time.perf_counter()
    cwd = Path(spec.fixture_path).expanduser() if spec.fixture_path else None
    env = dict(os.environ)
    env["SLM_HARNESS_EVAL_USE_MCP"] = "1" if spec.use_mcp else "0"
    try:
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=spec.timeout_ms / 1000.0,
        )
    except Exception as exc:
        return AgentSessionTrace(
            driver=name,
            scenario_id=spec.scenario_id,
            prompt=spec.prompt,
            use_mcp=spec.use_mcp,
            success=False,
            wall_time_ms=_elapsed_ms(started),
            error=str(exc),
        )

    logs = (stdout + stderr).decode("utf-8", errors="replace")
    success = process.returncode == 0 and _success_marker(logs, spec)
    return AgentSessionTrace(
        driver=name,
        scenario_id=spec.scenario_id,
        prompt=spec.prompt,
        use_mcp=spec.use_mcp,
        success=success,
        wall_time_ms=_elapsed_ms(started),
        tool_call_count=_count_markers(logs, ("tool", "Tool", "mcp", "MCP")),
        mcp_call_count=_count_markers(logs, ("slm_", "slm-harness", "MCP")),
        estimated_token_savings=0,
        logs=logs[-12_000:],
        metrics={"returncode": process.returncode},
        error=None if success else "agent session did not satisfy success marker",
    )


def _success_marker(logs: str, spec: AgentSessionSpec) -> bool:
    marker = spec.metadata.get("success_marker")
    if isinstance(marker, str) and marker:
        return marker in logs
    return spec.expected_success


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    return sum(text.count(marker) for marker in markers)


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
