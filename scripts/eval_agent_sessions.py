#!/usr/bin/env python3
"""Run agent-session evals with replay, Codex CLI, or Claude Code CLI drivers."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openharness.evals.agent_drivers import (  # noqa: E402
    AgentDriver,
    AgentSessionTrace,
    ClaudeCodeCliDriver,
    CodexCliDriver,
    ReplayDriver,
)
from openharness.evals.harness import (  # noqa: E402
    EvalScenario,
    load_jsonl_scenarios,
    summarize_traces,
    write_traces_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", choices=["replay", "codex", "claude"], default="replay")
    parser.add_argument(
        "--fixture",
        default=str(ROOT / "evals" / "fixtures" / "agent_sessions.jsonl"),
        help="JSONL scenario file",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "evals" / "agent_session_traces.jsonl"),
        help="JSONL trace output path",
    )
    parser.add_argument("--no-mcp", action="store_true", help="Run baseline without MCP enabled")
    args = parser.parse_args()

    scenarios = load_jsonl_scenarios(args.fixture)
    driver = _driver(args.driver)
    traces = asyncio.run(_run(driver, scenarios, use_mcp=not args.no_mcp))
    write_traces_jsonl(traces, args.output)
    summary = summarize_traces(traces)
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "summary": summary}, indent=2, sort_keys=True))


def _driver(name: str) -> AgentDriver:
    if name == "codex":
        return CodexCliDriver()
    if name == "claude":
        return ClaudeCodeCliDriver()
    return ReplayDriver()


async def _run(
    driver: AgentDriver,
    scenarios: list[EvalScenario],
    *,
    use_mcp: bool,
) -> list[AgentSessionTrace]:
    traces: list[AgentSessionTrace] = []
    for scenario in scenarios:
        traces.append(await driver.run(scenario.to_agent_spec(use_mcp=use_mcp)))
    return traces


if __name__ == "__main__":
    main()
