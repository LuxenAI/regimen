#!/usr/bin/env python3
"""Local smoke path for research-derived MCP subroutine executors."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openharness.orchestration.mcp_server import _run_subroutine_tool  # noqa: E402


async def main() -> None:
    cases = [
        ("Repair JSON", "json_repair", {"raw": "{'action': 'SEARCH', 'pattern': 'foo'}"}),
        (
            "Localize traceback",
            "trace_localize",
            {
                "project_prefix": "/workspace/app/",
                "traceback": "Traceback (most recent call last):\n  File \"/workspace/app/pkg/service.py\", line 42, in handle\n    user.name.lower()\nAttributeError: 'NoneType' object has no attribute 'name'",
            },
        ),
        ("Generate search queries", "search_query", {"task": "Find the retry budget helper"}),
        (
            "Rank search hits",
            "search_rank",
            {
                "query": "retry_budget",
                "hits": [
                    {"id": "import", "path": "pkg/a.py", "line": 1, "text": "from pkg.service import retry_budget"},
                    {"id": "def", "path": "pkg/service.py", "line": 20, "text": "def retry_budget(config):"},
                ],
            },
        ),
    ]
    results = []
    for goal, task_type, shared in cases:
        results.append(await _run_subroutine_tool(goal, task_type, shared))
    print(json.dumps({"ok": all(item["result"]["output"].get("success") for item in results), "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
