"""Trace storage for orchestration telemetry."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from openharness.orchestration.types import OrchestrationTrace


class InMemoryTraceStore:
    """Small bounded trace store used by the MCP server."""

    def __init__(self, *, limit: int = 100) -> None:
        self._limit = limit
        self._traces: OrderedDict[str, OrchestrationTrace] = OrderedDict()

    def add(self, trace: OrchestrationTrace) -> None:
        """Store a trace and evict the oldest trace when over capacity."""
        self._traces[trace.trace_id] = trace
        self._traces.move_to_end(trace.trace_id)
        while len(self._traces) > self._limit:
            self._traces.popitem(last=False)

    def get(self, trace_id: str) -> OrchestrationTrace | None:
        """Return one trace by id."""
        return self._traces.get(trace_id)

    def recent(self, *, limit: int = 5) -> list[OrchestrationTrace]:
        """Return recent traces, newest first."""
        return list(reversed(list(self._traces.values())))[0:limit]


class JsonlTraceSink:
    """Append-only JSONL trace sink for offline eval analysis."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, trace: OrchestrationTrace) -> None:
        """Append one trace as a JSON line."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.model_dump(mode="json"), sort_keys=True) + "\n")
