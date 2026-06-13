"""Rule-based workflow decomposition and subtask typing."""

from __future__ import annotations

import re

from openharness.orchestration.types import Subtask, TaskContext, TaskType


_SPLIT_RE = re.compile(r"(?:\n+|\s+\bthen\b\s+|;\s+)", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


class WorkflowDecomposer:
    """Convert a user goal into typed subtasks without making a model call."""

    def decompose(
        self,
        goal: str,
        *,
        task_type: TaskType | None = None,
        context: TaskContext | None = None,
    ) -> list[Subtask]:
        """Return typed subtasks for a goal."""
        normalized_goal = " ".join(goal.split())
        if not normalized_goal:
            return []
        parts = self._split_goal(goal)
        if not parts:
            parts = [normalized_goal]
        subtasks: list[Subtask] = []
        for index, part in enumerate(parts):
            inferred = task_type or self.infer_task_type(part, context=context)
            subtasks.append(
                Subtask(
                    task_type=inferred,
                    goal=part,
                    input=part,
                    priority=index,
                    metadata={"decomposer": "rule_based_v1"},
                )
            )
        return subtasks

    def infer_task_type(self, text: str, *, context: TaskContext | None = None) -> TaskType:
        """Infer a coarse task type from keywords and available context."""
        del context
        lowered = text.lower()
        if _has_any(lowered, ("repair json", "json repair", "malformed json", "broken json")):
            return "json_repair"
        if _has_any(lowered, ("traceback", "stack trace", "localize failure", "culprit frame")):
            return "trace_localize"
        if _has_any(lowered, ("search query", "generate query", "grep pattern")):
            return "search_query"
        if _has_any(lowered, ("rank search", "search hit", "rank hits")):
            return "search_rank"
        if _has_any(lowered, ("extract", "parse", "pull", "field", "email", "url", "json")):
            return "extract"
        if _has_any(lowered, ("classify", "label", "detect", "intent", "sentiment", "route")):
            return "classify"
        if _has_any(lowered, ("verify", "validate", "check", "review", "confirm", "is correct")):
            return "verify"
        if _has_any(lowered, ("code", "function", "bug", "test", "repo", "diff", "implement", "file")):
            return "code"
        if _has_any(lowered, ("search", "browser", "github", "slack", "calendar", "mcp", "tool")):
            return "tool"
        if _has_any(lowered, ("explain", "plan", "decide", "compare", "why", "how", "design")):
            return "reason"
        return "unknown"

    def _split_goal(self, goal: str) -> list[str]:
        raw_parts = _SPLIT_RE.split(goal)
        parts: list[str] = []
        for raw in raw_parts:
            cleaned = _BULLET_RE.sub("", raw).strip()
            if cleaned:
                parts.append(" ".join(cleaned.split()))
        return parts


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
