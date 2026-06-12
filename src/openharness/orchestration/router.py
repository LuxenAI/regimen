"""Cheapest-reliable executor routing."""

from __future__ import annotations

from openharness.orchestration.registry import ExecutorRegistry
from openharness.orchestration.types import RouteCandidate, RouteDecision, Subtask


class NoRouteError(Exception):
    """Raised when no executor can handle a subtask."""


class CheapestReliableRouter:
    """Pick the cheapest executor that clears a reliability threshold."""

    def route(
        self,
        subtask: Subtask,
        registry: ExecutorRegistry,
        *,
        min_reliability: float = 0.7,
        max_cost_usd: float | None = None,
        exclude: set[str] | None = None,
    ) -> RouteDecision:
        """Route one subtask to an executor."""
        profiles = registry.candidates_for(subtask.task_type, exclude=exclude)
        if max_cost_usd is not None:
            profiles = [profile for profile in profiles if profile.cost_per_call_usd <= max_cost_usd]
        candidates = [
            RouteCandidate(
                executor_name=profile.name,
                kind=profile.kind,
                reliability=profile.reliability,
                expected_cost_usd=profile.cost_per_call_usd,
                expected_latency_ms=profile.p50_latency_ms,
                local=profile.local,
                reliable=profile.reliability >= min_reliability,
                reason=(
                    "meets reliability threshold"
                    if profile.reliability >= min_reliability
                    else "below reliability threshold"
                ),
            )
            for profile in profiles
        ]
        if not candidates:
            raise NoRouteError(f"No executor supports task type: {subtask.task_type}")

        reliable = [candidate for candidate in candidates if candidate.reliable]
        pool = reliable or candidates
        pool.sort(
            key=lambda candidate: (
                candidate.expected_cost_usd,
                candidate.expected_latency_ms,
                not candidate.local,
                -candidate.reliability,
                candidate.executor_name,
            )
        )
        selected = pool[0]
        reason = (
            "selected cheapest executor above reliability threshold"
            if selected.reliable
            else "no executor met reliability threshold; selected best available fallback"
        )
        if selected.kind == "frontier_llm":
            reason = "selected frontier executor because cheaper local candidates were not reliable enough"
        return RouteDecision(
            subtask_id=subtask.id,
            selected_executor=selected.executor_name,
            selected_kind=selected.kind,
            should_escalate=selected.kind == "frontier_llm" or not selected.local,
            reason=reason,
            candidates=sorted(
                candidates,
                key=lambda candidate: (
                    not candidate.reliable,
                    candidate.expected_cost_usd,
                    candidate.expected_latency_ms,
                    candidate.executor_name,
                ),
            ),
        )
