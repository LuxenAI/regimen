"""Local-first orchestration engine."""

from __future__ import annotations

from typing import Any

from openharness.orchestration.decomposer import WorkflowDecomposer
from openharness.orchestration.registry import ExecutorRegistry, build_default_executor_registry
from openharness.orchestration.router import CheapestReliableRouter
from openharness.orchestration.telemetry import InMemoryTraceStore
from openharness.orchestration.types import (
    ExecutorResult,
    OrchestrationTrace,
    RouteDecision,
    Subtask,
    TaskContext,
    TaskType,
)
from openharness.orchestration.verifier import ResultVerifier


class OrchestrationEngine:
    """Decompose, route, execute, verify, escalate, and trace subtasks."""

    def __init__(
        self,
        *,
        registry: ExecutorRegistry | None = None,
        decomposer: WorkflowDecomposer | None = None,
        router: CheapestReliableRouter | None = None,
        verifier: ResultVerifier | None = None,
        trace_store: InMemoryTraceStore | None = None,
    ) -> None:
        self.registry = registry or build_default_executor_registry()
        self.decomposer = decomposer or WorkflowDecomposer()
        self.router = router or CheapestReliableRouter()
        self.verifier = verifier or ResultVerifier()
        self.trace_store = trace_store or InMemoryTraceStore()

    def list_executors(self) -> list[dict[str, Any]]:
        """Return executor profiles as JSON-ready dictionaries."""
        profiles: list[dict[str, Any]] = []
        for profile in self.registry.profiles():
            profiles.append(dict(profile.model_dump(mode="json")))
        return profiles

    def decompose(
        self,
        goal: str,
        *,
        task_type: TaskType | None = None,
        context: TaskContext | None = None,
    ) -> list[Subtask]:
        """Decompose a goal into typed subtasks."""
        return self.decomposer.decompose(goal, task_type=task_type, context=context)

    def route_subtask(
        self,
        subtask: Subtask,
        *,
        min_reliability: float = 0.7,
        max_cost_usd: float | None = None,
        exclude: set[str] | None = None,
    ) -> RouteDecision:
        """Route a prepared subtask."""
        return self.router.route(
            subtask,
            self.registry,
            min_reliability=min_reliability,
            max_cost_usd=max_cost_usd,
            exclude=exclude,
        )

    def route_goal(
        self,
        goal: str,
        *,
        task_type: TaskType | None = None,
        min_reliability: float = 0.7,
        max_cost_usd: float | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> tuple[Subtask, RouteDecision]:
        """Decompose a goal and route the first subtask."""
        context = self._context(goal, context_data)
        subtasks = self.decompose(goal, task_type=task_type, context=context)
        if not subtasks:
            raise ValueError("Cannot route an empty goal.")
        decision = self.route_subtask(
            subtasks[0],
            min_reliability=min_reliability,
            max_cost_usd=max_cost_usd,
        )
        return subtasks[0], decision

    async def run_goal(
        self,
        goal: str,
        *,
        task_type: TaskType | None = None,
        min_reliability: float = 0.7,
        verify: bool = True,
        max_escalations: int = 1,
        context_data: dict[str, Any] | None = None,
    ) -> OrchestrationTrace:
        """Run all decomposed subtasks and store a trace."""
        context = self._context(goal, context_data)
        trace = OrchestrationTrace(trace_id=context.trace_id, root_goal=goal)
        subtasks = self.decompose(goal, task_type=task_type, context=context)
        trace.subtasks = subtasks
        trace.add_event("decompose", {"subtasks": [subtask.model_dump(mode="json") for subtask in subtasks]})

        for subtask in subtasks:
            await self._run_subtask(
                subtask,
                context,
                trace,
                min_reliability=min_reliability,
                verify=verify,
                max_escalations=max_escalations,
            )

        trace.recompute_metrics()
        trace.add_event("complete", {"metrics": trace.metrics.model_dump(mode="json")})
        self.trace_store.add(trace)
        return trace

    async def _run_subtask(
        self,
        subtask: Subtask,
        context: TaskContext,
        trace: OrchestrationTrace,
        *,
        min_reliability: float,
        verify: bool,
        max_escalations: int,
    ) -> None:
        excluded: set[str] = set()
        attempts = 0
        while True:
            decision = self.route_subtask(
                subtask,
                min_reliability=min_reliability,
                exclude=excluded,
            )
            trace.decisions.append(decision)
            trace.add_event("route", decision.model_dump(mode="json"))

            result = await self._execute_decision(subtask, context, decision)
            trace.results.append(result)
            trace.add_event("execute", result.model_dump(mode="json"))

            if not verify:
                return
            verification = self.verifier.verify(subtask, result)
            trace.verifications.append(verification)
            trace.add_event("verify", verification.model_dump(mode="json"))
            if verification.accepted or attempts >= max_escalations or result.escalated:
                return

            attempts += 1
            excluded.add(decision.selected_executor)
            trace.add_event(
                "escalate",
                {
                    "subtask_id": subtask.id,
                    "from_executor": decision.selected_executor,
                    "reason": verification.reason,
                    "attempt": attempts,
                },
            )

    async def _execute_decision(
        self,
        subtask: Subtask,
        context: TaskContext,
        decision: RouteDecision,
    ) -> ExecutorResult:
        executor = self.registry.require(decision.selected_executor)
        return await executor.execute(subtask, context)

    def _context(self, goal: str, context_data: dict[str, Any] | None) -> TaskContext:
        shared = dict(context_data or {})
        budget = shared.get("budget_usd")
        return TaskContext(
            root_goal=goal,
            shared=shared,
            budget_usd=float(budget) if isinstance(budget, (int, float)) else None,
        )


def build_default_orchestration_engine() -> OrchestrationEngine:
    """Return a production-shaped default engine with local placeholder executors."""
    return OrchestrationEngine()
