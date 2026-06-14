"""Executor registry for orchestration."""

from __future__ import annotations

from openharness.orchestration.executors import (
    BaseExecutor,
    CodeHeuristicExecutor,
    FailureClassifierSlmExecutor,
    FrontierHandoffExecutor,
    HeuristicVerifierExecutor,
    JsonRepairSlmExecutor,
    PatchRiskClassifierSlmExecutor,
    KeywordClassifierExecutor,
    RegexExtractorExecutor,
    SearchHitRankerSlmExecutor,
    SearchQueryGenSlmExecutor,
    TinySlmStubExecutor,
    TraceLocalizerSlmExecutor,
    VerifierEscalationClassifierExecutor,
)
from openharness.orchestration.types import ExecutorProfile, TaskType


class ExecutorRegistry:
    """In-memory registry of available subtask executors."""

    def __init__(self) -> None:
        self._executors: dict[str, BaseExecutor] = {}

    def register(self, executor: BaseExecutor) -> None:
        """Register or replace an executor."""
        self._executors[executor.profile.name] = executor

    def get(self, name: str) -> BaseExecutor | None:
        """Return an executor by name."""
        return self._executors.get(name)

    def require(self, name: str) -> BaseExecutor:
        """Return an executor or raise when it is missing."""
        executor = self.get(name)
        if executor is None:
            raise KeyError(f"Unknown executor: {name}")
        return executor

    def profiles(self) -> list[ExecutorProfile]:
        """Return all executor profiles sorted by name."""
        return [self._executors[name].profile for name in sorted(self._executors)]

    def candidates_for(self, task_type: TaskType, *, exclude: set[str] | None = None) -> list[ExecutorProfile]:
        """Return profiles that can handle the task type."""
        excluded = exclude or set()
        return [
            profile
            for profile in self.profiles()
            if profile.name not in excluded and profile.supports(task_type)
        ]


def build_default_executor_registry() -> ExecutorRegistry:
    """Build the default local-first executor registry."""
    registry = ExecutorRegistry()
    registry.register(KeywordClassifierExecutor())
    registry.register(RegexExtractorExecutor())
    registry.register(JsonRepairSlmExecutor())
    registry.register(TraceLocalizerSlmExecutor())
    registry.register(SearchQueryGenSlmExecutor())
    registry.register(SearchHitRankerSlmExecutor())
    registry.register(FailureClassifierSlmExecutor())
    registry.register(PatchRiskClassifierSlmExecutor())
    registry.register(VerifierEscalationClassifierExecutor())
    registry.register(HeuristicVerifierExecutor())
    registry.register(TinySlmStubExecutor())
    registry.register(CodeHeuristicExecutor())
    registry.register(FrontierHandoffExecutor())
    return registry
