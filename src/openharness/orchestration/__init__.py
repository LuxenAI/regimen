"""Local-first orchestration layer for slm-harness."""

from openharness.orchestration.engine import OrchestrationEngine, build_default_orchestration_engine
from openharness.orchestration.registry import ExecutorRegistry, build_default_executor_registry
from openharness.orchestration.types import (
    ExecutorProfile,
    ExecutorResult,
    OrchestrationTrace,
    RouteDecision,
    Subtask,
    TaskContext,
    VerificationResult,
)

__all__ = [
    "ExecutorProfile",
    "ExecutorRegistry",
    "ExecutorResult",
    "OrchestrationEngine",
    "OrchestrationTrace",
    "RouteDecision",
    "Subtask",
    "TaskContext",
    "VerificationResult",
    "build_default_executor_registry",
    "build_default_orchestration_engine",
]
