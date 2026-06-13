"""Tests for the verifier/escalation classifier contract."""

from __future__ import annotations

import pytest

from openharness.orchestration.executors import VerifierEscalationClassifierExecutor
from openharness.orchestration.types import Subtask, TaskContext
from openharness.orchestration.verifier import ResultVerifier
from openharness.orchestration.verifier_model import HeuristicVerifierModel, VerifierInput


def test_heuristic_verifier_accepts_good_result() -> None:
    model = HeuristicVerifierModel()

    prediction = model.predict(
        VerifierInput(
            task="Extract an email address",
            task_type="extract",
            executor_output={"emails": ["ops@example.com"]},
            executor_confidence=0.91,
            logs="1 passed",
        )
    )

    assert prediction.accepted is True
    assert prediction.escalate is False
    assert prediction.confidence >= 0.9


def test_heuristic_verifier_escalates_error_logs() -> None:
    model = HeuristicVerifierModel()

    prediction = model.predict(
        VerifierInput(
            task="Run the tests",
            task_type="tool",
            executor_output="pytest output",
            executor_confidence=0.95,
            logs="Traceback: ModuleNotFoundError: No module named pytest",
        )
    )

    assert prediction.accepted is False
    assert prediction.escalate is True
    assert prediction.label == "escalate_error"


def test_heuristic_verifier_escalates_low_confidence_result() -> None:
    model = HeuristicVerifierModel()

    prediction = model.predict(
        VerifierInput(
            task="Classify a failure",
            task_type="classify",
            executor_output={"label": "type_error"},
            executor_confidence=0.41,
        )
    )

    assert prediction.accepted is False
    assert prediction.escalate is True
    assert prediction.label == "escalate_low_confidence"


@pytest.mark.asyncio
async def test_verifier_escalation_executor_emits_structured_decision() -> None:
    executor = VerifierEscalationClassifierExecutor(model=HeuristicVerifierModel())
    subtask = Subtask(goal="Verify extraction", input="Extract email", task_type="verify")
    context = TaskContext(
        root_goal="Extract email",
        shared={
            "candidate_task_type": "extract",
            "candidate_output": {"emails": ["admin@example.com"]},
            "executor_confidence": 0.9,
            "logs": "1 passed",
        },
    )

    result = await executor.execute(subtask, context)
    verification = ResultVerifier().verify(subtask, result)

    assert result.output["accepted"] is True
    assert result.output["escalate"] is False
    assert verification.accepted is True


@pytest.mark.asyncio
async def test_verifier_escalation_executor_marks_escalation() -> None:
    executor = VerifierEscalationClassifierExecutor(model=HeuristicVerifierModel())
    subtask = Subtask(goal="Verify patch", input="Apply a patch", task_type="verify")
    context = TaskContext(
        root_goal="Apply a patch",
        shared={
            "candidate_task_type": "code",
            "candidate_output": "Patch applied",
            "executor_confidence": 0.87,
            "diff": "subprocess.run(command, shell=True)",
        },
    )

    result = await executor.execute(subtask, context)
    verification = ResultVerifier().verify(subtask, result)

    assert result.output["accepted"] is False
    assert result.output["escalate"] is True
    assert result.escalated is True
    assert verification.accepted is False
