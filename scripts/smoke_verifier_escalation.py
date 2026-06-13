#!/usr/bin/env python3
"""Smoke-test the verifier/escalation classifier path."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openharness.orchestration.executors import VerifierEscalationClassifierExecutor  # noqa: E402
from openharness.orchestration.types import Subtask, TaskContext  # noqa: E402
from openharness.orchestration.verifier import ResultVerifier  # noqa: E402
from openharness.orchestration.verifier_model import (  # noqa: E402
    HeuristicVerifierModel,
    TransformersVerifierModel,
    VerifierClassifier,
    VerifierInput,
)


@dataclass(frozen=True)
class SmokeCase:
    name: str
    item: VerifierInput
    expected_escalate: bool


def main() -> None:
    args = parse_args()
    model = build_model(args.model_dir)
    cases = build_cases()
    failures: list[str] = []
    predictions: dict[str, dict[str, object]] = {}

    for case in cases:
        prediction = model.predict(case.item)
        predictions[case.name] = prediction.model_dump(mode="json")
        if prediction.escalate != case.expected_escalate:
            failures.append(
                f"{case.name}: expected escalate={case.expected_escalate}, got {prediction.escalate}"
            )

    executor_payload = asyncio.run(run_executor_smoke(model))
    if executor_payload["verification"]["accepted"] is not True:
        failures.append("executor path did not accept a good extraction result")

    report = {
        "model": args.model_dir or "heuristic",
        "predictions": predictions,
        "executor_path": executor_payload,
        "ok": not failures,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


def build_model(model_dir: str | None) -> VerifierClassifier:
    if model_dir:
        return TransformersVerifierModel(model_dir)
    env_model_dir = os.getenv("SLM_HARNESS_VERIFIER_MODEL_DIR")
    if env_model_dir:
        return TransformersVerifierModel(env_model_dir)
    return HeuristicVerifierModel()


async def run_executor_smoke(model: VerifierClassifier) -> dict[str, object]:
    executor = VerifierEscalationClassifierExecutor(model=model)
    subtask = Subtask(goal="Verify extraction", input="Extract email", task_type="verify")
    context = TaskContext(
        root_goal="Extract email",
        shared={
            "candidate_task_type": "extract",
            "candidate_output": {"emails": ["ops@example.com"]},
            "executor_confidence": 0.91,
            "logs": "1 passed",
        },
    )
    result = await executor.execute(subtask, context)
    verification = ResultVerifier().verify(subtask, result)
    return {
        "result": result.model_dump(mode="json"),
        "verification": verification.model_dump(mode="json"),
    }


def build_cases() -> list[SmokeCase]:
    return [
        SmokeCase(
            name="accepted_extraction",
            item=VerifierInput(
                task="Extract email",
                task_type="extract",
                executor_output={"emails": ["admin@example.com"]},
                executor_confidence=0.91,
                logs="1 passed",
            ),
            expected_escalate=False,
        ),
        SmokeCase(
            name="empty_output",
            item=VerifierInput(
                task="Classify CI failure",
                task_type="classify",
                executor_output="",
                executor_confidence=0.8,
            ),
            expected_escalate=True,
        ),
        SmokeCase(
            name="traceback",
            item=VerifierInput(
                task="Run tests",
                task_type="tool",
                executor_output="pytest output",
                executor_confidence=0.95,
                logs="Traceback: ModuleNotFoundError: No module named fastapi",
            ),
            expected_escalate=True,
        ),
        SmokeCase(
            name="low_confidence",
            item=VerifierInput(
                task="Route subtask",
                task_type="route",
                executor_output={"task_type": "unknown"},
                executor_confidence=0.35,
            ),
            expected_escalate=True,
        ),
        SmokeCase(
            name="risky_diff_with_tests",
            item=VerifierInput(
                task="Patch auth middleware",
                task_type="code",
                executor_output="Patch added and tests pass",
                executor_confidence=0.89,
                diff="auth middleware update",
                logs="12 passed, 0 failed",
            ),
            expected_escalate=False,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
