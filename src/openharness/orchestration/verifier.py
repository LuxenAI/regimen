"""Result verification for routed subtasks."""

from __future__ import annotations

from typing import Any

from openharness.orchestration.types import ExecutorResult, Subtask, VerificationResult


class ResultVerifier:
    """Confidence and shape checks for executor results."""

    def __init__(self, *, threshold: float = 0.65) -> None:
        self.threshold = threshold

    def verify(
        self,
        subtask: Subtask,
        result: ExecutorResult,
        *,
        threshold: float | None = None,
    ) -> VerificationResult:
        """Return an accept/reject decision for a result."""
        effective_threshold = self.threshold if threshold is None else threshold
        structured = _structured_prediction(result.output)
        if structured is not None:
            confidence = _bounded_float(structured.get("confidence"), result.confidence)
            escalate = bool(structured.get("escalate", False))
            accepted = bool(structured.get("accepted", False)) and not escalate
            if confidence < effective_threshold:
                accepted = False
            return VerificationResult(
                subtask_id=subtask.id,
                accepted=accepted,
                confidence=confidence,
                threshold=effective_threshold,
                verifier=str(structured.get("source") or result.executor_name),
                reason=(
                    str(structured.get("reason") or structured.get("label") or "structured_verifier")
                    if accepted
                    else str(
                        structured.get("reason")
                        or structured.get("label")
                        or "structured_verifier_rejected"
                    )
                ),
            )
        if result.error:
            return VerificationResult(
                subtask_id=subtask.id,
                accepted=False,
                confidence=0.0,
                threshold=effective_threshold,
                verifier="local.result_verifier",
                reason=f"executor_error: {result.error}",
            )
        if result.escalated:
            return VerificationResult(
                subtask_id=subtask.id,
                accepted=False,
                confidence=result.confidence,
                threshold=effective_threshold,
                verifier="local.result_verifier",
                reason="frontier_handoff_required",
            )
        has_output = result.output not in ("", None, {}, [])
        accepted = has_output and result.confidence >= effective_threshold
        return VerificationResult(
            subtask_id=subtask.id,
            accepted=accepted,
            confidence=result.confidence,
            threshold=effective_threshold,
            verifier="local.result_verifier",
            reason=(
                "confidence_and_output_shape_passed"
                if accepted
                else "low_confidence_or_empty_output"
            ),
        )


def _structured_prediction(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, dict):
        return None
    required = {"accepted", "confidence", "escalate"}
    if required.issubset(output.keys()):
        return output
    return None


def _bounded_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default
