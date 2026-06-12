"""Result verification for routed subtasks."""

from __future__ import annotations

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
