#!/usr/bin/env python3
"""Benchmark verifier/escalation backends on labeled harness cases."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openharness.orchestration.verifier_model import (  # noqa: E402
    HeuristicVerifierModel,
    TransformersVerifierModel,
    VerifierClassifier,
    VerifierInput,
    VerifierPrediction,
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    item: VerifierInput
    expected_escalate: bool
    expected_label: str


class QwenVerifierJudge:
    """Generative Qwen judge adapted to the verifier classifier protocol."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_new_tokens: int = 96,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for Qwen benchmarking") from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, item: VerifierInput) -> VerifierPrediction:
        prompt = build_qwen_prompt(item)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict verifier/escalation classifier for coding-agent "
                    "tool results. Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self._render_prompt(messages)
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0][encoded["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        payload = parse_json_object(response)
        accepted = bool(payload.get("accepted", False))
        escalate = bool(payload.get("escalate", not accepted))
        confidence = bounded_float(payload.get("confidence"), 0.0)
        label = str(payload.get("label") or ("accept" if accepted and not escalate else "escalate"))
        reason = str(payload.get("reason") or response.strip()[:160] or "qwen_judge")
        return VerifierPrediction(
            accepted=accepted and not escalate,
            confidence=confidence,
            escalate=escalate or not accepted,
            label=label,
            reason=reason,
            source=f"qwen:{self.model_name}",
            metadata={"device": self.device, "raw_response": response.strip()},
        )

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered = "".join(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                for message in messages
            )
            rendered += "<|im_start|>assistant\n"
        return str(rendered)


def main() -> None:
    args = parse_args()
    cases = build_cases()
    if args.limit:
        cases = cases[: args.limit]

    backends: list[tuple[str, VerifierClassifier]] = [("heuristic", HeuristicVerifierModel())]
    if args.tiny_model_dir:
        backends.append(("tiny_verifier", TransformersVerifierModel(args.tiny_model_dir)))
    if args.qwen_model:
        backends.append(("qwen_judge", QwenVerifierJudge(args.qwen_model, device=args.device)))

    report = {
        "cases": len(cases),
        "backends": {
            name: run_backend(name, backend, cases, warmup=args.warmup)
            for name, backend in backends
        },
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def run_backend(
    name: str,
    backend: VerifierClassifier,
    cases: list[BenchmarkCase],
    *,
    warmup: int,
) -> dict[str, Any]:
    for case in cases[:warmup]:
        try:
            backend.predict(case.item)
        except Exception:
            pass

    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for case in cases:
        started = time.perf_counter()
        try:
            prediction = backend.predict(case.item)
            error = None
        except Exception as exc:
            prediction = VerifierPrediction(
                accepted=False,
                confidence=0.0,
                escalate=True,
                label="backend_error",
                reason=str(exc),
                source=name,
            )
            error = str(exc)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(latency_ms)
        correct = prediction.escalate == case.expected_escalate
        false_accept = not prediction.escalate and case.expected_escalate
        false_escalation = prediction.escalate and not case.expected_escalate
        rows.append(
            {
                "case": case.name,
                "expected_escalate": case.expected_escalate,
                "expected_label": case.expected_label,
                "predicted_escalate": prediction.escalate,
                "predicted_label": prediction.label,
                "confidence": prediction.confidence,
                "latency_ms": latency_ms,
                "correct": correct,
                "false_accept": false_accept,
                "false_escalation": false_escalation,
                "reason": prediction.reason,
                "error": error,
            }
        )

    correct_count = sum(1 for row in rows if row["correct"])
    false_accepts = sum(1 for row in rows if row["false_accept"])
    false_escalations = sum(1 for row in rows if row["false_escalation"])
    return {
        "accuracy": correct_count / len(rows) if rows else 0.0,
        "correct": correct_count,
        "total": len(rows),
        "false_accepts": false_accepts,
        "false_escalations": false_escalations,
        "escalation_rate": sum(1 for row in rows if row["predicted_escalate"]) / len(rows)
        if rows
        else 0.0,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "max": max(latencies_ms) if latencies_ms else 0.0,
        },
        "rows": rows,
    }


def build_qwen_prompt(item: VerifierInput) -> str:
    return (
        "Decide if the executor result should be accepted or escalated to a stronger "
        "coding agent. Output exactly one JSON object with keys accepted, confidence, "
        "escalate, label, reason.\n\n"
        "Labels: accept, escalate_empty_output, escalate_error, escalate_low_confidence, "
        "escalate_risky_diff, escalate_visual_failure, escalate_incomplete.\n\n"
        "Rules:\n"
        "- Escalate empty outputs.\n"
        "- Escalate traceback, compiler, test, dependency, or runtime errors.\n"
        "- Escalate low executor confidence below 0.65.\n"
        "- Escalate risky code diffs unless logs show passing tests.\n"
        "- Accept only when there is useful output and no failure signal.\n\n"
        f"Verifier input:\n{item.to_model_text(max_chars=4000)}"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            "accepted_email_extraction",
            VerifierInput(
                task="Extract contact emails",
                task_type="extract",
                executor_name="local.regex_extractor",
                executor_kind="deterministic",
                executor_output={"emails": ["ops@example.com"]},
                executor_confidence=0.92,
                logs="1 passed",
            ),
            False,
            "accept",
        ),
        BenchmarkCase(
            "accepted_url_extraction",
            VerifierInput(
                task="Extract URL from docs",
                task_type="extract",
                executor_output={"urls": ["https://example.com/docs"]},
                executor_confidence=0.88,
            ),
            False,
            "accept",
        ),
        BenchmarkCase(
            "accepted_patch_with_tests",
            VerifierInput(
                task="Patch auth middleware",
                task_type="code",
                executor_output="Patch added and tests pass",
                executor_confidence=0.87,
                diff="auth middleware update",
                logs="12 passed, 0 failed",
            ),
            False,
            "accept",
        ),
        BenchmarkCase(
            "accepted_typecheck",
            VerifierInput(
                task="Run typecheck",
                task_type="tool",
                executor_output="mypy success",
                executor_confidence=0.9,
                logs="Success: no issues found in 15 source files",
            ),
            False,
            "accept",
        ),
        BenchmarkCase(
            "accepted_lint",
            VerifierInput(
                task="Run lint",
                task_type="tool",
                executor_output="All checks passed",
                executor_confidence=0.89,
                logs="All checks passed!",
            ),
            False,
            "accept",
        ),
        BenchmarkCase(
            "accepted_mcp_probe",
            VerifierInput(
                task="Check MCP probe response",
                task_type="verify",
                executor_output={"server": "slm-harness", "transport": "stdio"},
                executor_confidence=0.86,
            ),
            False,
            "accept",
        ),
        BenchmarkCase(
            "empty_string",
            VerifierInput(
                task="Classify CI failure",
                task_type="classify",
                executor_output="",
                executor_confidence=0.82,
            ),
            True,
            "escalate_empty_output",
        ),
        BenchmarkCase(
            "empty_dict",
            VerifierInput(
                task="Extract emails from text",
                task_type="extract",
                executor_output={},
                executor_confidence=0.9,
            ),
            True,
            "escalate_empty_output",
        ),
        BenchmarkCase(
            "traceback_module_missing",
            VerifierInput(
                task="Run pytest",
                task_type="tool",
                executor_output="pytest output",
                executor_confidence=0.95,
                logs="Traceback: ModuleNotFoundError: No module named fastapi",
            ),
            True,
            "escalate_error",
        ),
        BenchmarkCase(
            "syntax_error",
            VerifierInput(
                task="Run Python script",
                task_type="tool",
                executor_output="python output",
                executor_confidence=0.94,
                logs="SyntaxError: invalid syntax",
            ),
            True,
            "escalate_error",
        ),
        BenchmarkCase(
            "type_error",
            VerifierInput(
                task="Run unit tests",
                task_type="tool",
                executor_output="pytest output",
                executor_confidence=0.93,
                logs="TypeError: expected str, got None",
            ),
            True,
            "escalate_error",
        ),
        BenchmarkCase(
            "npm_error",
            VerifierInput(
                task="Build frontend",
                task_type="tool",
                executor_output="npm run build",
                executor_confidence=0.9,
                logs="npm ERR! missing script: build",
            ),
            True,
            "escalate_error",
        ),
        BenchmarkCase(
            "low_confidence_route",
            VerifierInput(
                task="Route unknown workflow",
                task_type="route",
                executor_output={"task_type": "unknown"},
                executor_confidence=0.31,
            ),
            True,
            "escalate_low_confidence",
        ),
        BenchmarkCase(
            "low_confidence_classifier",
            VerifierInput(
                task="Classify failure",
                task_type="classify",
                executor_output={"label": "type_error"},
                executor_confidence=0.44,
            ),
            True,
            "escalate_low_confidence",
        ),
        BenchmarkCase(
            "risky_shell_diff",
            VerifierInput(
                task="Review code patch",
                task_type="code",
                executor_output="Patch created",
                executor_confidence=0.84,
                diff="subprocess.run(user_command, shell=True)",
            ),
            True,
            "escalate_risky_diff",
        ),
        BenchmarkCase(
            "risky_payment_diff",
            VerifierInput(
                task="Review payment patch",
                task_type="code",
                executor_output="Patch updated billing flow",
                executor_confidence=0.86,
                diff="payment token delete path changed",
            ),
            True,
            "escalate_risky_diff",
        ),
        BenchmarkCase(
            "visual_blank",
            VerifierInput(
                task="Check browser screenshot",
                task_type="verify",
                executor_output="Rendered page",
                executor_confidence=0.89,
                screenshot_summary="The page is blank with a loading spinner.",
            ),
            True,
            "escalate_visual_failure",
        ),
        BenchmarkCase(
            "visual_overlap",
            VerifierInput(
                task="Check responsive layout",
                task_type="verify",
                executor_output="Rendered page",
                executor_confidence=0.88,
                screenshot_summary="Main button is clipped and text overlaps.",
            ),
            True,
            "escalate_visual_failure",
        ),
        BenchmarkCase(
            "incomplete_todo",
            VerifierInput(
                task="Summarize implementation",
                task_type="reason",
                executor_output="TODO: wire the final handler",
                executor_confidence=0.8,
            ),
            True,
            "escalate_incomplete",
        ),
        BenchmarkCase(
            "not_implemented",
            VerifierInput(
                task="Implement feature",
                task_type="code",
                executor_output="Placeholder only; not implemented",
                executor_confidence=0.81,
            ),
            True,
            "escalate_incomplete",
        ),
    ]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def bounded_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-model-dir")
    parser.add_argument("--qwen-model", default="")
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
