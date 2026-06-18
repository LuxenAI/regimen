#!/usr/bin/env python3
"""Benchmark rule-based and trained remaining subroutine classifiers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_remaining_subroutines import (  # noqa: E402
    TASK_SPECS,
    Example,
    classification_metrics,
    load_examples,
)

_SUBROUTINE_MODELS_PATH = SRC_ROOT / "openharness" / "orchestration" / "subroutine_models.py"
_SUBROUTINE_SPEC = importlib.util.spec_from_file_location("_slm_subroutine_models", _SUBROUTINE_MODELS_PATH)
if _SUBROUTINE_SPEC is None or _SUBROUTINE_SPEC.loader is None:
    raise RuntimeError(f"Could not load {_SUBROUTINE_MODELS_PATH}")
_SUBROUTINE_MODULE = importlib.util.module_from_spec(_SUBROUTINE_SPEC)
sys.modules[_SUBROUTINE_SPEC.name] = _SUBROUTINE_MODULE
_SUBROUTINE_SPEC.loader.exec_module(_SUBROUTINE_MODULE)
classify_failure_rules = _SUBROUTINE_MODULE.classify_failure_rules
classify_patch_risk_rules = _SUBROUTINE_MODULE.classify_patch_risk_rules


class TransformersClassifier:
    """Lazy Hugging Face sequence-classifier wrapper."""

    def __init__(self, model_dir: Path, *, device: str | None = None, max_length: int = 512) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for trained classifier benchmarks") from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**encoded).logits, dim=-1)[0]
        score, index = self.torch.max(probs, dim=-1)
        label = self.model.config.id2label.get(int(index.item()), str(int(index.item())))
        return str(label), float(score.item())


def main() -> None:
    args = parse_args()
    report = {
        "failure_classify": benchmark_task(
            "failure_classify",
            args.cases_jsonl,
            args.samples,
            args.failure_model_dir,
            args.device,
        ),
        "patch_risk": benchmark_task(
            "patch_risk",
            args.cases_jsonl,
            args.samples,
            args.patch_risk_model_dir,
            args.device,
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def benchmark_task(
    task_type: str,
    cases_jsonl: Path | None,
    samples: int,
    model_dir: Path | None,
    device: str | None,
) -> dict[str, Any]:
    spec = TASK_SPECS[task_type]
    examples = load_examples(cases_jsonl, samples, task_type)
    backends: dict[str, Any] = {"rules": None}
    if model_dir:
        backends["trained"] = TransformersClassifier(model_dir, device=device, max_length=spec.max_length)

    results: dict[str, Any] = {"examples": len(examples), "labels": spec.labels, "backends": {}}
    for name, backend in backends.items():
        predictions: list[str] = []
        confidences: list[float] = []
        latencies_ms: list[float] = []
        rows: list[dict[str, Any]] = []
        for example in examples:
            started = time.perf_counter()
            label, confidence = predict(name, backend, task_type, example)
            latency_ms = (time.perf_counter() - started) * 1000.0
            predictions.append(label)
            confidences.append(confidence)
            latencies_ms.append(latency_ms)
            rows.append(
                {
                    "expected": example.label,
                    "predicted": label,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "correct": label == example.label,
                    "source": example.source,
                }
            )
        metrics = classification_metrics([example.label for example in examples], predictions, spec.labels)
        metrics["confidence_mean"] = statistics.mean(confidences) if confidences else 0.0
        metrics["latency_ms"] = {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "mean": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        }
        metrics["rows"] = rows
        results["backends"][name] = metrics
    return results


def predict(name: str, backend: Any, task_type: str, example: Example) -> tuple[str, float]:
    if name == "trained":
        return backend.predict(example.text)
    if task_type == "failure_classify":
        output = classify_failure_rules(example.text)
        return str(output["category"]), float(output["confidence"])
    if task_type == "patch_risk":
        output = classify_patch_risk_rules(example.text)
        return str(output["risk_level"]), float(output["confidence"])
    raise KeyError(task_type)


def percentile(values: list[float], frac: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * frac))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-jsonl", type=Path)
    parser.add_argument("--failure-model-dir", type=Path)
    parser.add_argument("--patch-risk-model-dir", type=Path)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--device")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
